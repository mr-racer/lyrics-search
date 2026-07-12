import React from 'react';
import * as ReactDOM from 'react-dom/client';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import './index.css';
import { registerSW } from 'virtual:pwa-register';

// PWA service worker: autoUpdate. Требует secure context (ts.net/localhost);
// по plain-HTTP LAN registerSW тихо не сработает — приложение живёт как сайт.
registerSW({ immediate: true });

const { useState, useEffect, useRef, useCallback, useMemo, useLayoutEffect, Fragment } = React;

const API = window.location.protocol === 'file:'
  ? 'http://localhost:8000/api/v1'
  : '/api/v1';

// Basic artist-name → slug. Backend (/artists/{slug}) re-slugifies through
// the canonical _slugify_artist (strips smart quotes, +, &, ., normalizes
// dashes), so this minimal client-side form is enough to resolve.
function slugifyArtistName(name) {
  return (name || '').toLowerCase().trim().split(/\s+/).join('-');
}

// Renders a track's artist credit with each participant individually clickable.
// Uses backend-provided `artist_refs` (aligned {name, slug} per participant) so a
// collaboration like "Calvin Harris, Dua Lipa" links EACH artist to their own
// page — primary first, then "feat." the rest. Falls back to the raw `artist`
// string (navigating to the primary slug) when artist_refs is absent, so older
// or un-enriched surfaces still work and never 404 on a combined slug.
function ArtistCredit({ track, navigateToArtist, lang, color = '#bba8ff', stopProp = true }) {
  const refs = (track && Array.isArray(track.artist_refs)) ? track.artist_refs : [];
  const go = (slug) => (e) => {
    if (stopProp) e.stopPropagation();
    if (slug && navigateToArtist) navigateToArtist(slug);
  };
  const linkStyle = { color, cursor: navigateToArtist ? 'pointer' : 'default' };
  const title = navigateToArtist ? (lang === 'ru' ? 'Открыть страницу артиста' : 'Open artist page') : undefined;
  if (refs.length > 0) {
    return (
      <>
        {refs.map((r, i) => (
          <Fragment key={`${r.slug}-${i}`}>
            {i === 1 ? ' feat. ' : i > 1 ? ', ' : ''}
            <span onClick={go(r.slug)} style={linkStyle} title={title}>{r.name}</span>
          </Fragment>
        ))}
      </>
    );
  }
  const name = (track && track.artist) || '';
  const slug = (track && track.primary_artist_slug) || slugifyArtistName(name);
  return <span onClick={go(slug)} style={linkStyle} title={title}>{name}</span>;
}

// Best-effort primary-artist slug for click-only targets (avatars, chips) where
// there's no room to list individual participants. Prefers the canonical
// per-participant slug so collaborations resolve to a real page, never a
// combined slug that 404s.
function primaryArtistSlug(track) {
  if (track && Array.isArray(track.artist_refs) && track.artist_refs.length) {
    return track.artist_refs[0].slug;
  }
  return (track && track.primary_artist_slug) || slugifyArtistName(track && track.artist);
}

// ─── Phase A: Auth helpers ─────────────────────────────────────────────────
function getStoredToken() {
  return localStorage.getItem('musix_token') || '';
}

function setStoredAuth({ token, user }) {
  if (token) localStorage.setItem('musix_token', token);
  if (user) {
    localStorage.setItem('musix_user_id',    user.id || '');
    localStorage.setItem('musix_user_email', user.email || '');
    localStorage.setItem('musix_user_role',  user.role || '');
    localStorage.setItem('musix_user_premium', user.premium ? '1' : '0');
  }
  // A successful auth ends any 401-kick state: reset the reload-loop guard
  // and drop the legacy '#login' hash older builds left in the URL.
  sessionStorage.removeItem('musix_auth_kick_ts');
  if (window.location.hash === '#login') {
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
}

function clearStoredAuth() {
  localStorage.removeItem('musix_token');
  localStorage.removeItem('musix_user_id');
  localStorage.removeItem('musix_user_email');
  localStorage.removeItem('musix_user_role');
  localStorage.removeItem('musix_user_premium');
  // Drop the cached stream token too — it must not survive a logout into
  // the next account's session (module state, declared below).
  _streamToken = '';
  _streamTokenExpMs = 0;
  // And the prefetched media blobs — another account's audio must not sit in
  // memory (or get promoted to the player) after the account switch.
  dropMediaPrefetch();
}

// ─── Stream token — auth for <audio> URLs ───────────────────────────────────
// Media elements can't send an Authorization header, so /stream URLs carry a
// short-lived scope-limited JWT in ?st= (minted by POST /auth/stream-token).
// Cached in module state; App refreshes it on mount + on an interval well
// inside the server TTL, so buildStreamUrl can stay synchronous.
let _streamToken = '';
let _streamTokenExpMs = 0;

async function refreshStreamToken() {
  if (!getStoredToken()) return;            // not logged in yet — nothing to mint
  try {
    const data = await apiFetch('/auth/stream-token', { method: 'POST' });
    _streamToken = data.token || '';
    _streamTokenExpMs = Date.now() + (data.expires_in || 3600) * 1000;
  } catch (e) {
    console.warn('[stream-token] refresh failed', e);
  }
}

// ─── Next-track media prefetch ───────────────────────────────────────────────
// On a slow connection every track change used to start from byte 0: token
// check, Qdrant lookup, (ffprobe), then the whole download while the user
// hears silence. Instead, once the CURRENT track is buffered to the end
// (canplaythrough — so prefetch never competes with a still-starving stream),
// App downloads the NEXT queue track into a Blob; buildStreamUrl then hands
// the blob: URL to <audio> and the switch is instant and offline-proof.
// At most two blobs are alive: the one playing and the one prefetched.
let _playingBlob = { trackId: null, url: null };
let _nextPrefetch = { trackId: null, blobUrl: null, ctrl: null };

function prefetchNextTrack(trackId) {
  if (!trackId) return;
  if (_nextPrefetch.trackId === trackId || _playingBlob.trackId === trackId) return;
  if (_nextPrefetch.ctrl) _nextPrefetch.ctrl.abort();
  if (_nextPrefetch.blobUrl && _nextPrefetch.blobUrl !== _playingBlob.url) {
    URL.revokeObjectURL(_nextPrefetch.blobUrl);
  }
  const ctrl = new AbortController();
  const entry = { trackId, blobUrl: null, ctrl };
  _nextPrefetch = entry;
  // forPrefetch: the network-URL branch below must not treat this fetch as
  // "now playing something else" and revoke the blob the <audio> is using.
  fetch(buildStreamUrl(trackId, { forPrefetch: true }), { signal: ctrl.signal })
    .then(r => (r.ok ? r.blob() : null))
    .then(blob => {
      // A newer prefetch may have replaced this entry while we downloaded.
      if (!blob || _nextPrefetch !== entry) return;
      entry.blobUrl = URL.createObjectURL(blob);
    })
    .catch(() => {});   // aborted / offline — playback falls back to the network URL
}

// ── Current-track warmup ─────────────────────────────────────────────────────
// The NEXT track is prefetched whole (above), but the CURRENT one — the first
// track after a click, or any track whose prefetch didn't finish — streams
// over the network with the browser's conservative buffer. Over the internet
// a lossless FLAC rides close to the link bandwidth, so hiccups surface as
// mid-track stutter. Once playback is safely started we download the file
// fully on the side:
//  * the response lands in the browser HTTP cache (the stream endpoint sends
//    Cache-Control now), so the element's own later Range requests come from
//    disk instead of the network;
//  * if the element still starves ('waiting' mid-play) and the Blob is ready,
//    App hot-swaps src to the Blob at the same position — one micro-pause
//    instead of a stutter series.
let _currentWarmup = { trackId: null, blobUrl: null, ctrl: null };

function warmupCurrentTrack(trackId) {
  if (!trackId) return;
  if (_currentWarmup.trackId === trackId) return;            // already running/done
  if (_playingBlob.trackId === trackId) return;              // already playing offline
  if (_nextPrefetch.trackId === trackId && _nextPrefetch.blobUrl) return;
  dropCurrentWarmup();
  const ctrl = new AbortController();
  const entry = { trackId, blobUrl: null, ctrl };
  _currentWarmup = entry;
  fetch(buildStreamUrl(trackId, { forPrefetch: true }), { signal: ctrl.signal })
    .then(r => (r.ok ? r.blob() : null))
    .then(blob => {
      if (!blob || _currentWarmup !== entry) return;
      entry.blobUrl = URL.createObjectURL(blob);
    })
    .catch(() => {});   // aborted / offline — the element keeps its network src
}

function dropCurrentWarmup() {
  if (_currentWarmup.ctrl) _currentWarmup.ctrl.abort();
  if (_currentWarmup.blobUrl && _currentWarmup.blobUrl !== _playingBlob.url) {
    URL.revokeObjectURL(_currentWarmup.blobUrl);
  }
  _currentWarmup = { trackId: null, blobUrl: null, ctrl: null };
}

// Stall recovery: hand the completed warmup Blob over to _playingBlob (so
// buildStreamUrl keeps returning it for this track) and return its URL, or
// null when the download hasn't finished. Ownership moves — the caller swaps
// the <audio> src, buildStreamUrl's same-track path then reuses the same URL.
function takeWarmupBlob(trackId) {
  if (_currentWarmup.trackId !== trackId || !_currentWarmup.blobUrl) return null;
  const url = _currentWarmup.blobUrl;
  _currentWarmup = { trackId: null, blobUrl: null, ctrl: null };
  if (_playingBlob.url && _playingBlob.url !== url) URL.revokeObjectURL(_playingBlob.url);
  _playingBlob = { trackId, url };
  return url;
}

function dropMediaPrefetch() {
  if (_nextPrefetch.ctrl) _nextPrefetch.ctrl.abort();
  if (_nextPrefetch.blobUrl && _nextPrefetch.blobUrl !== _playingBlob.url) {
    URL.revokeObjectURL(_nextPrefetch.blobUrl);
  }
  _nextPrefetch = { trackId: null, blobUrl: null, ctrl: null };
  dropCurrentWarmup();
  if (_playingBlob.url) URL.revokeObjectURL(_playingBlob.url);
  _playingBlob = { trackId: null, url: null };
}

function buildStreamUrl(trackId, { forPrefetch = false } = {}) {
  // Completed prefetch for this track — promote it to "playing" and serve the
  // blob. The previous playing blob (if different) is no longer the element's
  // src after this switch, so it's safe to revoke.
  if (!forPrefetch && _nextPrefetch.trackId === trackId && _nextPrefetch.blobUrl) {
    if (_playingBlob.url && _playingBlob.url !== _nextPrefetch.blobUrl) {
      URL.revokeObjectURL(_playingBlob.url);
    }
    _playingBlob = { trackId, url: _nextPrefetch.blobUrl };
    // Ownership moves to _playingBlob. Leaving the entry here would hand out
    // this URL again AFTER the network path revoked it (play unprefetched →
    // prev back to this track) — a dead blob: src the element can't load.
    _nextPrefetch = { trackId: null, blobUrl: null, ctrl: null };
    return _playingBlob.url;
  }
  // Re-pointing the already-playing blob track (section remount, queue
  // re-sync) must return the SAME url so setSrc's same-src guard holds.
  if (!forPrefetch && _playingBlob.trackId === trackId && _playingBlob.url) {
    return _playingBlob.url;
  }
  // Network path. A real track switch away from the blob (manual pick of an
  // unprefetched track) frees the stale blob so at most one lingers.
  if (!forPrefetch && _playingBlob.url && _playingBlob.trackId !== trackId) {
    URL.revokeObjectURL(_playingBlob.url);
    _playingBlob = { trackId: null, url: null };
  }
  // Self-heal: if the cached token is missing or in its last 5 minutes,
  // kick off a background refresh. The CURRENT url may still 401 in the
  // missing-token edge (first play racing the boot refresh) — the next
  // track change picks up the fresh token.
  if (!_streamToken || Date.now() > _streamTokenExpMs - 5 * 60 * 1000) {
    refreshStreamToken();
  }
  const base = `${API}/search/tracks/${trackId}/stream`;
  return _streamToken ? `${base}?st=${encodeURIComponent(_streamToken)}` : base;
}

// ─── Phase D — one-time localStorage migration ──────────────────────────────
// Pre-Phase-D keys carried the collection name (acct_<id>) in their suffix;
// post-Phase-D every per-user store is keyed by the JWT user.id (the server is
// the source of identity). Runs once per browser, idempotent via the marker
// 'musix_localstorage_migration_d'. Foreign-account stores left over from a
// shared machine are dropped — they were never ours to keep.
function runCollectionLocalStorageMigration_d(userId) {
  if (!userId) return;
  if (localStorage.getItem('musix_localstorage_migration_d') === 'done') return;

  const userCol = `acct_${userId}`;
  // Snapshot keys before each pass — we mutate localStorage while iterating.
  const snapshot = () => {
    const ks = [];
    for (let i = 0; i < localStorage.length; i++) ks.push(localStorage.key(i));
    return ks;
  };

  // 1. active_collection — obsolete (server derives the collection from the JWT).
  localStorage.removeItem('active_collection');

  // 2. musix_chat_<col> → musix_chat_<userId> (ours); drop foreign accounts'.
  for (const key of snapshot()) {
    if (!key || !key.startsWith('musix_chat_')) continue;
    const suffix = key.slice('musix_chat_'.length);
    const target = `musix_chat_${userId}`;
    if (key === target) continue;  // already migrated
    if (suffix === userCol || suffix === 'default') {
      if (!localStorage.getItem(target)) localStorage.setItem(target, localStorage.getItem(key));
    }
    localStorage.removeItem(key);
  }

  // 3. recentSearches:<col> → recentSearches:<userId>; drop foreign.
  for (const key of snapshot()) {
    if (!key || !key.startsWith('recentSearches:')) continue;
    const suffix = key.slice('recentSearches:'.length);
    const target = `recentSearches:${userId}`;
    if (key === target) continue;
    if (suffix === userCol || suffix === '_default') {
      if (!localStorage.getItem(target)) localStorage.setItem(target, localStorage.getItem(key));
    }
    localStorage.removeItem(key);
  }

  // 4. chatHistory:track:<id> → chatHistory:track:<id>:<userId> (only unsuffixed).
  for (const key of snapshot()) {
    if (!key || !key.startsWith('chatHistory:track:')) continue;
    const tail = key.slice('chatHistory:track:'.length);
    if (tail.includes(':')) continue;  // already has a :<userId> suffix
    const newKey = `${key}:${userId}`;
    if (!localStorage.getItem(newKey)) localStorage.setItem(newKey, localStorage.getItem(key));
    localStorage.removeItem(key);
  }

  localStorage.setItem('musix_localstorage_migration_d', 'done');
}

// Soft-redirect on 401 — clear storage and force a full reload so the App
// re-renders into the LoginScreen branch. window.location.reload is the
// simplest "remount everything" lever and sidesteps stale React state.
//
// Reload-loop guard: a timestamp in sessionStorage, NOT the URL hash. The old
// hash check ('#login') had a trap: nothing cleared the hash after a
// successful login, so the NEXT 401 cleared the token WITHOUT reloading —
// the app stayed mounted token-less and every request 401'd with
// "missing or invalid Authorization header" until a manual reload.
const AUTH_KICK_GUARD_MS = 5000;
function _onAuthFailure() {
  clearStoredAuth();
  const last = parseInt(sessionStorage.getItem('musix_auth_kick_ts') || '0', 10);
  if (Date.now() - last < AUTH_KICK_GUARD_MS) return;  // just kicked — don't loop
  sessionStorage.setItem('musix_auth_kick_ts', String(Date.now()));
  window.location.reload();
}

async function apiFetch(path, opts = {}) {
  const token = getStoredToken();
  // For FormData bodies (file uploads) we must NOT set Content-Type — the browser
  // sets multipart/form-data with the correct boundary. Forcing application/json
  // here would make FastAPI's UploadFile parse fail.
  const isFormData = (typeof FormData !== 'undefined') && (opts.body instanceof FormData);
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(opts.headers || {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(API + path, { ...opts, headers });
  if (res.status === 401) {
    _onAuthFailure();
    throw new Error("HTTP 401: not authenticated");
  }
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

// ─── apiStream — POST a JSON body, consume a text/event-stream response ───────
// Reads SSE `data:` frames off the fetch body reader and hands each parsed event
// to onEvent(). Used by the chat streaming endpoint (/chat/stream). Callers wrap
// this in try/catch and fall back to the non-streaming endpoint on any failure.
async function apiStream(path, body, onEvent, signal) {
  const token = getStoredToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(API + path, { method: "POST", headers, body: JSON.stringify(body), signal });
  if (res.status === 401) { _onAuthFailure(); throw new Error("HTTP 401: not authenticated"); }
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${t.slice(0, 200)}`);
  }
  if (!res.body || !res.body.getReader) throw new Error("stream unsupported");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    // Frames are separated by a blank line; a frame may hold multiple data: lines.
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dataLines = frame.split("\n").filter(l => l.startsWith("data:"));
      if (!dataLines.length) continue;  // heartbeat / comment frame
      const payload = dataLines.map(l => l.slice(5).replace(/^ /, "")).join("\n");
      try { onEvent(JSON.parse(payload)); } catch { /* skip malformed frame */ }
    }
  }
}

// ─── MarkdownText — safe renderer for LLM outputs ────────────────────────────
// Used by chat answers (search + drawer), lyric explanations, refined facts,
// artist bios — any place where the backend returns LLM-generated text that
// may include markdown (**bold**, *italic*, lists, code, links).
//
// Safety: marked → DOMPurify pipeline so LLM-emitted HTML/script can't run.
// Renders an empty fragment if the text is falsy. Falls back to plain-text
// rendering if `marked` or `DOMPurify` haven't loaded yet (defensive — both
// come from CDN <script> tags in <head>).
function MarkdownText({ text, className }) {
  if (!text) return null;
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    return <span className={className}>{text}</span>;
  }
  const html = DOMPurify.sanitize(marked.parse(String(text), { breaks: true, gfm: true }));
  return <div className={`md-content${className ? ' ' + className : ''}`} dangerouslySetInnerHTML={{ __html: html }} />;
}

// ─── Playback event helpers ───────────────────────────────────────────────────

// Mint or retrieve a per-tab session ID. Cleared when the tab closes.
function getSessionId() {
  let id = sessionStorage.getItem('musix_session_id');
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() :
      'sess-' + Math.random().toString(36).slice(2) + '-' + Date.now());
    sessionStorage.setItem('musix_session_id', id);
  }
  return id;
}

async function postPlaybackEvent({ trackId, playedSec, totalDur, interacted, influence }) {
  // No collection guard: the server derives the collection from the JWT and
  // ignores any client-supplied value (Phase D-soft). Requiring a non-empty
  // collectionName here silently dropped EVERY event after Phase D removed
  // activeCollection from the player — that's why playback_events stayed empty.
  if (!trackId || playedSec == null) return;
  const body = JSON.stringify({
    session_id: getSessionId(),
    track_id: trackId,
    played_sec: playedSec,
    total_dur: totalDur ?? null,
    // Stream RecSys idle rule: did the user touch any control during this
    // listen (like/dislike/skip/pause/seek)? null = unknown (legacy semantics).
    interacted: interacted ?? null,
    influence: influence === undefined ? true : influence,
  });
  // fetch + keepalive instead of navigator.sendBeacon: /playback/events sits
  // behind the JWT gate and beacons CANNOT carry an Authorization header —
  // every beacon 401'd and ALL listen history was silently dropped. keepalive
  // gives the same survives-unload semantics while allowing headers.
  const token = getStoredToken();
  if (!token) return;  // logged out — the server would only 401 it anyway
  try {
    await fetch(`${API}/playback/events`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body,
      keepalive: true,
    });
  } catch (e) {
    console.warn('[playback] failed to post event', e);
  }
}

// Огонёк/Вода: record an ephemeral taste gesture (fire = «больше такого»,
// water = «остудить»). Fire-and-forget; the wave rebuild is triggered separately
// via onStreamSignal so the UI feels instant.
async function postTasteSignal(trackId, kind) {
  const token = getStoredToken();
  if (!token || !trackId) return;
  try {
    await fetch(`${API}/recommend/taste-signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ session_id: getSessionId(), track_id: trackId, kind }),
      keepalive: true,
    });
  } catch (e) {
    console.warn('[taste] failed to post signal', e);
  }
}

// ─── Listening-time accumulator ──────────────────────────────────────────────
// A single continuous listen of a track must produce exactly ONE playback event
// whose played_sec is the real number of seconds the user actually heard — NOT
// the absolute playhead position. The previous model emitted el.currentTime on
// pause / ended / track-change / tab-hide and the backend SUMmed those, so any
// track that fired >1 event (pause→resume, tab switch, etc.) double-counted the
// overlapping seconds and inflated the total. We instead accumulate true played
// time from 'timeupdate' deltas (which only tick while audio is actually
// playing) onto el.dataset, then flush once at the end of the listen.
//
// dataset fields (strings, persisted on the <audio> element):
//   playAcc      — accumulated real played seconds for the current listen
//   playLastPos  — last currentTime observed, to compute the next delta
//   playEmitted  — '1' once this listen has been recorded (dedupe guard)
function accumulatePlayedTime(el) {
  const t = el.currentTime || 0;
  const last = parseFloat(el.dataset.playLastPos || '0');
  const dt = t - last;
  // Count only forward steps taken while playing. dt >= 2 means a seek jump (or
  // a background-tab throttle gap) — not real listening — so it's ignored;
  // dt <= 0 is a rewind/seek-back. Either way we don't credit the time.
  if (!el.paused && dt > 0 && dt < 2) {
    el.dataset.playAcc = String(parseFloat(el.dataset.playAcc || '0') + dt);
  }
  el.dataset.playLastPos = String(t);
}

function resetPlaySession(el) {
  if (!el) return;
  el.dataset.playAcc = '0';
  el.dataset.playEmitted = '0';
  el.dataset.playLastPos = String(el.currentTime || 0);
  el.dataset.playInteracted = '0';
  // playNoInfluence is (re)stamped by setSrc per track; default off here.
  if (el.dataset.playNoInfluence === undefined) el.dataset.playNoInfluence = '0';
}

// Mark the CURRENT listen as user-touched (like/dislike/skip/pause/seek).
// Feeds the stream's idle rule: 5 untouched tracks in a row → subsequent
// events stop moving the taste profile until the user acts again.
function markPlaybackInteracted(el) {
  if (el) el.dataset.playInteracted = '1';
}

// Emit the accumulated listen for `el` exactly once. Safe to call from every
// terminal trigger (ended / track-change / unload) — the playEmitted guard
// makes repeat calls no-ops until the next listen resets the session.
function flushAccumulatedListen(el) {
  if (!el || el.dataset.playEmitted === '1') return;
  const trackId = el.dataset.playbackTrackId;
  if (!trackId) return;  // collection is server-derived; only the track id is required
  const acc = parseFloat(el.dataset.playAcc || '0');
  if (acc < 1) return;  // nothing meaningful heard yet
  el.dataset.playEmitted = '1';
  postPlaybackEvent({
    trackId,
    playedSec: acc,
    totalDur: el.duration || null,
    interacted: el.dataset.playInteracted === '1',
    influence: el.dataset.playNoInfluence !== '1',
  });
}

// ─── Lossless logo (inline SVG — scales perfectly at any size) ────────────────
const LOSSLESS_LOGO_SRC = (function() {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 215 64">' +
    '<svg x="0" y="12" width="66" height="40" viewBox="0 0 15 9">' +
    '<path fill="black" d="M8.184,0.35C9.944,0.35 10.703,3.296 11.338,5.238C11.673,3.842 11.497,3.542 11.857,3.542C11.99,3.542 12.126,3.633 12.126,3.798C12.126,3.809 12.123,3.839 12.117,3.883L12.091,4.058C12.02,4.522 11.845,5.494 11.654,6.144C13.198,10.191 14.345,4.861 14.474,3.772C14.493,3.615 14.612,3.542 14.731,3.542C14.891,3.542 15.022,3.662 14.997,3.843C14.72,5.605 14.295,8.35 12.547,8.35C11.582,8.35 11.04,7.595 10.611,6.73C9.54,4.626 9.047,1.093 7.997,1.093C7.66,1.093 7.411,1.444 7.394,1.444C7.362,1.444 7.337,1.301 7.023,0.909C7.322,0.567 7.734,0.35 8.184,0.35ZM2.458,0.354C5.211,0.354 5.456,7.618 7.014,7.618C7.197,7.618 7.394,7.507 7.61,7.256C7.729,7.458 7.851,7.638 7.978,7.796C7.667,8.151 7.28,8.35 6.795,8.35C5.054,8.349 4.306,5.434 3.663,3.466C3.511,4.097 3.432,4.669 3.402,4.925C3.382,5.088 3.263,5.163 3.143,5.163C3.009,5.163 2.874,5.071 2.874,4.908L2.874,4.908L2.877,4.87C2.966,4.223 3.146,3.243 3.347,2.56C3.079,1.858 2.745,1.091 2.252,1.091C1.257,1.091 0.687,3.591 0.527,4.925C0.508,5.088 0.388,5.163 0.268,5.163C0.135,5.163 0,5.071 0,4.908C0,4.896 0.001,4.883 0.002,4.87C0.283,2.836 0.808,0.354 2.458,0.354ZM5.315,0.35C5.809,0.35 6.339,0.608 6.797,1.211C6.822,1.241 7.078,1.639 7.159,1.777C8.277,3.802 8.818,7.627 9.881,7.627C10.065,7.627 10.264,7.513 10.484,7.256C10.604,7.458 10.726,7.638 10.852,7.796C10.542,8.15 10.155,8.35 9.67,8.35C6.933,8.349 6.636,1.09 5.128,1.09C4.788,1.09 4.536,1.444 4.519,1.444C4.487,1.444 4.462,1.301 4.148,0.909C4.455,0.558 4.87,0.35 5.315,0.35Z"/>' +
    '</svg>' +
    '<text x="80" y="44" font-family="Helvetica Neue,Helvetica,Arial,sans-serif" font-size="29" font-weight="400" fill="black">Lossless</text>' +
    '</svg>';
  return 'data:image/svg+xml;base64,' + btoa(svg);
})();
const _LOSSLESS_PNG_DELETED = 'xAAAFuklEQVR42u2ZeWwVVRTGfzOvC4WytBQbjZQ2D7VKABcsRU0t0YgimhaDSFqtJlRTdxMwiolLjBjxDzQEBWLcUYkLUVFRIoqoiQtKFBUVFEwjWpBiEVpQOv7Rb+LN5PX1tX0C2vMlk8m9c+723XPOPecOGAwGg8FgMBgMBoPBYDAYDAaDwWAwGAwAXg/kfb0DoN2pQ+UAiKkcAAeM5oO3QW47z+j7B7FuEBdIoy8BLgAGAz8ApwK1wBhgO5AF1AMVkv+pl5vWJ7XXl9sYDrwC/K7yROADYAuQK5IzgNXAHuB44CTgRaAOaNXGBUZ7aqRnAp8Cs536u0XgWpWHA03y4ZNU97RknuumZfVpZOh9E/C8Q1w58CDwqki9R89dKrcAhUAe8IvqLjbiU3c/MWANEHeilAVyL5UidA8wVRbxpeoekOy9Kn8DZPfiYM1wxj9cgoR/xa0AnOa4Bw8YACwTwdk6TAORC3CLyjul6UVAm+qmRywo2cK9/7O/7upbJfC2E8GcDGwF/gT2OW5nJjAIeFKanwdcrujlXbWtd+L5KIIk5RhwDXBCgnn72sSMyGbGnLpYAtks1VUAHwMDtcZMR8ZL0pcXGddLYzjpLQZKnfprgRlOudTR5KtU97qIXa/yDH3fl4S4/hFXkuMspJ/aXxmxFK8b7qIz2Sr1PTQNffVa0w90uI8gkAsJZY8CNkRkv9fEL1N5qSY2GhgLvAzskHZVi+AbnfYjpck4Lq1OfYYLbFHYGc0djgUeUmha71jJFFnhs4qmAqAYWCjZ2VrTflltK1AAzNf3WdL6QNbwDPCCXGQgy7hTdXOB/FTPK7+L3S0EdmlirtZtdWTP00HrAWUiISTZF3l7gbc02XOBEyOafXRkssUa03Uz7kEavkvkGkYAXyiimqeE7SVgFfC15lMmCxyk8ixgjvIOT/NZAxwHrAQaFDCUqJ91wDvAU9qERcBkjTNJ9UE6SD8C/CZH83NFRosjk6fJ7RQxV8inr9QkLtQ4j6nNGKBGJLkkb3PKRUBjFxFVIOtoA85XuFoDXCeCM4FmkRNXEjdCCvARME7fhkqxquT67tcZNF/ucpwsdLc2K66cJS4L2SSrmpnkvOoW6fl+x8RD5OP7rY4G5kpDvnU0+SK1f1zvODBBWrJFmlYAvOccSiEpOInWj50ctoGzsGItGhHzuSxxldzYHOArJWkxBQGFwJvS6jKdM4Hc5l7gDkVr1cAK4H1pdp02a7nGPUdh8DJZQV3k8q/HcesQ+Soc13Jk5MArkfxZDilnavDvRNDDkp+n7584m+7JCoapbqAWkuWMkw384SwsPGRvUH/hnG6XXIXcXjjHNoW0U9UWKcVm4GxZabX6ijsub7oChSpnLpuBJyRfpLoGte1/MMPd0OQ/FclLVX+bJtMo4kqBvyQzXu2O0SLCfs4AHolEKQN0mbZdG7lJz2gdZE3y7a3ANN37tAAfSutD2TeAX5VJ/yyyJmqOOcAS4DdZ7W4pQ7HGXa/2jZr7Ao23Qpa+qLeanuxaNlE5JKc2cg0wTP4y0Dd08ASaNMD1wM2RO52GCOk+MEruoFzPBFkesqwaEYRjMdN0K+qGg5WKssaqPFgHe0hWuSxqvNMmR5pdK9cX4hT1dfqhui7w5RLCa4D7Ii5ltW4dF0pDdsgtvK6IJ0xA1jpm6/Ughfc7aeenUOen+C/AP1wy6PBQDGPiZpHXT1npfuBRhWSLJbM8Ep9fLXeR6GLMTfC4Y8cidZ5T7yWR9RIkasn66kr2kBEfXnKt6wg7qVR4tUshWIH8YABcKp89Wb65NF2+sa/dSgLcqsSjWaHWXMW37cBGHVLh1fA2JVxTUsiU++x/z1R/7w1xkogwGRql+H6jNmWkEqwN8vV+KgmGIbX/r16Kh6D9I01jyNkeKQfOz27s/6nBYDAYDAaDwWAwGAwGg8FgMBgMBoPBYDAYDIb/Lv4GnKtX5Ea5WS0AAAAASUVORK5CYII=';

// ─── Color tokens ─────────────────────────────────────────────────────────────
function useColors(isDark) {
  return useMemo(() => ({
    bg:           isDark ? '#0d0d10'    : '#f2f1f6',
    bgDeep:       isDark ? '#08080b'    : '#e8e7ed',
    surface:      isDark ? '#17171b'    : '#ffffff',
    surface2:     isDark ? '#1e1e24'    : '#ebebf0',
    surface3:     isDark ? '#26262d'    : '#e4e3e9',
    border:       isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.09)',
    borderStrong: isDark ? 'rgba(255,255,255,0.13)' : 'rgba(0,0,0,0.16)',
    text:         isDark ? '#eeeef3'    : '#161620',
    textMuted:    isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
    textSubtle:   isDark ? 'rgba(255,255,255,0.28)' : 'rgba(0,0,0,0.32)',
    accent:       'oklch(60% 0.18 270)',
    accentBg:     isDark ? 'oklch(60% 0.18 270 / 0.15)' : 'oklch(60% 0.18 270 / 0.10)',
    accentLight:  'oklch(70% 0.16 270)',
    amber:        isDark ? 'oklch(72% 0.13 75)' : 'oklch(58% 0.15 60)',
    amberGlow:    'rgba(212,165,90,0.35)',
    inputBg:      isDark ? '#10101a'    : '#ffffff',
    sidebarBg:    isDark ? 'linear-gradient(180deg,#131318 0%,#0c0c10 100%)' : 'linear-gradient(180deg,#f0eff5 0%,#e7e6ec 100%)',
    chatPanelBg:  isDark ? '#0f0f13'    : '#f8f8fc',
    userBubble:   'linear-gradient(135deg, oklch(60% 0.18 270), oklch(58% 0.18 305))',
    aiBubble:     isDark ? '#1e1e26'    : '#ebebf2',
    green:        'oklch(63% 0.17 142)',
    greenBg:      isDark ? 'oklch(63% 0.17 142 / 0.13)' : 'oklch(63% 0.17 142 / 0.10)',
    red:          'oklch(58% 0.21 25)',
    redBg:        isDark ? 'oklch(58% 0.21 25 / 0.14)' : 'oklch(58% 0.21 25 / 0.09)',
  }), [isDark]);
}
const ske = (kind, isDark) => `ske-${kind}-${isDark ? 'd' : 'l'}`;
const brushed = (isDark) => `brushed-${isDark ? 'd' : 'l'}`;

// Stable hue [0..360) from a string — the same genre/artist is always the same
// colour (replaces the old arbitrary hue-by-index rainbow in the charts).
const hueFromString = (s) => {
  let h = 0;
  const str = s || '';
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
  return h % 360;
};

// Tracks the OS "reduce motion" setting. Every stats animation branches on this
// so the redesign honours the same a11y contract as the rest of index.css.
function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return reduced;
}

// Eases a number 0 → value over `dur` ms (easeOutCubic). Jumps straight to the
// value when reduced motion is on. Used by the "∑ listened" readout.
function useCountUp(value, dur = 600, reduced = false) {
  const [n, setN] = useState(reduced ? value : 0);
  useEffect(() => {
    if (reduced || !value) { setN(value || 0); return; }
    let raf, t0 = null;
    const tick = (t) => {
      if (t0 == null) t0 = t;
      const p = Math.min(1, (t - t0) / dur);
      setN(value * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, dur, reduced]);
  return n;
}

// ─── timeStore — module-level external store for audio currentTime ──────────
// Audio elements fire 'timeupdate' ~4×/sec. If we stored currentTime in React
// state, every tick would re-render the whole App tree (statistics jumping,
// text selection clearing, etc.). Instead we keep currentTime OUTSIDE React
// and let only components that actually display time subscribe via the
// useCurrentTime() hook — those components re-render independently and the
// rest of the app sleeps.
const timeStore = (() => {
  let time = 0;
  const listeners = new Set();
  return {
    setTime(t) {
      if (t === time) return;
      time = t;
      listeners.forEach(fn => fn());
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getSnapshot() { return time; },
  };
})();

// useCurrentTime — subscribes to timeStore via useSyncExternalStore.
// Components that call this hook re-render on every audio timeupdate;
// components that don't never even see the update.
function useCurrentTime() {
  return React.useSyncExternalStore(timeStore.subscribe, timeStore.getSnapshot, timeStore.getSnapshot);
}

// ─── useAudioPlayer — shared audio controller (lives at App level) ──────────
function useAudioPlayer() {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [currentSrc, setCurrentSrc] = useState(null);
  const [duration, setDuration] = useState(0);

  // Last src string this hook assigned to the element — the same-src guard
  // reads it instead of React state so setSrc can stay fully synchronous.
  const lastSrcRef = useRef(null);

  // When src changes mid-playback, fire a playback event for the *previous*
  // track using audio.currentTime as played_sec, then attach the new track id
  // to the audio element via dataset for the next 'ended' to pick up.
  //
  // setSrc drives the <audio> element DIRECTLY (el.src / el.play()) instead of
  // round-tripping through React. The element used to be controlled
  // (<audio src={currentSrc}>) and callers scheduled play() via
  // setTimeout(..., 50) — an async gap right when the previous track ended.
  // On a phone with the screen off that gap is fatal: the instant a track ends
  // the page stops being audible, so mobile Chrome throttles its timers (up to
  // 1/min after 5 min hidden) and may freeze the page outright. The deferred
  // play() never ran, the media notification vanished, and playback only
  // resumed when the user lit the screen (unfreezing the page flushed the
  // queued timer). Switching src + play() synchronously INSIDE the 'ended'
  // handler leaves no silent gap, so the OS keeps the media session alive.
  // React state (currentSrc) stays as a passive mirror for the UI.
  const setSrc = useCallback((src, meta, { autoplay = false } = {}) => {
    const el = audioRef.current;
    const playNow = (target) =>
      target.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    if (!el) {
      // Element not mounted yet (boot paths) — stash for initAudio to apply.
      lastSrcRef.current = src;
      setCurrentSrc(src);
      return;
    }
    // Same track re-pointed (section remount, playlist re-sync after a stream
    // buffer trim): do NOT reset the in-flight listen accumulator — that would
    // silently drop the seconds heard (and the interacted mark) mid-listen.
    if (meta && el.dataset.playbackTrackId === String(meta.trackId || '')) {
      // Keep the existing src if only the rotating ?st= stream token changed.
      // buildStreamUrl bakes a refresh-able token into the URL, so re-pointing
      // the SAME track with a freshly-minted token would change the <audio>
      // src string and reload the media from 0 — that's the "pressing like
      // sometimes restarts the song" bug (a reaction rebuilds the playlist,
      // which re-runs this effect for the already-playing head track).
      const prev = lastSrcRef.current;
      if (!(prev && prev.split('?')[0] === src.split('?')[0])) {
        lastSrcRef.current = src;
        el.src = src;
        setCurrentSrc(src);
      }
      if (autoplay && el.paused) playNow(el);
      return;
    }
    // Switching to a different track ends the previous listen — flush it.
    if (el.dataset.playbackTrackId && meta && el.dataset.playbackTrackId !== meta.trackId) {
      flushAccumulatedListen(el);
    }
    if (meta) {
      el.dataset.playbackTrackId = meta.trackId || '';
      el.dataset.playbackCollection = meta.collectionName || '';
      el.dataset.playNoInfluence = meta.noInfluence ? '1' : '0';
      resetPlaySession(el);
    }
    // New media: show the buffering veil until 'canplay'/'playing' clears it
    // (covers m4a transcode + stream warm-up where 'waiting' may not fire).
    setIsBuffering(true);
    lastSrcRef.current = src;
    el.src = src;                 // assigning src starts the load synchronously
    setCurrentSrc(src);
    if (autoplay) playNow(el);
  }, []);

  // Read playback state from the live <audio> element rather than React state.
  // Closures over `currentSrc` / `isPlaying` are stale when callers invoke this
  // right after setSrc (state hasn't committed yet). Reading audioRef.current
  // at execution time avoids that race and lets the callback identities stay
  // stable across renders. Track switches should pass { autoplay: true } to
  // setSrc instead — that plays synchronously with the src assignment.
  const play = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    el.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
  }, []);

  const pause = useCallback(() => {
    if (!audioRef.current) return;
    // Public pause/togglePlay/seek are only reachable from user-initiated UI
    // paths (auto-advance drives the element directly) — mark the listen as
    // interacted for the stream's idle rule.
    markPlaybackInteracted(audioRef.current);
    audioRef.current.pause();
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    markPlaybackInteracted(el);
    if (el.paused) {
      el.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    } else {
      el.pause();
      setIsPlaying(false);
    }
  }, []);

  const seek = useCallback((time) => {
    if (!audioRef.current) return;
    markPlaybackInteracted(audioRef.current);
    audioRef.current.currentTime = time;
    timeStore.setTime(time);
  }, []);

  // Same-track src replacement preserving position/play state (stall recovery:
  // network src → completed warmup Blob). Goes through lastSrcRef so setSrc's
  // same-track guard doesn't later see a "different" src and reload from 0.
  // Does NOT touch playbackTrackId or the listen accumulator — same listen.
  const hotSwapSrc = useCallback((url) => {
    const el = audioRef.current;
    if (!el || !url) return;
    const pos = el.currentTime || 0;
    const wasPlaying = !el.paused;
    const restore = () => {
      try { el.currentTime = pos; } catch { /* metadata race — keep 0 */ }
      if (wasPlaying) el.play().catch(() => {});
    };
    el.addEventListener('loadedmetadata', restore, { once: true });
    lastSrcRef.current = url;
    el.src = url;
    setCurrentSrc(url);
  }, []);

  const setVolume = useCallback((vol) => {
    if (!audioRef.current) return;
    audioRef.current.volume = vol;
  }, []);

  // Callback ref bound to the <audio> element. Attaches event listeners the
  // moment the element first mounts. We can't use a useEffect for this because
  // App has early-return paths during boot ('checking'/'onboarding'/'no-qdrant')
  // that don't render <audio>, so the first commit happens with
  // audioRef.current=null and a useEffect with [] would bail out and never
  // re-fire when <audio> later appears.
  const listenersAttached = useRef(false);
  const initAudio = useCallback((el) => {
    audioRef.current = el;
    if (!el) { listenersAttached.current = false; return; }
    // The element is uncontrolled (setSrc writes el.src directly). If setSrc
    // ran before the element mounted, apply the stashed src now.
    if (lastSrcRef.current && !el.getAttribute('src')) el.src = lastSrcRef.current;
    if (listenersAttached.current) return;
    el.addEventListener('timeupdate', () => {
      timeStore.setTime(el.currentTime || 0);
      // Credit real played time from the inter-tick delta (only ticks while
      // actually playing; seeks/rewinds are filtered inside the helper).
      accumulatePlayedTime(el);
    });
    el.addEventListener('loadedmetadata', () => setDuration(el.duration || 0));
    el.addEventListener('durationchange', () => setDuration(el.duration || 0));
    el.addEventListener('ended', () => {
      setIsPlaying(false);
      // Track finished — end of this listen, record the accumulated time.
      flushAccumulatedListen(el);
    });
    el.addEventListener('play', () => {
      setIsPlaying(true);
      // If the previous listen was already recorded (track ended, then replayed),
      // this is a brand-new listen — start a fresh accumulator. Otherwise this
      // is a resume mid-track: keep the accumulator, but re-anchor playLastPos so
      // the paused gap (or a seek made while paused) isn't credited as listening.
      if (el.dataset.playEmitted === '1') resetPlaySession(el);
      else el.dataset.playLastPos = String(el.currentTime || 0);
    });
    el.addEventListener('pause', () => {
      // No emit on pause: the accumulator already holds the seconds heard so far,
      // and pausing is not the end of a listen (the user may resume). Emitting
      // here is exactly what used to double-count on resume→end.
      setIsPlaying(false);
      setIsBuffering(false);
    });
    // Buffering signal — the cover veil + spinner key off this. 'waiting' and
    // 'stalled' fire while the media element is starved (m4a transcode, stream
    // warm-up); 'playing'/'canplay' clear it the instant audio actually flows.
    el.addEventListener('waiting',      () => setIsBuffering(true));
    el.addEventListener('stalled',      () => setIsBuffering(true));
    el.addEventListener('playing',      () => setIsBuffering(false));
    el.addEventListener('canplay',      () => setIsBuffering(false));
    el.addEventListener('canplaythrough', () => setIsBuffering(false));
    el.addEventListener('ended',        () => setIsBuffering(false));
    el.addEventListener('error',        () => setIsBuffering(false));
    // Browser-close / tab-hide safety net — sendBeacon survives unload. We flush
    // only on real page teardown (pagehide), NOT on visibilitychange: audio keeps
    // playing (and accumulating) in a backgrounded tab, so flushing on tab-switch
    // would prematurely seal the listen and drop the seconds heard after return.
    window.addEventListener('pagehide', () => flushAccumulatedListen(el));
    listenersAttached.current = true;
  }, []);

  return {
    audioRef, initAudio,
    isPlaying, isBuffering, currentSrc, duration,
    play, pause, togglePlay, seek, setSrc, setVolume, hotSwapSrc,
    // NOTE: currentTime is intentionally NOT here. Use useCurrentTime() for
    // display, or read audioRef.current?.currentTime in event handlers.
  };
}

// ─── useAIStatus — global AI mode signal ────────────────────────────────────
// Two-signal model:
//   aiAvailable               — runtime LLM ping result (60s heartbeat)
//   aiEnabledForCollection    — per-collection opt-in from collection_settings
//   aiActive = both true      — gates live-LLM UI features only
//                                (chat-search, Ask AI, future lyric-explain).
// Cached AI artifacts (sonic_vibe / refined_facts / artist_bio in SQLite) are
// NOT gated — they're already on disk.
function useAIStatus() {
  const [aiAvailable, setAiAvailable] = useState(null); // null = probing
  const [aiEnabledForCollection, setAiEnabledForCollection] = useState(null);
  const [instanceAiAvailable, setInstanceAiAvailable] = useState(null); // admin policy
  const [llmError, setLlmError] = useState(null);
  const [llmInfo, setLlmInfo] = useState(null);  // {base_url, model}
  const probeTimer = useRef(null);

  const probe = useCallback(async () => {
    const baseUrl = (localStorage.getItem('llm_base_url') || '').trim();
    const model = (localStorage.getItem('llm_model') || '').trim();
    try {
      const r = await apiFetch('/system/llm-status', {
        method: 'POST',
        body: JSON.stringify({
          base_url: baseUrl || undefined,
          model: model || undefined,
        }),
      });
      setAiAvailable(!!r.available);
      setLlmError(r.available ? null : (r.error || 'AI assistant not responding'));
      setLlmInfo({ base_url: r.base_url, model: r.model });
    } catch (e) {
      setAiAvailable(false);
      setLlmError(e?.message || 'probe failed');
      setLlmInfo(null);
    }
  }, []);

  // Initial probe + 60s heartbeat
  useEffect(() => {
    probe();
    probeTimer.current = setInterval(probe, 60000);
    return () => clearInterval(probeTimer.current);
  }, [probe]);

  // Re-probe on llm_base_url change in other tabs
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key === 'llm_base_url' || e.key === 'llm_model') probe();
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [probe]);

  // Fetch per-collection ai_enabled
  useEffect(() => {
    apiFetch(`/library/settings`)
      .then(r => setAiEnabledForCollection(!!r.ai_enabled))
      .catch(() => setAiEnabledForCollection(true));  // graceful default
  }, []);

  // Instance AI policy (admin toggle + endpoint). Gates members so they don't
  // see AI when the admin disabled it. Owners are never gated by this — they
  // manage the policy. Default true (non-blocking) on older servers / errors.
  useEffect(() => {
    apiFetch('/instance/config')
      .then(r => setInstanceAiAvailable(r.ai_available !== false))
      .catch(() => setInstanceAiAvailable(true));
  }, []);

  // `null` on either signal means we haven't heard back yet — distinct from
  // a confirmed `false`. Consumers should treat `aiProbing` as "don't decide
  // yet" (show skeleton/spinner) rather than as "AI off".
  const aiProbing = aiAvailable === null || aiEnabledForCollection === null;
  const isOwner = (localStorage.getItem('musix_user_role') || '') === 'owner';
  // instanceAiAvailable===null (still loading) must NOT block — only a confirmed
  // false gates, and only for members.
  const instanceGate = isOwner || instanceAiAvailable !== false;
  const aiActive = !!(aiAvailable && aiEnabledForCollection && instanceGate);
  return {
    aiAvailable, aiEnabledForCollection, aiActive, aiProbing,
    instanceAiAvailable,
    llmError, llmInfo, refresh: probe,
    setAiEnabledForCollection,   // exposed so Settings toggle can update locally w/o re-fetch
  };
}

// ─── useMediaSession — expose now-playing to the OS ─────────────────────────
// Feeds the Media Session API → Windows SMTC (the media flyout by the clock,
// lock screen, and hardware/media keys), the same mechanism Spotify/Yandex web
// use. Title/artist/album + album art + play/pause/prev/next + a position
// scrubber. No-op where unsupported; Chromium/Edge give the full SMTC.
function useMediaSession({ currentTrack, isPlaying, audioRef, onPlay, onPause, onNext, onPrev, onSeek }) {
  // Action callbacks via a ref so handlers register once but always call the
  // latest closures (next/prev/seek are recreated each render).
  const cbs = useRef({});
  cbs.current = { onPlay, onPause, onNext, onPrev, onSeek };

  // Metadata on track change.
  useEffect(() => {
    if (!('mediaSession' in navigator)) return;
    if (!currentTrack) { navigator.mediaSession.metadata = null; return; }
    let art = null;
    const rel = currentTrack.cover_art_path;
    if (rel) {
      const u = rel.startsWith('http') ? rel : `${API}${rel}`;
      art = u.startsWith('http') ? u : window.location.origin + u;
    }
    try {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: currentTrack.title || '',
        artist: currentTrack.artist || '',
        album: currentTrack.album || '',
        artwork: art ? [
          { src: art, sizes: '256x256', type: 'image/jpeg' },
          { src: art, sizes: '512x512', type: 'image/jpeg' },
        ] : [],
      });
    } catch {}
  }, [currentTrack?.track_id, currentTrack?.title, currentTrack?.artist, currentTrack?.album, currentTrack?.cover_art_path]);

  // Register action handlers once.
  useEffect(() => {
    if (!('mediaSession' in navigator)) return;
    const ms = navigator.mediaSession;
    const set = (action, fn) => { try { ms.setActionHandler(action, fn); } catch {} };
    set('play', () => cbs.current.onPlay?.());
    set('pause', () => cbs.current.onPause?.());
    set('previoustrack', () => cbs.current.onPrev?.());
    set('nexttrack', () => cbs.current.onNext?.());
    set('seekto', (e) => {
      if (e && typeof e.seekTime === 'number') {
        if (cbs.current.onSeek) cbs.current.onSeek(e.seekTime);
        else if (audioRef?.current) audioRef.current.currentTime = e.seekTime;
      }
    });
    return () => ['play', 'pause', 'previoustrack', 'nexttrack', 'seekto'].forEach(a => set(a, null));
  }, [audioRef]);

  // Mirror play/pause state.
  useEffect(() => {
    if ('mediaSession' in navigator) {
      navigator.mediaSession.playbackState = currentTrack ? (isPlaying ? 'playing' : 'paused') : 'none';
    }
  }, [isPlaying, currentTrack?.track_id]);

  // Position/duration for the system scrubber.
  useEffect(() => {
    const el = audioRef?.current;
    if (!el || !('mediaSession' in navigator) || !('setPositionState' in navigator.mediaSession)) return;
    const upd = () => {
      const d = el.duration;
      if (d && isFinite(d) && d > 0) {
        try {
          navigator.mediaSession.setPositionState({
            duration: d,
            position: Math.min(Math.max(el.currentTime || 0, 0), d),
            playbackRate: el.playbackRate || 1,
          });
        } catch {}
      }
    };
    el.addEventListener('timeupdate', upd);
    el.addEventListener('loadedmetadata', upd);
    el.addEventListener('durationchange', upd);
    upd();
    return () => {
      el.removeEventListener('timeupdate', upd);
      el.removeEventListener('loadedmetadata', upd);
      el.removeEventListener('durationchange', upd);
    };
  }, [audioRef, currentTrack?.track_id]);
}

// Single-owner flag for the Space shortcut. PlayerSection sets this true while
// mounted so its own keydown handler (which adds the cover flash) is the sole
// toggler; the global handler below then acts only as a fallback (e.g. on the
// home screen, where PlayerSection is unmounted). Without this, BOTH handlers
// fired on one Space press and toggled twice — net no-op playback while the
// icon still flashed. See PlayerSection's "_playerOwnsSpace" effect.
let _playerOwnsSpace = false;

// ─── useGlobalKeyboardShortcuts — document-level playback + nav shortcuts ───
function useGlobalKeyboardShortcuts({ audio, onNavToSection, onToggleLyrics, onCloseLyrics }) {
  useEffect(() => {
    function onKey(e) {
      // Skip if focus is in an input/textarea/contentEditable
      const t = e.target;
      const tag = (t && t.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (t && t.isContentEditable)) return;

      // Space: play/pause. Defer to PlayerSection's own handler when it's
      // mounted (it adds the cover flash) so we don't toggle twice; act as the
      // global fallback only when the player isn't mounted (home screen).
      if (e.code === 'Space') {
        e.preventDefault();
        if (!_playerOwnsSpace) audio?.togglePlay?.();
        return;
      }

      // Arrows: seek ±10s (no shift), prev/next track (shift)
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (e.shiftKey) {
          // TODO Plan 4: prev track in queue
        } else {
          const t = (audio?.audioRef?.current?.currentTime || 0) - 10;
          audio?.seek?.(Math.max(0, t));
        }
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (e.shiftKey) {
          // TODO Plan 4: next track in queue
        } else {
          const t = (audio?.audioRef?.current?.currentTime || 0) + 10;
          audio?.seek?.(Math.min(audio?.duration || t, t));
        }
        return;
      }

      // M: mute toggle (volume 0 / 0.85)
      if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        if (!audio?.audioRef?.current) return;
        const el = audio.audioRef.current;
        if (el.volume > 0) { el.dataset._lastVol = el.volume; el.volume = 0; }
        else { el.volume = parseFloat(el.dataset._lastVol || '0.85'); }
        return;
      }

      // Plan 4: Shift+L toggles lyrics view, Esc closes lyrics.
      // Lyrics + Like both want 'L'; Shift+L is for lyrics so the bare-L like
      // handler below is not triggered when the Shift modifier is held.
      if ((e.key === 'l' || e.key === 'L') && e.shiftKey) {
        e.preventDefault();
        onToggleLyrics?.();
        return;
      }

      // L/D: like/dislike (stubbed until Plan 4 player redesign)
      if (e.key === 'l' || e.key === 'L') { /* TODO Plan 4: like */ return; }
      if (e.key === 'd' || e.key === 'D') { /* TODO Plan 4: dislike */ return; }

      // Esc: close lyrics view (don't preventDefault — let Esc propagate for modals etc.)
      if (e.key === 'Escape') {
        onCloseLyrics?.();
        return;
      }

      // /: focus Search bar (navigate to Search if not there)
      if (e.key === '/') {
        e.preventDefault();
        onNavToSection?.('search');
        return;
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [audio, onNavToSection, onToggleLyrics, onCloseLyrics]);
}

function secsToMMSS(s) {
  if (s == null || isNaN(s)) return s || '—';
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}
// Fisher-Yates partial shuffle — return up to n unique random elements from arr.
// If arr has <= n items, returns a shuffled copy of the whole array.
function sampleN(arr, n) {
  if (!Array.isArray(arr)) return [];
  if (arr.length <= n) return arr.slice();
  const copy = arr.slice();
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (copy.length - i));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

const ARTIST_SAMPLE_SIZE = 5;

// ─── FactsRail (right-side panel: per-track "Did you know" facts) ─────────────
// Two variants:
//   - landing: combined song+artist items, random-collection fallback, type chips
//   - player:  explicit scope toggle (SONG | ARTIST). No mixing, no random
//              fallback. Sticky scope across tracks with auto fall-back when
//              the chosen scope has no data for the current track.
function FactsRail({ trackId, isDark, lang, accent, variant = "landing" }) {
  const c = useColors(isDark);
  const isPlayer = variant === "player";

  // Raw source data — kept separate so the player variant can toggle between
  // them without remixing. Landing variant combines them at render time.
  const [songFacts, setSongFacts] = useState([]);
  const [artistFactsAll, setArtistFactsAll] = useState([]);
  // Random-sampled subset of artist facts (≤5). Reshuffled per track only,
  // so the rotation through them within a single track is stable.
  const [artistFactsSampled, setArtistFactsSampled] = useState([]);
  // Landing-only fallback to /metadata/random-facts when the track has none.
  const [fallbackItems, setFallbackItems] = useState([]);
  const [status, setStatus] = useState('loading');
  // Player scope is sticky across tracks. effectiveScope (derived below) may
  // differ when the chosen scope has no data for the current track.
  const [scope, setScope] = useState('song');
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);

  const intervalMs = isPlayer ? 12000 : 6500;
  const factFontSize = isPlayer ? 14 : 15;

  // Type → label/color (used by landing variant's type chip + dot accent)
  const TYPE_META = {
    song:   { label: lang==='ru'?'О ПЕСНЕ':'ABOUT SONG',     color: 'oklch(60% 0.18 270)' },
    artist: { label: lang==='ru'?'АРТИСТ':'ARTIST',          color: 'oklch(70% 0.13 75)'  },
    misc:   { label: lang==='ru'?'ИЗ БИБЛИОТЕКИ':'LIBRARY', color: c.textMuted          },
  };

  // Fetch facts on track change
  useEffect(() => {
    if (!trackId) {
      setStatus('empty');
      setSongFacts([]); setArtistFactsAll([]); setArtistFactsSampled([]); setFallbackItems([]);
      return;
    }
    let cancelled = false;
    setStatus('loading');
    setIdx(0);

    apiFetch(`/metadata/tracks/${encodeURIComponent(trackId)}/facts?lang=${encodeURIComponent(lang)}`)
      .then(res => {
        if (cancelled) return;
        const songs = res.song_facts || [];
        const artists = res.artist_facts || [];
        setSongFacts(songs);
        setArtistFactsAll(artists);
        setArtistFactsSampled(sampleN(artists, ARTIST_SAMPLE_SIZE));

        if (songs.length || artists.length) {
          setStatus('loaded');
          setFallbackItems([]);
          return;
        }
        // Player variant: no random-collection fallback — user explicitly
        // complained about misleading "random facts on the song header".
        // Landing variant keeps the legacy fallback to fill the rail.
        if (isPlayer) {
          setStatus('empty');
          setFallbackItems([]);
          return;
        }
        apiFetch(`/metadata/random-facts?limit=5`)
          .then(rand => {
            if (cancelled) return;
            const arr = (Array.isArray(rand) ? rand : []).map(r => ({
              text: `${r.context} — ${r.fact}`,
              type: r.type === 'artist' ? 'artist' : (r.type === 'song' ? 'song' : 'misc'),
            }));
            setFallbackItems(arr);
            setStatus(arr.length ? 'loaded' : 'empty');
          })
          .catch(() => { setFallbackItems([]); setStatus('empty'); });
      })
      .catch(() => {
        if (cancelled) return;
        setSongFacts([]); setArtistFactsAll([]); setArtistFactsSampled([]); setFallbackItems([]);
        setStatus('empty');
      });

    return () => { cancelled = true; };
  }, [trackId, isPlayer]);

  // Compute what the rail actually shows. Player variant: respects toggle
  // with auto-fall-back to the other scope when the chosen one is empty.
  const haveSong = songFacts.length > 0;
  const haveArtist = artistFactsSampled.length > 0;
  const effectiveScope =
    (scope === 'song'   && !haveSong   && haveArtist) ? 'artist' :
    (scope === 'artist' && !haveArtist && haveSong)   ? 'song'   :
    scope;

  // Items used for rotation/render. Player: single-scope; landing: combined.
  let items;
  if (isPlayer) {
    items = effectiveScope === 'song'
      ? songFacts.map(f => ({ text: f, type: 'song'   }))
      : artistFactsSampled.map(f => ({ text: f, type: 'artist' }));
  } else if (fallbackItems.length) {
    items = fallbackItems;
  } else {
    items = [
      ...songFacts.map(f => ({ text: f, type: 'song'   })),
      ...artistFactsAll.map(f => ({ text: f, type: 'artist' })),
    ];
  }

  // Reset idx whenever the source list changes (scope flip or new track)
  useEffect(() => { setIdx(0); }, [effectiveScope, trackId]);

  // Auto-rotate (paused on hover)
  useEffect(() => {
    if (status !== 'loaded' || items.length < 2 || paused) return;
    const t = setInterval(() => setIdx(i => (i + 1) % items.length), intervalMs);
    return () => clearInterval(t);
    // items.length is the load-bearing dep — `items` itself is a fresh array
    // every render, so depending on it would restart the interval each tick.
  }, [status, items.length, paused, intervalMs]);

  const containerStyle = {
    borderLeft: isPlayer ? 'none' : `1px solid ${c.border}`,
    background: isPlayer ? 'transparent' : (isDark ? 'rgba(0,0,0,0.28)' : 'rgba(255,255,255,0.58)'),
    backdropFilter: isPlayer ? 'none' : 'blur(12px)',
    WebkitBackdropFilter: isPlayer ? 'none' : 'blur(12px)',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    minHeight: 0,
    padding: isPlayer ? 0 : undefined,
    border: isPlayer ? 'none' : undefined,
  };

  const headerStyle = {
    padding: isPlayer ? '0 0 2px' : '18px 22px 12px',
    borderBottom: isPlayer ? 'none' : `1px solid ${c.border}`,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 8,
  };

  // Body content selector
  let body;
  if (status === 'loading') {
    body = (
      <div style={{ padding:'24px 22px', display:'flex', flexDirection:'column', gap:10 }}>
        {[0,1,2,3].map(i => (
          <div key={i} style={{
            height: i === 0 ? 14 : 10,
            width: ['85%','95%','78%','60%'][i],
            borderRadius: 6,
            background: `linear-gradient(90deg, ${isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)'} 0%, ${isDark?'rgba(255,255,255,0.12)':'rgba(0,0,0,0.10)'} 50%, ${isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)'} 100%)`,
            backgroundSize: '200% 100%',
            animation: 'factShimmer 1.6s linear infinite',
          }} />
        ))}
      </div>
    );
  } else if (status === 'empty' || !items.length) {
    const emptyPad = isPlayer ? '10px 22px 18px' : '28px 22px';
    // Player variant: tailor message to the active scope so user knows what's missing.
    const emptyMsg = isPlayer
      ? (effectiveScope === 'artist'
          ? (lang==='ru' ? 'Фактов об артисте пока нет.' : 'No artist facts yet.')
          : (lang==='ru' ? 'Фактов о песне пока нет.'    : 'No song facts yet.'))
      : (lang==='ru'
          ? 'Факты появятся, когда библиотека будет обогащена.'
          : 'Facts will appear here once your library is enriched.');
    body = (
      <div style={{ padding: emptyPad, display:'flex', flexDirection:'column', gap:10, color: c.textMuted, fontSize:13, lineHeight:1.55 }}>
        <span style={{ width:24, height:1, background:c.border, marginBottom:4 }} />
        <span>{emptyMsg}</span>
      </div>
    );
  } else {
    const cur = items[idx] || items[0];
    const meta = TYPE_META[cur.type] || TYPE_META.misc;
    body = (
      <div style={{ position:'relative', flex:1, display:'flex', flexDirection:'column', minHeight:0 }}>
        {/* Type chip — landing only; player has the scope toggle in the header */}
        {!isPlayer && (
        <div style={{
          position:'absolute', top:14, right:18, zIndex:2,
          padding:'3px 9px', borderRadius:99,
          border:`1px solid ${c.border}`,
          fontSize:9.5, letterSpacing:'0.18em',
          color: meta.color,
          fontFamily: "'JetBrains Mono', ui-monospace, monospace",
          background: isDark ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.6)',
        }}>{meta.label}</div>
        )}

        {/* Fact body — key includes scope+trackId so factIn animation replays
            on EVERY transition, not just idx changes. Without scope in the key,
            toggling tabs while at idx=0 reused the same React node and the
            new fact text appeared silently with no transition. */}
        <div key={`${trackId}-${effectiveScope}-${idx}`}
          onMouseEnter={() => setPaused(true)}
          onMouseLeave={() => setPaused(false)}
          style={{
            padding: isPlayer ? '2px 22px 18px' : '24px 22px 18px',
            flex:1,
            display:'flex', flexDirection:'column', gap: isPlayer ? 6 : 10,
            animation: 'factIn 0.55s cubic-bezier(.22,.9,.3,1) both',
            overflowY: 'auto',
          }}
        >
          <span style={{ width:24, height:1, background:c.border, marginTop: isPlayer ? 0 : 6 }} />
          <div className="serif" style={{
            fontSize:factFontSize, lineHeight:1.55, color:c.text, letterSpacing:'-0.005em',
            fontWeight:400,
          }}><MarkdownText text={cur.text} /></div>
        </div>

        {/* Dot indicator — landing variant only (player dots are in header) */}
        {!isPlayer && items.length > 1 && (
          <div style={{
            display:'flex', justifyContent:'center', alignItems:'center', gap:7,
            padding:'10px 22px 16px',
            borderTop:`1px solid ${c.border}`,
          }}>
            {items.slice(0, 8).map((_, i) => {
              const active = i === idx;
              return (
                <button key={i}
                  onClick={() => setIdx(i)}
                  aria-label={`Fact ${i+1}`}
                  style={{
                    width: active ? 16 : 5, height: 5, borderRadius: 99,
                    background: active ? (accent || meta.color) : (isDark?'rgba(255,255,255,0.2)':'rgba(0,0,0,0.18)'),
                    border:'none', padding:0, cursor:'pointer',
                    transition:'all 0.35s cubic-bezier(.22,.9,.3,1)',
                  }}
                />
              );
            })}
            {items.length > 8 && (
              <span className="mono" style={{ fontSize:10, color:c.textSubtle, marginLeft:4 }}>
                +{items.length - 8}
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <aside style={containerStyle}>
      <div style={headerStyle}>
        {isPlayer ? (
          <div className="facts-scope-toggle">
            <button
              type="button"
              data-active={effectiveScope === 'song'}
              disabled={!haveSong}
              onClick={() => { if (haveSong) setScope('song'); }}
              aria-pressed={effectiveScope === 'song'}
              title={haveSong ? '' : (lang==='ru'?'Нет фактов о песне':'No song facts')}
            >{lang==='ru'?'ПЕСНЯ':'SONG'}</button>
            <button
              type="button"
              data-active={effectiveScope === 'artist'}
              disabled={!haveArtist}
              onClick={() => { if (haveArtist) setScope('artist'); }}
              aria-pressed={effectiveScope === 'artist'}
              title={haveArtist ? '' : (lang==='ru'?'Нет фактов об артисте':'No artist facts')}
            >{lang==='ru'?'АРТИСТ':'ARTIST'}</button>
          </div>
        ) : (
          <span className="mono-label" style={{
            fontSize: 11,
            letterSpacing: '0.22em',
            color: c.textSubtle,
          }}>
            {lang==='ru'?'※ ЗНАЕШЬ ЛИ ТЫ':'※ DID YOU KNOW'}
          </span>
        )}
        {/* Player variant: prev/next arrows + "n / m" counter. Manual nav
            pauses the auto-rotation briefly so the chosen fact stays put. */}
        {isPlayer && status === 'loaded' && items.length > 1 && (() => {
          const nudge = (delta) => {
            setIdx(i => (i + delta + items.length) % items.length);
            setPaused(true);
            setTimeout(() => setPaused(false), 5000);
          };
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <button className="fact-nav-btn" onClick={() => nudge(-1)}
                aria-label={lang==='ru'?'Предыдущий факт':'Previous fact'}>‹</button>
              <span className="mono" style={{
                fontSize: 10, color: c.textSubtle, letterSpacing: '0.1em',
                minWidth: 32, textAlign: 'center',
              }}>{idx + 1} / {items.length}</span>
              <button className="fact-nav-btn" onClick={() => nudge(1)}
                aria-label={lang==='ru'?'Следующий факт':'Next fact'}>›</button>
            </div>
          );
        })()}
        {/* Landing variant: numeric counter */}
        {!isPlayer && status === 'loaded' && items.length > 1 && (
          <span className="mono" style={{ fontSize:10, color:c.textSubtle, letterSpacing:'0.1em' }}>
            {String(idx+1).padStart(2,'0')} / {String(items.length).padStart(2,'0')}
          </span>
        )}
      </div>
      {body}
    </aside>
  );
}

// ─── Spinner ──────────────────────────────────────────────────────────────────
function Spinner({ size = 16, color }) {
  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      border: `2px solid ${color ? color + '30' : 'rgba(255,255,255,0.12)'}`,
      borderTopColor: color || 'oklch(60% 0.18 270)',
      borderRadius: '50%', animation: 'spin 0.7s linear infinite',
    }} />
  );
}

// ─── SkeRange ───────────────────────────────────────────────────────────────
// Skeuomorphic slider: a carved groove + a raised knob. The native <input>
// rides transparently on top so drag / track-click / keyboard / a11y all work
// for free, while the visible fill + knob mirror its value. Knob & fill are
// inset by the knob radius so the knob never spills past the rail.
//   • accent   — per-instance fill/glow colour (purple, gold, …)
//   • animated — knob & fill glide between stops (use for fixed-value sliders;
//                leave off for continuous drags so the knob tracks the finger)
//   • bipolar  — fill grows out from the centre (zero) instead of the left
const SKE_RNG_KNOB = 18; // px — keep in sync with --rng-knob
function SkeRange({ value, min = 0, max = 100, step = 1, onChange,
                    accent = 'oklch(62% 0.2 275)', animated = false,
                    bipolar = false, disabled = false, ariaLabel, style }) {
  const range = (max - min) || 1;
  const clamp = p => Math.max(0, Math.min(100, p));
  const pct = clamp(((value - min) / range) * 100);
  // Centre position of the knob at fraction p (inset by the knob radius).
  const center = p => `calc(${SKE_RNG_KNOB / 2}px + (100% - ${SKE_RNG_KNOB}px) * ${p / 100})`;

  let fillStyle;
  if (bipolar) {
    const zero = clamp(((0 - min) / range) * 100);
    const lo = Math.min(pct, zero);
    const hi = Math.max(pct, zero);
    fillStyle = { left: center(lo), width: `calc((100% - ${SKE_RNG_KNOB}px) * ${(hi - lo) / 100})` };
  } else {
    fillStyle = { left: 0, width: center(pct) };
  }

  const cls = 'ske-rng'
    + (animated ? ' is-animated' : '')
    + (disabled ? ' is-disabled' : '');

  return (
    <div className={cls} style={{ '--rng-accent': accent, ...style }}>
      <div className="ske-rng-track" />
      <div className="ske-rng-fill" style={fillStyle} />
      <div className="ske-rng-knob" style={{ left: center(pct) }} />
      <input type="range" className="ske-rng-input"
        min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        aria-label={ariaLabel} />
    </div>
  );
}

// ─── Cover thumbnails ─────────────────────────────────────────────────────────
// The covers endpoint generates and caches downscaled variants on ?w=320.
// Grids, rows and mosaics use them; the player (hero art, ambient wash, color
// sampling) keeps the full-size original — those three must share one URL so
// the browser cache is hit once and canvas sampling reuses the same entry.
function thumbCoverUrl(url, w = 320) {
  if (!url || typeof url !== 'string') return url;
  if (url.startsWith('blob:') || url.startsWith('data:')) return url;
  if (/[?&]w=\d/.test(url)) return url;              // already a thumbnail URL
  return url.includes('?') ? `${url}&w=${w}` : `${url}?w=${w}`;
}

// ─── AlbumCover ───────────────────────────────────────────────────────────────
function AlbumCover({ title='', artist='', size=44, isDark, coverPath, radius, fluid, eager }) {
  const hue = ((title.charCodeAt(0)||65)*37 + (artist.charCodeAt(0)||65)*17) % 360;
  const br = radius ?? (typeof size === 'number' ? (size > 100 ? 14 : size > 60 ? 10 : 8) : 0);
  const boxStyle = fluid
    ? { width:'100%', height:'100%', borderRadius:`${br}px`, overflow:'hidden',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.45), inset 0 0 0 1px rgba(0,0,0,0.5), 0 3px 8px rgba(0,0,0,0.55)'
          : 'inset 0 1px 0 rgba(255,255,255,0.85), inset 0 -1px 0 rgba(0,0,0,0.1), inset 0 0 0 1px rgba(0,0,0,0.08), 0 3px 7px rgba(40,30,60,0.13)' }
    : { width: size, height: size, borderRadius:`${br}px`, flexShrink:0, overflow:'hidden',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.14), inset 0 -1px 0 rgba(0,0,0,0.45), inset 0 0 0 1px rgba(0,0,0,0.5), 0 3px 8px rgba(0,0,0,0.55)'
          : 'inset 0 1px 0 rgba(255,255,255,0.85), inset 0 -1px 0 rgba(0,0,0,0.1), inset 0 0 0 1px rgba(0,0,0,0.08), 0 3px 7px rgba(40,30,60,0.13)' };

  const fs = (typeof size === 'number' ? size : 200) > 60 ? '22px' : '12px';

  if (coverPath) {
    // `eager` marks the player's hero cover (and its transition snapshot) —
    // the one place that needs the full-resolution original. Everything else
    // (list rows, grids, chips) renders small → server thumbnail.
    const fullSrc = coverPath.startsWith('http') ? coverPath : `${API}${coverPath}`;
    const imgSrc = eager ? fullSrc : thumbCoverUrl(fullSrc);
    return (
      <div style={boxStyle}>
        <img
          src={imgSrc} alt=""
          // Native lazy-load: the browser only fetches covers in/near the
          // viewport. Sections stay mounted-but-hidden (visibility:hidden,
          // width:0) across navigation, and an <img src> in a hidden section
          // STILL downloads eagerly without this — so a 5000-album library
          // (singles-heavy Yandex import) would fire 5000 cover requests the
          // moment you leave Home, starving the audio stream on the same
          // origin. loading="lazy" defers every off-screen/hidden cover so
          // playback wins the connection pool. decoding="async" keeps decode
          // off the main thread.
          // `eager` opts back into sync paint for the player's hero cover and
          // its outgoing transition snapshot: both remount mid-animation with
          // an already-cached URL, and lazy+async there costs a blank frame
          // (the cover "blinks" before flying out on track change).
          loading={eager ? 'eager' : 'lazy'}
          decoding={eager ? 'sync' : 'async'}
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          onError={e => {
            e.currentTarget.style.display = 'none';
            const parent = e.currentTarget.parentElement;
            const span = document.createElement('span');
            span.style.cssText = `display:flex;align-items:center;justify-content:center;width:100%;height:100%;
              background:linear-gradient(135deg, oklch(38% 0.13 ${hue}), oklch(52% 0.18 ${(hue+45)%360}));
              font-size:${fs};font-weight:700;color:rgba(255,255,255,0.65);
              font-family:'JetBrains Mono', monospace;letter-spacing:0.5px;`;
            span.textContent = (title||'?').slice(0,2).toUpperCase();
            parent.appendChild(span);
          }}
        />
      </div>
    );
  }
  return (
    <div style={{
      ...boxStyle,
      background: `linear-gradient(135deg, oklch(38% 0.13 ${hue}), oklch(52% 0.18 ${(hue+45)%360}))`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: fs, fontWeight: '700',
      color: 'rgba(255,255,255,0.65)', fontFamily: "'JetBrains Mono', monospace",
      letterSpacing: '0.5px',
    }}>
      {(title||'?').slice(0,2).toUpperCase()}
    </div>
  );
}

// ─── LAZY COVER ───────────────────────────────────────────────────────────────
// CSS background-image covers download EAGERLY the moment the element renders
// (even in a hidden section) — loading="lazy" only exists on a real <img>.
// List artwork goes through this so off-screen covers never compete with the
// audio stream for the 6-per-origin connection pool (same reasoning as the
// loading="lazy" in AlbumCover above). No artwork → a plain gradient div.
function LazyCover({ url, className, style, fallback = 'linear-gradient(135deg, rgba(124,91,255,.35) 0%, rgba(255,120,200,.25) 100%)' }) {
  // Key order matters in the fallback: `background` (shorthand) spread AFTER
  // style wins over any backgroundColor placeholder carried in `style`.
  if (!url) return <div className={className} style={{ ...style, background: fallback }} />;
  return (
    <img src={thumbCoverUrl(url)} alt="" loading="lazy" decoding="async" className={className}
         // The gradient doubles as a loading placeholder behind the not-yet-
         // decoded image; a 404 swaps in a transparent pixel (data: can't
         // re-error) so the gradient shows instead of a broken-image glyph.
         style={{ objectFit: 'cover', background: fallback, ...style }}
         onError={e => { e.currentTarget.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'; }} />
  );
}

// ─── MOSAIC COVER ─────────────────────────────────────────────────────────────
function MosaicCover({ trackIds = [], coverPaths = [], size = 220, radius = 12 }) {
  const n = Math.min(trackIds.length, 4);
  // A numeric size is a fixed px square. A flexible size (e.g. '100%') would
  // collapse to 0 height if we naively set height:'100%' and the parent has no
  // defined height (the playlist card's relative wrapper is auto-height) — so
  // for non-numeric sizes we enforce squareness with aspect-ratio instead.
  const dim = typeof size === 'number'
    ? { width: size, height: size }
    : { width: size, aspectRatio: '1 / 1' };
  if (n === 0) {
    return (
      <div style={{
        ...dim, borderRadius: radius, overflow: 'hidden',
        background: 'linear-gradient(135deg, rgba(124,91,255,.22) 0%, rgba(255,120,200,.14) 100%)',
        display: 'grid', placeItems: 'center', color: 'rgba(255,255,255,.45)',
        fontSize: typeof size === 'number' ? Math.round(size * 0.34) : 36,
      }}>♫</div>
    );
  }

  const grid = {
    ...dim, borderRadius: radius, overflow: 'hidden',
    display: 'grid', gap: 1, background: 'rgba(0,0,0,.4)',
  };
  if (n === 1) Object.assign(grid, { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' });
  if (n === 2) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr' });
  if (n === 3) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' });
  if (n === 4) Object.assign(grid, { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' });

  return (
    <div style={grid}>
      {Array.from({ length: n }).map((_, i) => (
        <LazyCover key={i}
          url={coverPaths[i] ? `${API}${coverPaths[i]}` : null}
          style={{
            width: '100%', height: '100%', minWidth: 0, minHeight: 0, display: 'block',
            backgroundColor: 'rgba(255,255,255,.04)',
            ...(n === 3 && i === 0 ? { gridColumn: '1 / -1' } : {}),
          }}
        />
      ))}
    </div>
  );
}

// ─── ToggleSwitch (rocker button) ─────────────────────────────────────────────
function ToggleSwitch({ checked, onChange, isDark }) {
  const leftActive = !checked;
  const rightActive = checked;
  const active = 'linear-gradient(180deg, oklch(67% 0.18 270), oklch(54% 0.22 280))';
  const activeShadow = 'inset 0 1px 3px rgba(0,0,0,0.35), 0 0 8px oklch(60% 0.18 270 / 0.4)';
  const inactiveD = {
    background: 'linear-gradient(180deg, #2c2c34, #1c1c23)',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.12), 0 1px 2px rgba(0,0,0,0.4)',
  };
  const inactiveL = {
    background: 'linear-gradient(180deg, #fff, #e1e0ea)',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,1), 0 1px 2px rgba(40,30,60,0.1)',
  };

  return (
    <button
      onClick={() => onChange(!checked)}
      style={{
        width: '44px', height: '22px', borderRadius: '8px',
        display: 'flex', flexShrink: 0, padding: '2px',
        background: isDark ? '#15151a' : '#d8d7df',
        boxShadow: isDark
          ? 'inset 0 2px 4px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.04)'
          : 'inset 0 2px 4px rgba(40,30,60,0.15), 0 1px 0 rgba(255,255,255,0.8)',
        border: 'none', cursor: 'pointer', gap: '2px',
      }}
    >
      <span style={{
        flex: 1, borderRadius: '6px', transition: 'all 0.2s',
        ...(leftActive
          ? { background: active, boxShadow: activeShadow }
          : (isDark ? inactiveD : inactiveL)),
      }} />
      <span style={{
        flex: 1, borderRadius: '6px', transition: 'all 0.2s',
        ...(rightActive
          ? { background: active, boxShadow: activeShadow }
          : (isDark ? inactiveD : inactiveL)),
      }} />
    </button>
  );
}

// ─── Knob (skeuomorphic toggle) ───────────────────────────────────────────────
function Knob({ angle = -45, size = 44, isDark, label, onClick, glow }) {
  const c = useColors(isDark);
  return (
    <button
      onClick={onClick}
      title={label}
      style={{
        width: size, height: size, borderRadius: '50%', flexShrink: 0,
        position: 'relative', cursor: 'pointer', padding: 0,
        background: isDark
          ? 'radial-gradient(circle at 30% 25%, #38383f 0%, #1a1a20 65%, #0c0c10 100%)'
          : 'radial-gradient(circle at 30% 25%, #ffffff 0%, #d8d7df 65%, #b8b7c1 100%)',
        boxShadow: isDark
          ? 'inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.55), 0 0 0 1px rgba(0,0,0,0.6)'
          : 'inset 0 1px 0 rgba(255,255,255,1), inset 0 -1px 0 rgba(0,0,0,0.13), 0 2px 5px rgba(40,30,60,0.18), 0 0 0 1px rgba(0,0,0,0.1)',
      }}
    >
      <div className="knob" style={{
        position:'absolute', inset:'0', borderRadius:'50%',
        transform:`rotate(${angle}deg)`,
      }}>
        <div style={{
          position:'absolute', top:'12%', left:'50%', transform:'translateX(-50%)',
          width:'2.5px', height:'30%', borderRadius:'2px',
          background: glow ? c.amber : (isDark ? '#9a99a4' : '#6e6c7a'),
          boxShadow: glow ? `0 0 8px ${c.amberGlow}` : 'none',
        }} />
      </div>
    </button>
  );
}

// ─── ResultCard ───────────────────────────────────────────────────────────────
function ResultCard({ hit, isDark, c, compact=false, onClick, onPlay }) {
  const track = hit.track || {};
  const modeColor = { lyrics: c.accent, audio: 'oklch(60% 0.18 310)', hybrid: c.amber }[hit.matched_on] || c.accent;
  return (
    <div
      onClick={onClick}
      onMouseEnter={e => { e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
      style={{ display:'flex', alignItems:'center', gap:'12px', padding: compact?'9px 12px':'12px 14px',
        borderRadius:'12px', cursor: onClick?'pointer':'default', transition:'background 0.15s' }}
    >
      <AlbumCover title={track.title} artist={track.artist} size={compact?40:50} isDark={isDark} coverPath={track.cover_art_path} />
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:'14px', fontWeight:'600', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', letterSpacing:'-0.01em' }}>{track.title||'—'}</div>
        <div style={{ fontSize:'14px', color:c.textMuted, marginTop:'2px' }}>{track.artist||'—'}{track.year?` · ${track.year}`:''}</div>
      </div>
      {onPlay && (
        <button onClick={(e) => { e.stopPropagation(); onPlay(hit); }}
          title="Play"
          style={{
            width:'34px', height:'34px', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center',
            flexShrink:0, background:'linear-gradient(180deg, oklch(60% 0.21 270), oklch(48% 0.22 285))',
            boxShadow:'inset 0 1px 0 rgba(255,255,255,0.2), inset 0 -1px 0 rgba(0,0,0,0.25), 0 2px 8px oklch(58% 0.21 270 / 0.4)',
            fontSize:'13px', color:'white',
          }}>
          ▶
        </button>
      )}
      <div style={{ display:'flex', flexDirection:'column', alignItems:'flex-end', gap:'4px', flexShrink:0 }}>
        <div className="mono" style={{ padding:'3px 9px', borderRadius:'20px', fontSize:'14px', fontWeight:'500',
          background: modeColor.replace(')', ' / 0.14)'),
          color: modeColor, border: `1px solid ${modeColor.replace(')', ' / 0.3)')}` }}>
          {Math.round((hit.score||0)*100)}% · {hit.matched_on}
        </div>
        {track.genre && <span className="mono" style={{ fontSize:'14px', color:c.textSubtle }}>{track.genre}</span>}
      </div>
    </div>
  );
}

// ─── Chat history ─────────────────────────────────────────────────────────────
function useChatHistory(userId) {
  const storageKey = `musix_chat_${userId || 'default'}`;
  const load = useCallback(() => {
    try { const raw = localStorage.getItem(storageKey); return raw ? JSON.parse(raw) : []; }
    catch { return []; }
  }, [storageKey]);
  const [sessions, setSessions] = useState(load);
  useEffect(() => { setSessions(load()); }, [storageKey, load]);
  // Upserts: pass the session id to update an ongoing conversation in place
  // (and bump it to the top) instead of minting a duplicate per answer.
  // Returns the session id so the caller can keep threading saves into it.
  const saveSession = useCallback((msgs, id = null) => {
    const first = (msgs || []).find(m => m.role === 'user');
    if (!first) return null;  // nothing worth keeping without a user message
    const ts = Date.now();
    const sid = id || ts;
    const title = first.text?.slice(0, 48) || 'Чат';
    setSessions(prev => {
      const entry = { id: sid, title, time: ts, messages: msgs };
      const rest = prev.filter(s => s.id !== sid);
      const updated = [entry, ...rest.slice(0, 19)];
      try { localStorage.setItem(storageKey, JSON.stringify(updated)); } catch {}
      return updated;
    });
    return sid;
  }, [storageKey]);
  const deleteSession = useCallback((id) => {
    setSessions(prev => {
      const updated = prev.filter(s => s.id !== id);
      try { localStorage.setItem(storageKey, JSON.stringify(updated)); } catch {}
      return updated;
    });
  }, [storageKey]);
  return { sessions, saveSession, deleteSession };
}

// ─── useTrackChat — per-track localStorage chat history ──────────────────────
function useTrackChat(trackId, userId, lang) {
  const storageKey = `chatHistory:track:${trackId || '_none'}:${userId || '_anon'}`;

  const readMsgs = useCallback(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }, [storageKey]);

  const [messages, setMessages] = useState(readMsgs);

  useEffect(() => { setMessages(readMsgs()); }, [storageKey, readMsgs]);

  const persist = useCallback((msgs) => {
    try { localStorage.setItem(storageKey, JSON.stringify(msgs)); } catch {}
  }, [storageKey]);

  // Streams /chat/track-chat/stream: `status` frames update the trailing
  // loading message's `activity` timeline (thinking → web_search → reading…)
  // so the drawer can narrate what's happening; the terminal `answer` frame
  // replaces it. Falls back to the non-streaming endpoint on any failure.
  const sendMessage = useCallback(async (text, trackContext, llmKw) => {
    const stamp = () => new Date().toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' });
    const userMsg = { role: 'user', text, time: stamp() };
    const loadingMsg = { role: 'assistant', loading: true, activity: [{ stage: 'thinking' }], time: userMsg.time };
    setMessages([...messages, userMsg, loadingMsg]);
    persist([...messages, userMsg]);  // never persist the transient loading row

    const body = {
      track_context: trackContext,
      mode: 'song',
      message: text,
      lang,
      history: messages.filter(m => !m.loading).map(m => ({
        role: m.role, content: m.text || '',
      })),
      ...(llmKw || {}),
    };

    const finalize = (aiMsg) => {
      const finalMsgs = [...messages, userMsg, aiMsg];
      setMessages(finalMsgs);
      persist(finalMsgs);
    };
    const patchLoading = (fn) => setMessages(prev => {
      const copy = prev.slice();
      const li = copy.length - 1;
      if (!copy[li] || !copy[li].loading) return prev;
      copy[li] = fn(copy[li]);
      return copy;
    });

    let answered = false;
    const onEvent = (ev) => {
      if (ev.type === 'answer') {
        answered = true;
        finalize({ role: 'assistant', text: ev.message || '…', web_search_used: ev.web_search_used, time: stamp() });
      } else if (ev.type === 'error') {
        answered = true;
        finalize({ role: 'assistant', text: `Ошибка: ${ev.message || ''}`, time: stamp() });
      } else if (ev.type === 'status' && ev.stage) {
        patchLoading(m => {
          const acts = m.activity || [];
          const last = acts[acts.length - 1];
          if (last && last.stage === ev.stage && last.query === ev.query) return m;
          return { ...m, activity: [...acts, { stage: ev.stage, query: ev.query }] };
        });
      }
    };

    try {
      await apiStream('/chat/track-chat/stream', body, onEvent);
      if (!answered) throw new Error('stream ended without answer');
    } catch (e) {
      if (answered) return;
      try {
        const res = await apiFetch('/chat/track-chat', { method: 'POST', body: JSON.stringify(body) });
        finalize({ role: 'assistant', text: res.message || '…', web_search_used: res.web_search_used, time: stamp() });
      } catch (e2) {
        finalize({ role: 'assistant', text: `Ошибка: ${e2.message}`, time: stamp() });
      }
    }
  }, [messages, persist, lang]);

  const clearChat = useCallback(() => {
    setMessages([]);
    persist([]);
  }, [persist]);

  return { messages, sendMessage, clearChat };
}

// ─── Brand mark ───────────────────────────────────────────────────────────────
const BRAND_EQ_BARS = [
  { h: 0.44, c: 'oklch(64% 0.19 268)' },
  { h: 0.70, c: 'oklch(69% 0.17 278)' },
  { h: 0.94, c: 'oklch(74% 0.14 292)' },
  { h: 0.60, c: 'oklch(72% 0.09 350)' },
  { h: 0.36, c: 'oklch(76% 0.13 80)' },
];
function BrandMark({ size=34, isDark }) {
  const barW = 3.1, gap = 1.3;
  const totalW = BRAND_EQ_BARS.length * barW + (BRAND_EQ_BARS.length - 1) * gap;
  const startX = (24 - totalW) / 2;
  return (
    <div style={{
      width:size, height:size, borderRadius:`${size*0.28}px`, flexShrink:0, position:'relative',
      background:'linear-gradient(150deg, #211c30, #0c0a13)',
      display:'flex', alignItems:'center', justifyContent:'center',
      boxShadow: isDark
        ? 'inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -1px 0 rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,0,0,0.4)'
        : 'inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.3), 0 3px 10px rgba(0,0,0,0.35), 0 0 0 1px rgba(0,0,0,0.08)',
    }}>
      <svg width={size*0.6} height={size*0.6} viewBox="0 0 24 24" fill="none">
        {BRAND_EQ_BARS.map((b, i) => {
          const h = b.h * 19;
          const x = startX + i * (barW + gap);
          const y = 12 - h / 2;
          return <rect key={i} x={x} y={y} width={barW} height={h} rx={barW/2} fill={b.c} />;
        })}
      </svg>
    </div>
  );
}

// ─── Top right corner: settings + theme + lang (used on landing) ──────────────
function TopRightControls({ isDark, lang, onLang, onTheme, onSettings, floating=false, showTheme=true, showSettings=true, showLang=true }) {
  const c = useColors(isDark);
  const wrap = floating ? {
    position:'absolute', top:'24px', right:'28px', zIndex:5,
  } : {};
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', ...wrap }}>
      {/* Lang switch — segmented physical control */}
      {showLang && (
      <div className={ske('inset', isDark)} style={{ display:'flex', padding:'3px', borderRadius:'10px', gap:'2px' }}>
        {['ru','en'].map(l => {
          const active = lang === l;
          return (
            <button key={l} onClick={() => onLang(l)} className={active ? ske('btn', isDark) : ''}
              style={{
                padding:'5px 11px', borderRadius:'7px', fontFamily:"'JetBrains Mono', monospace",
                fontSize:'15px', fontWeight:'600', letterSpacing:'0.4px',
                color: active ? c.text : c.textSubtle,
                transition:'color 0.2s',
              }}>{l.toUpperCase()}</button>
          );
        })}
      </div>
      )}
      {/* Theme knob */}
      {showTheme && (
        <Knob size={36} isDark={isDark} angle={isDark ? -55 : 55} glow={!isDark}
          label={isDark ? 'Light theme' : 'Dark theme'} onClick={onTheme} />
      )}
      {/* Settings — round button */}
      {showSettings && (
        <button onClick={onSettings} className={ske('btn', isDark)} title={lang==='ru'?'Настройки':'Settings'}
          style={{
            width:'36px', height:'36px', borderRadius:'50%',
            display:'flex', alignItems:'center', justifyContent:'center',
            color: c.textMuted, transition:'transform 0.5s ease',
          }}
          onMouseEnter={e => { e.currentTarget.firstChild.style.transform = 'rotate(60deg)'; }}
          onMouseLeave={e => { e.currentTarget.firstChild.style.transform = 'rotate(0)'; }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{ transition:'transform 0.5s cubic-bezier(.22,.9,.3,1)' }}>
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
        </button>
      )}
    </div>
  );
}

// ─── Home (Discovery Magazine) shared helpers + blocks ───────────────────────
function homeCoverUrl(path) {
  if (!path) return null;
  // Home cards / rec rows render covers small — the ?w=320 server thumbnail
  // is ~25× lighter than the full embedded art (mobile data + battery).
  return thumbCoverUrl(path.startsWith('http') ? path : `${API}${path}`);
}

// Shimmer skeleton block. Sizes accept px numbers or CSS strings (e.g. '80%').
function Skel({ w, h, r, isDark, style }) {
  return (
    <div className={isDark ? 'load-skel' : 'load-skel load-skel--l'}
         style={{ width:w, height:h, borderRadius:(r==null?8:r), ...(style||{}) }} />
  );
}

// Thin progress arc around the orb — a watch-bezel "dial" for the current
// track: status is visible with zero text. Isolated component on purpose:
// useCurrentTime ticks every frame, and this way only the SVG re-renders,
// not the whole hero.
function OrbProgressArc({ audio }) {
  const currentTime = useCurrentTime();
  const duration = (audio && audio.duration) || 0;
  const p = duration > 0 ? Math.min(1, currentTime / duration) : 0;
  const R = 35;
  const C = 2 * Math.PI * R;
  return (
    <svg className="fy-dial" viewBox="0 0 76 76" aria-hidden="true">
      <circle cx="38" cy="38" r={R} fill="none" stroke="rgba(255,255,255,.16)" strokeWidth="1.3" />
      <circle cx="38" cy="38" r={R} fill="none" stroke="rgba(255,255,255,.92)" strokeWidth="1.6"
        strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - p)}
        transform="rotate(-90 38 38)" style={{ transition: 'stroke-dashoffset .35s linear' }} />
    </svg>
  );
}

// ─── Cursor grammar (shared: landing + recommend) ───────────────────────────
// Cursor spotlight (--mx/--my) + optional 3D tilt (--rx/--ry) — the hover
// grammar: clickable cards tilt, static panels only get the light spot.
// CSS vars are written straight on the node — no state, no re-renders.
const spotHandlers = (tilt) => ({
  onPointerMove: (e) => {
    const el = e.currentTarget, r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    el.style.setProperty('--mx', `${px * 100}%`);
    el.style.setProperty('--my', `${py * 100}%`);
    if (tilt) {
      el.style.setProperty('--ry', `${((px - 0.5) * 6).toFixed(2)}deg`);
      el.style.setProperty('--rx', `${((0.5 - py) * 6).toFixed(2)}deg`);
    }
  },
  onPointerLeave: (e) => {
    e.currentTarget.style.setProperty('--rx', '0deg');
    e.currentTarget.style.setProperty('--ry', '0deg');
  },
});
// Liquid refraction inside the stream button: blobs at different "depths"
// shift toward the cursor with different strengths (--lx/--ly ∈ -.5…+.5).
const lqHandlers = {
  onPointerMove: (e) => {
    const el = e.currentTarget, r = el.getBoundingClientRect();
    el.style.setProperty('--lx', ((e.clientX - r.left) / r.width - 0.5).toFixed(3));
    el.style.setProperty('--ly', ((e.clientY - r.top) / r.height - 0.5).toFixed(3));
  },
  onPointerLeave: (e) => {
    e.currentTarget.style.setProperty('--lx', '0');
    e.currentTarget.style.setProperty('--ly', '0');
  },
};

// ─── Orb palette from the 6 sonic axes ──────────────────────────────────────
// Long-term taste → color: each axis has a fixed hue; the two axes with the
// largest |z| paint the orb gradient (top-1 = base hue, top-2 = second stop)
// and the page aurora. Saturation follows |z| clamped into a dusty band —
// the orb "sounds like" the listener without going neon. No axes → null
// (callers fall back to the brand violet).
const AXIS_HUES = {
  energy: 20, brightness: 48, acousticness: 145,
  spacious: 200, experimental: 275, vocal_lead: 330,
};
function axisPalette(axes) {
  const entries = Object.entries(AXIS_HUES)
    .map(([name, hue]) => {
      const a = axes && axes[name];
      const z = a && typeof a.z === 'number' ? a.z : null;
      return z == null ? null : { hue, az: Math.abs(z) };
    })
    .filter(Boolean)
    .sort((a, b) => b.az - a.az);
  if (!entries.length) return null;
  const sat = (az) => Math.round(Math.max(38, Math.min(55, 38 + az * 12)));
  const top = entries[0], second = entries[1] || entries[0];
  return {
    h1: top.hue, s1: sat(top.az),
    h2: second.hue, s2: sat(second.az),
  };
}

function ForYouHero({ isDark, lang, onStartStream, streamActive, audio, navigateToArtist, onPlayTrack, onPalette }) {
  const c = useColors(isDark);
  const isMobile = useIsMobile();
  const [profile, setProfile] = useState(null);
  const [vibe, setVibe] = useState(null);          // { phrase, source } | null
  const [vibeLoading, setVibeLoading] = useState(true);
  const [hoverCtl, setHoverCtl] = useState(false);
  const [loading, setLoading] = useState(true);
  const [likedShare, setLikedShare] = useState(0.3);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const shareTimer = useRef(null);

  // Derived BEFORE the hooks that depend on them — @babel/standalone hoists
  // const→var, so a later declaration would silently read as undefined in deps.
  const islands = (profile && profile.islands) || [];
  const hasProfile = islands.length > 0;
  const anchors = islands
    .map(isl => (isl.tracks && isl.tracks[0]) || null)
    .filter(Boolean);
  // «Вайбики» — the days-scale mood layer; server sends them strongest-first
  // (max 3). Empty → the row is simply absent, no empty state.
  const vibes = (profile && profile.vibes) || [];
  // Taste axes → dusty palette for the orb + page aurora (see axisPalette).
  const pal = axisPalette(profile && profile.axes);
  // The top anchor's cover hue stays as a third accent in the aurora, so the
  // wave still carries a trace of the actual records. Hook count is fixed:
  // a missing anchor yields a null URL.
  const topCoverUrl = homeCoverUrl(anchors[0] && anchors[0].cover_art_path);
  const color = useCoverColor(topCoverUrl);

  useEffect(() => {
    let alive = true;
    apiFetch(`/recommend/profile?lang=${encodeURIComponent(lang)}`)
      .then(r => {
        if (!alive || !r) return;
        setProfile(r);
        if (typeof r.liked_share === 'number') setLikedShare(r.liked_share);
      })
      .catch(() => {})
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  // The wave/vibe phrase — a short AI line over long- + short-term taste. The
  // endpoint returns an instant phrase (cached AI or deterministic fallback)
  // and warms the LLM cache in the background, so the hero never blocks.
  useEffect(() => {
    let alive = true;
    setVibeLoading(true);
    apiFetch(`/recommend/taste-vibe?lang=${encodeURIComponent(lang)}`)
      .then(r => { if (alive) setVibe(r && r.phrase ? r : null); })
      .catch(() => { if (alive) setVibe(null); })
      .finally(() => { if (alive) setVibeLoading(false); });
    return () => { alive = false; };
  }, [lang]);

  // Persist the liked/new slider (debounced) — the stream reads it server-side
  // on every chunk, so moving it mid-stream takes effect on the next chunk.
  const setShare = (v) => {
    setLikedShare(v);
    if (shareTimer.current) clearTimeout(shareTimer.current);
    shareTimer.current = setTimeout(() => {
      apiFetch('/recommend/stream/settings', {
        method: 'PUT', body: JSON.stringify({ liked_share: v }),
      }).catch(() => {});
    }, 350);
  };

  const startStream = () => { if (onStartStream) onStartStream(); };
  // For-You orb behaviour. When the live queue IS the wave (streamActive), the
  // orb becomes a play/pause control for it (no re-navigation). Otherwise — an
  // empty queue, or one the user hand-picked — a click starts a fresh wave
  // (which throws into the player and plays). See App.startStream / streamActive.
  const isPlaying = !!(audio && audio.isPlaying);
  const waveLive = !!streamActive;          // the wave is the active queue
  const wavePlaying = waveLive && isPlaying; // …and it's actually playing now
  const handleOrb = () => {
    if (waveLive) { if (audio && audio.togglePlay) audio.togglePlay(); }
    else { startStream(); }
  };
  // Cursor parallax for the orb blobs — the refraction move: liquid pulls
  // toward the pointer. CSS vars are written straight on the node (no state →
  // no re-renders at mousemove rate); per-blob depth lives in index.css.
  const orbRef = useRef(null);
  const orbMove = (e) => {
    const el = orbRef.current;
    if (!el || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--px', (((e.clientX - r.left) / r.width) - 0.5).toFixed(3));
    el.style.setProperty('--py', (((e.clientY - r.top) / r.height) - 0.5).toFixed(3));
  };
  const orbLeave = () => {
    setHoverCtl(false);
    const el = orbRef.current;
    if (el) { el.style.setProperty('--px', '0'); el.style.setProperty('--py', '0'); }
  };
  const phrase = vibe && vibe.phrase;

  // Launch spring: a short press-in bounce on the orb when a fresh wave starts
  // (the playing state then takes over). Class is applied off-state via the
  // ref — no re-render for a 650ms cosmetic.
  const launchSpring = () => {
    const el = orbRef.current;
    if (!el || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
    el.classList.add('fy-launching');
    setTimeout(() => { el.classList.remove('fy-launching'); }, 650);
  };
  const handleOrbClick = () => {
    if (!waveLive) launchSpring();
    handleOrb();
  };

  // «Вайбик» → radio: CLAP-neighbour queue seeded by the vibe's anchor track,
  // the anchor itself leads (same recipe as the recommend islands).
  const vibeRadio = async (v) => {
    try {
      const data = await apiFetch(
        `/recommend/autoplay-queue?seed_track_id=${encodeURIComponent(v.track_id)}&limit=20`);
      const lead = v.tracks && v.tracks[0];
      const hits = [
        ...(lead ? [{ track: lead, score: 0, matched_on: 'audio' }] : []),
        ...((data.tracks || []).map(t => ({ track: t, score: 0, matched_on: 'audio' }))),
      ];
      if (hits.length && onPlayTrack) onPlayTrack(hits[0], hits);
    } catch (e) {}
  };
  const vibeName = (v) =>
    v.name || (v.tracks && v.tracks[0] && v.tracks[0].artist) ||
    (lang === 'ru' ? 'Вайбик' : 'Vibe');

  // Dusty axis palette → 4 blob colors for the orb + the page aurora. The
  // lightness bands keep the hue readable on both themes; blob 3 keeps the top
  // anchor's cover hue as a trace of the actual records. No axes → brand set.
  const dusty = (h, s, l) => `hsl(${h} ${s}% ${l}%)`;
  const waveBlobs = pal ? [
    dusty(pal.h1, pal.s1, 58),
    dusty(pal.h2, pal.s2, 60),
    color ? dusty(Math.round(color.h), Math.min(60, Math.round(color.s)), 56) : dusty((pal.h1 + 40) % 360, pal.s1, 56),
    dusty((pal.h1 + 30) % 360, Math.max(36, pal.s1 - 6), 54),
  ] : ['#7c5bff', '#ff78c8', '#e0b341', '#b06bff'];
  // Kicker + halo tinted by the dominant axis so the whole hero "sounds" alike.
  const kickerColor = pal
    ? dusty(pal.h1, pal.s1, isDark ? 78 : 38)
    : (isDark ? '#c9b8ff' : 'oklch(46% 0.19 280)');
  const haloColor = pal
    ? `hsla(${pal.h1}, ${pal.s1}%, 60%, .55)`
    : 'rgba(255,150,210,.6)';
  // Palette CSS vars for the orb blobs/ring (index.css reads --fy-c1..c4).
  const orbVars = {
    '--fy-c1': waveBlobs[0], '--fy-c2': waveBlobs[1],
    '--fy-c3': waveBlobs[2], '--fy-c4': waveBlobs[3],
    '--fy-halo': haloColor,
  };

  // Hand the palette up: the page aurora lives in LandingScreen (the hero is
  // borderless now — its colors must bleed page-wide, not stop at a card edge).
  const paletteKey = waveBlobs.join('|');
  useEffect(() => {
    if (onPalette) onPalette(waveBlobs);
  }, [paletteKey]);

  return (
    // The lead element — borderless now: no glass card, no frame. The page
    // aurora behind it lives in LandingScreen (fed via onPalette); the hero is
    // pure content in the page's light. Flex column, natural content flow —
    // the vibes row sits right under the wave rather than pinned to the hero
    // bottom (that pushed it out of view on desktop and overlapped the search
    // box below on mobile, since a shrunk flex box doesn't clip an
    // auto-margin-pushed child by default).
    <div className="efir-hero" style={{
      position:'relative', display:'flex', flexDirection:'column', flex:1, minHeight:0, overflow:'hidden',
      padding: isMobile ? '4px 2px 4px' : '4px 0 0',
      animation:'fadeInUp .55s cubic-bezier(.22,.9,.3,1)',
    }}>
        {/* Top: kicker + the wave/vibe phrase headline (no concrete song) */}
        <div>
          <div className="mono" style={{ display:'flex', alignItems:'center', fontSize:11.5, letterSpacing:'.28em', color:kickerColor, transition:'color 1.6s ease' }}>
            <span className="hero-eq" aria-hidden="true"><span /><span /><span /><span /><span /></span>
            {lang==='ru'?'ТВОЙ ВАЙБ':'YOUR VIBE'}
          </div>
          {loading || vibeLoading ? (
            <div style={{ marginTop:16 }}>
              <Skel w={'78%'} h={28} r={8} isDark={isDark} />
              <Skel w={'52%'} h={28} r={8} isDark={isDark} style={{ marginTop:10 }} />
            </div>
          ) : phrase ? (
            <div className="vibe-serif" style={{ fontSize:'clamp(20px,2.2vw,28px)', fontWeight:400, lineHeight:1.28, letterSpacing:'0', marginTop:12, color:c.text, maxWidth:620 }}>
              {phrase}
            </div>
          ) : (
            <Fragment>
              <div className="vibe-serif" style={{ fontSize:'clamp(20px,2.2vw,28px)', fontWeight:400, marginTop:12, color:c.text }}>
                {lang==='ru'?'Начнём с разведки':'Let’s start exploring'}
              </div>
              <div style={{ fontSize:13, color:c.textMuted, marginTop:8, maxWidth:520, lineHeight:1.5 }}>
                {lang==='ru'
                  ? 'Истории пока мало — поток начнёт с неизученных уголков библиотеки и подстроится под ваши реакции.'
                  : 'Not much history yet — the stream starts from unexplored corners and adapts to your reactions.'}
              </div>
            </Fragment>
          )}
        </div>

        {/* Orb row — left-aligned: the orb + caption column with the wave
            settings disclosure right under the caption (not at the hero
            bottom — the bottom now belongs to the vibes). */}
        <div style={{ display:'flex', alignItems:'flex-start', gap:'clamp(18px,1.8vw,28px)', marginTop: isMobile ? 'clamp(14px,2.2vh,22px)' : 'clamp(22px,3.4vh,38px)' }}>
            <div style={{ width: isMobile ? 84 : 112, height: isMobile ? 84 : 112, flex:'none', display:'grid', placeItems:'center' }}>
              <div style={{ transform:`scale(${isMobile ? 1.25 : 1.7})` }}>
                <div ref={orbRef} className={`fy-hybrid tint-irid${wavePlaying ? ' fy-playing' : ''}`}
                     style={orbVars}
                     onMouseEnter={() => setHoverCtl(true)} onMouseLeave={orbLeave} onMouseMove={orbMove}
                     onClick={handleOrbClick} role="button" tabIndex={0}
                     onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleOrbClick(); } }}
                     aria-label={wavePlaying ? (lang==='ru'?'Поставить волну на паузу':'Pause your stream')
                                 : waveLive ? (lang==='ru'?'Продолжить волну':'Resume your stream')
                                 : (lang==='ru'?'Включить ваш поток':'Start your stream')}>
                  <span className="fy-ring" />
                  {wavePlaying && <OrbProgressArc audio={audio} />}
                  {/* .fy-breathe — idle-only slow swell (CSS drops it while playing) */}
                  <span className="fy-clip"><span className="fy-breathe"><span className="fy-blob fy-bg1" /><span className="fy-blob fy-bg2" /><span className="fy-blob fy-bg3" /><span className="fy-blob fy-bg4" /></span></span>
                  <span className="fy-glasscap" />
                  {wavePlaying
                    ? <span className="fy-glyph" style={{ marginLeft: 0 }}><svg width="15" height="15" viewBox="0 0 24 24" fill="#fff"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg></span>
                    : <span className="fy-glyph">▶</span>}
                  <span className="fy-halo" />
                </div>
              </div>
            </div>
            <div style={{ minWidth:0, maxWidth:360 }}>
              <div className="mono" style={{ fontSize:'clamp(12px,1vw,14px)', letterSpacing:'.2em', color:hoverCtl?(isDark?'#f3ecff':'#2c2440'):c.text, transition:'color .3s', cursor:'pointer' }}
                   onClick={handleOrbClick}>
                {wavePlaying ? (lang==='ru'?'ВОЛНА ИГРАЕТ':'YOUR STREAM IS LIVE')
                 : waveLive ? (lang==='ru'?'ВОЛНА НА ПАУЗЕ':'STREAM PAUSED')
                 : (lang==='ru'?'ВКЛЮЧИТЬ ПОТОК':'START YOUR STREAM')}
              </div>
              <div style={{ fontSize:'clamp(12px,0.9vw,13.5px)', color:c.textMuted, marginTop:7, lineHeight:1.45 }}>
                {waveLive
                  ? (wavePlaying
                      ? (lang==='ru' ? 'Нажмите, чтобы поставить волну на паузу' : 'Tap to pause your stream')
                      : (lang==='ru' ? 'Нажмите, чтобы продолжить волну' : 'Tap to resume your stream'))
                  : hasProfile
                    ? (lang==='ru' ? 'Волна под ваш вкус — подстраивается под реакции' : 'A wave tuned to your taste — it adapts to your reactions')
                    : (lang==='ru' ? 'Режим разведки — начнём с неизученного и подстроимся' : 'Exploration mode — we start from the unknown and adapt')}
              </div>
              {/* Wave settings — disclosure pill; the panel unfolds in place
                  below (0fr→1fr grid row, no fixed-height reserve needed since
                  nothing is vertically centered anymore). */}
              <button onClick={() => setSettingsOpen(o => !o)} title={lang==='ru'?'Настроить волну':'Tune your wave'}
                aria-expanded={settingsOpen}
                style={{ display:'inline-flex', alignItems:'center', gap:8, marginTop:14, padding:'8px 14px', borderRadius:999, cursor:'pointer',
                  background: settingsOpen ? (isDark?'rgba(255,255,255,.07)':'rgba(0,0,0,.05)') : (isDark?'rgba(255,255,255,.04)':'rgba(0,0,0,.03)'),
                  border:`1px solid ${settingsOpen ? c.border : 'transparent'}`, color:c.textMuted,
                  boxShadow:'inset 0 1px 0 rgba(255,255,255,.08)', transition:'all .3s ease' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                <span className="mono" style={{ fontSize:11.5, letterSpacing:'.16em' }}>{lang==='ru'?'НАСТРОИТЬ ВОЛНУ':'TUNE THE WAVE'}</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                     style={{ transform: settingsOpen ? 'rotate(180deg)' : 'none', transition:'transform .3s ease' }}><path d="m6 9 6 6 6-6"/></svg>
              </button>
              <div aria-hidden={!settingsOpen} style={{ display:'grid',
                gridTemplateRows: settingsOpen ? '1fr' : '0fr',
                opacity: settingsOpen ? 1 : 0,
                transition:'grid-template-rows .35s cubic-bezier(.22,.9,.3,1), opacity .35s ease' }}>
                <div style={{ overflow:'hidden', pointerEvents: settingsOpen ? 'auto' : 'none' }}>
                  <div style={{ marginTop:12, padding:'16px 18px', borderRadius:14,
                    background: isDark?'rgba(255,255,255,.04)':'rgba(0,0,0,.03)',
                    boxShadow:'inset 0 1px 0 rgba(255,255,255,.07)' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:13 }}>
                      <span className="mono" style={{ fontSize:11.5, letterSpacing:'.18em', color:c.textMuted, flex:'none' }}>
                        {lang==='ru'?'РЕДКОЕ':'RARE'}
                      </span>
                      <SkeRange min={0} max={100} step={10} value={Math.round(likedShare*100)}
                             onChange={v => setShare(v / 100)} accent="oklch(62% 0.2 275)" style={{ flex:1 }}
                             ariaLabel={lang==='ru'?'Баланс редкого и любимого в потоке':'Balance of rare vs liked in the stream'} />
                      <span className="mono" style={{ fontSize:11.5, letterSpacing:'.18em', color:c.textMuted, flex:'none' }}>
                        {lang==='ru'?'ЛЮБИМОЕ':'LIKED'}
                      </span>
                    </div>
                    <div className="mono" style={{ fontSize:10.5, letterSpacing:'.12em', color:c.textSubtle, textAlign:'center', marginTop:11 }}>
                      {lang==='ru'?'ПО УМОЛЧАНИЮ — 70% РЕДКОГО · 30% ЛЮБИМОГО':'DEFAULT — 70% RARE · 30% LIKED'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        {/* Taste anchors — desktop only (the compact home keeps launch + search). */}
        {hasProfile && anchors.length > 0 && !isMobile && (
          <div style={{ display:'flex', alignItems:'center', gap:13, marginTop:'clamp(18px,2.6vh,30px)' }}>
            <span className="mono" style={{ fontSize:10.5, letterSpacing:'.2em', color:c.textSubtle }}>{lang==='ru'?'ЯКОРЯ ВКУСА':'TASTE ANCHORS'}</span>
            <div style={{ display:'flex' }}>
              {anchors.slice(0, 5).map((t, i) => {
                const src = homeCoverUrl(t.cover_art_path);
                return (
                  <div key={t.track_id || i} title={`${t.title || ''}${t.artist ? ' — ' + t.artist : ''}`}
                       onClick={() => { const s = primaryArtistSlug(t); if (s && navigateToArtist) navigateToArtist(s); }}
                       style={{ width:40, height:40, borderRadius:10, overflow:'hidden', marginLeft: i ? -11 : 0, cursor:'pointer',
                                position:'relative', zIndex: 10 - i, background:'#241d38',
                                border:`2px solid ${isDark ? '#26262d' : '#fff'}`, boxShadow:'0 6px 14px rgba(0,0,0,.35)' }}>
                    {src && <img src={src} alt="" style={{ width:'100%', height:'100%', objectFit:'cover' }} />}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* «Вайбики» — the days-scale mood chips, right under the wave/orb
            (natural flow, not pinned to the hero bottom anymore). Absent
            entirely when the fast layer has nothing (optional layer, no
            empty state). */}
        {vibes.length > 0 && (
          <div style={{ marginTop: isMobile ? 'clamp(14px,2.2vh,20px)' : 'clamp(18px,2.6vh,28px)' }}>
            <div className="mono" style={{ fontSize:10.5, letterSpacing:'.2em', color:c.textSubtle, marginBottom:11 }}>
              {lang==='ru' ? 'ВАЙБИКИ' : 'VIBES'} · <span style={{ color:c.textSubtle, opacity:.7, letterSpacing:'.08em' }}>
                {lang==='ru' ? 'то, что держит тебя сейчас' : 'what holds you right now'}</span>
            </div>
            <div style={{ display:'flex', gap:11, flexWrap: isMobile ? 'nowrap' : 'wrap',
              overflowX: isMobile ? 'auto' : 'visible', paddingBottom: isMobile ? 6 : 0 }}>
              {vibes.slice(0, 3).map(v => {
                const cov = homeCoverUrl(v.tracks && v.tracks[0] && v.tracks[0].cover_art_path);
                return (
                  <div key={v.track_id} className="efir-vibe-chip" {...spotHandlers(true)}
                       onClick={() => vibeRadio(v)} role="button" tabIndex={0}
                       onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); vibeRadio(v); } }}
                       title={lang==='ru' ? 'Включить радио этого вайба' : 'Play this vibe as a radio'}
                       style={{ display:'flex', alignItems:'center', gap:9, padding:'9px 14px 9px 9px',
                         borderRadius:999, cursor:'pointer', flex:'none',
                         background: isDark
                           ? 'linear-gradient(150deg, rgba(255,255,255,.09), rgba(255,255,255,.03))'
                           : 'linear-gradient(150deg, rgba(255,255,255,.9), rgba(255,255,255,.55))',
                         boxShadow: isDark
                           ? '0 10px 24px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.12)'
                           : '0 8px 20px rgba(60,45,100,.12), inset 0 1px 0 rgba(255,255,255,.9)',
                         color:c.text, fontSize:12.5 }}>
                    <span style={{ width:26, height:26, borderRadius:'50%', overflow:'hidden', flex:'none',
                      background:'linear-gradient(135deg,#7c5cff,#b06bff)',
                      boxShadow:'inset 0 0 0 2px rgba(0,0,0,.35), 0 4px 10px rgba(0,0,0,.3)' }}>
                      {cov && <img src={cov} alt="" style={{ width:'100%', height:'100%', objectFit:'cover' }} />}
                    </span>
                    <span style={{ whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', maxWidth:170 }}>{vibeName(v)}</span>
                    <span style={{ opacity:.55, fontSize:11 }}>▶</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
    </div>
  );
}

// ─── Right wing of the landing: the two non-stream paths ────────────────────
// Kickers + captions live straight on the page background (borderless zones);
// glass is reserved for the interactive elements themselves.

// ✦ Lyrics search — a live input that hands the query off to the search screen
// (AI chat when the assistant is up, classic grid otherwise). No hover glow on
// purpose (user pref): focus/hover is a soft border tint only.
function LyricsSearchPath({ isDark, lang, aiActive, onSubmit, compact=false }) {
  const c = useColors(isDark);
  const [q, setQ] = useState('');
  const [focus, setFocus] = useState(false);
  const inputRef = useRef(null);
  const submit = () => { const t = q.trim(); if (t && onSubmit) onSubmit(t); };
  const kicker = isDark ? '#c9b8ff' : 'oklch(46% 0.19 280)';
  return (
    <div style={{ width:'100%' }}>
      <div className="mono" style={{ fontSize:11.5, letterSpacing:'.24em', color:kicker, marginBottom:6 }}>
        ✦ {lang==='ru'?'ПОИСК ПО ТЕКСТУ':'LYRICS SEARCH'}
      </div>
      <div style={{ fontSize:13.5, color:c.textMuted, marginBottom:13, lineHeight:1.45 }}>
        {lang==='ru'
          ? (aiActive ? 'Помнишь строчку, а не название? ИИ найдёт песню по словам' : 'Помнишь строчку, а не название? Найдём песню по словам')
          : (aiActive ? 'Remember a line, not the title? AI finds the song by its words' : 'Remember a line, not the title? Find the song by its words')}
      </div>
      <div onClick={() => inputRef.current && inputRef.current.focus()}
        style={{ display:'flex', alignItems:'center', gap:12, padding:'16px 18px', borderRadius:18, cursor:'text',
          border:`1px solid ${focus ? (isDark?'rgba(154,123,255,.45)':'rgba(124,91,255,.4)') : c.border}`,
          background: isDark
            ? 'linear-gradient(150deg, rgba(38,34,54,.55), rgba(18,17,26,.45))'
            : 'linear-gradient(150deg, rgba(255,255,255,.85), rgba(245,244,250,.6))',
          boxShadow: isDark
            ? '0 18px 44px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.09)'
            : '0 14px 34px rgba(60,45,100,.12), inset 0 1px 0 rgba(255,255,255,.9)',
          backdropFilter:'blur(14px)', WebkitBackdropFilter:'blur(14px)',
          transition:'border-color .3s ease' }}>
        <span aria-hidden="true" style={{ color:kicker, flex:'none', display:'flex' }}>
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-5.5-5.5"/></svg>
        </span>
        <input ref={inputRef} value={q} onChange={e => setQ(e.target.value)}
          onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
          onKeyDown={e => { if (e.key === 'Enter') submit(); }}
          placeholder={lang==='ru'?'строчка из песни…':'a line from the lyrics…'}
          aria-label={lang==='ru'?'Поиск песни по тексту':'Search a song by its lyrics'}
          style={{ flex:1, minWidth:0, background:'transparent', border:0, outline:'none',
            color:c.text, fontSize:14.5, fontFamily:'inherit' }} />
        {!compact && (
          <span className="mono" aria-hidden="true" style={{ flex:'none', fontSize:10, letterSpacing:'.1em',
            color:c.textSubtle, whiteSpace:'nowrap' }}>
            {lang==='ru'?'найти песню':'find a song'}
          </span>
        )}
        {aiActive ? (
          <button onClick={(e) => { e.stopPropagation(); submit(); }} title={lang==='ru'?'Найти с ИИ':'Search with AI'}
            className="mono" style={{ flex:'none', fontSize:9, letterSpacing:'.1em', color:'#9a7bff', cursor:'pointer',
              border:'1px solid rgba(154,123,255,.4)', borderRadius:6, padding:'3px 7px', background:'transparent' }}>
            {lang==='ru'?'ИИ':'AI'}
          </button>
        ) : (
          <button onClick={(e) => { e.stopPropagation(); submit(); }} title={lang==='ru'?'Найти':'Search'}
            style={{ flex:'none', color:c.textMuted, fontSize:16, cursor:'pointer', background:'transparent', border:0 }}>→</button>
        )}
      </div>
    </div>
  );
}

// ◉ Library — the "fan of covers" portal card: the three freshest album covers
// peek from the right edge and fan out on hover; stats ride /library/stats.
function LibraryPathCard({ isDark, lang, stats, onClick }) {
  const c = useColors(isDark);
  const [covers, setCovers] = useState([]);
  useEffect(() => {
    let alive = true;
    apiFetch(`/library/albums?sort=year_desc`)
      .then(d => {
        if (!alive) return;
        const cs = ((d && d.albums) || [])
          .map(a => homeCoverUrl(a.cover_art_path)).filter(Boolean).slice(0, 3);
        setCovers(cs);
      })
      .catch(() => {});
    return () => { alive = false; };
  }, []);
  const albums = stats && typeof stats.unique_albums === 'number' ? stats.unique_albums : null;
  const tracks = stats && typeof stats.total_tracks === 'number' ? stats.total_tracks : null;
  const fmt = (n) => (n == null ? '—' : (n.toLocaleString ? n.toLocaleString() : n));
  const summary = (albums != null || tracks != null)
    ? `${fmt(albums)} ${lang==='ru'?'АЛЬБОМОВ':'ALBUMS'} · ${fmt(tracks)} ${lang==='ru'?'ТРЕКОВ':'TRACKS'}`
    : (lang==='ru'?'ОТКРЫТЬ ФОНОТЕКУ':'BROWSE THE SHELVES');
  const kicker = isDark ? '#c9b8ff' : 'oklch(46% 0.19 280)';
  // Placeholder gradients keep the fan alive while covers load / are missing.
  const fills = ['linear-gradient(135deg,#6e5a2e,#3a2f14)', 'linear-gradient(135deg,#3d6258,#1e332c)', 'linear-gradient(135deg,#7a4360,#3f2038)'];
  return (
    <div style={{ width:'100%' }}>
      <div className="mono" style={{ fontSize:11.5, letterSpacing:'.24em', color:kicker, marginBottom:13 }}>
        ◉ {lang==='ru'?'ФОНОТЕКА':'LIBRARY'}
      </div>
      <div className="efir-lib-card" {...spotHandlers(true)} onClick={onClick} role="button" tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick && onClick(); } }}
        title={lang==='ru'?'Перейти в библиотеку':'Go to library'}
        style={{ position:'relative', borderRadius:18, padding:'20px', overflow:'hidden', cursor:'pointer',
          border:`1px solid ${c.border}`,
          background: isDark
            ? 'linear-gradient(150deg, rgba(38,34,54,.55), rgba(18,17,26,.45))'
            : 'linear-gradient(150deg, rgba(255,255,255,.85), rgba(245,244,250,.6))',
          boxShadow: isDark
            ? '0 18px 44px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.09)'
            : '0 14px 34px rgba(60,45,100,.12), inset 0 1px 0 rgba(255,255,255,.9)',
          backdropFilter:'blur(14px)', WebkitBackdropFilter:'blur(14px)' }}>
        <div className="efir-lib-fan" aria-hidden="true">
          {[0, 1, 2].map(i => (
            <span key={i} className={`efir-lib-cov efir-lib-cov-${i}`}
              style={{ background: covers[i] ? undefined : fills[i] }}>
              {covers[i] && <img src={covers[i]} alt="" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }} />}
            </span>
          ))}
        </div>
        <div style={{ position:'relative', maxWidth:'58%' }}>
          <div style={{ fontSize:18, fontWeight:600, color:c.text }}>{lang==='ru'?'Библиотека':'Library'} <span style={{ color:c.textMuted }}>→</span></div>
          <div className="mono" style={{ fontSize:10.5, letterSpacing:'.08em', color:c.textMuted, marginTop:7 }}>{summary}</div>
        </div>
      </div>
    </div>
  );
}

// ─── HeaderNowPlaying — compact quick-resume player pill in the landing header.
// Sits beside the Library pill; null when nothing is loaded. The full player
// (scrub, like, queue) opens on a click of the cover/title.
function HeaderNowPlaying({ track, audio, isDark, lang, onOpenPlayer, playlist, onTrackChange }) {
  const c = useColors(isDark);
  const currentTime = useCurrentTime();
  const [hover, setHover] = useState(false);
  const duration = (audio && audio.duration) || 0;
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;
  const cover = homeCoverUrl(track && track.cover_art_path);
  const isPlaying = !!(audio && audio.isPlaying);

  // Prev/next wiring — unwrap HIT[] → flat tracks and locate the current one,
  // exactly like MiniPlaybackPopout. onTrackChange propagates the chosen
  // neighbour up to App.playerTrack, which drives the actual setSrc + play.
  const flatTracks = (playlist || []).map(h => (h && h.track) ? h.track : h);
  const curIdx = track ? flatTracks.findIndex(t => t && t.track_id === track.track_id) : -1;
  const canPrev = curIdx > 0;
  const canNext = curIdx >= 0 && curIdx < flatTracks.length - 1;
  const goPrev = () => { if (canPrev && onTrackChange) onTrackChange(flatTracks[curIdx - 1]); };
  const goNext = () => { if (canNext && onTrackChange) onTrackChange(flatTracks[curIdx + 1]); };

  if (!track) return null;

  // Small round control. `lead` = the central play/pause (slightly larger);
  // prev/next dim + go not-allowed at the ends of the queue.
  const ctrlBtnStyle = (enabled, lead) => ({
    width: lead ? 32 : 28, height: lead ? 32 : 28, borderRadius: '50%',
    flex: 'none', display: 'grid', placeItems: 'center', color: c.text,
    opacity: enabled ? 1 : 0.32, cursor: enabled ? 'pointer' : 'not-allowed',
  });

  // Liquid glass only while sound is actually on; paused keeps the quiet pill
  // so the header doesn't shout when nothing plays. The glass variant has NO
  // sweeping hover glint (user pref — see .hnp-glass in index.css): hover is
  // the same shadow "lift" as the plain pill. When glass is active the inline
  // border/background/shadow are dropped so the CSS class owns the surface.
  const glassOn = isPlaying;

  return (
    <div onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      className={glassOn ? 'liquid-glass hnp-glass' : undefined}
      style={{ display:'flex', alignItems:'center', gap:8, padding:'6px 8px 6px 6px', borderRadius:14, maxWidth:390,
        ...(glassOn ? {} : {
          border:`1px solid ${hover ? c.borderStrong : c.border}`,
          background: isDark ? (hover?'rgba(255,255,255,.07)':'rgba(255,255,255,.04)') : (hover?'rgba(255,255,255,.92)':'rgba(255,255,255,.6)'),
          backdropFilter:'blur(16px)', WebkitBackdropFilter:'blur(16px)',
          boxShadow: hover ? '0 8px 22px rgba(60,45,100,.16)' : '0 4px 12px rgba(60,45,100,.08)',
        }),
        transition:'all .35s cubic-bezier(.22,.9,.3,1)' }}>
      <div onClick={onOpenPlayer} title={lang==='ru'?'Открыть плеер':'Open player'}
        style={{ width:34, height:34, borderRadius:9, overflow:'hidden', flex:'none', cursor:'pointer', background:'#241d38' }}>
        {cover && <img src={cover} alt="" style={{ width:'100%', height:'100%', objectFit:'cover' }} />}
      </div>
      <div onClick={onOpenPlayer} style={{ width:128, minWidth:0, cursor:'pointer' }}>
        <div style={{ fontSize:12.5, fontWeight:600, color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{track.title || '—'}</div>
        <div style={{ position:'relative', height:3, borderRadius:2, marginTop:4, background:isDark?'rgba(255,255,255,.12)':'rgba(0,0,0,.12)' }}>
          <div style={{ position:'absolute', left:0, top:0, bottom:0, width:`${progress*100}%`, borderRadius:2, background:'oklch(62% 0.18 275)' }} />
        </div>
      </div>
      <button className={ske('btn', isDark)} onClick={goPrev} disabled={!canPrev}
        title={lang==='ru'?'Предыдущий':'Previous'} style={ctrlBtnStyle(canPrev, false)}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="19,4 8,12 19,20"/><rect x="5" y="4" width="2" height="16"/></svg>
      </button>
      <button className={ske('btn', isDark)} onClick={() => audio && audio.togglePlay && audio.togglePlay()}
        title={isPlaying ? (lang==='ru'?'Пауза':'Pause') : (lang==='ru'?'Играть':'Play')}
        style={ctrlBtnStyle(true, true)}>
        {isPlaying
          ? <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          : <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft:2 }}><polygon points="6,4 20,12 6,20"/></svg>}
      </button>
      <button className={ske('btn', isDark)} onClick={goNext} disabled={!canNext}
        title={lang==='ru'?'Следующий':'Next'} style={ctrlBtnStyle(canNext, false)}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,4 16,12 5,20"/><rect x="17" y="4" width="2" height="16"/></svg>
      </button>
    </div>
  );
}

// ─── HintBadge — embossed "i" + skeuomorphic hover/focus tooltip ────────────
// Reusable affordance for explaining an unobvious UI element. `size` is the dot
// diameter in px (callers scale it relative to whatever it annotates); the
// tooltip itself scales with the viewport in CSS. Keyboard-reachable (tabIndex)
// so the explanation isn't mouse-only. `label` is the tooltip body (string or
// JSX with <strong>); `ariaLabel` is the screen-reader summary. `placement`
// flips the tooltip below the dot ('down') for badges that sit at the top edge
// of an overflow:hidden container, where an upward pop would be clipped.
function HintBadge({ size = 20, label, ariaLabel, placement = 'up' }) {
  const px = Math.max(14, size);
  return (
    <span className={`hint-badge${placement === 'down' ? ' hint-badge--down' : ''}`}
      tabIndex={0} role="note"
      aria-label={ariaLabel || (typeof label === 'string' ? label : 'Info')}>
      <span className="hint-badge__dot" aria-hidden="true"
        style={{ width: px, height: px, fontSize: Math.max(10, Math.round(px * 0.58)) }}>i</span>
      <span className="hint-badge__pop" role="tooltip">{label}</span>
    </span>
  );
}

// ─── SpotlightSearch — global find-and-play overlay (🔍 in the header, Ctrl/Cmd+K) ──
// macOS-Spotlight-style: dim + blur under a centered glass bar; instant rows
// from /library/browse. Click a row (or ↵) = play it with the other matches
// queued behind; «+» = add to a playlist without playing; Tab / «ещё» hands
// the query to the full search screen. One shared highlight pill slides
// between rows (hover and ↑↓) with a light bounce — no per-row glow.
function SpotlightSearch({ open, onClose, isDark, lang, onPlayTrack, onAddToPlaylist, onMore }) {
  const c = useColors(isDark);
  const [q, setQ] = useState('');
  const [rows, setRows] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [searched, setSearched] = useState(false);
  const [active, setActive] = useState(0);
  const [hl, setHl] = useState(null);          // { y, h } of the highlight pill
  const inputRef = useRef(null);
  const rowRefs = useRef([]);

  // Fresh sheet on every open (Spotlight muscle memory: open → type).
  useEffect(() => {
    if (!open) return;
    setQ(''); setRows([]); setSearched(false); setActive(0); setHl(null);
    const t = setTimeout(() => { if (inputRef.current) inputRef.current.focus(); }, 30);
    return () => clearTimeout(t);
  }, [open]);

  // Debounced instant search. limit=9, show 8 — the 9th only signals «ещё».
  useEffect(() => {
    if (!open) return;
    const term = q.trim();
    if (term.length < 2) { setRows([]); setSearched(false); setHasMore(false); return; }
    let alive = true;
    const timer = setTimeout(() => {
      apiFetch(`/library/browse?q=${encodeURIComponent(term)}&limit=9`)
        .then(d => {
          if (!alive) return;
          const list = Array.isArray(d) ? d : [];
          setRows(list.slice(0, 8));
          setHasMore(list.length > 8);
          setSearched(true);
          setActive(0);
        })
        .catch(() => { if (alive) { setRows([]); setHasMore(false); setSearched(true); } });
    }, 150);
    return () => { alive = false; clearTimeout(timer); };
  }, [q, open]);

  // The sliding pill tracks the active row (hover writes active too).
  useEffect(() => {
    const el = rowRefs.current[active];
    if (!el) { setHl(null); return; }
    setHl({ y: el.offsetTop, h: el.offsetHeight });
  }, [active, rows]);

  // Escape closes even if focus wandered off the input.
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); onClose && onClose(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const playRow = (i) => {
    const r = rows[i];
    if (!r) return;
    const hits = rows.map(t => ({ track: t, score: t.score || 0, matched_on: 'browse' }));
    if (onPlayTrack) onPlayTrack(hits[i], hits);
    if (onClose) onClose();
  };
  const goMore = () => {
    const term = q.trim();
    if (term && onMore) { onMore(term); if (onClose) onClose(); }
  };
  const onInputKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(i => Math.min(rows.length - 1, i + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(i => Math.max(0, i - 1)); }
    else if (e.key === 'Enter') { e.preventDefault(); playRow(active); }
    else if (e.key === 'Tab') { e.preventDefault(); goMore(); }
  };
  const fmtDur = (s) => {
    if (typeof s !== 'number' || !isFinite(s) || s <= 0) return null;
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
  };
  const glass = {
    background: isDark
      ? 'linear-gradient(150deg, rgba(38,34,54,.72), rgba(18,17,26,.66))'
      : 'linear-gradient(150deg, rgba(255,255,255,.92), rgba(245,244,250,.82))',
    backdropFilter:'blur(18px)', WebkitBackdropFilter:'blur(18px)',
    border:`1px solid ${isDark ? 'rgba(255,255,255,.09)' : 'rgba(0,0,0,.08)'}`,
  };
  const term = q.trim();

  return (
    <div onMouseDown={(e) => { if (e.target === e.currentTarget && onClose) onClose(); }}
      role="dialog" aria-modal="true" aria-label={lang==='ru'?'Быстрый поиск по песням':'Quick song search'}
      style={{ position:'fixed', inset:0, zIndex:1500,
        background: isDark ? 'rgba(5,4,10,.55)' : 'rgba(30,25,50,.28)',
        backdropFilter:'blur(3px)', WebkitBackdropFilter:'blur(3px)',
        display:'flex', flexDirection:'column', alignItems:'center',
        paddingTop:'min(14vh, 130px)', animation:'fadeIn .18s ease' }}>
      <div style={{ width:'min(92vw, 680px)' }} onMouseDown={(e) => e.stopPropagation()}>
        {/* Search bar */}
        <div style={{ ...glass, display:'flex', alignItems:'center', gap:12, padding:'16px 20px', borderRadius:20,
          boxShadow:'0 30px 80px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.14)' }}>
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke={c.textMuted} strokeWidth="2" strokeLinecap="round" style={{ flex:'none' }}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
          <input ref={inputRef} value={q} onChange={e => setQ(e.target.value)} onKeyDown={onInputKey}
            placeholder={lang==='ru'?'название, артист или альбом…':'song, artist or album…'}
            aria-label={lang==='ru'?'Найти песню':'Find a song'}
            style={{ flex:1, minWidth:0, background:'transparent', border:0, outline:'none',
              color:c.text, fontSize:15.5, fontFamily:'inherit' }} />
          <span className="mono" style={{ flex:'none', fontSize:9, letterSpacing:'.1em', color:c.textSubtle }}>ESC</span>
        </div>

        {/* Results panel — only once there is a real query */}
        {term.length >= 2 && (
          <div style={{ ...glass, marginTop:12, borderRadius:16, padding:10,
            boxShadow:'0 30px 80px rgba(0,0,0,.45)' }}>
            {rows.length > 0 ? (
              <Fragment>
                <div style={{ position:'relative' }}>
                  {hl && (
                    <div className="spotlight-hl" aria-hidden="true" style={{
                      transform:`translateY(${hl.y}px)`, height:hl.h,
                      background: isDark ? 'rgba(154,123,255,.14)' : 'rgba(124,91,255,.12)',
                    }} />
                  )}
                  {rows.map((r, i) => {
                    const cover = homeCoverUrl(r.cover_art_path);
                    const dur = fmtDur(r.duration);
                    const sub = [r.artist, r.album ? `${lang==='ru'?'альбом ':''}${r.album}` : null, r.year || null]
                      .filter(Boolean).join(' · ');
                    return (
                      <div key={r.track_id || i} ref={el => { rowRefs.current[i] = el; }}
                        onMouseEnter={() => setActive(i)} onClick={() => playRow(i)}
                        style={{ position:'relative', display:'flex', alignItems:'center', gap:12,
                          padding:'11px 14px', borderRadius:12, cursor:'pointer' }}>
                        <div style={{ width:44, height:44, borderRadius:8, overflow:'hidden', flex:'none',
                          background:'linear-gradient(135deg,#4a3d78,#251d3f)',
                          boxShadow:'0 6px 16px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.15)' }}>
                          {cover && <img src={cover} alt="" loading="lazy" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }} />}
                        </div>
                        <div style={{ minWidth:0, flex:1 }}>
                          <div style={{ color:c.text, fontSize:14.5, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{r.title || '—'}</div>
                          <div style={{ color:c.textMuted, fontSize:11.5, marginTop:2, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{sub}</div>
                        </div>
                        {r.genre && (
                          <span className="mono" style={{ flex:'none', fontSize:9, letterSpacing:'.1em', color:isDark?'#a79cc9':'#6a5b96',
                            border:`1px solid ${isDark?'rgba(167,156,201,.3)':'rgba(106,91,150,.3)'}`, borderRadius:6, padding:'3px 8px',
                            maxWidth:130, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', textTransform:'uppercase' }}>{r.genre}</span>
                        )}
                        <span style={{ width:14, flex:'none' }} />
                        {dur && <span className="mono" style={{ flex:'none', fontSize:10.5, letterSpacing:'.05em', color:isDark?'#a79cc9':'#6a5b96' }}>{dur}</span>}
                        <button onClick={(e) => { e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(r.track_id, e.currentTarget); }}
                          title={lang==='ru'?'Добавить в плейлист':'Add to playlist'} className="spotlight-add"
                          style={{ flex:'none', width:32, height:32, borderRadius:'50%', display:'grid', placeItems:'center',
                            background:'linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.02))',
                            boxShadow:'0 5px 12px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.16)',
                            border:0, color:c.textMuted, fontSize:15, cursor:'pointer' }}>＋</button>
                        <span className="mono" style={{ flex:'none', width:12, fontSize:9, color:'#9a7bff', opacity: i === active ? 1 : 0 }}>↵</span>
                      </div>
                    );
                  })}
                </div>
                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', gap:10,
                  padding:'9px 14px 4px', marginTop:6, borderTop:`1px solid ${isDark?'rgba(255,255,255,.05)':'rgba(0,0,0,.06)'}` }}>
                  <span className="mono" style={{ fontSize:9, letterSpacing:'.08em', color:c.textSubtle }}>
                    {lang==='ru'?'↑↓ ВЫБОР · ↵ ИГРАТЬ · TAB — ВЕСЬ ПОИСК':'↑↓ SELECT · ↵ PLAY · TAB — FULL SEARCH'}
                  </span>
                  {hasMore && (
                    <button onClick={goMore} className="mono"
                      style={{ fontSize:9, letterSpacing:'.08em', color:c.textMuted, background:'transparent', border:0, cursor:'pointer' }}>
                      {lang==='ru'?'ЕЩЁ СОВПАДЕНИЯ →':'MORE MATCHES →'}
                    </button>
                  )}
                </div>
              </Fragment>
            ) : searched ? (
              <div style={{ padding:'18px 14px', fontSize:13.5, color:c.textMuted }}>
                {lang==='ru'?'Ничего не нашлось в библиотеке':'Nothing found in your library'}
              </div>
            ) : (
              <div style={{ padding:'18px 14px', fontSize:13.5, color:c.textSubtle }}>…</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LandingScreen({ isDark, lang, onLang, onTheme, onSettings, onNav, hasLibrary, stats, playerTrack, playerPlaylist, onTrackChange, onPlayTrack, onStartStream, streamActive, audio, navigateToArtist, onOpenSpotlight, onSearchLyrics, aiActive }) {
  const c = useColors(isDark);
  const isMobile = useIsMobile();

  // «Эфир»: one airy screen, three paths — the wave (hero), lyrics search and
  // the library. No glass zone frames: the zones share the page and are split
  // by air; the aurora below bleeds page-wide in the hero's taste palette
  // (fed back via onPalette). Brand set until the profile arrives.
  const [auroraBlobs, setAuroraBlobs] = useState(null);
  const blobs = auroraBlobs || ['#7c5bff', '#ff78c8', '#e0b341', '#b06bff'];

  // Sizing: 100% of the app shell, NOT 100vw/100vh — on mobile the shell
  // stacks MiniPlayerBar + BottomTabBar below this screen, and a viewport-
  // sized root overflowed under them (the wave block "danced" into the
  // header because flex centering overflowed both directions).
  return (
    <div className="grain" style={{
      flex:1, minWidth:0, minHeight:0, width:'100%', overflow:'hidden auto', position:'relative',
      display:'flex', flexDirection:'column',
      background: isDark
        ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
        : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
      color: c.text,
      animation: 'themeFade 0.45s ease',
    }}>
      {/* Page aurora — the hero's liquid wave promoted to the page layer: it
          starts behind the hero and dissolves toward the right wing, erasing
          any sense of a card boundary. Blobs + caustics under one static SVG
          displacement lens (near-zero extra GPU cost; the motion is the blobs'). */}
      <div className="efir-wave" style={{ position:'absolute', top:0, left:0, right:0, height:'62%',
        opacity: isDark ? 0.34 : 0.24, pointerEvents:'none',
        WebkitMaskImage:'linear-gradient(180deg, #000 0%, #000 40%, transparent 100%)',
        maskImage:'linear-gradient(180deg, #000 0%, #000 40%, transparent 100%)' }}>
        <div className="efir-liquid">
          <span className="fy-blob fy-bg1" style={{ background:blobs[0], transition:'background 1.6s ease' }} />
          <span className="fy-blob fy-bg2" style={{ background:blobs[1], transition:'background 1.6s ease' }} />
          <span className="fy-blob fy-bg3" style={{ background:blobs[2], transition:'background 1.6s ease' }} />
          <span className="fy-blob fy-bg4" style={{ background:blobs[3], transition:'background 1.6s ease' }} />
          <span className="efir-caustics" />
        </div>
      </div>
      {/* displacement source for the liquid-glass refraction (defs only, 0×0) */}
      <svg width="0" height="0" style={{ position:'absolute' }} aria-hidden="true" focusable="false">
        <filter id="efir-liquid-lens" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" baseFrequency="0.007 0.013" numOctaves="2" seed="7" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="90" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
      {/* Subtle bg pattern: concentric vinyl-like rings */}
      <div style={{
        position:'absolute', top:'-25%', right:'-15%', width:'820px', height:'820px',
        borderRadius:'50%', pointerEvents:'none', opacity:0.4,
        background: `radial-gradient(circle,
          transparent 0%, transparent 32%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 32.5%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 33%,
          transparent 33.5%, transparent 38%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 38.5%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 39%,
          transparent 39.5%, transparent 47%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 47.5%,
          ${isDark?'rgba(255,255,255,0.012)':'rgba(0,0,0,0.022)'} 48%,
          transparent 48.5%)`,
      }} />

      {/* Header: brand + mini-player (center) + spotlight + controls. The
          Library pill is gone — the library is a full path on the page now. */}
      <div style={{
        position:'relative', display:'flex', alignItems:'center', justifyContent:'space-between', gap:16,
        padding: isMobile ? 'clamp(10px,1.6vh,16px) clamp(14px,3vw,22px)' : 'clamp(18px,2.4vh,26px) clamp(20px,3vw,40px)',
        flexWrap:'wrap',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:'14px', flex:'1 1 0', minWidth:0 }}>
          <BrandMark size={42} isDark={isDark} />
          <div style={{ minWidth:0 }}>
            <div className="serif" style={{ fontSize:'clamp(26px,2.6vw,34px)', lineHeight:'0.95', letterSpacing:'-0.02em' }}>
              Musi<i style={{ color: 'oklch(62% 0.2 275)' }}>X</i>
            </div>
            <PremiumMark />
          </div>
        </div>
        {/* Centered mini-player — its own header zone (flanked by two equal
            flex:1 0 zones) so it sits in the middle of the bar. Collapses to
            null when nothing is playing. */}
        {!isMobile && (
        <div style={{ display:'flex', justifyContent:'center', flex:'0 1 auto', minWidth:0 }}>
          <HeaderNowPlaying track={playerTrack} audio={audio} isDark={isDark} lang={lang}
            playlist={playerPlaylist} onTrackChange={onTrackChange}
            onOpenPlayer={() => onNav('player')} />
        </div>
        )}
        <div style={{ display:'flex', alignItems:'center', gap:12, flexWrap:'wrap', flex:'1 1 0', minWidth:0, justifyContent:'flex-end' }}>
          {/* Song spotlight (⌘K) — quick find-and-play over the library;
              lyrics search lives on the page as its own path. */}
          <button onClick={onOpenSpotlight}
            title={lang==='ru'?'Найти песню (Ctrl+K)':'Find a song (Ctrl+K)'} className={ske('btn', isDark)}
            style={{ width:38, height:38, borderRadius:'50%', display:'grid', placeItems:'center', color:c.textMuted }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
          </button>
          <TopRightControls isDark={isDark} lang={lang} onLang={onLang} onTheme={onTheme} onSettings={onSettings} showLang={false} showTheme={false} />
        </div>
      </div>

      {isMobile ? (
        /* Mobile «Эфир»: one column, same three paths — hero (vibe + orb +
           vibes strip inside), lyrics search, library. Normal document flow,
           so nothing can overlap the header. */
        <div style={{ position:'relative', flex:'1 1 auto', minHeight:0, overflowY:'auto',
          padding:'2px 16px 18px', display:'flex', flexDirection:'column', gap:22 }}>
          <ForYouHero isDark={isDark} lang={lang}
                      onStartStream={onStartStream} streamActive={streamActive} audio={audio}
                      navigateToArtist={navigateToArtist} onPlayTrack={onPlayTrack}
                      onPalette={setAuroraBlobs} />
          <LyricsSearchPath isDark={isDark} lang={lang} aiActive={aiActive} onSubmit={onSearchLyrics} compact />
          <LibraryPathCard isDark={isDark} lang={lang} stats={stats} onClick={() => onNav('library')} />
        </div>
      ) : (
      /* «Эфир» split — takes exactly the space left under the header (no page
          scroll on desktop; .efir-main's min-height + the root's auto overflow
          remain the fallback for very short windows) */
      <div style={{ position:'relative', flex:'1 1 auto', minHeight:0, display:'flex', flexDirection:'column', justifyContent:'center' }}>
        <div className="efir-main" style={{ position:'relative' }}>
          <ForYouHero isDark={isDark} lang={lang}
                      onStartStream={onStartStream} streamActive={streamActive} audio={audio}
                      navigateToArtist={navigateToArtist} onPlayTrack={onPlayTrack}
                      onPalette={setAuroraBlobs} />
          {/* Right wing — no cards, just the two paths separated by air. The
              top padding drops the wing's optical start to the hero phrase. */}
          <div className="efir-wing">
            <LyricsSearchPath isDark={isDark} lang={lang} aiActive={aiActive} onSubmit={onSearchLyrics} />
            <LibraryPathCard isDark={isDark} lang={lang} stats={stats} onClick={() => onNav('library')} />
          </div>
        </div>
      </div>
      )}

      {/* Under-the-fold extras — desktop only, deliberately quiet so they never
          compete with the wave/search/library trio above. Mobile has no room
          and stays exactly as-is (per design: no scroll needed on the phone). */}
      {!isMobile && hasLibrary && <HomeDailyExtras isDark={isDark} lang={lang} />}
    </div>
  );
}

// ─── Home daily extras (desktop only) — a couple of static facts + a quiet
// weekly pulse, both fixed for the day (cached under a date key) so they
// don't reshuffle every time the page is revisited.
function HomeDailyExtras({ isDark, lang }) {
  const c = useColors(isDark);
  const [facts, setFacts] = useState(null);
  const [pulse, setPulse] = useState(null);

  useEffect(() => {
    let alive = true;
    const today = new Date().toDateString();
    const cacheKey = 'musix_daily_facts';
    let cached = null;
    try { cached = JSON.parse(localStorage.getItem(cacheKey) || 'null'); } catch {}
    if (cached && cached.date === today && Array.isArray(cached.facts)) {
      setFacts(cached.facts);
    } else {
      apiFetch('/metadata/random-facts?limit=3').then(r => {
        if (!alive) return;
        const list = Array.isArray(r) ? r : [];
        setFacts(list);
        try { localStorage.setItem(cacheKey, JSON.stringify({ date: today, facts: list })); } catch {}
      }).catch(() => { if (alive) setFacts([]); });
    }
    const tzOffset = -new Date().getTimezoneOffset();
    apiFetch(`/library/weekly-pulse?tz_offset_minutes=${tzOffset}`)
      .then(r => { if (alive) setPulse(r || null); })
      .catch(() => { if (alive) setPulse(null); });
    return () => { alive = false; };
  }, []);

  const fmtDur = (sec) => {
    if (!sec) return null;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const hU = lang === 'ru' ? 'ч' : 'h', mU = lang === 'ru' ? 'м' : 'm';
    return h > 0 ? `${h}${hU} ${m}${mU}` : `${m}${mU}`;
  };

  const hasFacts = !!(facts && facts.length > 0);
  const hasPulse = !!(pulse && (pulse.seconds_listened > 0 || pulse.top_genre));
  if (facts === null && pulse === null) return null;   // still loading — nothing to flash in
  if (!hasFacts && !hasPulse) return null;

  const cardStyle = {
    borderRadius: 14, padding: '14px 16px', flex: '1 1 220px', minWidth: 200,
    background: isDark ? 'rgba(255,255,255,.025)' : 'rgba(0,0,0,.02)',
    border: `1px solid ${c.border}`,
  };

  return (
    <div style={{ padding: '10px clamp(20px,3vw,40px) 40px', marginTop: 18 }}>
      <div style={{ display:'flex', flexWrap:'wrap', gap:14 }}>
        {hasFacts && facts.slice(0, 3).map((f, i) => (
          <div key={i} style={cardStyle}>
            <div className="mono" style={{ fontSize:9.5, letterSpacing:'.16em', color:c.textSubtle, marginBottom:6 }}>
              {f.type === 'artist'
                ? (lang==='ru'?'ОБ АРТИСТЕ':'ABOUT THE ARTIST')
                : (lang==='ru'?'О ПЕСНЕ':'ABOUT THE SONG')}
            </div>
            <div style={{ fontSize:12.5, color:c.textMuted, lineHeight:1.5 }}>{f.fact}</div>
            {f.context && (
              <div style={{ fontSize:11, color:c.textSubtle, marginTop:6 }}>{f.context}</div>
            )}
          </div>
        ))}
        {hasPulse && (
          <div style={cardStyle}>
            <div className="mono" style={{ fontSize:9.5, letterSpacing:'.16em', color:c.textSubtle, marginBottom:6 }}>
              {lang==='ru'?'ЗА ЭТУ НЕДЕЛЮ':'THIS WEEK'}
            </div>
            <div style={{ fontSize:12.5, color:c.textMuted, lineHeight:1.5 }}>
              {fmtDur(pulse.seconds_listened)
                ? (lang==='ru' ? `Прослушано ${fmtDur(pulse.seconds_listened)}` : `${fmtDur(pulse.seconds_listened)} listened`)
                : (lang==='ru' ? 'Пока тихо' : 'Quiet so far')}
              {pulse.top_genre && (lang==='ru' ? ` · любимый жанр — ${pulse.top_genre}` : ` · favorite genre — ${pulse.top_genre}`)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── FloatingIconNav (v4) — liquid-glass floating islands: brand coin, tab capsule, pebble dock ──
function FloatingIconNav({ section, onNav, isDark, lang, onSettings, currentTrack, audio, playlist, onTrackChange }) {
  const c = useColors(isDark);
  const [popoutOpen, setPopoutOpen] = useState(false);
  const popoutTimer = useRef(null);

  // Theme-aware tokens so nav stays readable on both dark and light backgrounds.
  // Glass surfaces themselves live in CSS (.lg-island / .lg-blob, themed via
  // the .lg-light wrapper class) — only text/icon colors stay inline.
  const inactiveColor = isDark ? 'rgba(238,238,243,0.55)' : 'rgba(22,22,32,0.55)';
  const hoverColor    = c.text;
  const settingsColor = isDark ? 'rgba(238,238,243,0.55)' : 'rgba(22,22,32,0.55)';
  const activeFg      = isDark ? '#fff' : c.text;

  // 80ms enter delay swallows accidental mouse-bys (cursor briefly passing over
  // the pebble); 200ms close delay gives the user time to traverse from pebble
  // into the popout without it snapping shut.
  const openPopout = () => {
    if (popoutTimer.current) { clearTimeout(popoutTimer.current); popoutTimer.current = null; }
    popoutTimer.current = setTimeout(() => setPopoutOpen(true), 80);
  };
  const closePopoutSoon = () => {
    if (popoutTimer.current) { clearTimeout(popoutTimer.current); popoutTimer.current = null; }
    popoutTimer.current = setTimeout(() => setPopoutOpen(false), 200);
  };

  // Tab order: player first (primary surface), then library → recommend → search.
  // 'home' is no longer a tab — the brand coin above the capsule handles it.
  const navItems = [
    { id:'player',    label: lang==='ru'?'ПЛЕЕР':'PLAY',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none"/></svg> },
    { id:'library',   label: lang==='ru'?'БИБЛ':'LIB',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg> },
    { id:'recommend', label: lang==='ru'?'РЕК':'REC',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2 15 8l6 1-4.5 4.5L18 20l-6-3-6 3 1.5-6.5L3 9l6-1z"/></svg> },
    { id:'search',    label: lang==='ru'?'ПОИСК':'SRCH',
      icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> },
  ];

  // Geometry shared by tabs and the sliding glass blob.
  const TAB_H = 56, TAB_GAP = 6;
  const activeIdx = navItems.findIndex(i => i.id === section);
  // Remember the last tab position so the blob shrinks in place (instead of
  // jumping to the top) when a non-tab section like 'artist' is open.
  const lastIdxRef = useRef(0);
  if (activeIdx >= 0) lastIdxRef.current = activeIdx;
  const blobY = (activeIdx >= 0 ? activeIdx : lastIdxRef.current) * (TAB_H + TAB_GAP);

  // On the player tab the bottom pebble is redundant (the player itself shows
  // the cover full-size), so the rail stays clean there. Also nothing to show
  // without a track.
  const pebbleHidden = section === 'player' || !currentTrack;

  return (
    <div className={isDark ? undefined : 'lg-light'} style={{
      width:'92px', minWidth:'92px', height:'100vh', display:'flex', flexDirection:'column',
      alignItems:'center', padding:'14px 0 16px',
      background:'transparent', zIndex:10, flexShrink:0, position:'relative',
    }}>
      {/* Brand mark — returns to the landing */}
      <button className="lg-logo" onClick={() => onNav('home')} title={lang==='ru'?'На главную':'Home'}
        style={{ width:50, height:50, display:'grid', placeItems:'center', cursor:'pointer',
                 background:'transparent', border:0, color:c.text, padding:0 }}>
        <BrandMark size={36} isDark={isDark} />
      </button>

      <div style={{ flex:1 }} />

      {/* Floating glass capsule with the four tabs + sliding blob indicator */}
      {/* overflow:clip keeps the springy blob inside the glass capsule: its
          overshoot bezier overshoots ~8-10% of travel, which on the long
          player↔search hop (186px) pokes ~9px past the rounded end. Clipping to
          the capsule's border-radius contains it — the droplet now squashes
          against the glass wall on arrival instead of escaping, while the full
          spring wobble is preserved. */}
      <nav className="lg-island" style={{ width:TAB_H + 12, borderRadius:(TAB_H + 12) / 2, padding:6,
        position:'relative', overflow:'clip', display:'flex', flexDirection:'column', gap:TAB_GAP }}>
        <div className="lg-blob" style={{
          width:TAB_H, height:TAB_H,
          transform:`translateY(${blobY}px) scale(${activeIdx >= 0 ? 1 : 0.4})`,
          opacity: activeIdx >= 0 ? 1 : 0,
        }} />
        {navItems.map(item => {
          const active = section === item.id;
          return (
            <button key={item.id} className="lg-tab" onClick={() => onNav(item.id)} title={item.label}
              style={{ width:TAB_H, height:TAB_H, color: active ? activeFg : inactiveColor }}
              onMouseEnter={e => { if(!active) e.currentTarget.style.color = hoverColor; }}
              onMouseLeave={e => { if(!active) e.currentTarget.style.color = inactiveColor; }}>
              <span style={{ display:'flex' }}>{item.icon}</span>
              <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontWeight:500, textTransform:'uppercase', fontSize:8, color:'inherit', letterSpacing:'0.16em' }}>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div style={{ flex:1 }} />

      {/* Bottom glass island: settings + Now Playing pebble. The pebble slot
          collapses (flies up + fades) on the player view to keep the rail clean.
          Pinned absolute at the bottom (out of flex flow) so the pebble's
          collapse/expand changes only this island's own height — it grows
          upward into empty space and never shifts the centered tab capsule. */}
      <div className="lg-island" style={{ width:60, borderRadius:30, padding:6,
        position:'absolute', left:16, bottom:16,
        display:'flex', flexDirection:'column', alignItems:'center', gap:2 }}>
        <button onClick={onSettings} title={lang==='ru'?'Настройки':'Settings'}
          style={{ width:44, height:44, borderRadius:22, display:'grid', placeItems:'center', cursor:'pointer',
                   background:'transparent', border:0, color:settingsColor, transition:'color 0.2s ease' }}
          onMouseEnter={e => e.currentTarget.style.color=c.text}
          onMouseLeave={e => e.currentTarget.style.color=settingsColor}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
          </svg>
        </button>

        {/* Now Playing Pebble — hidden on the player view (the player shows the cover itself) */}
        <div className="lg-pebble-slot" data-hidden={pebbleHidden ? 'true' : 'false'} data-pebble-slot="active">
          <NowPlayingPebble
            track={currentTrack}
            isPlaying={!!(audio && audio.isPlaying)}
            isDark={isDark}
            onClick={() => { if (currentTrack) onNav('player'); }}
            onHoverChange={(hovered) => {
              if (!currentTrack) return;  // no point opening an empty popout
              if (hovered) openPopout(); else closePopoutSoon();
            }}
          />
        </div>
      </div>

      {/* Mini playback popout — always mounted so the exit transition can play.
          data-state drives entrance/exit via .mini-popout-shell CSS. */}
      <div
        className="mini-popout-shell"
        data-state={popoutOpen && currentTrack ? 'open' : 'closed'}
        style={{ position: 'fixed', left: 96, bottom: 18, zIndex: 1000 }}
        onMouseEnter={openPopout}
        onMouseLeave={closePopoutSoon}
      >
        <MiniPlaybackPopout
          track={currentTrack}
          audio={audio}
          isDark={isDark}
          playlist={playlist}
          onTrackChange={onTrackChange}
          onOpenPlayer={() => { setPopoutOpen(false); onNav('player'); }}
          onClose={closePopoutSoon}
        />
      </div>
    </div>
  );
}

// ─── NowPlayingPebble — 40×40 circular cover indicator at bottom of sidebar ─────
function NowPlayingPebble({ track, isPlaying, isDark = true, onClick, onHoverChange }) {
  const hasTrack = !!track;
  // cover_art_path is a relative URL like "/covers/abc.jpg"; AlbumCover prepends ${API}
  // (e.g. "/api/v1") and we do the same here so the browser doesn't 404 against the
  // frontend origin. http(s) URLs pass through untouched.
  const rawCover = track && (track.cover_art_path || track.coverArt) || null;
  const cover = rawCover ? thumbCoverUrl(rawCover.startsWith('http') ? rawCover : `${API}${rawCover}`) : null;
  const animation = (hasTrack && isPlaying) ? 'pebblePulse 2.4s ease-in-out infinite' : 'none';

  // Theme-aware tokens
  const emptyBg     = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)';
  const filledBg    = isDark ? '#1a1a22'                : '#f6f5fa';
  const filledBorder= 'rgba(212,165,90,0.5)';
  const emptyBorder = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)';
  const placeholderColor = isDark ? 'rgba(238,238,243,0.4)' : 'rgba(22,22,32,0.45)';

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => onHoverChange?.(true)}
      onMouseLeave={() => onHoverChange?.(false)}
      onFocus={() => onHoverChange?.(true)}
      onBlur={() => onHoverChange?.(false)}
      title={hasTrack ? `${track.title || ''} — ${track.artist || ''}` : 'No track playing'}
      style={{
        width: 40, height: 40, borderRadius: '50%', padding: 0,
        background: hasTrack ? filledBg : emptyBg,
        border: '1px solid ' + (hasTrack ? filledBorder : emptyBorder),
        cursor: hasTrack ? 'pointer' : 'default',
        display: 'grid', placeItems: 'center', overflow: 'hidden',
        animation, transition: 'border-color 0.25s ease',
      }}>
      {hasTrack && cover ? (
        <img src={cover} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
      ) : hasTrack ? (
        <span className="serif-display" style={{ color: '#d4a55a', fontSize: 18, fontStyle: 'normal' }}>
          {(track.title || '?')[0]?.toUpperCase()}
        </span>
      ) : (
        <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontWeight:500, letterSpacing:'0.22em', textTransform:'uppercase', fontSize: 7, color: placeholderColor }}>MX</span>
      )}
    </button>
  );
}

// ─── MiniPlaybackPopout — 320×80 hover panel anchored to NowPlayingPebble ────
function MiniPlaybackPopout({ track, audio, isDark = true, onOpenPlayer, onClose, playlist, onTrackChange }) {
  const c = useColors(isDark);
  const safe = track || {};
  const title = safe.title || '—';
  const artist = safe.artist || '';
  // Prefix relative cover paths with ${API} (matches AlbumCover behavior).
  const rawCover = safe.cover_art_path || safe.coverArt || null;
  const cover = rawCover ? thumbCoverUrl(rawCover.startsWith('http') ? rawCover : `${API}${rawCover}`) : null;
  const duration = audio?.duration || 0;
  // Subscribe to time store so this popout updates smoothly without re-rendering
  // the rest of the app on every tick.
  const currentTime = useCurrentTime();
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  // Skip wiring: playlist is HIT[] (from search/recommend) so unwrap to FLAT
  // tracks before findIndex. onTrackChange propagates the chosen neighbour up
  // to App.playerTrack — PlayerSection's effect[initialPlaylist, initialTrack]
  // then handles the actual audio.setSrc + play.
  const flatTracks = (playlist || []).map(h => (h && h.track) ? h.track : h);
  const curIdx = track ? flatTracks.findIndex(t => t && t.track_id === track.track_id) : -1;
  const canPrev = curIdx > 0;
  const canNext = curIdx >= 0 && curIdx < flatTracks.length - 1;
  const goPrev = () => { if (canPrev && onTrackChange) onTrackChange(flatTracks[curIdx - 1]); };
  const goNext = () => { if (canNext && onTrackChange) onTrackChange(flatTracks[curIdx + 1]); };

  // Theme-aware tokens
  const panelBg = isDark
    ? 'linear-gradient(180deg, rgba(30,30,38,0.92) 0%, rgba(22,22,28,0.92) 100%)'
    : 'linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,244,250,0.92) 100%)';
  const panelBorder = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)';
  const coverBg = isDark ? '#1a1a22' : '#f0eff5';
  const scrubberTrack = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.10)';

  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
  };

  return (
    <div
      className="mini-popout-inner"
      style={{
      width: 320, padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 10,
      borderRadius: 14,
      background: panelBg,
      backdropFilter: 'blur(22px) saturate(1.1)',
      WebkitBackdropFilter: 'blur(22px) saturate(1.1)',
      border: `1px solid ${panelBorder}`,
      boxShadow: isDark
        ? 'inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 26px rgba(0,0,0,0.5)'
        : 'inset 0 1px 0 rgba(255,255,255,0.95), 0 8px 22px rgba(40,30,60,0.18)',
    }}
      onMouseLeave={onClose}
    >
      {/* Cover thumb */}
      <div style={{ width: 56, height: 56, borderRadius: 10, overflow: 'hidden', background: coverBg, flexShrink: 0, display: 'grid', placeItems: 'center' }}>
        {cover ? (
          <img src={cover} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <span className="serif-display" style={{ color: '#d4a55a', fontSize: 22, fontStyle: 'normal' }}>{title[0]?.toUpperCase() || '?'}</span>
        )}
      </div>

      {/* Meta + scrubber + controls */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: c.text, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{title}</div>
        <div style={{ fontSize: 11, color: c.textMuted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{artist}</div>

        {/* Scrubber — native range for reliable click+drag in a tiny space.
            Gradient on the background renders the played portion; thumb styled
            via .mini-scrubber CSS class. */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
          <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontSize: 9, letterSpacing:'0.08em', color: c.textSubtle }}>{fmt(currentTime)}</span>
          <input
            type="range" min={0} max={1} step={0.001} value={progress}
            onChange={(e) => {
              if (!audio || !duration) return;
              const val = parseFloat(e.target.value);
              audio.seek(val * duration);
            }}
            className="mini-scrubber"
            style={{
              flex: 1, height: 4, borderRadius: 2,
              WebkitAppearance: 'none', appearance: 'none', outline: 'none', border: 'none', cursor: 'pointer',
              background: `linear-gradient(90deg, oklch(60% 0.18 270) ${progress*100}%, ${scrubberTrack} ${progress*100}%)`,
            }}
          />
          <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontSize: 9, letterSpacing:'0.08em', color: c.textSubtle }}>{fmt(duration)}</span>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
          <button title="Previous" disabled={!canPrev}
                  style={{ ...ctrlBtn(false, isDark), opacity: canPrev ? 1 : 0.32, cursor: canPrev ? 'pointer' : 'not-allowed' }}
                  onClick={goPrev}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="19,4 8,12 19,20" /><rect x="5" y="4" width="2" height="16" /></svg>
          </button>
          <button title={audio?.isPlaying ? 'Pause' : 'Play'} style={ctrlBtn(true, isDark)} onClick={() => audio?.togglePlay?.()}>
            {audio?.isPlaying
              ? <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
              : <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="6,4 20,12 6,20"/></svg>}
          </button>
          <button title="Next" disabled={!canNext}
                  style={{ ...ctrlBtn(false, isDark), opacity: canNext ? 1 : 0.32, cursor: canNext ? 'pointer' : 'not-allowed' }}
                  onClick={goNext}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,4 16,12 5,20" /><rect x="17" y="4" width="2" height="16" /></svg>
          </button>
          <div style={{ flex: 1 }} />
          <button onClick={onOpenPlayer} title="Open Player"
                  style={{ background: 'transparent', border: 0, color: 'oklch(70% 0.16 75)', cursor: 'pointer', fontSize: 9, padding: '4px 6px',
                           fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontWeight:500, letterSpacing:'0.18em', textTransform:'uppercase' }}>
            OPEN →
          </button>
        </div>
      </div>
    </div>
  );
}

function ctrlBtn(primary = false, isDark = true) {
  const fgInactive = isDark ? 'rgba(238,238,243,0.6)' : 'rgba(22,22,32,0.6)';
  const primaryBg  = isDark
    ? 'linear-gradient(180deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05))'
    : 'linear-gradient(180deg, rgba(0,0,0,0.08), rgba(0,0,0,0.03))';
  const primaryBorder = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';
  const primaryFg = isDark ? '#fff' : '#161620';
  return {
    width: primary ? 28 : 24, height: primary ? 28 : 24, borderRadius: '50%',
    background: primary ? primaryBg : 'transparent',
    border: primary ? `1px solid ${primaryBorder}` : '0',
    color: primary ? primaryFg : fgInactive,
    display: 'grid', placeItems: 'center', cursor: 'pointer', padding: 0,
  };
}

// ─── Section Header (used inside all sub-sections) ────────────────────────────
function SectionHeader({ isDark, lang, kicker, title, accent, right, children }) {
  const c = useColors(isDark);
  // The kicker + big serif title were removed app-wide to free vertical
  // space (sidebar nav already shows which section is active). The header
  // remains only as a host for `right` (tab toggles, filters) and `children`.
  // When nothing fills those slots, the component collapses entirely.
  if (!right && !children) return null;
  return (
    <div style={{ padding:'14px 36px 10px', borderBottom:`1px solid ${c.border}`,
      background: isDark ? 'linear-gradient(180deg, rgba(255,255,255,0.018) 0%, transparent 100%)' : 'linear-gradient(180deg, rgba(255,255,255,0.6) 0%, transparent 100%)' }}>
      {right && (
        <div style={{ display:'flex', justifyContent:'flex-end' }}>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

// ─── Segmented control ────────────────────────────────────────────────────────
function Segmented({ value, onChange, options, isDark, size='md', style } = {}) {
  const c = useColors(isDark);
  const padY = size === 'sm' ? '5px' : '7px';
  return (
    <div className={ske('inset', isDark)} style={{ display:'inline-flex', padding:'3px', borderRadius:'10px', gap:'2px', ...style }}>
      {options.map(o => {
        const active = value === o.value;
        return (
          <button key={o.value} onClick={() => onChange(o.value)}
            className={active ? ske('btn', isDark) : ''}
            style={{
              padding:`${padY} 14px`, borderRadius:'8px',
              fontSize: size === 'sm' ? '11px' : '12px', fontWeight:'600',
              fontFamily:"'JetBrains Mono', monospace", letterSpacing:'0.06em',
              color: active ? c.text : c.textSubtle,
              transition:'all 0.18s', whiteSpace:'nowrap',
            }}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// ─── SCORE BREAKDOWN TOOLTIP ──────────────────────────────────────────────────
function ScoreBreakdownTooltip({ hit, breakdown, isDark, onPlay, onAddToPlaylist, lang }) {
  const c = useColors(isDark);
  if (!breakdown && !hit) return null;
  const rows = breakdown ? [
    { label: 'Text',  v: breakdown.text_dense_score },
    { label: 'Audio', v: breakdown.audio_score },
  ].filter(r => r.v != null) : [];
  return (
    <div className="panel-v3" style={{
      position:'absolute', top:'100%', left:'50%', transform:'translateX(-50%)',
      marginTop:'8px', padding:'10px 12px', minWidth:'200px', maxWidth:'calc(100vw - 28px)',
      zIndex:50, fontSize:'12px', color:c.text,
      boxShadow:'0 8px 24px rgba(0,0,0,0.32)',
    }}>
      {rows.length > 0 && (
        <div style={{ marginBottom:'8px' }}>
          {rows.map(r => (
            <div key={r.label} style={{ display:'flex', justifyContent:'space-between', gap:'12px', padding:'2px 0' }}>
              <span style={{ color:c.textSubtle }}>{r.label}</span>
              <span style={{ fontFamily:'ui-monospace, monospace' }}>{(r.v * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
      <div style={{
        display:'flex', gap:'6px', justifyContent:'center',
        borderTop: rows.length ? `1px solid ${c.border}` : 'none',
        paddingTop: rows.length ? '8px' : 0,
      }}>
        <button
          className="pill-v3"
          onClick={e => { e.stopPropagation(); onPlay && onPlay(hit); }}
          style={{ padding:'4px 10px', fontSize:'11px', cursor:'pointer' }}
        >▶ Play</button>
        <button
          className="pill-v3"
          onClick={e => { e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(hit.track?.track_id || hit.track_id, e.currentTarget); }}
          style={{ padding:'4px 10px', fontSize:'11px', cursor:'pointer' }}
          title={lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}
        >＋ {lang === 'ru' ? 'В плейлист' : 'Add'}</button>
      </div>
    </div>
  );
}

// ─── RECENT SEARCHES HOOK ─────────────────────────────────────────────────────
function useRecentSearches(userId) {
  const storageKey = `recentSearches:${userId || '_default'}`;
  const [items, setItems] = useState(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(0, 8) : [];
    } catch { return []; }
  });

  const push = useCallback((query) => {
    const q = (query || '').trim();
    if (!q) return;
    setItems(prev => {
      const next = [q, ...prev.filter(x => x.toLowerCase() !== q.toLowerCase())].slice(0, 8);
      try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch {}
      return next;
    });
  }, [storageKey]);

  const remove = useCallback((query) => {
    setItems(prev => {
      const next = prev.filter(x => x !== query);
      try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch {}
      return next;
    });
  }, [storageKey]);

  const clearAll = useCallback(() => {
    setItems([]);
    try { localStorage.removeItem(storageKey); } catch {}
  }, [storageKey]);

  // Re-read when userId changes (different storageKey).
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      const parsed = raw ? JSON.parse(raw) : [];
      setItems(Array.isArray(parsed) ? parsed.slice(0, 8) : []);
    } catch { setItems([]); }
  }, [storageKey]);

  return { items, push, remove, clearAll };
}

// ─── PLAYLISTS HOOK ───────────────────────────────────────────────────────────
function usePlaylists() {
  const [playlists, setPlaylists] = React.useState([]);
  const [loading, setLoading]     = React.useState(false);

  const refetchAll = React.useCallback(async () => {
    setLoading(true);
    try {
      const r = await apiFetch(`/playlists`);
      setPlaylists(r?.playlists || []);
    } catch (e) {
      console.warn('[usePlaylists] list failed', e);
      setPlaylists([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchWithMembership = React.useCallback(async (trackId) => {
    if (!trackId) return { playlists: []};
    try {
      return await apiFetch(`/playlists?include_track_id=${encodeURIComponent(trackId)}`);
    } catch (e) {
      console.warn('[usePlaylists] membership fetch failed', e);
      return { playlists: []};
    }
  }, []);

  const createPlaylist = async (name, description) => {
    const r = await apiFetch(`/playlists`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: description || null }),
    });
    // Background refresh: the create dialog closes on POST success; awaiting
    // the list refetch here kept the "Create" button spinning for seconds on
    // a busy connection (the modal waited on a request it didn't need).
    refetchAll();
    return r;
  };

  const renamePlaylist = async (id, patch) => {
    const r = await apiFetch(`/playlists/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    await refetchAll();
    return r;
  };

  const deletePlaylist = async (id) => {
    await apiFetch(`/playlists/${id}`, { method: 'DELETE' });
    await refetchAll();
  };

  const addTrack = async (id, trackId) => {
    const r = await apiFetch(`/playlists/${id}/tracks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_id: trackId }),
    });
    refetchAll();   // background — same reasoning as createPlaylist
    return r;
  };

  const removeTrack = async (id, trackId) => {
    await apiFetch(`/playlists/${id}/tracks/${encodeURIComponent(trackId)}`, { method: 'DELETE' });
    await refetchAll();
  };

  const reorderTracks = async (id, trackIds) => {
    const r = await apiFetch(`/playlists/${id}/reorder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_ids: trackIds }),
    });
    await refetchAll();
    return r;
  };

  React.useEffect(() => { refetchAll(); }, [refetchAll]);

  return {
    playlists, loading,
    refetchAll, fetchWithMembership,
    createPlaylist, renamePlaylist, deletePlaylist,
    addTrack, removeTrack, reorderTracks,
  };
}

// ─── RECENT SEARCHES CHIPS ────────────────────────────────────────────────────
function RecentSearchesChips({ items, onPick, onRemove, onClearAll, isDark }) {
  const c = useColors(isDark);
  if (!items || items.length === 0) return null;
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', flexWrap:'wrap', padding:'8px 0' }}>
      <span style={{ fontSize:'12px', color:c.textSubtle }}>Recent:</span>
      {items.map(q => (
        <button
          key={q}
          className="pill-v3"
          onClick={() => onPick(q)}
          style={{ padding:'4px 10px', fontSize:'13px', cursor:'pointer',
                   display:'inline-flex', alignItems:'center', gap:'6px' }}
        >
          <span>{q}</span>
          <span
            role="button"
            aria-label="remove"
            onClick={e => { e.stopPropagation(); onRemove(q); }}
            style={{ opacity:0.6, fontSize:'11px', cursor:'pointer' }}
          >×</span>
        </button>
      ))}
      <button
        onClick={onClearAll}
        style={{ background:'transparent', border:'none', color:c.textSubtle,
                 fontSize:'11px', cursor:'pointer', textDecoration:'underline' }}
      >Clear all</button>
    </div>
  );
}

// ─── SONIC FILTERS CHIPS ──────────────────────────────────────────────────────
function SonicFiltersChips({ selectedTags, onToggleTag, isDark }) {
  const c = useColors(isDark);
  const [facets, setFacets] = useState({ tags: [] });
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    setError(false);
    apiFetch('/library/sonic-facets?top_k=30')
      .then(data => { if (alive) setFacets(data || { tags: [] }); })
      .catch(() => { if (alive) setError(true); });
    return () => { alive = false; };
  }, []);

  if (error) return (
    <div style={{ fontSize:'12px', color:c.textSubtle, padding:'6px 0' }}>
      Sonic filters unavailable.
    </div>
  );
  if (!facets.tags || !facets.tags.length) return null;

  const renderChip = (label, count, active, onClick) => (
    <button
      key={label}
      onClick={onClick}
      className={`pill-v3${active ? ' pill-v3-active' : ''}`}
      style={{
        padding:'4px 10px', fontSize:'12px', cursor:'pointer',
        opacity: active ? 1 : 0.7,
        border: active ? `1px solid ${c.accent}` : '1px solid transparent',
      }}
    >
      {label} <span style={{ opacity:0.5, fontSize:'10px', marginLeft:'4px' }}>{count}</span>
    </button>
  );

  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', flexWrap:'wrap', padding:'8px 0' }}>
      <span style={{ fontSize:'12px', color:c.textSubtle, minWidth:'48px' }}>Tags:</span>
      {facets.tags.map(({ value, count }) =>
        renderChip(value, count, selectedTags.includes(value), () => onToggleTag(value))
      )}
    </div>
  );
}

// ─── DECADE FILTERS CHIPS ─────────────────────────────────────────────────────
function DecadeFiltersChips({ selectedRanges, onToggleRange, isDark }) {
  const c = useColors(isDark);
  const [facets, setFacets] = useState({ year_ranges: [] });
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    setError(false);
    const col = '';
    apiFetch(`/library/year-facets?top_k=30${col}`)
      .then(data => { if (alive) setFacets(data || { year_ranges: [] }); })
      .catch(() => { if (alive) setError(true); });
    return () => { alive = false; };
  }, []);

  if (error) return null;  // silent — decade filter is optional
  if (!facets.year_ranges || !facets.year_ranges.length) return null;

  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', flexWrap:'wrap', padding:'8px 0' }}>
      <span style={{ fontSize:'12px', color:c.textSubtle, minWidth:'48px' }}>Years:</span>
      {facets.year_ranges.map(({ value, count }) => {
        const active = selectedRanges.includes(value);
        return (
          <button
            key={value}
            onClick={() => onToggleRange(value)}
            className={`pill-v3${active ? ' pill-v3-active' : ''}`}
            style={{ padding:'4px 10px', fontSize:'12px', cursor:'pointer' }}
          >
            {value} <span style={{ opacity:0.5, fontSize:'10px', marginLeft:'4px' }}>{count}</span>
          </button>
        );
      })}
    </div>
  );
}

// ─── Chat: small inline icons ────────────────────────────────────────────────
const StepCheck = ({ size = 10 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
);

// Russian plural: (1 шаг, 2 шага, 5 шагов)
const pluralRu = (n, one, few, many) => {
  const m = Math.abs(n) % 100, d = m % 10;
  if (m >= 11 && m <= 14) return many;
  if (d === 1) return one;
  if (d >= 2 && d <= 4) return few;
  return many;
};

// Crossfades its text on change: the outgoing label floats up and out while
// the incoming one rises in (absolute overlap — never a hard swap).
function CrossfadeText({ text }) {
  const [pair, setPair] = useState({ curr: text, prev: null, k: 0 });
  useEffect(() => {
    setPair(p => (text === p.curr ? p : { curr: text, prev: p.curr, k: p.k + 1 }));
  }, [text]);
  return (
    <span className="xfade-wrap">
      {pair.prev != null && (
        <span key={`p${pair.k}`} className="xfade out"
          onAnimationEnd={() => setPair(p => ({ ...p, prev: null }))}>{pair.prev}</span>
      )}
      <span key={`c${pair.k}`} className="xfade in">{pair.curr}</span>
    </span>
  );
}

// ─── AgentSteps — live agent-work card, collapses after the answer ───────────
// While streaming: a large liquid-glass card sitting where the answer will
// appear — active step in big crossfading type, completed steps stacked small
// above, shimmer strip along the bottom. After the answer: one summary row
// that expands on click.
function AgentSteps({ steps, streaming, lang }) {
  const [open, setOpen] = useState(false);
  const list = steps || [];

  if (streaming) {
    const active = list.length ? list[list.length - 1] : { human: lang === 'ru' ? 'Думаю…' : 'Thinking…' };
    const done = list.slice(0, -1);
    return (
      <div className="liquid-glass agent-card" style={{ marginBottom: 6 }}>
        {done.length > 0 && (
          <div className="agent-card-done agent-steps">
            {done.map((s, i) => (
              <div key={i} className="agent-step">
                <div className="agent-step-rail">
                  <div className="agent-step-dot is-done"><span className="agent-step-check"><StepCheck /></span></div>
                  <div className="agent-step-connector" />
                </div>
                <div className="agent-step-label">{s.human}</div>
              </div>
            ))}
          </div>
        )}
        <div className="agent-card-active">
          <div className="agent-step-spin" />
          <CrossfadeText text={active.human} />
        </div>
        <div className="agent-card-shimmer" />
      </div>
    );
  }

  if (!list.length) return null;
  const summary = lang === 'ru'
    ? `Нашёл за ${list.length} ${pluralRu(list.length, 'шаг', 'шага', 'шагов')}`
    : `Found in ${list.length} step${list.length === 1 ? '' : 's'}`;
  return (
    <div style={{ marginBottom: 6 }}>
      <div className={`agent-steps-summary ${open ? 'open' : ''}`} onClick={() => setOpen(o => !o)}>
        <span className="agent-step-check" style={{ display: 'inline-flex' }}><StepCheck /></span>
        <span>{summary}</span>
        <svg className="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2.5" strokeLinecap="round"><path d="m6 9 6 6 6-6"/></svg>
      </div>
      <div className={`collapse-rows ${open ? 'open' : ''}`}>
        <div className="collapse-inner">
          <div className="agent-steps" style={{ paddingTop: 8 }}>
            {list.map((s, i) => (
              <div key={i} className="agent-step">
                <div className="agent-step-rail">
                  <div className="agent-step-dot is-done"><span className="agent-step-check"><StepCheck /></span></div>
                  {i < list.length - 1 && <div className="agent-step-connector" />}
                </div>
                <div className="agent-step-label">{s.human}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── LyricSnippet — matched lyric line + context, inline expand ───────────────
// Highlights the matched line (from hit.matched_line, else a word-overlap
// heuristic) with one neighbour above/below; "show more" fades to full lyrics.
// Excerpts sometimes arrive with mangled/absent line breaks — matching is
// whitespace-normalized, and when the "line" is a wall of text the highlight
// falls back to slicing context around the matched phrase (never the whole
// block).
const GIANT_LINE = 160;  // beyond this a "line" is a blob, not a lyric line
function LyricSnippet({ lyrics, query, matchedLine, lang, c }) {
  const reduced = usePrefersReducedMotion();
  const [open, setOpen] = useState(false);
  const text = String(lyrics || '').trim();
  const lines = useMemo(() => text.split('\n').map(l => l.trim()).filter(Boolean), [text]);

  // Locate the matched line: exact → normalized equality → normalized contains
  // → query-word-overlap heuristic. -1 = nothing matched (no highlight).
  const idx = useMemo(() => {
    if (!lines.length) return -1;
    const norm = s => s.toLowerCase().replace(/\s+/g, ' ').trim();
    if (matchedLine) {
      const m = String(matchedLine).trim();
      let i = lines.findIndex(l => l === m);
      if (i < 0) { const nm = norm(m); i = lines.findIndex(l => norm(l) === nm); }
      if (i < 0) { const nm = norm(m); i = lines.findIndex(l => norm(l).includes(nm)); }
      if (i >= 0) return i;
    }
    const terms = String(query || '').toLowerCase().split(/\s+/).filter(t => t.length > 2);
    if (!terms.length) return -1;
    let best = -1, bestScore = 0;
    lines.forEach((l, i) => {
      const low = l.toLowerCase();
      const score = terms.reduce((a, t) => a + (low.includes(t) ? 1 : 0), 0);
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return best;
  }, [lines, query, matchedLine]);

  // Blob mode: the highlight target is a giant block — find the matched
  // phrase inside the raw text (whitespace-tolerant regex) and cut context
  // around it instead of highlighting the whole block.
  const blob = useMemo(() => {
    const giant = idx >= 0
      ? lines[idx].length > GIANT_LINE
      : (lines.length > 0 && lines.length <= 2 && lines[0].length > GIANT_LINE);
    if (!giant) return null;
    if (!matchedLine) return { plain: true };
    const words = String(matchedLine).trim().split(/\s+/)
      .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    if (!words.length) return { plain: true };
    let m = null;
    try { m = new RegExp(words.join('\\s+'), 'i').exec(text); } catch { /* noop */ }
    if (!m) return { plain: true };
    const beforeFull = text.slice(0, m.index).replace(/\s+/g, ' ').trim();
    const afterFull  = text.slice(m.index + m[0].length).replace(/\s+/g, ' ').trim();
    const match = m[0].replace(/\s+/g, ' ');
    const beforeCtx = beforeFull.length > 110
      ? '…' + beforeFull.slice(-110).replace(/^\S*\s/, '') : beforeFull;
    const afterCtx = afterFull.length > 110
      ? afterFull.slice(0, 110).replace(/\s\S*$/, '') + '…' : afterFull;
    return { match, beforeFull, afterFull, beforeCtx, afterCtx };
  }, [lines, idx, text, matchedLine]);

  if (!lines.length) return null;
  const vars = {
    '--lyric-muted': c.textMuted,
    '--lyric-strong': c.text,
    '--lyric-hl': c.accentBg,
    '--lyric-accent': c.accent,
  };

  let rows, matchAt, expandable;
  if (blob && !blob.plain) {
    rows = (open
      ? [blob.beforeFull, blob.match, blob.afterFull]
      : [blob.beforeCtx, blob.match, blob.afterCtx]);
    matchAt = 1;
    // drop empty context rows, keeping the highlight on the match
    const keep = rows.map((r, i) => ({ r, hl: i === 1 })).filter(x => x.r);
    rows = keep.map(x => x.r);
    matchAt = keep.findIndex(x => x.hl);
    expandable = blob.beforeFull.length > blob.beforeCtx.length || blob.afterFull.length > blob.afterCtx.length;
  } else if (blob) {
    // wall of text, nothing to highlight — clipped plain preview
    const flat = text.replace(/\s+/g, ' ').trim();
    rows = [open ? flat : (flat.length > 220 ? flat.slice(0, 220).replace(/\s\S*$/, '') + '…' : flat)];
    matchAt = -1;
    expandable = flat.length > 220;
  } else {
    const anchor = idx >= 0 ? idx : 0;
    const from = Math.max(0, anchor - 1);
    const preview = lines.slice(from, Math.min(lines.length, anchor + 2));
    rows = open ? lines : preview;
    matchAt = idx < 0 ? -1 : (open ? idx : idx - from);
    expandable = lines.length > preview.length;
  }

  return (
    <div className="lyric-snippet" style={vars} onClick={e => e.stopPropagation()}>
      <div key={open ? 'full' : 'prev'} style={{ animation: reduced ? undefined : 'fadeIn .25s ease' }}>
        {rows.map((l, i) => (
          <div key={i} className={`lyric-line ${i === matchAt ? 'lyric-line-match' : ''}`}>{l}</div>
        ))}
      </div>
      {expandable && (
        <button className="lyric-more" onClick={() => setOpen(o => !o)}>
          {open ? (lang === 'ru' ? 'свернуть' : 'show less') : (lang === 'ru' ? 'показать больше' : 'show more')}
        </button>
      )}
    </div>
  );
}

// ─── SEARCH SECTION (redesigned) ──────────────────────────────────────────────
function SearchSection({ isDark, lang, onPlayTrack, navigateToArtist, aiStatus, onAddToPlaylist, searchHandoff }) {
  const c = useColors(isDark);
  const aiActive = !!(aiStatus && aiStatus.aiActive);
  const isMobileChat = useIsMobile();  // compact best-hit cover on phones
  const recent = useRecentSearches(localStorage.getItem('musix_user_id'));
  const { sessions, saveSession, deleteSession } = useChatHistory(localStorage.getItem('musix_user_id'));

  // Mode: 'chat' (AI dialog — the default whenever the assistant is up) or
  // 'search' (classic grid). Sections mount before the AI probe resolves, so
  // an effect below upgrades the default to 'chat' once aiActive lands —
  // unless the user already picked a tab or started working.
  const [tab, setTab] = useState(aiActive ? 'chat' : 'search');
  const tabTouchedRef = useRef(false);
  const pickTab = (id) => { tabTouchedRef.current = true; setTab(id); };

  // Chat state — an empty conversation renders the hero (slogan + centered
  // composer) instead of a greeting bubble, GPT-style.
  const initMsg = () => [];
  const [messages, setMessages] = useState(initMsg);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatMode, setChatMode] = useState('hybrid');
  const [autoMode, setAutoMode] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [historyClosing, setHistoryClosing] = useState(false);
  const [railOpen, setRailOpen] = useState(() => localStorage.getItem('musix_chat_rail') !== '0');
  const sessionIdRef = useRef(null);  // current conversation's history-session id (upsert target)
  const chatEndRef = useRef(null);
  const toggleRail = () => setRailOpen(v => {
    try { localStorage.setItem('musix_chat_rail', v ? '0' : '1'); } catch {}
    return !v;
  });

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState('hybrid');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [searchExecuted, setSearchExecuted] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({ artist:'', album:'', genre:'', sonic_tags:[], year_ranges:[] });
  const [filterSuggestions, setFilterSuggestions] = useState({ artist:[], album:[], genre:[] });
  const [activeFilterField, setActiveFilterField] = useState(null);
  const filterTimer = useRef(null);

  // Shared results
  const [results, setResults] = useState([]);
  const [hoveredHit, setHoveredHit] = useState(null);

  const getLLMSettings = () => ({
    llm_base_url: localStorage.getItem('llm_base_url') || undefined,
    llm_model:    localStorage.getItem('llm_model')    || undefined,
    planner_enabled: true,
  });

  useEffect(() => {
    if (chatEndRef.current) chatEndRef.current.scrollTop = chatEndRef.current.scrollHeight;
  }, [messages, tab]);

  // ── Snap back to search if AI becomes unavailable ──
  useEffect(() => {
    if (!aiActive && tab === 'chat') setTab('search');
  }, [aiActive, tab]);

  // ── AI probe landed → upgrade the default tab to the chat (unless the user
  // already picked a tab or has work in progress on the grid) ──
  useEffect(() => {
    if (aiActive && !tabTouchedRef.current && messages.length === 0 && !searchExecuted) setTab('chat');
  }, [aiActive]);

  // ── Chat handler (streaming) ──
  // POSTs /chat/stream and animates the agent's step events live; the trailing
  // assistant message accumulates steps, then the `answer` event fills its body.
  // Falls back to the non-streaming /chat/ endpoint if the stream can't open.
  const handleChat = async (textArg) => {
    const userText = (typeof textArg === 'string' ? textArg : input).trim();
    if (!userText || loading) return;
    const now = new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'});
    const effectiveMode = autoMode ? 'hybrid' : chatMode;
    const history = messages.filter(m=>!m.loading && !m.streaming)
      .map(m=>({ role:m.role==='assistant'?'assistant':'user', content:m.text||'' }));
    const body = { message:userText, history, mode: effectiveMode, auto_mode: autoMode, lang, ...getLLMSettings() };

    setMessages([...messages, { role:'user', text:userText, time:now },
                              { role:'assistant', streaming:true, steps:[], time:now }]);
    setInput(''); setLoading(true);

    // Patch the trailing assistant placeholder in place.
    const patchLast = (patch) => setMessages(prev => {
      const copy = prev.slice();
      const li = copy.length - 1;
      copy[li] = typeof patch === 'function' ? patch(copy[li]) : { ...copy[li], ...patch };
      return copy;
    });

    let answered = false;

    // Step events arrive in bursts (the backend emits classify+plan back to
    // back once the planner finishes), which made stages flash by unseen.
    // Pace the reveal: each queued step shows for ≥REVEAL_MS before the next;
    // whatever is still queued when the answer lands is folded into the
    // final message (it collapses into the summary row anyway).
    const REVEAL_MS = 750;
    const stepQueue = [];
    let queueTimer = null, lastReveal = 0;
    const pumpSteps = () => {
      if (queueTimer || !stepQueue.length || answered) return;
      const wait = Math.max(0, lastReveal + REVEAL_MS - Date.now());
      queueTimer = setTimeout(() => {
        queueTimer = null;
        if (answered) return;
        const ev = stepQueue.shift();
        if (ev) {
          lastReveal = Date.now();
          patchLast(m => ({ ...m, steps: [...(m.steps||[]), ev] }));
        }
        pumpSteps();
      }, wait);
    };

    const finalizeAnswer = (ev) => {
      answered = true;
      if (queueTimer) { clearTimeout(queueTimer); queueTimer = null; }
      const hits = ev.hits || [];
      setResults(hits);
      setMessages(prev => {
        const copy = prev.slice();
        const li = copy.length - 1;
        copy[li] = {
          role:'assistant', text:ev.message||'…', hits, best_hit:ev.best_hit,
          confidence:ev.confidence, song:ev.song, artist:ev.artist,
          attempts:ev.attempts, streaming:false,
          steps: [...(copy[li].steps || []), ...stepQueue.splice(0)],
          time:new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'})
        };
        sessionIdRef.current = saveSession(copy, sessionIdRef.current);
        return copy;
      });
    };

    const onEvent = (ev) => {
      if (ev.type === 'answer') { finalizeAnswer(ev); return; }
      if (ev.type === 'error') {
        if (queueTimer) { clearTimeout(queueTimer); queueTimer = null; }
        patchLast({ streaming:false, text:`${lang==='ru'?'Ошибка':'Error'}: ${ev.message||''}` });
        return;
      }
      stepQueue.push(ev);  // classify/plan/search/validate/retry
      pumpSteps();
    };

    try {
      await apiStream('/chat/stream', body, onEvent);
      if (!answered) throw new Error('stream ended without answer');
    } catch(e) {
      if (answered) { setLoading(false); return; }
      // Fallback: non-streaming endpoint — no live steps, but a real answer.
      try {
        const res = await apiFetch('/chat/', { method:'POST', body: JSON.stringify(body) });
        finalizeAnswer({ ...res });
      } catch(e2) {
        patchLast({ streaming:false, text:`${lang==='ru'?'Ошибка':'Error'}: ${e2.message}` });
      }
    } finally { setLoading(false); }
  };

  // ── Search handler ──
  // queryArg: the landing/spotlight handoff runs a query straight away —
  // state hasn't flushed yet at that point (same trick as handleChat's textArg;
  // a click event object falls through to the state value).
  const handleSearch = async (queryArg) => {
    const query = (typeof queryArg === 'string' ? queryArg : searchQuery).trim();
    if (!query || searchLoading) return;
    setSearchLoading(true); setSearchError('');
    try {
      const f = {};
      ['artist','album','genre'].forEach(k => { if (filters[k]) f[k] = filters[k]; });
      if (filters.sonic_tags.length) f.sonic_tags = filters.sonic_tags;
      if (filters.year_ranges.length) f.year_ranges = filters.year_ranges;
      const res = await apiFetch('/search/', { method:'POST', body: JSON.stringify({
        query, mode:searchMode, filters:Object.keys(f).length?f:null,
        limit:20
      }) });
      setResults(res.hits||[]);
      setSearchExecuted(true);
      recent.push(query);
    } catch(e) { setSearchError(e.message); }
    finally { setSearchLoading(false); }
  };

  // ── One-shot query handoff (landing lyrics field / spotlight «ещё») ──
  // ts deduping lets the same query re-fire on a second submit. mode 'grid'
  // forces the classic tab; 'auto' prefers the AI chat when it's up.
  const handledHandoffRef = useRef(null);
  useEffect(() => {
    if (!searchHandoff || !searchHandoff.query) return;
    if (handledHandoffRef.current === searchHandoff.ts) return;
    handledHandoffRef.current = searchHandoff.ts;
    tabTouchedRef.current = true;
    if (searchHandoff.mode !== 'grid' && aiActive) {
      setTab('chat');
      handleChat(searchHandoff.query);
    } else {
      setTab('search');
      setSearchQuery(searchHandoff.query);
      handleSearch(searchHandoff.query);
    }
  }, [searchHandoff]);

  // ── Filter autocomplete: fetch suggestions from /browse ──
  const fetchFilterSuggestions = async (field, query) => {
    if (!query || query.length < 2) {
      setFilterSuggestions(p => ({ ...p, [field]: [] }));
      return;
    }
    try {
      const col = '';
      const data = await apiFetch(`/library/browse?q=${encodeURIComponent(query)}&limit=3${col}`);
      // /browse returns: { track_id, title, artist, album, cover_art_path, score }
      const seen = new Set();
      const unique = [];
      for (const item of (data || [])) {
        const val = (item[field] || '').trim();
        if (val && !seen.has(val.toLowerCase())) {
          seen.add(val.toLowerCase());
          unique.push(val);
        }
      }
      setFilterSuggestions(p => ({ ...p, [field]: unique.slice(0, 3) }));
    } catch {
      setFilterSuggestions(p => ({ ...p, [field]: [] }));
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(p => ({ ...p, [field]: value }));
    setActiveFilterField(field);
    if (filterTimer.current) clearTimeout(filterTimer.current);
    filterTimer.current = setTimeout(() => fetchFilterSuggestions(field, value), 250);
  };

  const selectFilterSuggestion = (field, value) => {
    setFilters(p => ({ ...p, [field]: value }));
    setFilterSuggestions(p => ({ ...p, [field]: [] }));
    setActiveFilterField(null);
  };

  const toggleSonicTag = (value) => {
    setFilters(p => ({
      ...p,
      sonic_tags: p.sonic_tags.includes(value)
        ? p.sonic_tags.filter(x => x !== value)
        : [...p.sonic_tags, value],
    }));
  };

  const toggleYearRange = (value) => {
    setFilters(p => ({
      ...p,
      year_ranges: p.year_ranges.includes(value)
        ? p.year_ranges.filter(x => x !== value)
        : [...p.year_ranges, value],
    }));
  };

  // ── Render a secondary (non-best) result card ──
  const renderHitCard = (hit, hi, hits) => (
    <div key={hi} className="chat-result-card"
      style={{ background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)' }}
      onClick={() => { setResults(hits); onPlayTrack && onPlayTrack(hit, hits); }}
      onMouseEnter={e => { e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'; }}>
      <AlbumCover title={hit.track?.title||''} artist={hit.track?.artist||''} size={56} isDark={isDark} coverPath={hit.track?.cover_art_path} />
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:'14px', fontWeight:'600', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{hit.track?.title||'—'}</div>
        <div style={{ fontSize:'13px', color:c.textMuted, display: 'inline-block' }}>
          <ArtistCredit track={hit.track} navigateToArtist={navigateToArtist} lang={lang} color={c.textMuted} />
        </div>
      </div>
      <div className="mono" style={{ padding:'3px 8px', borderRadius:'16px', fontSize:'12px',
        background: c.accentBg, color: c.accent, flexShrink:0 }}>
        {Math.round((hit.score||0)*100)}%
      </div>
      <button onClick={e => { e.stopPropagation(); onPlayTrack && onPlayTrack(hit, hits); }}
        style={{ width:'28px', height:'28px', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center',
          flexShrink:0, background:'linear-gradient(180deg, oklch(60% 0.21 270), oklch(48% 0.22 285))',
          boxShadow:'0 2px 6px oklch(58% 0.21 270 / 0.3)', fontSize:'10px', color:'white' }}>
        ▶
      </button>
    </div>
  );

  // ── Render the accentuated best-hit card (liquid glass + lyric snippet) ──
  const renderBestHit = (hit, hits, conf, confColor, userQuery) => {
    // hit.lyrics is newline-flattened by the backend (LLM context form);
    // track.lyrics keeps the raw line breaks — prefer it so the snippet
    // renders as verse lines, not one run-together paragraph.
    const snippetLyrics = hit.track?.lyrics || hit.lyrics;
    const snippet = snippetLyrics && hit.matched_on !== 'audio';
    const coverSize = isMobileChat ? 72 : 96;
    return (
      <div className="liquid-glass best-hit-card"
        style={{ flexDirection:'column', alignItems:'stretch', gap:'11px' }}
        onClick={() => { setResults(hits); onPlayTrack && onPlayTrack(hit, hits); }}>
        {/* confidence accent bar (glass clips it) */}
        <div style={{ position:'absolute', left:0, top:0, bottom:0, width:'3px', background:confColor, opacity:0.9 }} />
        <div style={{ display:'flex', alignItems:'center', gap:'15px' }}>
          <AlbumCover title={hit.track?.title||''} artist={hit.track?.artist||''} size={coverSize} isDark={isDark} coverPath={hit.track?.cover_art_path} />
          <div style={{ flex:1, minWidth:0 }}>
            <div style={{ fontSize:'18px', fontWeight:'700', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', letterSpacing:'-0.01em' }}>{hit.track?.title||'—'}</div>
            <div style={{ fontSize:'14px', color:c.textMuted, display:'inline-block', marginTop:'2px' }}>
              <ArtistCredit track={hit.track} navigateToArtist={navigateToArtist} lang={lang} color={c.textMuted} />
            </div>
          </div>
          <div className="mono" style={{ padding:'4px 9px', borderRadius:'16px', fontSize:'12px', fontWeight:'600',
            background:c.accentBg, color:c.accent, flexShrink:0 }}>
            {Math.round((hit.score||0)*100)}%
          </div>
          <button onClick={e => { e.stopPropagation(); onPlayTrack && onPlayTrack(hit, hits); }}
            style={{ width:'34px', height:'34px', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center',
              flexShrink:0, background:'linear-gradient(180deg, oklch(60% 0.21 270), oklch(48% 0.22 285))',
              boxShadow:'0 3px 9px oklch(58% 0.21 270 / 0.4)', fontSize:'12px', color:'white' }}>
            ▶
          </button>
        </div>
        {snippet && <LyricSnippet lyrics={snippetLyrics} query={userQuery} matchedLine={hit.matched_line} lang={lang} c={c} />}
      </div>
    );
  };

  // ── Render chat message ──
  const renderMessage = (msg, i) => {
    const isUser = msg.role==='user';
    const conf = msg.confidence;
    const confColor = conf==='high' ? c.green : conf==='medium' ? c.amber : c.textSubtle;
    const isAssistant = !isUser;
    const showBubble = isUser || !!msg.text;
    const showSteps = isAssistant && (msg.streaming || (msg.steps && msg.steps.length > 0));
    const userQuery = (i > 0 && messages[i-1]?.role === 'user') ? messages[i-1].text : '';
    return (
      <div key={i} style={{ display:'flex', flexDirection:'column', alignItems:isUser?'flex-end':'flex-start', gap:'6px', animation:'fadeIn 0.3s ease', width:'100%' }}>
        {showSteps && (
          <div style={{ width:'100%' }}>
            <AgentSteps steps={msg.steps} streaming={!!msg.streaming} lang={lang} />
          </div>
        )}
        {showBubble && (
          <div style={{
            maxWidth: isUser ? '78%' : '100%', padding:'11px 15px',
            borderRadius: isUser?'16px 16px 5px 16px':'16px 16px 16px 5px',
            background: isUser ? c.userBubble : c.aiBubble,
            color: isUser?'white':c.text, fontSize:'15px', lineHeight:'1.6',
            boxShadow: isDark
              ? (isUser ? 'inset 0 1px 0 rgba(255,255,255,0.18), 0 3px 10px rgba(0,0,0,0.45)' : 'inset 0 1px 0 rgba(255,255,255,0.05), 0 1px 4px rgba(0,0,0,0.3)')
              : (isUser ? 'inset 0 1px 0 rgba(255,255,255,0.22), 0 3px 10px oklch(60% 0.18 270 / 0.25)' : 'inset 0 1px 0 rgba(255,255,255,0.95), 0 1px 4px rgba(40,30,60,0.08)'),
          }}>
            {isUser ? msg.text : <MarkdownText text={msg.text} />}
          </div>
        )}
        {msg.hits && msg.hits.length > 0 && (
          <div style={{ display:'flex', flexDirection:'column', gap:'6px', maxWidth:'96%', width:'100%', animation:'fadeIn 0.3s ease' }}>
            {msg.best_hit
              ? renderBestHit(msg.best_hit, msg.hits, conf, confColor, userQuery)
              : msg.hits.slice(0,5).map((hit, hi) => renderHitCard(hit, hi, msg.hits))}
          </div>
        )}
        {(showBubble || (msg.hits && msg.hits.length > 0)) && (
          <div className="mono" style={{ display:'flex', gap:'10px', alignItems:'center', fontSize:'12px', letterSpacing:'0.12em' }}>
            <span style={{ color:c.textSubtle }}>{msg.time}</span>
            {conf && (
              <span style={{ color: confColor }}>
                ● {lang==='ru'
                  ? `Уверенность: ${conf==='high'?'Высокая':conf==='medium'?'Средняя':'Низкая'}`
                  : `Confidence: ${conf==='high'?'High':conf==='medium'?'Medium':'Low'}`}
              </span>
            )}
          </div>
        )}
      </div>
    );
  };

  const isEmpty = results.length === 0;

  return (
    /* Section root is transparent: the colour field lives in SearchAmbient at
       app-shell level (full-bleed, behind the transparent floating nav), so the
       background flows continuously under the rail — no tone seam at its edge. */
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:'transparent' }}>
      {/* ── Section Header with mode toggle ── */}
      <SectionHeader
        isDark={isDark} lang={lang}
        kicker={lang==='ru'?'РАЗДЕЛ 01 · ПОИСК':'SECTION 01 · SEARCH'}
        title={lang==='ru'?'Поиск':'Search'}
        accent={lang==='ru'?'по смыслу':'by meaning'}
        right={
          <div className="pill-v3 pill-segmented">
            <button
              className={`pill-v3${tab==='search' ? ' pill-v3-active' : ''}`}
              onClick={() => pickTab('search')}
              style={{ fontFamily:"'JetBrains Mono', monospace", fontSize:'11px', letterSpacing:'0.06em', fontWeight:'600' }}>
              {lang==='ru'?'🔍 Поиск':'🔍 Search'}
            </button>
            {aiActive && (
              <button
                className={`pill-v3${tab==='chat' ? ' pill-v3-active' : ''}`}
                onClick={() => pickTab('chat')}
                style={{ fontFamily:"'JetBrains Mono', monospace", fontSize:'11px', letterSpacing:'0.06em', fontWeight:'600' }}>
                {lang==='ru'?'💬 Чат':'💬 Chat'}
              </button>
            )}
          </div>
        }
      />

      {/* ── Tab content ── */}
      <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
        {(tab === 'search' || !aiActive) ? (
          /* ═══════════════════════════════════════════════════════
             SEARCH MODE — Hero bar + grid
             ═══════════════════════════════════════════════════════ */
          <div className="tab-enter" style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
            {/* Hero search bar */}
            <div style={{ padding:'20px clamp(14px,4vw,32px) 16px' }}>
              <div className="hero-bar panel-v3" style={{
                borderRadius:'16px', padding:'6px', display:'flex', gap:'8px', alignItems:'center',
              }}>
                <div style={{ padding:'8px 12px', color:c.textSubtle, display:'flex' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-5.5-5.5"/></svg>
                </div>
                <input
                  value={searchQuery} onChange={e=>setSearchQuery(e.target.value)}
                  onKeyDown={e=>e.key==='Enter'&&handleSearch()}
                  placeholder={lang==='ru'?'Опишите, что ищете — текст, настроение, жанр…':'Describe what you want — lyrics, mood, genre…'}
                  style={{ flex:1, padding:'12px 0', border:'none', background:'transparent',
                    color:c.text, fontSize:'17px', outline:'none', fontFamily:"'Geist', sans-serif" }} />
                <button onClick={handleSearch} disabled={searchLoading||!searchQuery.trim()}
                  className={searchLoading||!searchQuery.trim() ? '' : 'cta-v3'}
                  style={{
                    width:'44px', height:'44px', borderRadius:'12px', flexShrink:0, padding:0,
                    background: searchLoading||!searchQuery.trim() ? (isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.05)') : undefined,
                    color: searchLoading||!searchQuery.trim() ? c.textSubtle : 'white',
                    cursor: searchLoading||!searchQuery.trim()?'not-allowed':'pointer',
                    display:'flex', alignItems:'center', justifyContent:'center',
                  }}>
                  {searchLoading ? <Spinner size={16} color="white" /> : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                  )}
                </button>
              </div>

              {/* Recent searches */}
              <RecentSearchesChips
                items={recent.items}
                onPick={(q) => { setSearchQuery(q); }}
                onRemove={recent.remove}
                onClearAll={recent.clearAll}
                isDark={isDark}
              />

              {/* Sub-row: mode + filters */}
              <div style={{ display:'flex', alignItems:'center', gap:'12px', marginTop:'10px', paddingLeft:'4px', flexWrap:'wrap' }}>
                <Segmented isDark={isDark} value={searchMode} onChange={setSearchMode} size="sm"
                  options={[
                    { value:'text',   label:lang==='ru'?'ТЕКСТ':'TEXT' },
                    { value:'audio',  label:lang==='ru'?'ЗВУК':'AUDIO' },
                    { value:'hybrid', label:lang==='ru'?'ГИБРИД':'HYBRID' },
                  ]} />
                <div style={{ flex:1 }} />
                <button onClick={() => setShowFilters(s=>!s)}
                  className={`pill-v3 mono${showFilters ? ' pill-v3-active' : ''}`}
                  style={{ fontSize:'11px', letterSpacing:'0.15em', fontWeight:'600' }}>
                  {showFilters ? '−' : '+'} {lang==='ru'?'ФИЛЬТРЫ':'FILTERS'}
                </button>
              </div>

              {/* Filters */}
              {showFilters && (
                <div className="filter-autocomplete" style={{ display:'flex', gap:'8px', marginTop:'10px', animation:'fadeIn 0.2s', flexWrap:'wrap' }}>
                  {[{k:'artist', l:lang==='ru'?'Артист':'Artist'},{k:'album', l:lang==='ru'?'Альбом':'Album'},{k:'genre', l:lang==='ru'?'Жанр':'Genre'}].map(f => (
                    <div key={f.k} style={{ position:'relative', width:'180px' }}>
                      <input
                        value={filters[f.k]}
                        onChange={e => handleFilterChange(f.k, e.target.value)}
                        onFocus={() => setActiveFilterField(f.k)}
                        onBlur={() => { setTimeout(() => setActiveFilterField(null), 200); }}
                        placeholder={f.l}
                        className="panel-v3"
                        style={{ width:'100%', padding:'8px 12px', borderRadius:'9px',
                          color:c.text, fontSize:'13px', outline:'none', fontFamily:"'JetBrains Mono', monospace", boxSizing:'border-box',
                          background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.7)',
                          border: `1px solid ${c.border}` }} />
                      {filterSuggestions[f.k]?.length > 0 && activeFilterField === f.k && (
                        <div className={`filter-autocomplete-dropdown ${ske('panel', isDark)}`}
                          style={{ borderRadius:'9px', border:`1px solid ${c.border}` }}>
                          {filterSuggestions[f.k].map((suggestion, si) => (
                            <div key={si} className="filter-autocomplete-item"
                              style={{ color:c.text, background: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)' }}
                              onMouseEnter={e => e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}
                              onMouseLeave={e => e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'}
                              onClick={() => selectFilterSuggestion(f.k, suggestion)}>
                              <span style={{ fontSize:'12px', color:c.textSubtle }}>●</span>
                              {suggestion}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Sonic tag filter chips */}
              <SonicFiltersChips

                selectedTags={filters.sonic_tags}
                onToggleTag={toggleSonicTag}
                isDark={isDark}
              />
              <DecadeFiltersChips

                selectedRanges={filters.year_ranges}
                onToggleRange={toggleYearRange}
                isDark={isDark}
              />
            </div>

            {/* Results area */}
            <div style={{ flex:1, overflowY:'auto', padding:'0 0 24px' }}>
              {searchError && (
                <div style={{ padding:'10px clamp(14px,4vw,32px)', borderRadius:'10px', fontSize:'14px', margin:'0 20px',
                  background: c.redBg, color: c.red, border:`1px solid ${c.red.replace(')',' / 0.3)')}` }}>
                  {searchError}
                </div>
              )}

              {isEmpty ? (
                <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
                  minHeight:'400px', gap:'18px', textAlign:'center', padding:'40px' }}>
                  <div className={ske('display', isDark)} style={{
                    width:'120px', height:'120px', borderRadius:'24px',
                    display:'flex', alignItems:'center', justifyContent:'center',
                    color: c.amber,
                  }}>
                    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="11" cy="11" r="7"/><path d="m21 21-5.5-5.5"/></svg>
                  </div>
                  <div className="serif" style={{ fontSize:'28px', letterSpacing:'-0.01em', color:c.text }}>
                    {searchExecuted
                      ? <i style={{ color: c.textMuted }}>{lang==='ru'?'Ничего не нашлось':'Nothing matched'}</i>
                      : <>{lang==='ru'?'Готов слушать.':'Ready to listen.'}</>}
                  </div>
                  <div style={{ fontSize:'15px', color:c.textSubtle, maxWidth:'380px' }}>
                    {searchExecuted
                      ? (lang==='ru'?'Попробуй другие слова или режим.':'Try different words or another mode.')
                      : (lang==='ru'?'Введите запрос выше и нажмите Enter.':'Type a query above and press Enter.')}
                  </div>
                </div>
              ) : (
                <>
                  <div className="mono" style={{ padding:'8px clamp(14px,4vw,32px) 12px', fontSize:'13px', color:c.textSubtle, letterSpacing:'0.22em' }}>
                    {results.length} {lang==='ru'?'НАЙДЕНО':'FOUND'}
                  </div>
                  {/* Grid of album cards */}
                  <div className="search-grid">
                    {results.map((hit, i) => (
                      <div key={i} className="grid-card panel-v3"
                        onClick={() => onPlayTrack && onPlayTrack(hit, results)}
                        onMouseEnter={() => setHoveredHit(hit.track?.track_id)}
                        onMouseLeave={() => setHoveredHit(null)}
                        style={{
                          borderRadius:'16px', overflow:'hidden', position:'relative',
                        }}>
                        {/* Album cover */}
                        <div style={{
                          width:'100%', aspectRatio:'1', position:'relative',
                        }}>
                          <AlbumCover
                            title={hit.track?.title||''} artist={hit.track?.artist||''}
                            size={200} isDark={isDark} coverPath={hit.track?.cover_art_path} radius={0} fluid
                          />
                          {/* Score badge */}
                          <div style={{
                            position:'absolute', top:'8px', right:'8px',
                            padding:'4px 10px', borderRadius:'12px', fontSize:'12px', fontWeight:'600',
                            fontFamily:"'JetBrains Mono', monospace",
                            background: isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.9)',
                            backdropFilter:'blur(8px)', color: c.accent,
                          }}>
                            {Math.round((hit.score||0)*100)}%
                          </div>
                        </div>
                        {/* Info */}
                        <div style={{ padding:'12px 14px' }}>
                          <div style={{ fontSize:'14px', fontWeight:'600', color:c.text,
                            whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                            letterSpacing:'-0.01em' }}>
                            {hit.track?.title||'—'}
                          </div>
                          <div style={{ fontSize:'13px', color:c.textMuted, marginTop:'3px',
                            whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                            <ArtistCredit track={hit.track} navigateToArtist={navigateToArtist} lang={lang} color={c.textMuted} />{hit.track?.year ? ` · ${hit.track?.year}` : ''}
                          </div>
                          {hit.track?.genre && (
                            <div className="mono" style={{ fontSize:'11px', color:c.textSubtle, marginTop:'6px', letterSpacing:'0.1em' }}>
                              {hit.track.genre.toUpperCase()}
                            </div>
                          )}
                        </div>
                        {hoveredHit === hit.track?.track_id && (
                          <ScoreBreakdownTooltip
                            hit={hit}
                            breakdown={hit.score_breakdown}
                            isDark={isDark}
                            onPlay={(h) => onPlayTrack && onPlayTrack(h, results)}
                            onAddToPlaylist={onAddToPlaylist}
                            lang={lang}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          /* ═══════════════════════════════════════════════════════
             CHAT MODE — GPT-style: hero (empty) / centered column +
             docked composer, persistent history rail on the right
             ═══════════════════════════════════════════════════════ */
          (() => {
            const hasConversation = messages.some(m => m.role === 'user');
            const startNewChat = () => {
              saveSession(messages, sessionIdRef.current);
              sessionIdRef.current = null;
              setMessages(initMsg()); setResults([]);
            };
            const openSession = (s) => {
              sessionIdRef.current = s.id;
              setMessages(s.messages);
              setShowHistory(false);
            };
            const suggestions = lang === 'ru'
              ? ['грустный синти-поп под ночь', 'песня про дорогу домой', 'энергичный рок для тренировки', 'как дождь за окном']
              : ['melancholic synth-pop for late nights', 'a song about the road home', 'high-energy rock for a workout', 'like rain on the window'];
            const renderComposer = (docked) => (
              <div className={`liquid-glass chat-composer${docked ? ' chat-composer-docked' : ''}`} style={{ padding:'8px', borderRadius:'22px' }}>
                <div style={{ display:'flex', gap:'8px', alignItems:'flex-end' }}>
                  <textarea
                    value={input} onChange={e=>setInput(e.target.value)}
                    onKeyDown={e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleChat();} }}
                    placeholder={lang==='ru'?'Опиши музыку…':'Describe the music…'}
                    rows={1} disabled={loading}
                    style={{
                      flex:1, padding:'10px 12px', borderRadius:'10px', border:'none',
                      background:'transparent', color:c.text, resize:'none', fontSize:'16px',
                      lineHeight:'1.5', outline:'none', minHeight:'42px', maxHeight:'120px',
                      fontFamily:"'Geist', sans-serif",
                    }} />
                  <button onClick={handleChat} disabled={loading||!input.trim()}
                    className={loading||!input.trim() ? '' : 'cta-v3'}
                    style={{
                      width:'44px', height:'44px', borderRadius:'13px', flexShrink:0, padding:0,
                      background: loading||!input.trim() ? (isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.05)') : undefined,
                      color: loading||!input.trim() ? c.textSubtle : 'white',
                      cursor: loading||!input.trim()?'not-allowed':'pointer',
                      display:'flex', alignItems:'center', justifyContent:'center',
                    }}>
                    {loading ? <Spinner size={15} color="white" /> : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4z"/></svg>}
                  </button>
                </div>
                {/* Tools row — mode controls live inside the capsule, GPT-style. */}
                <div className="composer-tools">
                  <button
                    onClick={() => { if (autoMode) { setAutoMode(false); setChatMode('hybrid'); } else { setAutoMode(true); } }}
                    className={autoMode ? 'ske-accent' : ske('btn', isDark)}
                    style={{
                      padding:'5px 12px', borderRadius:'8px',
                      fontSize:'11px', fontWeight:'600', fontFamily:"'JetBrains Mono', monospace",
                      letterSpacing:'0.06em', color: autoMode ? 'white' : c.textSubtle,
                      transition:'all 0.18s',
                    }}>
                    ✦ AUTO
                  </button>
                  <Segmented isDark={isDark} value={!autoMode ? chatMode : 'hybrid'}
                    onChange={v => { setAutoMode(false); setChatMode(v); }}
                    size="sm"
                    style={autoMode ? { opacity: 0.35, pointerEvents: 'none' } : {}}
                    options={[
                      { value:'text',   label:lang==='ru'?'ТЕКСТ':'TEXT' },
                      { value:'audio',  label:lang==='ru'?'ЗВУК':'AUDIO' },
                      { value:'hybrid', label:'HYB' },
                    ]} />
                  <div style={{ flex:1 }} />
                  {/* Narrow screens: history slide-over + new chat live here. */}
                  <button onClick={() => { setHistoryClosing(false); setShowHistory(true); }}
                    className={`chat-history-btn ${ske('btn', isDark)}`}
                    style={{ padding:'5px 10px', borderRadius:'8px', fontSize:'11px', fontWeight:'600',
                      fontFamily:"'JetBrains Mono', monospace", letterSpacing:'0.08em', color:c.textMuted,
                      display:'inline-flex', alignItems:'center', gap:'5px' }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
                    {lang==='ru'?'ЧАТЫ':'CHATS'}
                  </button>
                  {hasConversation && (
                    <button onClick={startNewChat}
                      className={`chat-history-btn ${ske('btn', isDark)}`}
                      style={{ padding:'5px 10px', borderRadius:'8px', fontSize:'11px', fontWeight:'600',
                        fontFamily:"'JetBrains Mono', monospace", letterSpacing:'0.08em', color:c.textMuted }}>
                      + {lang==='ru'?'НОВЫЙ':'NEW'}
                    </button>
                  )}
                </div>
              </div>
            );
            return (
              <div className="tab-enter" style={{ flex:1, display:'flex', overflow:'hidden', position:'relative' }}>
                {/* ── Main column ── */}
                <div style={{ flex:1, minWidth:0, display:'flex', flexDirection:'column', overflow:'hidden' }}>
                  {hasConversation ? (
                    <Fragment>
                      <div ref={chatEndRef} style={{ flex:1, overflowY:'auto', padding:'20px clamp(16px,3vw,32px)' }}>
                        <div className="chat-col" style={{ display:'flex', flexDirection:'column', gap:'16px' }}>
                          {messages.map(renderMessage)}
                        </div>
                      </div>
                      <div style={{ padding:'10px clamp(16px,3vw,32px) 20px' }}>
                        <div className="chat-col">{renderComposer(true)}</div>
                      </div>
                    </Fragment>
                  ) : (
                    /* Hero — slogan + centered composer + suggestion chips. */
                    <div className="chat-hero">
                      <div className="serif chat-hero-slogan" style={{ color:c.text }}>
                        {lang==='ru' ? 'Что послушаем?' : 'What shall we listen to?'}
                      </div>
                      <div className="chat-hero-sub" style={{ color:c.textMuted }}>
                        {lang==='ru'
                          ? 'Опиши настроение, текст, звук или жанр — я найду это в твоей библиотеке.'
                          : "Describe a mood, a lyric, a sound, or a genre — I'll find it in your library."}
                      </div>
                      {renderComposer(false)}
                      <div className="chat-suggestions">
                        {suggestions.map((s, i) => (
                          <button key={i} className="chat-suggestion" onClick={() => handleChat(s)}>{s}</button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* ── History rail (≥1100px): chat titles always in view ── */}
                {!railOpen && (
                  <button className={`chat-rail-toggle ${ske('btn', isDark)}`} onClick={toggleRail}
                    title={lang==='ru'?'История чатов':'Chat history'}
                    style={{ position:'absolute', top:'12px', right:'12px', zIndex:5,
                      width:'32px', height:'32px', borderRadius:'9px',
                      alignItems:'center', justifyContent:'center', color:c.textMuted }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
                  </button>
                )}
                <div className="chat-history-rail" data-collapsed={!railOpen}>
                  <div style={{ display:'flex', alignItems:'center', gap:'8px', padding:'13px 14px 9px' }}>
                    <span className="mono" style={{ fontSize:'11px', color:c.textMuted, letterSpacing:'0.18em', fontWeight:'600', flex:1 }}>
                      {lang==='ru'?'ЧАТЫ':'CHATS'}
                    </span>
                    <button onClick={startNewChat} title={lang==='ru'?'Новый чат':'New chat'}
                      className={ske('btn', isDark)}
                      style={{ width:'24px', height:'24px', borderRadius:'7px', display:'flex', alignItems:'center',
                        justifyContent:'center', color:c.textMuted, fontSize:'15px', lineHeight:1 }}>
                      +
                    </button>
                    <button onClick={toggleRail} title={lang==='ru'?'Свернуть':'Collapse'}
                      className={ske('btn', isDark)}
                      style={{ width:'24px', height:'24px', borderRadius:'7px', display:'flex', alignItems:'center',
                        justifyContent:'center', color:c.textMuted }}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="m9 6 6 6-6 6"/></svg>
                    </button>
                  </div>
                  <div className="chat-history-list">
                    {sessions.length === 0 ? (
                      <div style={{ padding:'26px 12px', textAlign:'center', fontSize:'13px', color:c.textSubtle, fontStyle:'italic' }}>
                        {lang==='ru'?'История пуста':'No chats yet'}
                      </div>
                    ) : sessions.map(s => (
                      <div key={s.id} className="chat-history-item" onClick={() => openSession(s)}
                        style={s.id === sessionIdRef.current ? { background: isDark?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.05)' } : undefined}>
                        <div style={{ flex:1, minWidth:0, fontSize:'14px', fontWeight:'500', color:c.text,
                          whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', alignSelf:'center' }}>
                          {s.title}
                        </div>
                        <button className="history-delete-btn"
                          onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                          style={{ width:'22px', height:'22px', borderRadius:'6px', flexShrink:0,
                            display:'flex', alignItems:'center', justifyContent:'center',
                            color:c.textSubtle, background: isDark?'rgba(255,255,255,0.06)':'rgba(0,0,0,0.05)',
                            fontSize:'13px', border:'none', cursor:'pointer' }}>
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()
        )}
      </div>

      {/* ── History Slide-Over Panel ── */}
      {(showHistory || historyClosing) && (
        <div className={`history-slide-overlay ${historyClosing ? 'history-closing' : 'history-open'}`}
          onClick={() => { setHistoryClosing(true); setTimeout(() => setShowHistory(false), 250); }} />
      )}
      {(showHistory || historyClosing) && (
        <div className={`history-slide-panel ${ske('panel', isDark)} ${historyClosing ? 'history-closing' : 'history-open'}`}
          onClick={e => e.stopPropagation()}>
          {/* Header */}
          <div style={{ position:'sticky', top:'0', zIndex:10, display:'flex', justifyContent:'space-between', alignItems:'center',
            padding:'16px 20px', borderBottom:`1px solid ${c.border}`,
            background: isDark ? 'rgba(13,13,16,0.9)' : 'rgba(242,241,246,0.9)',
            backdropFilter:'blur(12px)' }}>
            <span className="mono" style={{ fontSize:'13px', color:c.textMuted, letterSpacing:'0.18em', fontWeight:'600' }}>
              {lang==='ru'?'ИСТОРИЯ':'HISTORY'}
            </span>
            <button onClick={() => { setHistoryClosing(true); setTimeout(() => setShowHistory(false), 250); }}
              className={ske('btn', isDark)}
              style={{ width:'28px', height:'28px', borderRadius:'7px', display:'flex', alignItems:'center', justifyContent:'center',
                fontSize:'16px', color:c.textMuted }}>
              ×
            </button>
          </div>

          {/* Sessions list */}
          {sessions.length === 0 ? (
            <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
              padding:'60px 24px', gap:'14px', textAlign:'center' }}>
              <div className={ske('display', isDark)} style={{
                width:'64px', height:'64px', borderRadius:'16px',
                display:'flex', alignItems:'center', justifyContent:'center',
              }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={c.textSubtle} strokeWidth="1.5">
                  <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>
                </svg>
              </div>
              <div className="serif" style={{ fontSize:'15px', color:c.textSubtle, fontStyle:'italic' }}>
                {lang==='ru'?'История пуста':'No history yet'}
              </div>
            </div>
          ) : (
            sessions.map(s => {
              const firstMsg = s.messages?.[0]?.text || s.messages?.[1]?.text || '';
              const preview = firstMsg.slice(0, 80);
              return (
                <div key={s.id} className="history-slide-item"
                  style={{ borderBottom:`1px solid ${c.border}` }}
                  onMouseEnter={e=>e.currentTarget.style.background=isDark?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.02)'}
                  onMouseLeave={e=>e.currentTarget.style.background='transparent'}
                  onClick={() => { sessionIdRef.current = s.id; setMessages(s.messages); setShowHistory(false); }}>
                  <span style={{ width:'8px', height:'8px', borderRadius:'50%', background:c.accent, flexShrink:0, marginTop:'4px' }} />
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'14px', fontWeight:'600', color:c.text,
                      whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                      {s.title}
                    </div>
                    <div className="mono" style={{ fontSize:'11px', color:c.textSubtle, marginTop:'3px', letterSpacing:'0.1em' }}>
                      {new Date(s.time).toLocaleDateString('ru', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' })}
                    </div>
                    {preview && (
                      <div style={{ fontSize:'13px', color:c.textMuted, marginTop:'6px', lineHeight:'1.4',
                        display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical', overflow:'hidden' }}>
                        {preview}…
                      </div>
                    )}
                  </div>
                  <button className="history-delete-btn"
                    onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                    style={{ width:'24px', height:'24px', borderRadius:'6px', flexShrink:0,
                      display:'flex', alignItems:'center', justifyContent:'center',
                      color:c.textSubtle, background: isDark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)',
                      fontSize:'14px' }}>
                    ×
                  </button>
                </div>
              );
            })
          )}
        </div>
      )}

    </div>
  );
}

// ─── RECOMMEND SECTION ────────────────────────────────────────────────────────
// Rebuilt for Stream RecSys: zone A — one wish → AI-curated playlist (plan →
// execute → select, gated on aiStatus; quick-mix presets when AI is off);
// zone B — taste center: steerable 6-axis radar (knobs → /axis-playlist),
// LLM listener portrait, taste islands → anchor radio; zone C — similar
// tracks via /recommend/similar (CLAP + axes, replaces the old text hack).

const RECSYS_AXES = ['energy', 'vocal_lead', 'spacious', 'experimental', 'brightness', 'acousticness'];
const RECSYS_AXIS_LABELS = {
  energy:       { ru: 'ЭНЕРГИЯ',      en: 'ENERGY' },
  vocal_lead:   { ru: 'ВОКАЛ',        en: 'VOCALS' },
  spacious:     { ru: 'ПРОСТРАНСТВО', en: 'SPACE' },
  experimental: { ru: 'ЭКСПЕРИМЕНТ',  en: 'EXPERIMENTAL' },
  brightness:   { ru: 'ЯРКОСТЬ',      en: 'BRIGHTNESS' },
  acousticness: { ru: 'АКУСТИКА',     en: 'ACOUSTIC' },
};
const RECSYS_TOOL_LABELS = {
  clap_search:    { ru: 'поиск по звучанию', en: 'sound search' },
  library_search: { ru: 'поиск по смыслу',   en: 'meaning search' },
  similar_tracks: { ru: 'похожие на трек',   en: 'similar to a track' },
};

// Hexagonal taste radar: filled polygon = the listener's profile, dashed
// amber polygon = the current knob targets (shown once they diverge).
// z is clamped to ±2σ → radius; the solid middle ring marks z=0.
function AxisRadar({ values, targets, isDark, lang, size }) {
  const s = size || 240;
  const cx = s / 2, cy = s / 2, R = s * 0.34;
  const pt = (i, z) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / RECSYS_AXES.length;
    const r = R * (Math.max(-2, Math.min(2, z || 0)) + 2) / 4;
    return `${cx + r * Math.cos(ang)},${cy + r * Math.sin(ang)}`;
  };
  const ringPts = (z) => RECSYS_AXES.map((_, i) => pt(i, z)).join(' ');
  const polyPts = (vals) => RECSYS_AXES.map((a, i) => pt(i, vals[a] || 0)).join(' ');
  const grid = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.12)';
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} style={{ flex: 'none' }}>
      {[-1, 0, 1, 2].map(z => (
        <polygon key={z} points={ringPts(z)} fill="none" stroke={grid}
                 strokeWidth={z === 0 ? 1.2 : 0.6} strokeDasharray={z === 0 ? undefined : '3 3'} />
      ))}
      {RECSYS_AXES.map((a, i) => {
        const ang = -Math.PI / 2 + (i * 2 * Math.PI) / RECSYS_AXES.length;
        return <line key={a} x1={cx} y1={cy}
                     x2={cx + R * Math.cos(ang)} y2={cy + R * Math.sin(ang)}
                     stroke={grid} strokeWidth="0.6" />;
      })}
      {values && <polygon points={polyPts(values)} fill="oklch(60% 0.18 270 / 0.22)"
                          stroke="oklch(62% 0.2 275)" strokeWidth="1.6" />}
      {targets && <polygon points={polyPts(targets)} fill="none"
                           stroke="oklch(72% 0.13 75)" strokeWidth="1.4" strokeDasharray="5 4" />}
      {RECSYS_AXES.map((a, i) => {
        const ang = -Math.PI / 2 + (i * 2 * Math.PI) / RECSYS_AXES.length;
        const lx = cx + (R + 22) * Math.cos(ang);
        const ly = cy + (R + 16) * Math.sin(ang);
        return (
          <text key={a} x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                style={{ fontSize: 9, letterSpacing: '0.12em', fontFamily: 'inherit',
                         fill: isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)' }}>
            {RECSYS_AXIS_LABELS[a][lang === 'ru' ? 'ru' : 'en']}
          </text>
        );
      })}
    </svg>
  );
}

function RecommendSection({ isDark, lang, onPlayTrack, aiStatus, onStartStream, visible }) {
  const c = useColors(isDark);
  const aiOn = !!(aiStatus && aiStatus.aiActive);
  const llmKw = () => ({
    llm_base_url: localStorage.getItem('llm_base_url') || undefined,
    llm_model: localStorage.getItem('llm_model') || undefined,
  });

  // ── Zone A: one wish → AI playlist ──
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiBusy, setAiBusy] = useState(false);
  const [aiResult, setAiResult] = useState(null);   // {title, steps, tracks}
  const [aiError, setAiError] = useState(false);
  const [aiStepsShown, setAiStepsShown] = useState(0);

  // Derived BEFORE any hook that might depend on it (Babel const→var hoisting).
  const aiHits = (aiResult && aiResult.tracks ? aiResult.tracks : [])
    .map(t => ({ track: t, score: t.score || 0, matched_on: 'ai' }));

  // promptOverride: demo chips run a canned wish without waiting for state.
  const runAiPlaylist = async (promptOverride) => {
    const prompt = (typeof promptOverride === 'string' ? promptOverride : aiPrompt).trim();
    if (!prompt || aiBusy) return;
    setAiBusy(true); setAiResult(null); setAiError(false); setAiStepsShown(0);
    try {
      const res = await apiFetch('/recommend/ai-playlist', {
        method: 'POST',
        body: JSON.stringify({ prompt, lang, limit: 15, ...llmKw() }),
      });
      setAiResult(res);
      (res.steps || []).forEach((_, i) =>
        setTimeout(() => setAiStepsShown(s => Math.max(s, i + 1)), 220 * (i + 1)));
      setTimeout(() => setAiStepsShown(99), 220 * ((res.steps || []).length + 1));
    } catch (e) { setAiError(true); }
    finally { setAiBusy(false); }
  };

  // ── Zone A fallback (AI off): quick-mix presets over plain CLAP search ──
  const [presetLoading, setPresetLoading] = useState(false);
  const [presetResults, setPresetResults] = useState([]);
  const [activePreset, setActivePreset] = useState(null);
  const moodPresets = [
    { id:'late', label: lang==='ru'?'Под поздний вечер':'Late night',
      q:'Song with a smooth late-night jazz with velvet female vocals and soft piano', mode:'audio', hue:265, icon:'🌙' },
    { id:'walk', label: lang==='ru'?'Для прогулки':'For a walk',
      q:'Rhythmic lofi hip-hop beat for midnight walk through city streets', mode:'audio', hue:140, icon:'⊜' },
    { id:'focus', label: lang==='ru'?'Для фокуса':'Focus',
      q:'Atmospheric ambient piano track for focused study and mental clarity', mode:'audio', hue:200, icon:'◇' },
    { id:'rain', label: lang==='ru'?'Дождь и кофе':'Rain & coffee',
      q:'Acoustic ballad with poetic lyrics about rain and autumn longing', mode:'hybrid', hue:220, icon:'☁' },
    { id:'energy', label: lang==='ru'?'Заряд энергии':'Energy boost',
      q:'Powerful rock anthem with empowering lyrics and driving electric guitar', mode:'hybrid', hue:25, icon:'⚡' },
    { id:'love', label: lang==='ru'?'Романтика':'Romance',
      q:'Dreamy synth-pop track with poetic lyrics about falling in love', mode:'hybrid', hue:340, icon:'❤' },
  ];
  const runPreset = async (p) => {
    setActivePreset(p.id); setPresetLoading(true);
    try {
      const res = await apiFetch('/search/', { method:'POST', body: JSON.stringify({
        query: p.q, mode: p.mode, limit: 12 })});
      setPresetResults(res.hits || []);
    } catch (e) { setPresetResults([]); }
    finally { setPresetLoading(false); }
  };

  // ── Zone B: taste center (profile + knobs + portrait + islands) ──
  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [portraitBusy, setPortraitBusy] = useState(false);
  // Auto-fill of missing island names at the user (AI on). autoEnriching drives
  // the per-island shimmer; enrichAttempted (lang::trackIds) guards the effect
  // from re-triggering on its own setProfile.
  const [autoEnriching, setAutoEnriching] = useState(false);
  const enrichAttempted = useRef(null);

  const profileAxisValues = profile && profile.axes
    ? Object.fromEntries(RECSYS_AXES.map(a => [a, (profile.axes[a] && profile.axes[a].z) || 0]))
    : null;
  const islands = (profile && profile.islands) || [];

  // Re-fetch the long-term profile whenever the tab is (re)shown so islands,
  // anchors and axes stay current. Kept-alive sections only re-fetch on a
  // visible→true transition (the section never unmounts within non-home nav) —
  // mirrors LibrarySection. profileLoading is left untouched on re-fetch, so
  // stale data is replaced in place without a skeleton flash.
  useEffect(() => {
    if (visible === false) return;
    let alive = true;
    apiFetch(`/recommend/profile?lang=${encodeURIComponent(lang)}`)
      .then(r => {
        if (!alive || !r) return;
        setProfile(r);
      })
      .catch(() => {})
      .finally(() => { if (alive) setProfileLoading(false); });
    return () => { alive = false; };
  }, [lang, visible]);

  const regenPortrait = async () => {
    if (portraitBusy) return;
    setPortraitBusy(true);
    try {
      const res = await apiFetch('/recommend/profile/ai-enrich', {
        method: 'POST',
        body: JSON.stringify({ lang, ...llmKw() }),
      });
      setProfile(prev => prev ? {
        ...prev,
        portrait: res.portrait || prev.portrait,
        headline: res.headline || prev.headline,
        islands: (prev.islands || []).map(i => ({
          ...i, name: (res.island_names && res.island_names[i.track_id]) || i.name,
        })),
      } : prev);
    } catch (e) { /* LLM down — keep whatever we had */ }
    finally { setPortraitBusy(false); }
  };

  // When AI is on and any island lacks an LLM name, generate names (+ portrait)
  // now via the same endpoint regenPortrait uses, and merge them in. The server
  // returns name=null (never a stale name — its cache is invalidated by the
  // islands hash) when cold or after the taste drifted, so we never show a
  // stale label: either the fresh name or a shimmer. Guarded by a per-set key so
  // the setProfile below can't loop us. Re-runs for a new island set / language.
  useEffect(() => {
    if (!aiOn) return;
    const isls = (profile && profile.islands) || [];
    if (!isls.length) return;
    if (!isls.some(i => !i.name)) return;
    const key = lang + '::' + isls.map(i => i.track_id).join('|');
    if (enrichAttempted.current === key) return;
    enrichAttempted.current = key;
    let alive = true;
    setAutoEnriching(true);
    apiFetch('/recommend/profile/ai-enrich', {
      method: 'POST', body: JSON.stringify({ lang, ...llmKw() }),
    })
      .then(res => {
        if (!alive || !res) return;
        setProfile(prev => prev ? {
          ...prev,
          portrait: res.portrait || prev.portrait,
          headline: res.headline || prev.headline,
          islands: (prev.islands || []).map(i => ({
            ...i, name: (res.island_names && res.island_names[i.track_id]) || i.name,
          })),
        } : prev);
      })
      .catch(() => {})
      .finally(() => { if (alive) setAutoEnriching(false); });
    return () => { alive = false; };
  }, [profile, aiOn, lang]);

  const islandRadio = async (isl) => {
    try {
      const data = await apiFetch(
        `/recommend/autoplay-queue?seed_track_id=${encodeURIComponent(isl.track_id)}&limit=20`);
      const lead = isl.tracks && isl.tracks[0];
      const hits = [
        ...(lead ? [{ track: lead, score: 0, matched_on: 'audio' }] : []),
        ...((data.tracks || []).map(t => ({ track: t, score: 0, matched_on: 'audio' }))),
      ];
      if (hits.length && onPlayTrack) onPlayTrack(hits[0], hits);
    } catch (e) {}
  };

  // Re-present the agent's plan in plain words (never raw tool ids / queries).
  const friendlySteps = (steps) => (steps || [])
    .map(s => (RECSYS_TOOL_LABELS[s.tool] && RECSYS_TOOL_LABELS[s.tool][lang === 'ru' ? 'ru' : 'en']))
    .filter(Boolean);

  const islandName = (isl) =>
    isl.name || (isl.tracks && isl.tracks[0] && isl.tracks[0].artist) ||
    (lang === 'ru' ? 'Остров' : 'Island');

  const heroHeadline = (profile && profile.headline) ||
    (lang === 'ru' ? 'Твой музыкальный портрет' : 'Your music portrait');

  // Demo wishes under the hero field (AI on): show off what the builder can
  // parse — artist-flavoured and niche-genre asks, not generic moods.
  const aiDemoChips = lang === 'ru' ? [
    { icon:'🎸', text:'спокойная музыка как у Dire Straits' },
    { icon:'🌍', text:'восточный хип-хоп' },
    { icon:'🌙', text:'медленное и дымное под поздний вечер' },
    { icon:'⚡', text:'энергичный рок в дорогу' },
  ] : [
    { icon:'🎸', text:'calm music like Dire Straits' },
    { icon:'🌍', text:'oriental hip-hop' },
    { icon:'🌙', text:'slow and smoky for late night' },
    { icon:'⚡', text:'energetic rock for the road' },
  ];

  // Mosaic order: most-populated island first — it renders as the big 2×2 tile.
  const sortedIslands = [...islands].sort(
    (a, b) => ((b.tracks || []).length) - ((a.tracks || []).length));
  // «Вайбики» — the fast mood layer; server sends them strongest-first.
  const vibes = (profile && profile.vibes) || [];
  // One cover per album on the island/vibe walls: several songs off the same
  // album collapse into a single tile (tracks without an album stay separate).
  const uniqueAlbumTracks = (tracks) => {
    const seen = new Set();
    const out = [];
    (tracks || []).forEach(t => {
      const key = t.album
        ? `${(t.artist || '').toLowerCase()}||${t.album.toLowerCase()}`
        : (t.cover_art_path || t.track_id);
      if (!seen.has(key)) { seen.add(key); out.push(t); }
    });
    return out;
  };
  const islandArtists = (isl) => {
    const seen = [];
    (isl.tracks || []).forEach(t => { if (t.artist && !seen.includes(t.artist)) seen.push(t.artist); });
    return seen.slice(0, 3).join(' · ');
  };

  // A built sequence: one «Включить плейлист» on top, then a numbered queue
  // (no per-track play buttons — the whole list is the unit).
  const playlistResult = (title, sub, hits, withReasons) => (
    <div style={{ animation:'fadeIn 0.4s ease' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'14px', margin:'6px 2px 8px' }}>
        <div style={{ flex:1, minWidth:0 }}>
          <div className="serif" style={{ fontSize:'20px', fontWeight:600, color:c.text }}>{title}</div>
          {sub && <div style={{ fontSize:'12.5px', color:c.textSubtle, marginTop:'2px' }}>{sub}</div>}
        </div>
        <button className="rec-playall" onClick={() => hits.length && onPlayTrack && onPlayTrack(hits[0], hits)}>
          ▶ {lang === 'ru' ? 'Включить плейлист' : 'Play playlist'}
        </button>
      </div>
      {hits.map((h, i) => (
        <div key={(h.track && h.track.track_id) || i} className="rec-trk"
             onClick={() => onPlayTrack && onPlayTrack(h, hits)}>
          <div className="rec-trk__ix" style={{ color:c.textSubtle }}>{i + 1}</div>
          <LazyCover className="rec-trk__cov" url={homeCoverUrl(h.track && h.track.cover_art_path)}
                     fallback="linear-gradient(135deg,#7c5cff,#b06bff)" />
          <div style={{ flex:1, minWidth:0 }}>
            <div style={{ fontSize:'15px', fontWeight:700, color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{h.track && h.track.title}</div>
            <div style={{ fontSize:'12.5px', color:c.textSubtle }}>{h.track && h.track.artist}</div>
            {withReasons && h.track && h.track.reason && (
              <div className="serif" style={{ fontSize:'12.5px', fontStyle:'italic', color:c.textMuted, marginTop:'2px' }}>↳ {h.track.reason}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );

  const skeletonRows = (n) => (
    <div style={{ display:'flex', flexDirection:'column', gap:'8px', animation:'fadeIn 0.3s ease' }}>
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} style={{ height:'62px', borderRadius:'12px',
          background: isDark?'rgba(255,255,255,0.03)':'rgba(0,0,0,0.025)',
          boxShadow: isDark?'inset 0 1px 2px rgba(0,0,0,0.4)':'inset 0 1px 2px rgba(40,30,60,0.06)',
          animation:`pulse 1.4s ${i*0.1}s ease-in-out infinite` }} />
      ))}
    </div>
  );
  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:c.bg }}>
      <SectionHeader
        isDark={isDark} lang={lang}
        kicker={lang==='ru'?'РЕКОМЕНДАЦИИ':'RECOMMEND'}
        title={lang==='ru'?'Что включить':"What to play"}
        accent={lang==='ru'?'сегодня':'tonight'}
      />

      <div style={{ flex:1, overflowY:'auto', padding:'18px 28px 60px' }}>
        <div className="rec2-wrap">

          {/* ── HERO: одно желание → плейлист (AI) | быстрые миксы (no AI) ── */}
          {aiOn ? (
            <>
              <div className="rec2-rim">
                <div className="rec2-wish" style={{ color:c.text }}
                     onClick={e => { const inp = e.currentTarget.querySelector('input'); if (inp) inp.focus(); }}>
                  <span className="rec2-spark">✨</span>
                  <input value={aiPrompt} onChange={e => setAiPrompt(e.target.value)}
                         onKeyDown={e => { if (e.key === 'Enter') runAiPlaylist(); }}
                         placeholder={lang==='ru'?'опиши, что хочешь услышать…':'describe what you want to hear…'} />
                  {!aiPrompt && <span className="rec2-caret" />}
                  <button className="rec2-go" onClick={() => runAiPlaylist()} disabled={aiBusy || !aiPrompt.trim()}>
                    {aiBusy ? (lang==='ru'?'СОБИРАЮ…':'BUILDING…') : (lang==='ru'?'СОБРАТЬ ▸':'BUILD ▸')}
                  </button>
                </div>
              </div>
              <div className="rec2-chips">
                {aiDemoChips.map(ch => (
                  <span key={ch.text} className="rec2-chip" style={{ color:c.textMuted }}
                        onClick={() => { setAiPrompt(ch.text); runAiPlaylist(ch.text); }}>
                    <span style={{ fontSize:'14px' }}>{ch.icon}</span>{ch.text}
                  </span>
                ))}
              </div>
              {(aiBusy || aiError || aiResult) && (
                <div className="rec2-results">
                  {aiBusy && !aiResult && <div style={{ fontStyle:'italic', color:c.textSubtle, margin:'4px 2px', animation:'pulse 1.4s ease-in-out infinite' }}>{lang==='ru'?'Подбираю под твоё желание…':'Tuning to your wish…'}</div>}
                  {aiError && !aiBusy && <div style={{ color:c.textSubtle, margin:'4px 2px' }}>{lang==='ru'?'Не получилось собрать — попробуй переформулировать желание.':'Could not build — try rephrasing your wish.'}</div>}
                  {aiResult && !aiBusy && (
                    <>
                      <div style={{ fontSize:'12.5px', fontStyle:'italic', color:'#caa14a', margin:'4px 2px' }}>
                        {friendlySteps(aiResult.steps).slice(0, aiStepsShown).map((s,i) => <span key={i}>✓ {s}{'   ·   '}</span>)}
                        {aiStepsShown >= friendlySteps(aiResult.steps).length && <span>{lang==='ru'?`собрал ${aiHits.length} треков`:`built ${aiHits.length} tracks`}</span>}
                      </div>
                      {aiHits.length ? playlistResult(
                        aiResult.title,
                        `${aiHits.length} ${lang==='ru'?'треков · собрано по твоему запросу':'tracks · built from your wish'}`,
                        aiHits, true) : (
                        <div className="serif" style={{ fontSize:'14px', color:c.textSubtle, fontStyle:'italic', textAlign:'center', padding:'20px' }}>
                          {lang==='ru'?'Ничего не нашлось — попробуй переформулировать':'Nothing found — try rephrasing'}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <div style={{ textAlign:'center', marginTop:'14px' }}>
                <div className="serif" style={{ fontSize:'clamp(20px,4.5vw,26px)', fontWeight:600, color:c.text }}>{lang==='ru'?'Быстрые миксы':'Quick mixes'}</div>
                <div style={{ fontSize:'13.5px', color:c.textSubtle, marginTop:'6px' }}>{lang==='ru'?'Готовые подборки по настроению — подберут похожее по звучанию.':'Ready mixes by mood — they pull what sounds alike.'}</div>
              </div>
              <div className="rec2-chips">
                {moodPresets.map(p => (
                  <span key={p.id} className={`rec2-chip${activePreset===p.id?' rec2-chip--on':''}`}
                        style={{ color:c.textMuted }} onClick={() => runPreset(p)}>
                    <span style={{ fontSize:'14px' }}>{p.icon}</span>{p.label}
                  </span>
                ))}
              </div>
              {(presetLoading || presetResults.length > 0) && (
                <div className="rec2-results">
                  {presetLoading ? skeletonRows(3) : playlistResult(
                    (moodPresets.find(p=>p.id===activePreset)||{}).label || (lang==='ru'?'Микс':'Mix'),
                    `${presetResults.length} ${lang==='ru'?'треков':'tracks'}`,
                    presetResults, false)}
                </div>
              )}
            </>
          )}

          {/* ── КОНСОЛЬ: панель профиля (слева) + мозаика островов (справа) ── */}
          <div className="rec2-grid">
            <div className="rec2-panel" {...spotHandlers(false)}>
              <div className="rec2-screws"><span className="rec2-screw" /><span className="rec2-screw" /></div>
              <div className="rec2-lbl" style={{ color:c.textSubtle }}>{lang==='ru'?'твой звук':'your sound'}</div>
              {profileLoading ? skeletonRows(3) : (
                <div className="rec2-panel-body">
                  <div className="rec2-display">
                    <AxisRadar values={profileAxisValues} isDark={isDark} lang={lang} size={244} />
                  </div>
                  <div className="rec2-panel-info">
                    {aiOn ? (
                      <>
                        <div style={{ display:'flex', alignItems:'baseline', gap:'8px', marginTop:'12px' }}>
                          <div className="serif" style={{ fontSize:'16.5px', fontWeight:700, color:c.text, flex:1, minWidth:0 }}>{heroHeadline}</div>
                          <button onClick={regenPortrait} disabled={portraitBusy} title={lang==='ru'?'Обновить':'Refresh'}
                            style={{ border:'none', background:'transparent', cursor: portraitBusy?'wait':'pointer', color:c.textSubtle, fontSize:'15px', padding:'0 2px', flex:'none' }}>
                            {portraitBusy ? '…' : '↺'}
                          </button>
                        </div>
                        {profile && profile.portrait
                          ? <div className="serif rec2-portrait" style={{ color:c.textMuted, marginTop:'6px' }}>«{profile.portrait}»</div>
                          : <div className="rec2-portrait" style={{ color:c.textSubtle }}>{(portraitBusy || autoEnriching) ? (lang==='ru'?'Собираю твой портрет…':'Writing your portrait…') : ''}</div>}
                      </>
                    ) : (
                      <div className="rec-hint" style={{ color:c.textMuted, marginTop:'12px' }}>✨ {lang==='ru'?'Подключи ИИ, чтобы узнать свой музыкальный вкус словами':'Enable AI to learn your taste in words'}</div>
                    )}
                    {!profileAxisValues && (
                      <div style={{ fontSize:'12.5px', color:c.textSubtle, marginTop:'8px' }}>
                        {lang==='ru'?'Профиль появится после прослушивания':'Your profile appears as you listen'}
                      </div>
                    )}
                    {onStartStream && (
                      <button className="rec2-stream" onClick={onStartStream} {...lqHandlers}>
                        <span className="rec2-lq-pool" aria-hidden="true">
                          <span className="rec2-lq-hue">
                            <span className="rec2-lq-blob rec2-lq-b1" /><span className="rec2-lq-blob rec2-lq-b2" />
                            <span className="rec2-lq-blob rec2-lq-b3" /><span className="rec2-lq-blob rec2-lq-b4" />
                            <span className="rec2-lq-blob rec2-lq-b5" />
                          </span>
                        </span>
                        <span className="rec2-lq-cap" aria-hidden="true" />
                        <span className="rec2-lq-sheen" aria-hidden="true" />
                        <span className="rec2-lq-text">▶ {lang==='ru'?'Запустить поток':'Start stream'}</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div>
              <div className="rec2-isl-head">
                <span className="rec2-isl-title" style={{ color:c.text }}>{lang==='ru'?'Острова вкуса':'Taste islands'}</span>
                {sortedIslands.length > 0 && <span className="rec2-isl-count">{sortedIslands.length}</span>}
              </div>
              <div className="rec2-isl-about" style={{ color:c.textSubtle }}>
                {lang==='ru' ? (
                  <>Выраженные области твоего музыкального вкуса. Они живут вместе с тобой: крепнут от <b style={{ color:c.textMuted }}>дослушанных до конца треков</b> и <b style={{ color:c.textMuted }}>огоньков</b>, а без прослушиваний постепенно <b style={{ color:c.textMuted }}>тают</b>. Нажми на остров — заиграет радио в его духе.</>
                ) : (
                  <>Distinct regions of your music taste. They live with you: they grow from <b style={{ color:c.textMuted }}>tracks played to the end</b> and <b style={{ color:c.textMuted }}>fires</b>, and slowly <b style={{ color:c.textMuted }}>melt away</b> when unplayed. Tap an island to start a radio in its spirit.</>
                )}
              </div>
              {profileLoading ? skeletonRows(2) : sortedIslands.length ? (
                <div className="rec2-mosaic">
                  {sortedIslands.map((isl, ix) => {
                    const big = ix === 0;
                    const nameNode = isl.name ? isl.name
                      : (autoEnriching && aiOn) ? <Skel w={'80%'} h={14} r={6} isDark={isDark} />
                      : islandName(isl);
                    return (
                      <div key={isl.track_id} className={`rec2-isl${big?' rec2-isl--big':''}`}
                           style={{ color:c.text }} onClick={() => islandRadio(isl)} {...spotHandlers(true)}>
                        {big && <span className="rec2-badge">{lang==='ru'?'самый обитаемый':'most lived-in'}</span>}
                        <div className="rec2-covs">
                          {uniqueAlbumTracks(isl.tracks).slice(0, big ? 4 : 2).map((t, j) => (
                            <LazyCover key={t.track_id || j} className="rec2-cov" url={homeCoverUrl(t.cover_art_path)}
                                       fallback="linear-gradient(135deg,#7c5cff,#b06bff)" />
                          ))}
                        </div>
                        <div style={{ minWidth:0 }}>
                          <div className="rec2-isl-name">{nameNode}</div>
                          {big && <div className="serif rec2-isl-desc" style={{ color:c.textSubtle }}>{islandArtists(isl)}</div>}
                          <div className="rec2-isl-sub" style={{ color:c.textSubtle }}>
                            <span>{(isl.tracks||[]).length} {lang==='ru'?'треков':'tracks'}</span>
                            {big && <span className="rec2-isl-play">▶ {lang==='ru'?'радио':'radio'}</span>}
                          </div>
                        </div>
                        {!big && <span className="rec2-isl-play" style={{ position:'absolute', right:'12px', bottom:'12px' }}>▶</span>}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{ fontSize:'13.5px', color:c.textSubtle, marginTop:'4px' }}>{lang==='ru'?'Слушай и отмечай треки — со временем здесь проявятся твои музыкальные острова.':'Listen and react — your islands surface here over time.'}</div>
              )}

              {/* «Вайбики» — fast mood layer under the islands: same visual
                  language, but ephemeral (days, not months). Hidden entirely
                  when none are alive — it's an optional layer, no empty state. */}
              {vibes.length > 0 && (
                <>
                  <div className="rec-div rec-div--taste" style={{ margin:'20px 0 8px' }}>
                    <div className="rec-div__ln" />
                    <div className="rec-div__lbl" style={{ color:c.textSubtle }}>
                      <span className="rec-div__nd" />
                      {lang==='ru' ? 'Вайбики' : 'Vibes'}
                    </div>
                    <div className="rec-div__ln" />
                  </div>
                  <div className="rec2-isl-about" style={{ color:c.textSubtle }}>
                    {lang==='ru' ? (
                      <>То, что ты слушаешь <b style={{ color:c.textMuted }}>прямо сейчас</b>. Вайбики ловят твоё текущее настроение и, в отличие от островов, тают за пару дней. Скипы похожих песен и «остудить» ускоряют их уход.</>
                    ) : (
                      <>What you're listening to <b style={{ color:c.textMuted }}>right now</b>. Vibes catch your current mood and, unlike islands, melt away within days. Skipping similar songs or "cool it" speeds up their fade.</>
                    )}
                  </div>
                  <div className="rec2-mosaic">
                    {vibes.map(v => (
                      <div key={v.track_id} className="rec2-isl" style={{ color:c.text }}
                           onClick={() => islandRadio(v)} {...spotHandlers(true)}>
                        <div className="rec2-covs">
                          {uniqueAlbumTracks(v.tracks).slice(0, 2).map((t, j) => (
                            <LazyCover key={t.track_id || j} className="rec2-cov" url={homeCoverUrl(t.cover_art_path)}
                                       fallback="linear-gradient(135deg,#7c5cff,#b06bff)" />
                          ))}
                        </div>
                        <div style={{ minWidth:0 }}>
                          <div className="rec2-isl-name">{islandName(v)}</div>
                          <div className="rec2-isl-sub" style={{ color:c.textSubtle }}>
                            <span>{(v.tracks||[]).length} {lang==='ru'?'треков':'tracks'}</span>
                          </div>
                        </div>
                        <span className="rec2-isl-play" style={{ position:'absolute', right:'12px', bottom:'12px' }}>▶</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// ─── ALBUM CARD ───────────────────────────────────────────────────────────────
function AlbumCard({ album, isDark, onClick, navigateToArtist, lang, index = 0 }) {
  const c = useColors(isDark);
  const yearStr = album.year_range || (album.year ? String(album.year) : '');
  const topGenre = album.top_genres?.[0];
  const onArtistClick = (slug) => (e) => {
    e.stopPropagation();
    if (slug && navigateToArtist) navigateToArtist(slug);
  };
  return (
    <div
      onClick={onClick}
      className="lib-album-card"
      style={{
        '--lib-i': Math.min(index, 18),
        borderRadius:'14px', overflow:'hidden',
        background:'rgba(255,255,255,.04)',
        border:`1px solid ${c.border}`,
        cursor:'pointer',
      }}
    >
      <div style={{ position:'relative', aspectRatio:'1', overflow:'hidden' }}>
        <div className="lib-album-cover">
          <AlbumCover title={album.album_title} artist={album.primary_artist} size={264} isDark={isDark} coverPath={album.cover_art_path} radius={0} fluid />
        </div>
        <div style={{
          position:'absolute', top:'8px', right:'8px',
          padding:'3px 9px', borderRadius:'12px', fontSize:'12px', fontFamily:"'JetBrains Mono', monospace",
          background:'rgba(0,0,0,.7)', color:'#bba8ff', backdropFilter:'blur(6px)',
        }}>{album.track_count} {lang==='ru'?'тр':'tr'}</div>
      </div>
      <div className="lib-album-body">
        <div className="lib-album-title" style={{ fontWeight:600, color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{album.album_title}</div>
        <div className="lib-album-sub" style={{ marginTop:'3px', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
          <span style={{ color:'#bba8ff', cursor:'pointer' }} onClick={onArtistClick(album.primary_artist_slug)} title={lang==='ru'?'Открыть страницу артиста':'Open artist'}>{album.primary_artist}</span>
          {album.feat_artists?.slice(0, 2).map(f => (
            <React.Fragment key={f.slug}>
              <span style={{ color:c.textSubtle }}> · </span>
              <span style={{ color:'#bba8ff', cursor:'pointer' }} onClick={onArtistClick(f.slug)}>{f.name}</span>
            </React.Fragment>
          ))}
          {(album.feat_artists?.length || 0) > 2 && <span style={{ color:c.textSubtle }}> +{album.feat_artists.length - 2}</span>}
          {yearStr && <span style={{ color:c.textMuted }}> · {yearStr}</span>}
        </div>
        {topGenre && (
          <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, marginTop:'5px', letterSpacing:'0.1em', textTransform:'uppercase' }}>
            {topGenre}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── PLAYLIST CARD ────────────────────────────────────────────────────────────
// Same card anatomy/classes as AlbumCard (.lib-album-card / .lib-album-cover /
// .lib-album-body) so playlists inherit the album grid's sizing — including the
// 150px/132px mobile columns — instead of the old oversized 218px tiles.
function PlaylistCard({ playlist, onOpen, onPlayAll, lang, index = 0 }) {
  return (
    <div
      onClick={() => onOpen(playlist.id)}
      className="lib-album-card"
      style={{
        '--lib-i': Math.min(index, 18),
        borderRadius:'14px', overflow:'hidden',
        background:'rgba(255,255,255,.04)',
        border:'1px solid rgba(255,255,255,.06)',
        cursor:'pointer',
      }}
      onMouseEnter={(e) => {
        const playBtn = e.currentTarget.querySelector('[data-play]');
        if (playBtn) { playBtn.style.opacity = '1'; playBtn.style.transform = 'translateY(0)'; }
      }}
      onMouseLeave={(e) => {
        const playBtn = e.currentTarget.querySelector('[data-play]');
        if (playBtn) { playBtn.style.opacity = '0'; playBtn.style.transform = 'translateY(6px)'; }
      }}
    >
      <div style={{ position:'relative', aspectRatio:'1', overflow:'hidden' }}>
        <div className="lib-album-cover">
          <MosaicCover trackIds={playlist.cover_track_ids || []} coverPaths={playlist.cover_art_paths || []} size={'100%'} radius={0} />
        </div>
        <button
          data-play
          onClick={(e) => { e.stopPropagation(); onPlayAll(playlist.id); }}
          title={lang === 'ru' ? 'Слушать всё' : 'Play all'}
          style={{
            position: 'absolute', right: 10, bottom: 10,
            width: 44, height: 44, borderRadius: '50%',
            display: 'grid', placeItems: 'center',
            background: 'linear-gradient(180deg, oklch(72% 0.2 275) 0%, oklch(52% 0.24 282) 100%)',
            color: '#fff', fontSize: 18, paddingLeft: 3,
            border: 'none', cursor: 'pointer',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,.4), 0 6px 18px rgba(0,0,0,.4), 0 0 0 3px rgba(124,91,255,.2)',
            opacity: 0, transform: 'translateY(6px)',
            transition: 'opacity .22s, transform .22s',
          }}
        >▶</button>
      </div>
      <div className="lib-album-body">
        <div className="lib-album-title" style={{ fontWeight:600, color:'#eeeef3', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
          {playlist.name}
        </div>
        <div className="mono" style={{ fontSize: 10, letterSpacing: '0.15em', color: 'rgba(238,238,243,.45)', marginTop: 4, textTransform: 'uppercase' }}>
          {playlist.track_count} {lang === 'ru' ? (playlist.track_count === 1 ? 'трек' : 'треков') : 'tracks'}
        </div>
      </div>
    </div>
  );
}


function NewPlaylistTile({ onClick, lang }) {
  return (
    <div
      onClick={onClick}
      className="lib-album-card"
      style={{
        borderRadius: 14, overflow: 'hidden',
        border: '1px dashed rgba(124,91,255,.35)',
        background: 'linear-gradient(180deg, rgba(124,91,255,.04) 0%, rgba(124,91,255,.01) 100%)',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'linear-gradient(180deg, rgba(124,91,255,.10) 0%, rgba(124,91,255,.03) 100%)'; e.currentTarget.style.borderColor = 'rgba(124,91,255,.55)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'linear-gradient(180deg, rgba(124,91,255,.04) 0%, rgba(124,91,255,.01) 100%)'; e.currentTarget.style.borderColor = 'rgba(124,91,255,.35)'; }}
    >
      <div style={{ aspectRatio: '1', display: 'grid', placeItems: 'center', background: 'rgba(124,91,255,.06)' }}>
        <span style={{ fontSize: 44, color: 'rgba(124,91,255,.5)', fontWeight: 200 }}>＋</span>
      </div>
      <div className="lib-album-body">
        <div className="lib-album-title" style={{ fontWeight:500, color:'#d8ccff', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
          {lang === 'ru' ? 'Новый плейлист' : 'New playlist'}
        </div>
        <div className="mono" style={{ fontSize: 10, letterSpacing: '0.15em', color: 'rgba(238,238,243,.4)', marginTop: 4, textTransform: 'uppercase' }}>
          {lang === 'ru' ? 'создать с нуля' : 'create from scratch'}
        </div>
      </div>
    </div>
  );
}


function PlaylistsListView({ playlists, onOpen, onCreate, onPlayAll, lang }) {
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 18 }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: '0.18em', color: 'rgba(238,238,243,.5)' }}>
          {playlists.length} {lang === 'ru' ? 'ПЛЕЙЛИСТОВ' : 'PLAYLISTS'}
        </span>
      </div>
      {/* No --lib-grid-min override: inherit the album grid's column sizing
          (216px desktop, 150px/132px mobile via the shared media rules). */}
      <div className="lib-grid">
        <NewPlaylistTile onClick={onCreate} lang={lang} />
        {playlists.map((p, i) => (
          <PlaylistCard key={p.id} playlist={p} onOpen={onOpen} onPlayAll={onPlayAll} lang={lang} index={i + 1} />
        ))}
      </div>
    </>
  );
}

// Deterministic per-day pick: same formula spirit as the backend's
// featured-artist rotation (sha1(date) % N) — a tiny string hash seeded by
// today's date + a per-slot salt, linear-probed to avoid picking the same
// index twice. Everyone sees the same picks on a given day; they roll over
// at local midnight.
function dailyPickIndices(length, count, salt) {
  if (length <= 0) return [];
  const day = new Date().toDateString();
  const seen = new Set();
  const picks = [];
  for (let slot = 0; slot < count && seen.size < length; slot++) {
    const s = `${day}|${salt}|${slot}`;
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    let idx = h % length;
    while (seen.has(idx)) idx = (idx + 1) % length;
    seen.add(idx);
    picks.push(idx);
  }
  return picks;
}

// ─── ALBUMS GRID TAB ──────────────────────────────────────────────────────────
function AlbumsGridTab({ albums, sort, onSortChange, onAlbumOpen, isDark, lang, navigateToArtist }) {
  const c = useColors(isDark);
  const sortOpts = [
    {id:'alphabetical', label: lang==='ru' ? 'А-Я' : 'A-Z'},
    {id:'year_desc',    label: lang==='ru' ? 'год ↓' : 'year ↓'},
    {id:'year_asc',     label: lang==='ru' ? 'год ↑' : 'year ↑'},
    {id:'track_count_desc', label: lang==='ru' ? 'больше треков' : 'most tracks'},
  ];
  const todaysPicks = useMemo(() => (
    (albums && albums.length > 3)
      ? dailyPickIndices(albums.length, 3, 'lib-albums-today').map(i => albums[i])
      : []
  ), [albums]);
  if (!albums || albums.length === 0) {
    return <div style={{ padding:'64px 20px', textAlign:'center', color:c.textSubtle, fontSize:'14px' }}>{lang==='ru' ? 'Нет треков с album-тегом в этой библиотеке' : 'No tracks with album tag in this library'}</div>;
  }
  return (
    <>
      {todaysPicks.length > 0 && (
        <div style={{ marginBottom:24 }}>
          <div className="mono" style={{ fontSize:11, letterSpacing:'.18em', color:c.textSubtle, marginBottom:11 }}>
            {lang==='ru' ? 'СЕГОДНЯ В ПОДБОРКЕ' : "TODAY'S PICKS"}
          </div>
          <div className="lib-grid" style={{ '--lib-grid-min':'176px' }}>
            {todaysPicks.map((a, i) => (
              <AlbumCard
                key={`today-${a.primary_artist_slug}-${a.album_title}`}
                album={a} index={i} isDark={isDark} lang={lang}
                navigateToArtist={navigateToArtist}
                onClick={(e) => {
                  const coverEl = e?.currentTarget?.querySelector('.lib-album-cover');
                  const r = (coverEl || e?.currentTarget)?.getBoundingClientRect?.();
                  onAlbumOpen(a, r ? { top:r.top, left:r.left, width:r.width, height:r.height } : null);
                }}
              />
            ))}
          </div>
        </div>
      )}
      <div className="lib-sortrow" style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'6px 2px 18px', fontSize:'13px' }}>
        <span className="mono" style={{ color:c.textSubtle, letterSpacing:'0.06em' }}>{albums.length} {lang==='ru'?'альбомов':'albums'}</span>
        <div className="lib-sortpills" style={{ display:'flex', gap:'6px' }}>
          {sortOpts.map(o => (
            <button key={o.id} onClick={() => onSortChange(o.id)}
              className="mono lib-sortpill"
              style={{
                padding:'5px 13px', borderRadius:'14px', fontSize:'12px', cursor:'pointer',
                background: sort===o.id ? 'rgba(120,80,200,.18)' : 'rgba(255,255,255,.04)',
                color: sort===o.id ? '#bba8ff' : c.textMuted,
                border:'none',
                transition:'all .2s',
              }}>{o.label}</button>
          ))}
        </div>
      </div>
      <div className="lib-grid">
        {albums.map((a, i) => (
          <AlbumCard
            key={`${a.primary_artist_slug}-${a.album_title}`}
            album={a}
            index={i}
            isDark={isDark} lang={lang}
            navigateToArtist={navigateToArtist}
            onClick={(e) => {
              // Capture the cover square's on-screen rect so the modal can
              // fly out of this exact card (shared-element transition).
              const coverEl = e?.currentTarget?.querySelector('.lib-album-cover');
              const r = (coverEl || e?.currentTarget)?.getBoundingClientRect?.();
              onAlbumOpen(a, r ? { top: r.top, left: r.left, width: r.width, height: r.height } : null);
            }}
          />
        ))}
      </div>
    </>
  );
}

// ─── LIBRARY GLASSY ROW (shared by Liked + Recently) ──────────────────────────
function LibraryGlassyRow({ track, when, playCount, isDark, lang, onClick, navigateToArtist, onAlbumClick, isCurrent, onAddToPlaylist }) {
  const c = useColors(isDark);
  const fmtDur = (s) => {
    if (!s) return '—';
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
  };
  // Mobile: the desktop 5-column grid (spacer + fixed 72px duration) wasted a
  // third of a phone's width and squeezed title/album into ellipsis. Two full-
  // width text lines instead; duration folds into the meta line.
  const isMobile = useIsMobile();
  const coverSize = isMobile ? 46 : 52;
  return (
    <div
      onClick={onClick}
      style={{
        display:'grid',
        gridTemplateColumns: isMobile ? `${coverSize}px minmax(0,1fr) auto` : '52px 1fr auto 72px auto',
        gap: isMobile ? '10px' : '16px', alignItems:'center',
        padding: isMobile ? '8px 8px 8px 8px' : '11px 17px 11px 11px',
        borderRadius:'14px',
        background: isCurrent ? 'rgba(120,80,200,.13)' : 'rgba(255,255,255,.04)',
        // Mobile: a long list of blurred rows is pure GPU burn — the tint alone
        // reads the same on the flat section background.
        backdropFilter: isMobile ? 'none' : 'blur(22px) saturate(1.1)',
        WebkitBackdropFilter: isMobile ? 'none' : 'blur(22px) saturate(1.1)',
        border:`1px solid ${isCurrent ? 'rgba(120,80,200,.32)' : c.border}`,
        boxShadow:'inset 0 1px 0 rgba(255,255,255,.04)',
        cursor:'pointer',
        transition:'all .18s cubic-bezier(.22,.9,.3,1)',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = isCurrent ? 'rgba(120,80,200,.18)' : 'rgba(255,255,255,.07)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
      onMouseLeave={e => { e.currentTarget.style.background = isCurrent ? 'rgba(120,80,200,.13)' : 'rgba(255,255,255,.04)'; e.currentTarget.style.transform = 'translateY(0)'; }}
    >
      <div style={{ position:'relative', width:coverSize, height:coverSize, borderRadius:'10px', overflow:'hidden' }}>
        <AlbumCover title={track.title} artist={track.artist} size={coverSize} isDark={isDark} coverPath={track.cover_art_path} radius={10} fluid />
      </div>
      <div style={{ minWidth:0 }}>
        <div style={{ color:c.text, fontSize: isMobile ? '15px' : '17px', fontWeight:500, letterSpacing:'-0.01em', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{track.title}</div>
        <div style={{ color:c.textMuted, fontSize: isMobile ? '12.5px' : '14px', marginTop:'2px', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
          <ArtistCredit track={track} navigateToArtist={navigateToArtist} lang={lang} />
          {track.album && (<>
            <span style={{ color:c.textSubtle, margin:'0 7px' }}>·</span>
            <span onClick={(e) => { e.stopPropagation(); onAlbumClick && onAlbumClick(track.album); }} style={{ color:'#a8b8c8', cursor:'pointer' }}>{track.album}</span>
          </>)}
          {when && (<>
            <span style={{ color:c.textSubtle, margin:'0 7px' }}>·</span>
            <span className="mono" style={{ color:c.textSubtle, fontSize: isMobile ? '11.5px' : '13px', letterSpacing:'0.06em' }}>{when}</span>
          </>)}
          {playCount != null && playCount > 0 && (<>
            <span style={{ color:c.textSubtle, margin:'0 7px' }}>·</span>
            <span className="mono" style={{ color:c.textSubtle, fontSize: isMobile ? '11.5px' : '13px' }}>{playCount}×</span>
          </>)}
          {isMobile && track.duration ? (<>
            <span style={{ color:c.textSubtle, margin:'0 7px' }}>·</span>
            <span className="mono" style={{ color:c.textSubtle, fontSize:'11.5px' }}>{fmtDur(track.duration)}</span>
          </>) : null}
        </div>
      </div>
      {!isMobile && <div />}
      {!isMobile && <div className="mono" style={{ color:c.textMuted, fontSize:'13px', textAlign:'right' }}>{fmtDur(track.duration)}</div>}
      <div style={{ display:'flex', gap:'5px' }}>
        <button
          className="player-icon-btn"
          title={lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}
          onClick={(e) => { e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(track.track_id, e.currentTarget); }}
          style={{ width: 36, height: 36 }}
        >＋</button>
      </div>
    </div>
  );
}

// ─── formatRelativeTime helper ────────────────────────────────────────────────
function formatRelativeTime(iso, lang) {
  if (!iso) return '';
  const d = new Date(iso);
  const now = new Date();
  const diffMin = Math.floor((now - d) / 60000);
  if (diffMin < 1)  return lang==='ru' ? 'только что' : 'just now';
  if (diffMin < 60) return lang==='ru' ? `${diffMin} мин назад` : `${diffMin}m ago`;
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    return lang==='ru' ? `сегодня ${hh}:${mm}` : `today ${hh}:${mm}`;
  }
  const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return lang==='ru' ? 'вчера' : 'yesterday';
  return d.toLocaleDateString(lang==='ru'?'ru-RU':'en-US', { month:'short', day:'numeric' });
}

// ─── RECENTLY PLAYED TAB ──────────────────────────────────────────────────────
function RecentlyPlayedTab({ tracks, sort, onSortChange, isDark, lang, onPlayTrack, navigateToArtist, onAlbumClick, currentTrackId, onAddToPlaylist }) {
  const c = useColors(isDark);
  if (!tracks || tracks.length === 0) {
    return <div style={{ padding:'40px 20px', textAlign:'center', color:c.textSubtle, fontSize:'13px' }}>{lang==='ru' ? 'История пуста — выбери что-нибудь и сыграй' : 'No playback history yet — pick something and play'}</div>;
  }
  const sortOpts = [
    {id:'last_played', label: lang==='ru' ? 'недавно' : 'recent'},
    {id:'play_count',  label: lang==='ru' ? 'плеи ↓' : 'plays ↓'},
  ];
  const sorted = [...tracks].sort((a, b) => {
    if (sort === 'play_count') return (b.play_count||0) - (a.play_count||0);
    return new Date(b.last_played) - new Date(a.last_played);
  });
  return (
    <>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'4px 0 10px', fontSize:'11px' }}>
        <span className="mono" style={{ color:c.textSubtle, letterSpacing:'0.06em' }}>{tracks.length} {lang==='ru'?'треков · dedup':'tracks · dedup'}</span>
        <div style={{ display:'flex', gap:'5px' }}>
          {sortOpts.map(o => (
            <button key={o.id} onClick={() => onSortChange(o.id)}
              className="mono"
              style={{
                padding:'3px 9px', borderRadius:'12px', fontSize:'10px', cursor:'pointer',
                background: sort===o.id ? 'rgba(120,80,200,.18)' : 'rgba(255,255,255,.04)',
                color: sort===o.id ? '#bba8ff' : c.textMuted,
                border:'none',
              }}>{o.label}</button>
          ))}
        </div>
      </div>
      <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
        {sorted.map(t => (
          <LibraryGlassyRow
            key={t.track_id}
            track={t}
            when={formatRelativeTime(t.last_played, lang)}
            playCount={t.play_count}
            isDark={isDark} lang={lang}
            isCurrent={t.track_id === currentTrackId}
            onClick={() => onPlayTrack && onPlayTrack({ track: t }, tracks.map(tt => ({ track: tt })))}
            navigateToArtist={navigateToArtist}
            onAlbumClick={(name) => onAlbumClick && onAlbumClick(name)}
            onAddToPlaylist={onAddToPlaylist}
          />
        ))}
      </div>
    </>
  );
}

// ─── ADD TO PLAYLIST POPOVER (Task 19) ───────────────────────────────────────
function AddToPlaylistPopover({ trackId, anchor, onClose, listing, lang }) {
  const [data, setData] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [inlineMode, setInlineMode] = React.useState(false);
  const [inlineName, setInlineName] = React.useState('');

  React.useEffect(() => {
    listing.fetchWithMembership(trackId).then(setData);
  }, [trackId, listing]);

  React.useEffect(() => {
    const onEsc = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onEsc);
    return () => document.removeEventListener('keydown', onEsc);
  }, [onClose]);

  // Centered modal (not an anchored popover): a small-screen player toolbar
  // sits near the viewport bottom, so an anchored dropdown overflowed off
  // screen. A dimmed full-screen overlay + scrollable card always fits.
  if (!trackId) return null;

  const onAdd = async (pl) => {
    if (busy || pl.contains_track) return;
    setBusy(true);
    try {
      await listing.addTrack(pl.id, trackId);
      if (typeof showToast === 'function') showToast(lang === 'ru' ? `Добавлено в «${pl.name}»` : `Added to "${pl.name}"`, 'success');
      onClose();
    } catch (e) {
      if (typeof showToast === 'function') showToast(lang === 'ru' ? 'Не удалось добавить' : 'Failed to add', 'error');
    } finally {
      setBusy(false);
    }
  };

  const onCreateInline = async () => {
    const name = inlineName.trim();
    if (!name || busy) return;
    setBusy(true);
    try {
      const created = await listing.createPlaylist(name, null);
      await listing.addTrack(created.id, trackId);
      if (typeof showToast === 'function') showToast(lang === 'ru' ? `Создан «${name}», трек добавлен` : `Created "${name}", track added`, 'success');
      onClose();
    } catch (e) {
      if (typeof showToast === 'function') showToast(lang === 'ru' ? 'Не удалось создать' : 'Failed to create', 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      onMouseDown={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(8,6,14,0.55)',
        backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
        animation: 'fadeIn 0.18s ease',
      }}
    >
      <div
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(360px, calc(100vw - 32px))', maxHeight: '70vh',
          display: 'flex', flexDirection: 'column',
          padding: 10, borderRadius: 16,
          background: 'linear-gradient(180deg, rgba(28,24,40,0.97) 0%, rgba(18,16,28,0.97) 60%)',
          backdropFilter: 'blur(24px) saturate(1.3)', WebkitBackdropFilter: 'blur(24px) saturate(1.3)',
          border: '1px solid rgba(255,255,255,.08)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,.12), 0 24px 60px rgba(0,0,0,.6), 0 0 0 1px rgba(124,91,255,.12)',
          animation: 'toastIn 0.22s cubic-bezier(0.22,0.9,0.3,1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 6px 8px', borderBottom: '1px solid rgba(255,255,255,.05)', marginBottom: 4 }}>
          <span className="mono" style={{ fontSize: 9.5, letterSpacing: '0.24em', color: 'rgba(238,238,243,.45)', textTransform: 'uppercase' }}>
            {lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}
          </span>
          <button
            onClick={onClose}
            aria-label={lang === 'ru' ? 'Закрыть' : 'Close'}
            style={{ background: 'none', border: 'none', color: 'rgba(238,238,243,.5)', cursor: 'pointer', fontSize: 17, lineHeight: 1, padding: '2px 6px' }}
          >✕</button>
        </div>

        <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
          {data === null ? (
            <div style={{ padding: '14px 10px', color: 'rgba(238,238,243,.4)', fontSize: 12 }}>…</div>
          ) : data.playlists.length === 0 ? (
            <div style={{ padding: '14px 10px', color: 'rgba(238,238,243,.4)', fontSize: 12 }}>
              {lang === 'ru' ? 'У вас пока нет плейлистов' : 'No playlists yet'}
            </div>
          ) : (
            data.playlists.map(pl => (
              <div key={pl.id}
                onClick={() => onAdd(pl)}
                style={{
                  display: 'grid', gridTemplateColumns: '24px 1fr auto', gap: 8, alignItems: 'center',
                  padding: '9px 10px', borderRadius: 9,
                  cursor: pl.contains_track ? 'default' : 'pointer',
                  color: pl.contains_track ? 'rgba(238,238,243,.55)' : '#eeeef3',
                  transition: 'background .12s',
                }}
                onMouseEnter={(e) => { if (!pl.contains_track) e.currentTarget.style.background = 'rgba(124,91,255,.12)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
              >
                <span style={{ color: 'oklch(72% 0.13 145)', fontSize: 14 }}>{pl.contains_track ? '✓' : ''}</span>
                <span style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{pl.name}</span>
                <span className="mono" style={{ fontSize: 10, color: 'rgba(238,238,243,.4)', letterSpacing: '0.08em' }}>{pl.track_count}</span>
              </div>
            ))
          )}
        </div>

        <div style={{ height: 1, background: 'rgba(255,255,255,.05)', margin: '4px 0' }} />
        {inlineMode ? (
          <div style={{ display: 'flex', gap: 6, padding: 6 }}>
            <input
              autoFocus
              value={inlineName} onChange={(e) => setInlineName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') onCreateInline(); else if (e.key === 'Escape') setInlineMode(false); }}
              placeholder={lang === 'ru' ? 'Название плейлиста' : 'Playlist name'}
              style={{ flex: 1, background: 'rgba(0,0,0,.3)', border: '1px solid rgba(124,91,255,.35)', borderRadius: 8, padding: '8px 10px', color: '#fff', fontSize: 13, outline: 'none' }}
            />
            <button onClick={onCreateInline}
              style={{ background: 'linear-gradient(180deg, oklch(72% 0.2 275) 0%, oklch(52% 0.24 282) 100%)', color: '#fff', border: 'none', borderRadius: 8, padding: '0 12px', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>
              ▶
            </button>
          </div>
        ) : (
          <div
            onClick={() => setInlineMode(true)}
            style={{ padding: '9px 10px', borderRadius: 9, cursor: 'pointer', color: '#d8ccff', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}
            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(124,91,255,.15)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
            <span style={{ fontSize: 16, lineHeight: 1 }}>＋</span>
            {lang === 'ru' ? 'Новый плейлист…' : 'New playlist…'}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── PLAYLIST TRACK ROW (Task 18) ────────────────────────────────────────────
function PlaylistTrackRow({ track, isDragging, onDragStart, onDragOver, onDragLeave, onDrop, onPlay, navigateToArtist, onRemove, onAddToPlaylist, playlistId, listing, lang }) {
  const fmtDur = (s) => { if (!s) return '—'; const m = Math.floor(s/60), r = Math.floor(s%60); return `${m}:${String(r).padStart(2,'0')}`; };

  const burst = (btn, kind) => {
    const b = document.createElement('span');
    const kindForCss = kind === 'remove' ? 'dislike' : 'like';
    b.className = `player-icon-burst player-icon-burst--${kindForCss}`;
    if (kind === 'add') {
      b.style.background = 'radial-gradient(circle, rgba(124,91,255,0.65) 0%, rgba(124,91,255,0) 70%)';
    }
    btn.appendChild(b);
    setTimeout(() => b.remove(), 640);
  };

  const handleRemove = (e) => {
    burst(e.currentTarget, 'remove');
    onRemove();
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      style={{
        display: 'grid',
        gridTemplateColumns: '20px 44px 1fr auto 78px auto',
        gap: 14, alignItems: 'center',
        padding: '9px 14px 9px 6px', borderRadius: 12,
        background: 'rgba(255,255,255,.025)',
        border: '1px solid rgba(255,255,255,.04)',
        cursor: 'pointer',
        transition: 'all .15s cubic-bezier(.22,.9,.3,1)',
        opacity: isDragging ? 0.35 : 1,
        position: 'relative',
      }}
      onClick={onPlay}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,.06)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,.08)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,.025)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,.04)'; e.currentTarget.style.transform = ''; }}
    >
      <span title={lang === 'ru' ? 'Перетащите' : 'Drag to reorder'} style={{ cursor: 'grab', color: 'rgba(238,238,243,.25)', fontSize: 14, letterSpacing: -2, userSelect: 'none' }}>⋮⋮</span>
      <div style={{ width: 44, height: 44, borderRadius: 8, overflow: 'hidden' }}>
        <AlbumCover title={track.title} artist={track.artist} size={44} coverPath={track.cover_art_path} radius={8} fluid />
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 15, color: '#eeeef3', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', letterSpacing: '-0.005em' }}>{track.title}</div>
        <div style={{ fontSize: 12, color: 'rgba(238,238,243,.55)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          <ArtistCredit track={track} navigateToArtist={navigateToArtist} lang={lang} />
          {track.album && (<>{' · '}{track.album}</>)}
        </div>
      </div>
      <div />
      <div className="mono" style={{ fontSize: 12, color: 'rgba(238,238,243,.5)', textAlign: 'right' }}>{fmtDur(track.duration)}</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="player-icon-btn" style={{ width: 40, height: 40, fontSize: 22 }} onClick={(e) => { e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(track.track_id, e.currentTarget); }} title={lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}>＋</button>
        <button className="player-icon-btn" style={{ width: 40, height: 40, fontSize: 22 }} onClick={(e) => { e.stopPropagation(); handleRemove(e); }} title={lang === 'ru' ? 'Убрать из плейлиста' : 'Remove from playlist'}>⨯</button>
      </div>
    </div>
  );
}

// ─── PLAYLIST DETAIL VIEW (Task 17) ──────────────────────────────────────────
function PlaylistDetailView({ playlistId, lang, isDark, onClose, onPlayTrack, navigateToArtist, listing, onAddToPlaylist }) {
  const [detail, setDetail] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [dragId, setDragId] = React.useState(null);
  const reduced = usePrefersReducedMotion();

  const refetch = React.useCallback(async () => {
    try {
      const r = await apiFetch(`/playlists/${playlistId}`);
      setDetail(r);
    } catch (e) {
      setError(e?.detail || String(e));
    }
  }, [playlistId]);

  React.useEffect(() => { refetch(); }, [refetch]);

  if (error) return <div style={{ padding: 40, color: '#ff7a8a' }}>{error}</div>;
  if (!detail) return <div style={{ padding: 40, color: 'rgba(238,238,243,.5)' }}>…</div>;

  const onRenameBlur = async (e) => {
    const newName = (e.target.textContent || '').trim();
    if (newName && newName !== detail.name) {
      try {
        await listing.renamePlaylist(playlistId, { name: newName });
        await refetch();
      } catch (err) {
        e.target.textContent = detail.name;
        if (typeof showToast === 'function') showToast(lang === 'ru' ? 'Имя уже занято' : 'Name already taken', 'warn');
      }
    }
  };
  const onDescBlur = async (e) => {
    const v = (e.target.textContent || '').trim();
    if (v === (detail.description || '')) return;
    try {
      if (v === '') await listing.renamePlaylist(playlistId, { clear_description: true });
      else await listing.renamePlaylist(playlistId, { description: v });
      await refetch();
    } catch (err) {
      e.target.textContent = detail.description || '';
    }
  };

  const onDelete = async () => {
    if (!confirm(lang === 'ru' ? `Удалить плейлист «${detail.name}»?` : `Delete playlist "${detail.name}"?`)) return;
    await listing.deletePlaylist(playlistId);
    onClose();
  };

  const onRemoveTrack = async (track_id) => {
    setDetail({ ...detail, tracks: detail.tracks.filter(t => t.track_id !== track_id) });
    try { await listing.removeTrack(playlistId, track_id); await refetch(); }
    catch { refetch(); }
  };

  const onDragStart = (id) => (e) => { setDragId(id); e.dataTransfer.effectAllowed = 'move'; };
  const onDragOver  = (id) => (e) => { e.preventDefault(); if (id !== dragId) e.currentTarget.classList.add('drop-above'); };
  const onDragLeave = (e) => { e.currentTarget.classList.remove('drop-above'); };
  const onDrop = (targetId) => async (e) => {
    e.preventDefault();
    document.querySelectorAll('.drop-above').forEach(el => el.classList.remove('drop-above'));
    if (!dragId || dragId === targetId) return;
    const ids = detail.tracks.map(t => t.track_id);
    const from = ids.indexOf(dragId), to = ids.indexOf(targetId);
    if (from < 0 || to < 0) return;
    ids.splice(from, 1); ids.splice(to, 0, dragId);
    const reordered = ids.map(tid => detail.tracks.find(t => t.track_id === tid));
    setDetail({ ...detail, tracks: reordered });
    try { await listing.reorderTracks(playlistId, ids); }
    catch { refetch(); if (typeof showToast === 'function') showToast(lang === 'ru' ? 'Не удалось сохранить порядок' : 'Failed to save order', 'error'); }
  };

  const totalDur = detail.tracks.reduce((acc, t) => acc + (t.duration || 0), 0);
  const fmtTotal = (s) => {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h > 0 ? `${h}:${String(m).padStart(2,'0')}:00` : `${m}:${String(Math.floor(s % 60)).padStart(2,'0')}`;
  };

  return (
    <div>
      <button onClick={onClose} className="mono" style={{ background: 'none', border: 'none', color: 'rgba(238,238,243,.5)', cursor: 'pointer', fontSize: 11, letterSpacing: '0.22em', padding: '0 0 18px' }}>
        ← {lang === 'ru' ? 'К ПЛЕЙЛИСТАМ' : 'BACK TO PLAYLISTS'}
      </button>

      <div style={{
        display: 'grid', gridTemplateColumns: '240px 1fr', gap: 28,
        padding: 20, borderRadius: 18,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.015) 60%)',
        backdropFilter: 'blur(22px) saturate(1.1)', WebkitBackdropFilter: 'blur(22px) saturate(1.1)',
        border: '1px solid rgba(255,255,255,.07)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,.12), 0 6px 24px rgba(0,0,0,.32)',
        marginBottom: 22,
        animation: reduced ? undefined : 'fadeInUp 0.45s cubic-bezier(.22,.9,.3,1) both',
      }}>
        <MosaicCover trackIds={detail.tracks.slice(0,4).map(t => t.track_id)} coverPaths={detail.tracks.slice(0,4).map(t => t.cover_art_path)} size={240} radius={14} />
        <div style={{ display: 'flex', flexDirection: 'column', padding: '4px 0' }}>
          <div className="mono" style={{ fontSize: 10, letterSpacing: '0.22em', color: 'rgba(238,238,243,.5)', marginBottom: 12, textTransform: 'uppercase' }}>
            {lang === 'ru' ? 'ПЛЕЙЛИСТ' : 'PLAYLIST'}
          </div>
          <h1
            contentEditable suppressContentEditableWarning spellCheck={false}
            onBlur={onRenameBlur}
            style={{
              fontFamily: "'Noto Serif Display', Georgia, serif", fontStyle: 'italic', fontWeight: 300,
              fontSize: 48, lineHeight: 1.05, letterSpacing: '-0.015em', margin: '0 0 14px',
              color: isDark ? '#f1eeff' : '#241a3a',
              cursor: 'text', padding: 0, border: '1px solid transparent', borderRadius: 8, outline: 'none',
            }}
          >{detail.name}</h1>
          <div className="mono" style={{ fontSize: 11, letterSpacing: '0.16em', color: 'rgba(238,238,243,.55)', marginBottom: 16, textTransform: 'uppercase' }}>
            {detail.tracks.length} {lang === 'ru' ? 'ТРЕКОВ' : 'TRACKS'} · {fmtTotal(totalDur)}
          </div>
          <p
            contentEditable suppressContentEditableWarning spellCheck={false}
            onBlur={onDescBlur}
            style={{
              fontFamily: "'Noto Serif Display', Georgia, serif", fontStyle: 'italic', fontWeight: 300,
              color: detail.description ? 'rgba(216,204,255,.85)' : 'rgba(238,238,243,.3)',
              fontSize: 17, lineHeight: 1.45, maxWidth: 540, margin: '0 0 22px',
              cursor: 'text', padding: '2px 0', border: '1px solid transparent', borderRadius: 8, outline: 'none',
            }}
          >{detail.description || (lang === 'ru' ? 'добавьте описание…' : 'add a description…')}</p>
          <div style={{ display: 'flex', gap: 10, marginTop: 'auto', alignItems: 'center' }}>
            <button
              className="cta-v3" style={{ padding: '11px 24px' }}
              onClick={() => {
                if (!detail.tracks[0]) return;
                const queue = detail.tracks.map(tr => ({ track: tr }));
                onPlayTrack(queue[0], queue);
              }}>
              ▶ {lang === 'ru' ? 'Слушать всё' : 'Play all'}
            </button>
            <div style={{ flex: 1 }} />
            <button
              onClick={onDelete}
              title={lang === 'ru' ? 'Удалить плейлист' : 'Delete playlist'}
              style={{
                width: 40, height: 40, borderRadius: '50%',
                background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.06)',
                color: 'rgba(238,238,243,.7)', cursor: 'pointer', fontSize: 16,
              }}>🗑</button>
          </div>
        </div>
      </div>

      {detail.missing_track_ids.length > 0 && (
        <div className="mono" style={{
          padding: '10px 14px', borderRadius: 10, fontSize: 12, letterSpacing: '0.08em',
          background: 'rgba(255,180,80,.05)', border: '1px solid rgba(255,180,80,.18)',
          color: 'rgba(255,200,140,.7)', margin: '12px 0',
        }}>
          ⚠ {detail.missing_track_ids.length} {lang === 'ru' ? 'ТРЕК(А) НЕ НАЙДЕНЫ В ТЕКУЩЕЙ БИБЛИОТЕКЕ — СКРЫТЫ, НО НЕ УДАЛЕНЫ. ПОВТОРНОЕ ДОБАВЛЕНИЕ ВЕРНЁТ ИХ' : 'TRACK(S) NOT FOUND IN CURRENT LIBRARY — HIDDEN, NOT REMOVED. RE-ADD MUSIC TO RESTORE'}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6,
        animation: reduced ? undefined : 'fadeInUp 0.5s cubic-bezier(.22,.9,.3,1) 0.09s both' }}>
        {(() => {
          // Build the HIT-shaped queue once per render so every row's onPlay
          // sends the SAME array — App.handlePlayTrack uses it both as the
          // player's queue AND as the id-set for batch metadata enrichment.
          const queue = detail.tracks.map(tr => ({ track: tr }));
          return detail.tracks.map(t => (
            <PlaylistTrackRow
              key={t.track_id}
              track={t}
              isDragging={dragId === t.track_id}
              onDragStart={onDragStart(t.track_id)}
              onDragOver={onDragOver(t.track_id)}
              onDragLeave={onDragLeave}
              onDrop={onDrop(t.track_id)}
              onPlay={() => onPlayTrack({ track: t }, queue)}
              navigateToArtist={navigateToArtist}
              onRemove={() => onRemoveTrack(t.track_id)}
              onAddToPlaylist={onAddToPlaylist}
              playlistId={playlistId}
              listing={listing}
              lang={lang}
            />
          ));
        })()}
      </div>
    </div>
  );
}

// ─── PLAYLISTS TAB ────────────────────────────────────────────────────────────
function PlaylistsTab({ listing, activePlaylistId, onOpenPlaylist, onCloseDetail, onRequestCreate, onPlayTrack, isDark, lang, navigateToArtist, onAddToPlaylist }) {
  if (activePlaylistId == null) {
    return (
      <PlaylistsListView
        playlists={listing.playlists}
        lang={lang}
        onOpen={onOpenPlaylist}
        onCreate={onRequestCreate}
        onPlayAll={(id) => { /* wired in Task 17 via detail view */ }}
      />
    );
  }
  return (
    <PlaylistDetailView
      playlistId={activePlaylistId}
      lang={lang}
      isDark={isDark}
      onClose={onCloseDetail}
      onPlayTrack={onPlayTrack}
      navigateToArtist={navigateToArtist}
      listing={listing}
      onAddToPlaylist={onAddToPlaylist}
    />
  );
}

// ─── LIBRARY SECTION ──────────────────────────────────────────────────────────
// ── Catalog quick-search (non-LLM: title / album / artist in one field) ──────
function CatalogSearchBar({ value, onChange, isDark, lang, loading }) {
  const c = useColors(isDark);
  return (
    <div style={{ position:'relative' }}>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={lang==='ru' ? 'Поиск: песня, альбом или исполнитель…' : 'Search: song, album or artist…'}
        aria-label={lang==='ru' ? 'Поиск по библиотеке' : 'Search the library'}
        style={{
          width:'100%', boxSizing:'border-box',
          background:'rgba(255,255,255,.04)', border:`1px solid ${c.border}`,
          borderRadius:'14px', padding:'14px 44px 14px 16px',
          color:c.text, fontSize:'15px', outline:'none', backdropFilter:'blur(20px)',
        }}
      />
      <span style={{ position:'absolute', right:'16px', top:'50%', transform:'translateY(-50%)', color:c.textDim, fontSize:'14px', pointerEvents:'none' }}>
        {loading ? '⏳' : '🔎'}
      </span>
    </div>
  );
}

function CatalogResults({ hits, loading, onOpen, isDark, lang }) {
  const c = useColors(isDark);
  const badge = (t) => t === 'song' ? '♪' : t === 'album' ? '💿' : '🎤';
  const typeLabel = (t) => lang === 'ru'
    ? (t === 'song' ? 'Песня' : t === 'album' ? 'Альбом' : 'Исполнитель')
    : (t === 'song' ? 'Song' : t === 'album' ? 'Album' : 'Artist');

  // Per-type row tint. Songs stay neutral; albums get a pastel violet wash,
  // artists a pastel teal that complements the violet and keeps text legible.
  // Low-alpha oklch tints read as a soft wash over both light and dark themes,
  // so the foreground stays c.text either way.
  const tint = (t) => {
    if (t === 'album') return {
      bg: 'oklch(70% 0.11 295 / 0.14)', border: 'oklch(72% 0.13 295 / 0.34)',
      glow: 'oklch(62% 0.2 295 / 0.30)', ring: 'oklch(72% 0.13 295 / 0.42)',
    };
    if (t === 'artist') return {
      bg: 'oklch(74% 0.09 200 / 0.15)', border: 'oklch(74% 0.11 200 / 0.36)',
      glow: 'oklch(64% 0.13 200 / 0.30)', ring: 'oklch(74% 0.11 200 / 0.42)',
    };
    return { bg: 'rgba(255,255,255,.04)', border: c.border, glow: null, ring: null };
  };

  if (!hits.length) {
    return (
      <div className="lib-tab-pane" style={{ padding:'28px', textAlign:'center', color:c.textDim }}>
        {loading
          ? (lang === 'ru' ? 'Поиск…' : 'Searching…')
          : (lang === 'ru' ? 'Ничего не найдено' : 'Nothing found')}
      </div>
    );
  }

  return (
    <div className="lib-tab-pane" style={{ display:'flex', flexDirection:'column', gap:'8px' }}>
      {hits.map((h, i) => {
        const cover = h.type === 'artist' ? h.image : h.cover_art_path;
        const coverUrl = homeCoverUrl(cover);
        const primary = h.type === 'song' ? h.title : (h.type === 'album' ? h.album : h.artist);
        const secondary = h.type === 'song'
          ? h.artist
          : (h.type === 'album'
              ? h.artist
              : (h.track_count != null ? `${h.track_count} ${lang === 'ru' ? 'треков' : 'tracks'}` : ''));
        const key = `${h.type}-${h.track_id || h.artist_slug || ''}-${h.album || h.title || ''}-${i}`;
        const tn = tint(h.type);
        return (
          <div
            key={key}
            className="lift-row"
            onClick={() => onOpen(h)}
            style={{
              display:'flex', alignItems:'center', gap:'12px', cursor:'pointer',
              background:tn.bg, border:`1px solid ${tn.border}`,
              borderRadius:'12px', padding:'10px 14px',
              ...(tn.glow ? { '--lift-glow': tn.glow, '--lift-ring': tn.ring } : {}),
            }}
          >
            <div style={{
              width:'44px', height:'44px', flex:'0 0 auto',
              borderRadius: h.type === 'artist' ? '50%' : '8px',
              overflow:'hidden', background:'rgba(255,255,255,.06)',
              display:'flex', alignItems:'center', justifyContent:'center', fontSize:'18px',
            }}>
              {coverUrl
                ? <img src={coverUrl} alt="" style={{ width:'100%', height:'100%', objectFit:'cover' }} />
                : badge(h.type)}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ color:c.text, fontWeight:600, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{primary}</div>
              <div style={{ color:c.textMuted, fontSize:'13px', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{secondary}</div>
            </div>
            <div style={{
              color:c.textMuted, fontSize:'11px', textTransform:'uppercase', letterSpacing:'.5px',
              flex:'0 0 auto', display:'flex', alignItems:'center', gap:'7px', textAlign:'right',
            }}>
              <span>{typeLabel(h.type)}</span>
              <span style={{ fontSize:'14px', width:'1.2em', textAlign:'center', flexShrink:0 }}>{badge(h.type)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function LibrarySection({ isDark, lang, onPlayTrack, navigateToArtist, playerTrack, onAddToPlaylist, playlistsListing, visible }) {
  const c = useColors(isDark);

  // ── Data fetching ─────────────────────────────────────────────────────
  const [stats, setStats] = useState(null);                  // /library/stats
  const [albumsData, setAlbumsData] = useState(null);        // /library/albums
  const [recentData, setRecentData] = useState(null);        // /playback/recent
  const [listenData, setListenData] = useState(null);        // /library/listening-stats
  const [rhythmData, setRhythmData] = useState(null);        // /library/rhythm
  const [engagementData, setEngagementData] = useState(null); // /library/engagement
  const [tasteMap, setTasteMap] = useState(null);            // /library/taste-map (lazy)
  const [tasteMapLoading, setTasteMapLoading] = useState(false);
  const [albumSort, setAlbumSort] = useState('alphabetical');
  const [recentSort, setRecentSort] = useState('last_played');

  // ── UI state ──────────────────────────────────────────────────────────
  // Tab pick: a ?tab= deep link (spec 2026-07-10-spa-routing, phase 3) wins
  // over the localStorage-persisted choice.
  const [activeTab, setActiveTab] = useState(() => {
    const urlTab = new URLSearchParams(window.location.search).get('tab');
    // 'liked' removed 2026-07 (likes replaced by огоньки + add-to-playlist);
    // old deep links / persisted picks fall through to the albums default.
    if (urlTab && ['albums', 'recent', 'playlists', 'stats'].includes(urlTab)) return urlTab;
    const stored = localStorage.getItem('library_active_tab');
    return (stored && stored !== 'liked') ? stored : 'albums';
  });
  const [albumModal, setAlbumModal] = useState(null);  // { album: AlbumSummary, originRect: DOMRect|null }
  // playlistsListing is now provided by App-level usePlaylists (lifted in Plan 19 follow-up)
  const [activePlaylistId, setActivePlaylistId] = React.useState(null);
  const [showNewPlaylistModal, setShowNewPlaylistModal] = React.useState(false);

  // ── Catalog quick-search (replaces tabs while typing) ─────────────────
  const [catalogQuery, setCatalogQuery] = useState('');
  const [catalogHits, setCatalogHits] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const catalogTimer = useRef(null);

  useEffect(() => {
    localStorage.setItem('library_active_tab', activeTab);
    // Reflect the tab in the URL for shareable deep links. replaceState (not
    // push) — tab switches must not become history entries; the pathname
    // guard keeps a freshly-switched section's URL from being rewritten
    // (child effects run before App's URL-sync effect).
    if (window.location.pathname === '/library') {
      const url = new URL(window.location.href);
      if (url.searchParams.get('tab') !== activeTab) {
        url.searchParams.set('tab', activeTab);
        window.history.replaceState(window.history.state, '', url);
      }
    }
  }, [activeTab]);

  // ── Entrance animation epoch ──────────────────────────────────────────
  // The section stays mounted (visibility-toggled at App level), so CSS
  // entrance animations would fire once while hidden and never replay.
  // Bumping this key on every visible→true transition remounts the content
  // subtree, replaying the staggered .lib-rise reveal on each visit.
  const [enterEpoch, setEnterEpoch] = useState(0);
  useEffect(() => { if (visible !== false) setEnterEpoch(e => e + 1); }, [visible]);

  // ── Effect 1: 4 endpoints unrelated to albumSort ─────────────────────
  // Re-fires when the user navigates back to Library (visible→true) so that
  // liked-songs, recent plays, and listening-stats reflect activity from
  // the Player tab. LibrarySection stays mounted (visibility-toggled), so
  // without a `visible` dep this would only run once and the panel would
  // grow stale after each play / like.
  useEffect(() => {
    if (visible === false) return;
    apiFetch(`/library/stats`).then(setStats).catch(() => setStats(null));
    apiFetch(`/playback/recent?limit=50`).then(setRecentData).catch(() => setRecentData({tracks:[]}));
    // getTimezoneOffset() returns minutes where local = UTC - offset; negate it
    // so the backend gets the additive UTC→local shift (UTC+3 → +180) for
    // bucketing the peak hour in the user's own timezone.
    const tzOffset = -new Date().getTimezoneOffset();
    apiFetch(`/library/listening-stats?lang=${lang}&tz_offset_minutes=${tzOffset}`).then(setListenData).catch(() => setListenData(null));
    apiFetch(`/library/rhythm?lang=${lang}&tz_offset_minutes=${tzOffset}`).then(setRhythmData).catch(() => setRhythmData(null));
    apiFetch(`/library/engagement?lang=${lang}`).then(setEngagementData).catch(() => setEngagementData(null));
  }, [lang, visible]);

  // ── Taste map: lazy — only built/fetched the first time the Stats tab opens
  // (PCA + clustering is heavy, so we don't pay for it on every Library visit).
  useEffect(() => {
    if (activeTab !== 'stats' || tasteMap || tasteMapLoading) return;
    setTasteMapLoading(true);
    apiFetch(`/library/taste-map?lang=${lang}`)
      .then(d => setTasteMap(d || { points: [], clusters: [] }))
      .catch(() => setTasteMap({ points: [], clusters: [] }))
      .finally(() => setTasteMapLoading(false));
  }, [activeTab, lang, tasteMap, tasteMapLoading]);

  // ── Effect 2: only albums (refires on sort change) ───────────────────
  useEffect(() => {
    apiFetch(`/library/albums?sort=${albumSort}`)
      .then(setAlbumsData).catch(() => setAlbumsData({albums:[]}));
  }, [albumSort]);

  // ── Effect 3: debounced catalog search ───────────────────────────────
  useEffect(() => {
    const q = catalogQuery.trim();
    if (catalogTimer.current) clearTimeout(catalogTimer.current);
    if (q.length < 2) { setCatalogHits([]); setCatalogLoading(false); return; }
    setCatalogLoading(true);
    catalogTimer.current = setTimeout(() => {
      apiFetch(`/search/catalog?q=${encodeURIComponent(q)}&limit=12`)
        .then(hits => setCatalogHits(Array.isArray(hits) ? hits : []))
        .catch(() => setCatalogHits([]))
        .finally(() => setCatalogLoading(false));
    }, 150);
    return () => { if (catalogTimer.current) clearTimeout(catalogTimer.current); };
  }, [catalogQuery]);

  const openCatalogHit = (h) => {
    if (h.type === 'song') {
      const toHit = (x) => ({
        track: { track_id: x.track_id, title: x.title, artist: x.artist, album: x.album, cover_art_path: x.cover_art_path },
        score: x.score,
      });
      const songHits = catalogHits.filter(x => x.type === 'song');
      onPlayTrack(toHit(h), songHits.map(toHit));
    } else if (h.type === 'album') {
      const albums = albumsData?.albums || [];
      const found = albums.find(a => a.primary_artist_slug === h.artist_slug && a.album_title === h.album)
        || albums.find(a => a.album_title === h.album);
      if (found) setAlbumModal({ album: found, originRect: null });
    } else if (h.type === 'artist') {
      navigateToArtist(h.artist_slug);
    }
  };

  const albumsCount = albumsData?.albums?.length ?? 0;
  const recentCount = recentData?.tracks?.length ?? 0;
  const catalogActive = catalogQuery.trim().length >= 2;

  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:c.bg }}>
      <SectionHeader
        isDark={isDark} lang={lang}
        kicker={lang==='ru' ? 'РАЗДЕЛ 03 · БИБЛИОТЕКА' : 'SECTION 03 · LIBRARY'}
        title={lang==='ru' ? 'Твоя библиотека' : 'Your library'}
        accent={lang==='ru' ? 'в звуках' : 'in sounds'}
      />

      <div style={{ flex:1, overflowY:'auto', padding:'clamp(36px, 7vh, 72px) 32px 100px' }}>
        <div key={`lib-enter-${enterEpoch}`} style={{ maxWidth:'1180px', margin:'0 auto', display:'flex', flexDirection:'column', gap:'20px' }}>

          <div className="lib-rise">
            <LibraryHeroLine stats={stats} albumCount={albumsCount} isDark={isDark} lang={lang} />
          </div>

          <div className="lib-rise" style={{ '--lib-d':'0.16s' }}>
            <CatalogSearchBar
              value={catalogQuery}
              onChange={setCatalogQuery}
              loading={catalogLoading}
              isDark={isDark} lang={lang}
            />
          </div>

          {catalogActive ? (
            <CatalogResults
              hits={catalogHits}
              loading={catalogLoading}
              onOpen={openCatalogHit}
              isDark={isDark} lang={lang}
            />
          ) : (
          <React.Fragment>
          <div className="lib-rise" style={{ '--lib-d':'0.18s' }}>
            <LibraryTabsStrip
              active={activeTab}
              onChange={setActiveTab}
              counts={{ albums: albumsCount, recent: recentCount, playlists: playlistsListing.playlists.length }}
              isDark={isDark}
              lang={lang}
            />
          </div>

          {/* Tab content — keyed by tab so switching replays the pane reveal */}
          <div key={`lib-tab-${activeTab}`} className="lib-tab-pane">
          {activeTab === 'albums' && (
            <AlbumsGridTab
              albums={albumsData?.albums || []}
              sort={albumSort}
              onSortChange={setAlbumSort}
              onAlbumOpen={(a, rect) => setAlbumModal({ album: a, originRect: rect || null })}
              isDark={isDark} lang={lang}
              navigateToArtist={navigateToArtist}
            />
          )}
          {activeTab === 'recent' && (
            <RecentlyPlayedTab
              tracks={recentData?.tracks || []}
              sort={recentSort}
              onSortChange={setRecentSort}
              isDark={isDark} lang={lang}
              onPlayTrack={onPlayTrack}
              navigateToArtist={navigateToArtist}
              onAlbumClick={(albumTitle) => {
                const found = (albumsData?.albums || []).find(a => a.album_title === albumTitle);
                if (found) setAlbumModal({ album: found, originRect: null });
              }}
              currentTrackId={playerTrack?.track_id || null}
              onAddToPlaylist={onAddToPlaylist}
            />
          )}
          {activeTab === 'playlists' && (
            <PlaylistsTab
              listing={playlistsListing}
              activePlaylistId={activePlaylistId}
              onOpenPlaylist={setActivePlaylistId}
              onCloseDetail={() => setActivePlaylistId(null)}
              onRequestCreate={() => setShowNewPlaylistModal(true)}
              onPlayTrack={onPlayTrack}
              isDark={isDark}
              lang={lang}
              navigateToArtist={navigateToArtist}
              onAddToPlaylist={onAddToPlaylist}
            />
          )}
          {activeTab === 'stats' && (
            <StatsTab
              stats={stats}
              listenData={listenData}
              rhythm={rhythmData}
              engagement={engagementData}
              tasteMap={tasteMap}
              tasteMapLoading={tasteMapLoading}
              isDark={isDark} lang={lang}
              onPlayTrack={onPlayTrack}
              navigateToArtist={navigateToArtist}
            />
          )}
          </div>
          </React.Fragment>
          )}

        </div>
      </div>

      {albumModal && (
        <AlbumModal
          album={albumModal.album}
          originRect={albumModal.originRect}
          onClose={() => setAlbumModal(null)}
          onPlayTrack={onPlayTrack}
          navigateToArtist={navigateToArtist}
          isDark={isDark}
          lang={lang}
          onAddToPlaylist={onAddToPlaylist}
        />
      )}
      {showNewPlaylistModal && (
        <NewPlaylistModal
          lang={lang}
          onCancel={() => setShowNewPlaylistModal(false)}
          onSubmit={async (name, desc) => {
            const created = await playlistsListing.createPlaylist(name, desc);
            setShowNewPlaylistModal(false);
            setActivePlaylistId(created.id);
          }}
        />
      )}
      {/* AddToPlaylistPopover is now rendered at App level (Plan 19 follow-up) */}
    </div>
  );
}

function LibraryHeroLine({ stats, albumCount, isDark, lang }) {
  const c = useColors(isDark);
  const total = stats?.total_tracks ?? '—';
  const artists = stats?.unique_artists ?? '—';
  const genres = stats?.unique_genres ?? '—';
  const yr = stats?.year_range;
  const yearText = yr ? `${yr.min}—${yr.max}` : '—';
  const dot = <span style={{ color: 'rgba(255,255,255,.18)', margin:'0 9px' }}>·</span>;
  return (
    <div style={{
      background: 'linear-gradient(90deg, transparent 0%, rgba(120,80,200,.13) 50%, transparent 100%)',
      border: `1px solid ${c.border}`,
      borderRadius: '16px',
      padding: '17px 24px',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
    }}>
      <div className="mono" style={{ fontSize:'clamp(13px, 1.3vw, 15px)', color:c.textMuted, letterSpacing:'0.04em', textAlign:'center' }}>
        <b style={{ color:c.text, fontWeight:600 }}>{total.toLocaleString ? total.toLocaleString() : total}</b> {lang==='ru' ? 'треков' : 'tracks'}
        {dot}<b style={{ color:c.text, fontWeight:600 }}>{albumCount}</b> {lang==='ru' ? 'альбомов' : 'albums'}
        {dot}<b style={{ color:c.text, fontWeight:600 }}>{artists}</b> {lang==='ru' ? 'артистов' : 'artists'}
        {dot}<b style={{ color:c.text, fontWeight:600 }}>{genres}</b> {lang==='ru' ? 'жанров' : 'genres'}
        {dot}<b style={{ color:c.text, fontWeight:600 }}>{yearText}</b>
      </div>
    </div>
  );
}
// ─── Statistics tab ───────────────────────────────────────────────────────
// Empty-state hint shared by every distribution chart.
function Empty({ lang }) {
  return <div style={{ fontSize:11, fontStyle:'italic', opacity:0.5 }}>{lang==='ru'?'Нет данных':'No data'}</div>;
}

// Labelled seam between stats sections (reuses the .rec-div divider language).
function StatsDivider({ label, hue }) {
  const col = `oklch(70% 0.15 ${hue})`;
  const ln = { background:`linear-gradient(90deg,transparent,oklch(70% 0.15 ${hue} / .45),transparent)` };
  return (
    <div className="rec-div" style={{ margin:'6px 4px 0' }}>
      <div className="rec-div__ln" style={ln} />
      <div className="rec-div__lbl" style={{ color: col, fontSize:'clamp(12px, 1.1vw, 14px)' }}>
        <span className="rec-div__nd" style={{ background: col, boxShadow:`0 0 8px ${col}` }} />
        {label}
      </div>
      <div className="rec-div__ln" style={ln} />
    </div>
  );
}

// The whole "◷ Статистика" tab: readout metrics rail → collection map.
// (Sonar / rhythm / engagement sections slot in above the distributions in
// later phases.)
function StatsTab({ stats, listenData, rhythm, engagement, tasteMap, tasteMapLoading, isDark, lang, onPlayTrack, navigateToArtist }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:22 }}>
      <ListeningWidgetsRow data={listenData} rhythm={rhythm} isDark={isDark} lang={lang}
        onPlayTrack={onPlayTrack} navigateToArtist={navigateToArtist} />
      <StatsDivider label={lang==='ru'?'сонар вкуса':'taste sonar'} hue={275} />
      <SonarSection data={tasteMap} loading={tasteMapLoading} isDark={isDark} lang={lang} onPlayTrack={onPlayTrack} />
      <StatsDivider label={lang==='ru'?'таймлайн прослушиваний':'listening timeline'} hue={150} />
      <RhythmSection rhythm={rhythm} isDark={isDark} lang={lang} />
      <StatsDivider label={lang==='ru'?'что ты дослушиваешь':'what you finish'} hue={275} />
      <EngagementSection engagement={engagement} isDark={isDark} lang={lang} onPlayTrack={onPlayTrack} />
      <StatsDivider label={lang==='ru'?'карта коллекции':'collection map'} hue={75} />
      <DistributionsPanel stats={stats} isDark={isDark} lang={lang} navigateToArtist={navigateToArtist} />
    </div>
  );
}

// "By decade" as proportional vertical bars — evenly distributed regardless of
// how many decades there are, with the peak era highlighted and captioned.
// (Replaces a stretched ridgeline that distorted under preserveAspectRatio.)
function EraBars({ decades, isDark, lang, reduced, labelStyle }) {
  const c = useColors(isDark);
  const [drawn, setDrawn] = useState(reduced);
  useEffect(() => {
    if (reduced) { setDrawn(true); return; }
    const id = requestAnimationFrame(() => setDrawn(true));
    return () => cancelAnimationFrame(id);
  }, [reduced]);
  if (!decades.length) return (
    <div><div style={labelStyle}>{lang==='ru'?'ПО ЭПОХАМ':'BY DECADE'}</div><Empty lang={lang} /></div>
  );
  const maxDec = decades.reduce((m,d)=>Math.max(m,d.count||0),0)||1;
  const peak = decades.reduce((p,d)=>(d.count>(p?.count||0)?d:p), null);
  const total = decades.reduce((s,d)=>s+(d.count||0),0)||1;
  const peakPct = peak ? Math.round((peak.count/total)*100) : 0;
  const hue=268, H=140, GAP='clamp(6px, 1.4vw, 16px)';
  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:14, gap:12, flexWrap:'wrap' }}>
        <div style={labelStyle}>{lang==='ru'?'ПО ЭПОХАМ':'BY DECADE'}</div>
        {peak && <div style={{ fontSize:'clamp(13px, 1.2vw, 15px)', color:c.textMuted }}>
          {lang==='ru'?'Ядро коллекции — ':'Core of your library — '}
          <b style={{ color:c.amber, fontWeight:700 }}>{peak.decade}s · {peakPct}%</b>
        </div>}
      </div>
      <div style={{ display:'flex', alignItems:'flex-end', gap:GAP, height:H }}>
        {decades.map((d,i)=>{
          const isPk = peak && d.decade===peak.decade;
          const h = Math.max(4, Math.round((d.count/maxDec)*(H-24)));
          const pc = isPk ? c.amber : `oklch(70% 0.16 ${hue})`;
          return (
            <div key={d.decade} style={{ flex:1, minWidth:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'flex-end', height:'100%' }}>
              <div className="mono" style={{ fontSize:'clamp(10px, 0.95vw, 12.5px)', color:isPk?c.amber:c.textMuted, marginBottom:6, fontWeight:isPk?700:400 }}>{d.count}</div>
              <div style={{ width:'100%', maxWidth:56, height: drawn?h:0, borderRadius:'7px 7px 3px 3px',
                background:`linear-gradient(180deg, ${isPk?'oklch(80% 0.15 80)':`oklch(72% 0.17 ${hue})`}, ${isPk?'oklch(62% 0.16 58)':`oklch(54% 0.17 ${hue})`})`,
                boxShadow:`inset 0 1px 0 rgba(255,255,255,.35), 0 0 14px -4px ${pc}`,
                transition: reduced?'none':`height .7s cubic-bezier(.22,.9,.3,1) ${(i*0.05).toFixed(2)}s` }} />
            </div>
          );
        })}
      </div>
      <div style={{ display:'flex', gap:GAP, marginTop:8 }}>
        {decades.map((d)=>{
          const isPk = peak && d.decade===peak.decade;
          return <div key={d.decade} className="mono" style={{ flex:1, minWidth:0, textAlign:'center',
            fontSize:'clamp(10px, 0.9vw, 12px)', color:isPk?c.amber:c.textSubtle, fontWeight:isPk?700:400 }}>{String(d.decade).slice(2)}s</div>;
        })}
      </div>
    </div>
  );
}

// "By genre" — carved grooves with a glass fill that grows in via meterTick.
// Colour is hashed from the genre name (stable), top 6 + expander.
function GenreBars({ genres, isDark, lang, reduced, labelStyle }) {
  const c = useColors(isDark);
  const [expanded, setExpanded] = useState(false);
  if (!genres.length) return (
    <div><div style={labelStyle}>{lang==='ru'?'ПО ЖАНРАМ':'BY GENRE'}</div><Empty lang={lang} /></div>
  );
  const max = genres.reduce((m,g)=>Math.max(m,g.count||0),0)||1;
  const shown = expanded ? genres : genres.slice(0,6);
  return (
    <div>
      <div style={labelStyle}>{lang==='ru'?'ПО ЖАНРАМ':'BY GENRE'}</div>
      {shown.map((g,i)=>{
        const hue = hueFromString(g.genre), w = Math.round((g.count/max)*100);
        return (
          <div key={g.genre} style={{ marginBottom:13 }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:6, gap:8 }}>
              <span style={{ color:c.text, fontSize:'clamp(13px, 1.2vw, 15px)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:'62%' }}>{g.genre}</span>
              <span className="mono" style={{ color:c.textMuted, fontSize:'clamp(11px, 1vw, 13px)' }}>{g.count} · {g.pct}%</span>
            </div>
            <div className={ske('inset', isDark)} style={{ height:12, borderRadius:7, overflow:'hidden' }}>
              <div style={{ height:'100%', width:`${w}%`, borderRadius:7, transformOrigin:'left',
                background:`linear-gradient(90deg, oklch(58% 0.18 ${hue}), oklch(70% 0.17 ${hue+22}))`,
                boxShadow:`inset 0 1px 1px rgba(255,255,255,.5), inset 0 -1px 1px rgba(0,0,0,.2), 0 0 8px -2px oklch(65% 0.18 ${hue})`,
                animation: reduced?'none':'meterTick 0.55s cubic-bezier(.22,.9,.3,1) both',
                animationDelay: reduced?'0s':`${(i*0.06).toFixed(2)}s` }} />
            </div>
          </div>
        );
      })}
      {genres.length>6 && (
        <button onClick={()=>setExpanded(e=>!e)} className="mono"
          style={{ fontSize:'clamp(11px, 1vw, 12px)', color:c.textMuted, letterSpacing:'0.08em', marginTop:4 }}>
          {expanded ? (lang==='ru'?'СВЕРНУТЬ':'LESS') : (lang==='ru'?`ЕЩЁ ${genres.length-6}`:`+${genres.length-6} MORE`)}
        </button>
      )}
    </div>
  );
}

// "By artist" — tactile monogram chips (a different idiom from the bars so the
// panel doesn't read as three identical lists). Covers/navigation land later.
function ArtistMosaic({ artists, isDark, lang, labelStyle, navigateToArtist }) {
  const c = useColors(isDark);
  if (!artists.length) return (
    <div><div style={labelStyle}>{lang==='ru'?'ПО АРТИСТАМ':'BY ARTIST'}</div><Empty lang={lang} /></div>
  );
  return (
    <div>
      <div style={labelStyle}>{lang==='ru'?'ПО АРТИСТАМ':'BY ARTIST'}</div>
      <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
        {artists.map((a,i)=>{
          const clickable = !!(a.slug && navigateToArtist);
          return (
            <div key={a.artist} onClick={()=>{ if (clickable) navigateToArtist(a.slug); }}
              style={{ display:'flex', alignItems:'center', gap:12, padding:'9px 11px', borderRadius:13, cursor: clickable?'pointer':'default',
              background:'rgba(255,255,255,.04)', border:`1px solid ${c.border}`, transition:'transform .16s ease, background .16s ease' }}
              onMouseEnter={e=>{ e.currentTarget.style.transform='translateY(-2px)'; e.currentTarget.style.background='rgba(255,255,255,.08)'; }}
              onMouseLeave={e=>{ e.currentTarget.style.transform='translateY(0)'; e.currentTarget.style.background='rgba(255,255,255,.04)'; }}>
              <AlbumCover title={a.artist} artist={a.artist} coverPath={a.image} size={42} radius={11} isDark={isDark} />
              <div style={{ minWidth:0, flex:1 }}>
                <div style={{ fontSize:'clamp(14px, 1.3vw, 16px)', color:c.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{a.artist}</div>
                <div className="mono" style={{ fontSize:'clamp(11px, 1vw, 12.5px)', color:c.textSubtle }}>{a.count} {lang==='ru'?'треков':'tracks'}</div>
              </div>
              <div className="mono" style={{ fontSize:'clamp(12px, 1.1vw, 14px)', color:c.textMuted, fontWeight:700 }}>#{i+1}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// NEW fragment — "track length" from the previously-unused duration_buckets.
function DurationBars({ buckets, isDark, lang, reduced, labelStyle }) {
  const c = useColors(isDark);
  if (!buckets.length) return (
    <div><div style={labelStyle}>{lang==='ru'?'ДЛИНА ТРЕКА':'TRACK LENGTH'}</div><Empty lang={lang} /></div>
  );
  const max = buckets.reduce((m,b)=>Math.max(m,b.count||0),0)||1;
  const fmtRange = (r) => {
    const m = String(r).match(/(\d+)\s*-\s*(\d+)/);
    if (m) return `${Math.round(+m[1]/60)}–${Math.round(+m[2]/60)} ${lang==='ru'?'мин':'min'}`;
    return String(r);
  };
  const hue = 150;
  return (
    <div>
      <div style={labelStyle}>{lang==='ru'?'ДЛИНА ТРЕКА':'TRACK LENGTH'}</div>
      {buckets.map((b,i)=>{
        const w = Math.round((b.count/max)*100);
        return (
          <div key={b.range||i} style={{ marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5, gap:8 }}>
              <span style={{ color:c.textMuted, fontSize:'clamp(12px, 1.1vw, 14px)' }}>{fmtRange(b.range)}</span>
              <span className="mono" style={{ color:c.textSubtle, fontSize:'clamp(11px, 1vw, 12px)' }}>{b.count}</span>
            </div>
            <div className={ske('inset', isDark)} style={{ height:11, borderRadius:6, overflow:'hidden' }}>
              <div style={{ height:'100%', width:`${w}%`, borderRadius:6, transformOrigin:'left',
                background:`linear-gradient(90deg, oklch(60% 0.13 ${hue}), oklch(72% 0.14 ${hue+15}))`,
                boxShadow:'inset 0 1px 1px rgba(255,255,255,.4)',
                animation: reduced?'none':'meterTick 0.55s cubic-bezier(.22,.9,.3,1) both',
                animationDelay: reduced?'0s':`${(i*0.06).toFixed(2)}s` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Russian-aware plural picker. ru = [one, few, many]; en = [one, other].
const plural = (n, lang, ru, en) => {
  if (lang !== 'ru') return n === 1 ? en[0] : en[1];
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return ru[0];
  if (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) return ru[1];
  return ru[2];
};

// 14-day trailing sparkline for the "∑ listened" readout.
function Sparkline({ days, width=92, height=24, hue=145 }) {
  const pts = useMemo(() => {
    const map = new Map((days||[]).map(d => [d.date, d.count]));
    const today = new Date(); today.setHours(0,0,0,0);
    const vals = [];
    for (let i=13;i>=0;i--){
      const dt = new Date(today); dt.setDate(today.getDate()-i);
      const iso = `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`;
      vals.push(map.get(iso) || 0);
    }
    return vals;
  }, [days]);
  const max = Math.max(1, ...pts);
  const stepX = width/(pts.length-1);
  const line = pts.map((v,i)=>`${i?'L':'M'}${(i*stepX).toFixed(1)},${(height-(v/max)*(height-3)-1).toFixed(1)}`).join(' ');
  const area = `${line} L${width},${height} L0,${height} Z`;
  const gid = `spk${hue}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display:'block', flex:'none' }}>
      <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={`oklch(72% 0.16 ${hue})`} stopOpacity="0.32" />
        <stop offset="100%" stopColor={`oklch(72% 0.16 ${hue})`} stopOpacity="0" />
      </linearGradient></defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={line} fill="none" stroke={`oklch(74% 0.16 ${hue})`} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Glass readout chip used in the Rhythm section header (streak / busiest day).
// Optional `foot` adds a clarifying third line.
function RhythmReadout({ icon, value, label, foot, isDark, hue, grow }) {
  const c = useColors(isDark);
  const glass = isDark ? {
    background:'linear-gradient(165deg, rgba(255,255,255,0.13), rgba(255,255,255,0.03)), rgba(24,24,32,0.36)',
    border:'1px solid rgba(255,255,255,0.13)',
    boxShadow:'inset 0 1px 0 rgba(255,255,255,0.22), 0 10px 26px rgba(0,0,0,0.32)',
  } : {
    background:'linear-gradient(165deg, rgba(255,255,255,0.92), rgba(255,255,255,0.5)), rgba(244,243,250,0.4)',
    border:'1px solid rgba(255,255,255,0.8)',
    boxShadow:'inset 0 1px 0 rgba(255,255,255,0.95), 0 10px 26px rgba(46,36,86,0.12)',
  };
  return (
    <div style={{ ...glass, flex: grow?'1 1 280px':'0 1 auto', display:'flex', alignItems:'center', gap:14,
      padding:'14px 18px', borderRadius:15, minWidth:0, backdropFilter:'blur(14px) saturate(1.5)', WebkitBackdropFilter:'blur(14px) saturate(1.5)' }}>
      <span style={{ fontSize:'clamp(22px, 2.2vw, 28px)', flex:'none', lineHeight:1 }}>{icon}</span>
      <div style={{ minWidth:0 }}>
        <div style={{ fontSize:'clamp(18px, 1.9vw, 23px)', fontWeight:700, color:c.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{value}</div>
        <div className="mono" style={{ fontSize:'clamp(10px, 1vw, 12px)', letterSpacing:'0.1em', textTransform:'uppercase', color:`oklch(70% 0.13 ${hue})`, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{label}</div>
        {foot && <div style={{ fontSize:'clamp(12px, 1.05vw, 13.5px)', color:c.textMuted, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', marginTop:3 }}>{foot}</div>}
      </div>
    </div>
  );
}

// 24h polar histogram — a tactile clock-dial of when you listen. The disc is
// split into a cool night half (top) and a warm day half (bottom) so it's
// obvious at a glance whether the peak falls in the day or the night.
function RhythmDial({ byHour, isDark, lang, reduced }) {
  const c = useColors(isDark);
  const arr = (byHour && byHour.length===24) ? byHour : new Array(24).fill(0);
  const max = Math.max(1, ...arr);
  const anyData = arr.some(v=>v>0);
  const peak = anyData ? arr.indexOf(Math.max(...arr)) : 0;
  const S=176, cx=S/2, cy=S/2, r0=42, r1=80;
  const dayHue=95, nightHue=265;
  const isDayHour = (h) => h>=4 && h<22;     // night = 22:00–04:00
  const R = r1+8;
  // Day/night as true sectors anchored to the clock (midnight at top), so the
  // shaded night arc spans exactly 22:00→04:00 rather than a fixed half.
  const ang = (h) => (h/24)*2*Math.PI - Math.PI/2;
  const pt = (h, rr=R) => `${(cx+rr*Math.cos(ang(h))).toFixed(2)} ${(cy+rr*Math.sin(ang(h))).toFixed(2)}`;
  const wedge = (h0, h1) => {
    const large = ((h1 - h0 + 24) % 24) > 12 ? 1 : 0;
    return `M ${cx} ${cy} L ${pt(h0)} A ${R} ${R} 0 ${large} 1 ${pt(h1)} Z`;
  };
  return (
    <div style={{ display:'inline-flex', flexDirection:'column', alignItems:'center', gap:8 }}>
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} style={{ overflow:'visible' }}>
        <path d={wedge(22, 4)} fill={`oklch(60% 0.12 ${nightHue})`} opacity={isDark?0.16:0.11} />
        <path d={wedge(4, 22)} fill={`oklch(72% 0.13 ${dayHue})`}  opacity={isDark?0.12:0.09} />
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={c.border} strokeWidth="1" />
        <circle cx={cx} cy={cy} r={r0-4} fill="none" stroke={c.border} strokeWidth="1" />
        {arr.map((v,i)=>{
          const a=(i/24)*2*Math.PI - Math.PI/2;
          const len = r0 + (v/max)*(r1-r0);
          const isPk = anyData && i===peak;
          const bh = isDayHour(i)?dayHue:nightHue;
          return <line key={i} x1={cx+r0*Math.cos(a)} y1={cy+r0*Math.sin(a)} x2={cx+len*Math.cos(a)} y2={cy+len*Math.sin(a)}
            stroke={isPk?'oklch(82% 0.15 80)':`oklch(${56+(v/max)*20}% ${(0.08+(v/max)*0.08).toFixed(3)} ${bh})`}
            strokeWidth={5} strokeLinecap="round"
            style={{ filter:isPk?'drop-shadow(0 0 6px oklch(82% 0.15 80))':'none',
                     opacity: reduced?1:0, animation: reduced?'none':'fadeIn 0.4s ease both', animationDelay: reduced?'0s':`${(i*0.015).toFixed(3)}s` }} />;
        })}
        {[0,6,12,18].map(h=>{ const a=(h/24)*2*Math.PI-Math.PI/2, rr=R+10;
          return <text key={h} x={cx+rr*Math.cos(a)} y={cy+rr*Math.sin(a)+3.5} textAnchor="middle" className="mono" style={{ fontSize:10, fill:c.textSubtle }}>{h}</text>; })}
        <text x={cx} y={cy-2} textAnchor="middle" style={{ fontSize:22, fontWeight:800, fill:c.text }}>{anyData?`${String(peak).padStart(2,'0')}:00`:'—'}</text>
        <text x={cx} y={cy+15} textAnchor="middle" className="mono" style={{ fontSize:9, letterSpacing:'0.12em', fill:c.textSubtle }}>{lang==='ru'?'ПИК':'PEAK'}</text>
      </svg>
      <div style={{ display:'flex', gap:16, marginTop:14, fontSize:'clamp(11px, 1vw, 12.5px)', color:c.textMuted }}>
        <span style={{ display:'inline-flex', alignItems:'center', gap:6 }}><span style={{width:11,height:11,borderRadius:3,background:`oklch(72% 0.13 ${dayHue})`}} />{lang==='ru'?'☀ день':'☀ day'}</span>
        <span style={{ display:'inline-flex', alignItems:'center', gap:6 }}><span style={{width:11,height:11,borderRadius:3,background:`oklch(60% 0.12 ${nightHue})`}} />{lang==='ru'?'☾ ночь':'☾ night'}</span>
      </div>
    </div>
  );
}

// GitHub-style calendar heatmap of plays per day over the last ~year, with
// month labels along the top. The tooltip is positioned relative to the
// component (not the viewport) so a transformed ancestor can't fling it across
// the screen.
function CalendarHeatmap({ days, isDark, lang, reduced }) {
  const c = useColors(isDark);
  const [hov, setHov] = useState(null);
  const wrapRef = useRef(null);
  const GAP=3, WEEKS=53;
  const { cells, months } = useMemo(() => {
    const map = new Map((days||[]).map(d => [d.date, d.count]));
    const max = (days||[]).reduce((m,d)=>Math.max(m,d.count||0),0) || 1;
    const today=new Date(); today.setHours(0,0,0,0);
    const end=new Date(today); end.setDate(end.getDate()+(6-today.getDay())); // align to Saturday
    const total=WEEKS*7, out=[];
    for (let i=total-1;i>=0;i--){
      const dt=new Date(end); dt.setDate(end.getDate()-i);
      const iso=`${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,'0')}-${String(dt.getDate()).padStart(2,'0')}`;
      const cnt=map.get(iso)||0, future=dt>today;
      const level=cnt===0?0:Math.min(4, 1+Math.floor((cnt/max)*3.999));
      out.push({iso,cnt,level,future,mon:dt.getMonth()});
    }
    const mons=[]; let prev=-1;
    for (let col=0; col<WEEKS; col++){
      const top=out[col*7];
      if (top && top.mon!==prev){
        mons.push({ col, label:new Date(top.iso+'T00:00:00').toLocaleDateString(lang==='ru'?'ru-RU':'en-US',{month:'short'}) });
        prev=top.mon;
      }
    }
    return { cells: out, months: mons };
  }, [days, lang]);
  const color=(lvl)=> lvl===0 ? (isDark?'rgba(255,255,255,.06)':'rgba(0,0,0,.06)') : `oklch(${50+lvl*6}% ${(0.05+lvl*0.045).toFixed(3)} 150)`;
  const fmtDate=(iso)=>{ try{ return new Date(iso+'T00:00:00').toLocaleDateString(lang==='ru'?'ru-RU':'en-US',{day:'numeric',month:'short'}); }catch{ return iso; } };
  const onCell=(cell,e)=>{
    if (cell.future) return;
    const r = wrapRef.current?.getBoundingClientRect();
    setHov({ iso:cell.iso, cnt:cell.cnt, x: r? e.clientX-r.left : 0, y: r? e.clientY-r.top : 0 });
  };
  return (
    <div ref={wrapRef} style={{ position:'relative' }}>
      <div className={ske('inset', isDark)} style={{ borderRadius:12, padding:'10px 12px' }}>
        <div style={{ width:'100%' }}>
          <div style={{ position:'relative', height:16, marginBottom:4, width:'100%' }}>
            {months.map(m=>(
              <span key={m.col} className="mono" style={{ position:'absolute', left:`${(m.col/WEEKS)*100}%`, top:0,
                fontSize:'clamp(9px, 0.85vw, 11px)', color:c.textSubtle, whiteSpace:'nowrap' }}>{m.label}</span>
            ))}
          </div>
          <div style={{ display:'grid', gridAutoFlow:'column', gridTemplateRows:'repeat(7, 1fr)',
            gridTemplateColumns:`repeat(${WEEKS}, minmax(0, 1fr))`, gap:GAP, width:'100%', aspectRatio:`${WEEKS} / 7` }}>
            {cells.map((cell,i)=>(
              <div key={cell.iso}
                onMouseEnter={e=>onCell(cell,e)}
                onMouseLeave={()=>setHov(null)}
                style={{ borderRadius:3,
                  background: cell.future?'transparent':color(cell.level),
                  boxShadow: cell.level>0?`inset 0 0 0 1px oklch(${50+cell.level*6}% ${(0.05+cell.level*0.045).toFixed(3)} 150 / .5)`:'none',
                  opacity: reduced?1:0,
                  animation: (reduced||cell.future)?'none':'scaleIn 0.3s ease both',
                  animationDelay: reduced?'0s':`${(Math.floor(i/7)*0.012).toFixed(3)}s`,
                  cursor: cell.future?'default':'pointer' }} />
            ))}
          </div>
        </div>
      </div>
      <div style={{ display:'flex', alignItems:'center', gap:6, justifyContent:'flex-end', marginTop:8 }}>
        <span className="mono" style={{ fontSize:'clamp(10px, 0.9vw, 12px)', color:c.textSubtle }}>{lang==='ru'?'меньше':'less'}</span>
        {[0,1,2,3,4].map(l=> <span key={l} style={{ width:12, height:12, borderRadius:3, background:color(l) }} />)}
        <span className="mono" style={{ fontSize:'clamp(10px, 0.9vw, 12px)', color:c.textSubtle }}>{lang==='ru'?'больше':'more'}</span>
      </div>
      {hov && (
        <div className="mono" style={{ position:'absolute', zIndex:50, pointerEvents:'none',
          left:Math.max(0, Math.min(hov.x+12, (wrapRef.current?.clientWidth||320)-160)), top:Math.max(0, hov.y-38),
          padding:'7px 10px', borderRadius:8, fontSize:'clamp(11px, 1vw, 12.5px)', whiteSpace:'nowrap',
          background:isDark?'rgba(20,18,30,0.97)':'rgba(255,255,255,0.98)', color:c.text,
          border:`1px solid ${c.border}`, boxShadow:'0 8px 22px rgba(0,0,0,.35)' }}>
          {fmtDate(hov.iso)} · {hov.cnt} {plural(hov.cnt, lang, ['плей','плея','плеев'], ['play','plays'])}
        </div>
      )}
    </div>
  );
}

// "Ритм" section: streak/busiest readout + year-pulse heatmap + 24h dial.
function RhythmSection({ rhythm, isDark, lang }) {
  const c = useColors(isDark);
  const reduced = usePrefersReducedMotion();
  const days = rhythm?.days || [];
  const lbl = { fontFamily:"'JetBrains Mono', monospace", fontSize:'clamp(10px, 1vw, 12px)', color:c.textSubtle, letterSpacing:'0.2em', textTransform:'uppercase', marginBottom:12 };
  if (!days.length) {
    return (
      <div className={brushed(isDark)} style={{ borderRadius:18, padding:'28px 24px', textAlign:'center' }}>
        <div style={{ fontSize:14, color:c.textMuted, fontStyle:'italic' }}>
          {lang==='ru' ? 'Заглядывай сюда — твой ритм проявится со временем' : 'Come back soon — your rhythm will appear over time'}
        </div>
      </div>
    );
  }
  const dayWord = ['день','дня','дней'], dayWordEn = ['day','days'];
  const busiest = rhythm?.busiest_day;
  const fmtLong=(iso)=>{ try{ return new Date(iso+'T00:00:00').toLocaleDateString(lang==='ru'?'ru-RU':'en-US',{day:'numeric',month:'long'}); }catch{ return iso; } };
  return (
    <div className={brushed(isDark)} style={{ borderRadius:18, padding:'22px 24px', display:'flex', flexDirection:'column', gap:20,
      animation: reduced?'none':'fadeIn 0.4s cubic-bezier(.22,.9,.3,1)' }}>
      <div style={{ display:'flex', flexWrap:'wrap', gap:12 }}>
        <RhythmReadout icon="🔥" isDark={isDark} hue={30}
          value={`${rhythm.streak_current} ${plural(rhythm.streak_current, lang, dayWord, dayWordEn)}`}
          label={lang==='ru'?'подряд сейчас':'current streak'} />
        <RhythmReadout icon="🏆" isDark={isDark} hue={75}
          value={`${rhythm.streak_best} ${plural(rhythm.streak_best, lang, dayWord, dayWordEn)}`}
          label={lang==='ru'?'лучшая серия':'best streak'} />
        {busiest && (
          <RhythmReadout icon="⚡" isDark={isDark} hue={275} grow
            value={fmtLong(busiest.date)}
            label={lang==='ru'?'самый активный день':'most active day'}
            foot={busiest.top_track
              ? `${busiest.count} ${plural(busiest.count, lang, ['прослушивание','прослушивания','прослушиваний'], ['play','plays'])} · ${lang==='ru'?'хит дня':'top'}: ${busiest.top_track.title}`
              : `${busiest.count} ${plural(busiest.count, lang, ['прослушивание','прослушивания','прослушиваний'], ['play','plays'])}`} />
        )}
      </div>
      <div style={{ display:'flex', gap:24, alignItems:'flex-start', flexWrap:'wrap' }}>
        <div style={{ flex:'1 1 420px', minWidth:0 }}>
          <div style={lbl}>{lang==='ru'?'ПУЛЬС ГОДА':'YEAR PULSE'}</div>
          <CalendarHeatmap days={days} isDark={isDark} lang={lang} reduced={reduced} />
        </div>
        <div style={{ flex:'0 0 auto', margin:'0 auto', textAlign:'center' }}>
          <div style={{ ...lbl, marginBottom:22 }}>{lang==='ru'?'АКТИВНЫЕ ЧАСЫ':'ACTIVE HOURS'}</div>
          <RhythmDial byHour={rhythm?.by_hour} isDark={isDark} lang={lang} reduced={reduced} />
        </div>
      </div>
    </div>
  );
}
// "Сонар вкуса" wrapper — handles loading / empty / loaded states.
function SonarSection({ data, loading, isDark, lang, onPlayTrack }) {
  const c = useColors(isDark);
  const hasData = data && Array.isArray(data.points) && data.points.length > 0;
  return (
    <div className={brushed(isDark)} style={{ borderRadius:18, padding:'24px 26px' }}>
      {hasData
        ? <TasteMapScope data={data} isDark={isDark} lang={lang} onPlayTrack={onPlayTrack} />
        : (
          <div style={{ textAlign:'center', padding:'34px 10px', color:c.textMuted, fontStyle:'italic', fontSize:'clamp(13px, 1.2vw, 15px)' }}>
            {loading
              ? (lang==='ru' ? 'Строю карту твоего звука…' : 'Mapping your sound…')
              : (lang==='ru' ? 'Карта появится, когда в библиотеке наберётся больше треков' : 'The map appears once your library has more tracks')}
          </div>
        )}
    </div>
  );
}

// The sonar scope: the point cloud is painted IMPERATIVELY on <canvas> (outside
// React's render) — thousands of tracks as DOM nodes would be fatal. React owns
// only the glass legend + tooltip. The draw effect depends on data/theme, never
// on hover/active (those are read from a ref inside the loop), so hovering never
// tears down the rAF loop or the ResizeObserver.
function TasteMapScope({ data, isDark, lang, onPlayTrack }) {
  const c = useColors(isDark);
  const reduced = usePrefersReducedMotion();
  const wrapRef = useRef(null), canvasRef = useRef(null), drawRef = useRef(()=>{}), rafRef = useRef(0);
  const stateRef = useRef({ hover:null, active:null, progress: reduced?1:0, size:0 });
  const [hover, setHover] = useState(null);
  const [active, setActive] = useState(null);
  const points = data?.points || [];
  const clusters = data?.clusters || [];
  const hueOf = useMemo(() => {
    const m = {}; clusters.forEach(cl => { m[cl.id] = hueFromString(cl.name || `c${cl.id}`); }); return m;
  }, [clusters]);
  const glass = isDark ? {
    background:'linear-gradient(165deg, rgba(255,255,255,0.12), rgba(255,255,255,0.03)), rgba(20,20,28,0.55)',
    border:'1px solid rgba(255,255,255,0.13)', boxShadow:'inset 0 1px 0 rgba(255,255,255,0.2), 0 12px 30px rgba(0,0,0,0.4)',
    backdropFilter:'blur(16px) saturate(1.5)', WebkitBackdropFilter:'blur(16px) saturate(1.5)',
  } : {
    background:'linear-gradient(165deg, rgba(255,255,255,0.95), rgba(255,255,255,0.7)), rgba(245,244,250,0.6)',
    border:'1px solid rgba(255,255,255,0.85)', boxShadow:'inset 0 1px 0 rgba(255,255,255,1), 0 12px 30px rgba(46,36,86,0.14)',
    backdropFilter:'blur(16px) saturate(1.5)', WebkitBackdropFilter:'blur(16px) saturate(1.5)',
  };

  useEffect(()=>{ stateRef.current.active = active; }, [active]);
  useEffect(()=>{ stateRef.current.hover = hover; }, [hover]);

  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext('2d');
    const col = (id, a) => `oklch(70% 0.16 ${hueOf[id] ?? 0} / ${a})`;
    const resize = () => {
      const r = wrap.getBoundingClientRect();
      const size = Math.max(120, r.width);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      stateRef.current.size = size;
      canvas.width = size*dpr; canvas.height = size*dpr;
      canvas.style.width = size+'px'; canvas.style.height = size+'px';
      ctx.setTransform(dpr,0,0,dpr,0,0);
      drawRef.current();
    };
    drawRef.current = () => {
      const st = stateRef.current, size = st.size; if (!size) return;
      const cx=size/2, cy=size/2, R=size*0.46, prog=st.progress;
      ctx.clearRect(0,0,size,size);
      ctx.strokeStyle = isDark?'rgba(255,255,255,.05)':'rgba(0,0,0,.05)'; ctx.lineWidth=1;
      [0.34,0.68,1].forEach(f=>{ ctx.beginPath(); ctx.arc(cx,cy,R*f,0,7); ctx.stroke(); });
      ctx.beginPath(); ctx.moveTo(cx-R,cy); ctx.lineTo(cx+R,cy); ctx.moveTo(cx,cy-R); ctx.lineTo(cx,cy+R); ctx.stroke();
      clusters.forEach(cl=>{
        const px=cx+cl.cx*R, py=cy+cl.cy*R;
        const dim = st.active!=null && st.active!==cl.id ? 0.22 : 1;
        const rad = Math.max(24, (cl.spread||0.2)*R*1.5);
        const g = ctx.createRadialGradient(px,py,0,px,py,rad);
        g.addColorStop(0, col(cl.id, 0.20*dim)); g.addColorStop(1, col(cl.id, 0));
        ctx.fillStyle=g; ctx.beginPath(); ctx.arc(px,py,rad,0,7); ctx.fill();
      });
      const hv = st.hover;
      points.forEach(p=>{
        const px=cx+p.x*R, py=cy+p.y*R;
        const dim = st.active!=null && st.active!==p.cluster ? 0.15 : 1;
        const isHov = hv && hv.track_id===p.track_id;
        ctx.globalAlpha = dim*prog;
        ctx.fillStyle = col(p.cluster, 0.9);
        ctx.beginPath(); ctx.arc(px,py, isHov?5:2, 0, 7); ctx.fill();
        if (isHov){ ctx.globalAlpha=dim; ctx.lineWidth=1.5; ctx.strokeStyle='#fff'; ctx.stroke(); }
      });
      ctx.globalAlpha=1;
      // Soft square vignette: erase the outer rim to transparent so the cloud
      // reads as a full square that dissolves at the edges — no circular crop,
      // no hard border. destination-out fades existing pixels by the gradient's
      // alpha, revealing the panel behind.
      const fade = size*0.13;
      ctx.globalCompositeOperation='destination-out';
      const wipe=(x0,y0,x1,y1,rx,ry,rw,rh)=>{ const g=ctx.createLinearGradient(x0,y0,x1,y1);
        g.addColorStop(0,'rgba(0,0,0,1)'); g.addColorStop(1,'rgba(0,0,0,0)'); ctx.fillStyle=g; ctx.fillRect(rx,ry,rw,rh); };
      wipe(0,0,fade,0, 0,0,fade,size);                 // left
      wipe(size,0,size-fade,0, size-fade,0,fade,size); // right
      wipe(0,0,0,fade, 0,0,size,fade);                 // top
      wipe(0,size,0,size-fade, 0,size-fade,size,fade); // bottom
      ctx.globalCompositeOperation='source-over';
    };
    const ro = new ResizeObserver(resize); ro.observe(wrap); resize();
    if (!reduced){
      const t0 = performance.now();
      const tick = (t)=>{ const p=Math.min(1,(t-t0)/900); stateRef.current.progress=p; drawRef.current(); if(p<1) rafRef.current=requestAnimationFrame(tick); };
      rafRef.current = requestAnimationFrame(tick);
    } else { stateRef.current.progress=1; drawRef.current(); }
    return ()=>{ cancelAnimationFrame(rafRef.current); ro.disconnect(); };
  }, [points, clusters, hueOf, isDark, reduced]);

  useEffect(()=>{ drawRef.current(); }, [active, hover]);

  const pick = (e) => {
    const st=stateRef.current, wrap=wrapRef.current; if(!wrap||!st.size) return null;
    const r = wrap.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top, size=st.size, cx=size/2, cy=size/2, R=size*0.46;
    let best=null, bd=12*12;
    for (const p of points){ const px=cx+p.x*R, py=cy+p.y*R; const d=(px-mx)**2+(py-my)**2; if(d<bd){bd=d;best={p,px,py};} }
    return best;
  };
  const onMove = (e) => {
    const b = pick(e); const id = b?b.p.track_id:null;
    setHover(h => (h?.track_id||null)===id ? h : (b ? { track_id:b.p.track_id, title:b.p.title, artist:b.p.artist, sx:b.px, sy:b.py } : null));
  };
  const onClick = (e) => { const b=pick(e); if (b && onPlayTrack) onPlayTrack({ track:{ track_id:b.p.track_id, title:b.p.title, artist:b.p.artist }, score:1 }, []); };

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:14, gap:12, flexWrap:'wrap' }}>
        <div className="mono" style={{ fontSize:'clamp(10px, 1vw, 12px)', color:c.textSubtle, letterSpacing:'0.2em', textTransform:'uppercase' }}>{lang==='ru'?'КАРТА ТВОЕГО ЗВУКА':'MAP OF YOUR SOUND'}</div>
        <div style={{ fontSize:'clamp(12px, 1.1vw, 14px)', color:c.textMuted }}>{points.length} {plural(points.length, lang, ['трек','трека','треков'], ['track','tracks'])} · {clusters.length} {plural(clusters.length, lang, ['район','района','районов'], ['region','regions'])}</div>
      </div>
      <div style={{ display:'flex', gap:'clamp(20px, 4vw, 52px)', alignItems:'center', flexWrap:'wrap', justifyContent:'center' }}>
        <div ref={wrapRef} style={{ position:'relative', flex:'1 1 320px', maxWidth:460, aspectRatio:'1 / 1', minWidth:0 }}>
          <canvas ref={canvasRef} onPointerMove={onMove} onPointerLeave={()=>setHover(null)} onClick={onClick}
            style={{ position:'absolute', inset:0, width:'100%', height:'100%', cursor:'pointer' }} />
          {hover && (
            <div style={{ ...glass, position:'absolute', zIndex:3, pointerEvents:'none', borderRadius:12, padding:'7px 11px',
              left:Math.min(hover.sx+10, (stateRef.current.size||320)-150), top:Math.max(0, hover.sy-44) }}>
              <div style={{ fontSize:'clamp(12px, 1.1vw, 13.5px)', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', maxWidth:160 }}>{hover.title}</div>
              <div style={{ fontSize:'clamp(11px, 1vw, 12px)', color:c.textMuted, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', maxWidth:160 }}>{hover.artist}</div>
            </div>
          )}
        </div>
        <div style={{ ...glass, flex:'0 1 300px', minWidth:220, padding:'14px 16px', borderRadius:16 }}>
          <div className="mono" style={{ fontSize:'clamp(9px, 0.9vw, 11px)', color:c.textSubtle, letterSpacing:'0.16em', textTransform:'uppercase', marginBottom:10 }}>{lang==='ru'?'РАЙОНЫ':'REGIONS'}</div>
          {[...clusters].sort((a,b)=>b.size-a.size).map(cl=>{
            const on = active===cl.id;
            return (
              <button key={cl.id} onClick={()=>setActive(a=>a===cl.id?null:cl.id)}
                style={{ display:'flex', alignItems:'center', gap:10, width:'100%', padding:'7px 6px', borderRadius:8,
                  background: on?'rgba(124,91,255,.14)':'transparent', transition:'background .15s', textAlign:'left' }}>
                <span style={{ width:11, height:11, borderRadius:3, flex:'none', background:`oklch(70% 0.16 ${hueOf[cl.id]??0})`, boxShadow:`0 0 8px oklch(70% 0.16 ${hueOf[cl.id]??0} / .6)` }} />
                <span style={{ flex:1, minWidth:0, fontSize:'clamp(12px, 1.1vw, 14px)', color:c.text, lineHeight:1.25, wordBreak:'break-word' }}>{cl.name}</span>
                <span className="mono" style={{ fontSize:'clamp(10px, 0.9vw, 11px)', color:c.textSubtle, flex:'none' }}>{cl.size}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Tactile semicircular VU gauge: carved track + glass dash-fill + glowing
// needle. Fills/sweeps on first scroll into view (IntersectionObserver), so it
// never animates off-screen.
function SkeuoArcGauge({ value=0, hue=145, label, sub, isDark }) {
  const c = useColors(isDark);
  const ref = useRef(null);
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced);
  useEffect(() => {
    if (reduced) { setShown(true); return; }
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setShown(true); io.disconnect(); } }, { threshold: 0.4 });
    io.observe(el); return () => io.disconnect();
  }, [reduced]);
  const pct = Math.max(0, Math.min(1, value));
  const r=54, cx=64, cy=64, len = Math.PI * r;
  const fill = shown ? pct : 0;
  const a = Math.PI - pct*Math.PI;
  const [hx, hy] = shown ? [cx + r*Math.cos(a), cy - r*Math.sin(a)] : [cx - r, cy];
  return (
    <div ref={ref} style={{ textAlign:'center', flex:'none' }}>
      <svg viewBox="0 0 128 78" width="168" height="103">
        <path d="M10,64 A54,54 0 0 1 118,64" fill="none"
          stroke={isDark?'rgba(0,0,0,.5)':'rgba(40,30,60,.16)'} strokeWidth="9" strokeLinecap="round" />
        <path d="M10,64 A54,54 0 0 1 118,64" fill="none"
          stroke={`oklch(70% 0.17 ${hue})`} strokeWidth="9" strokeLinecap="round"
          strokeDasharray={len} strokeDashoffset={len*(1-fill)}
          style={{ transition: reduced?'none':'stroke-dashoffset 0.9s cubic-bezier(.22,.9,.3,1)',
                   filter:`drop-shadow(0 0 6px oklch(70% 0.17 ${hue} / .5))` }} />
        <circle cx={hx} cy={hy} r="4.5" fill="#fff"
          style={{ filter:`drop-shadow(0 0 5px oklch(75% 0.18 ${hue}))`,
                   transition: reduced?'none':'cx 0.9s cubic-bezier(.22,.9,.3,1), cy 0.9s cubic-bezier(.22,.9,.3,1)' }} />
      </svg>
      <div style={{ fontSize:'clamp(28px, 3vw, 36px)', fontWeight:800, color:c.text, marginTop:-12 }}>{Math.round(pct*100)}%</div>
      {label && <div className="mono" style={{ fontSize:'clamp(10px, 1vw, 12px)', letterSpacing:'0.2em', textTransform:'uppercase', color:c.textSubtle }}>{label}</div>}
      {sub && <div style={{ fontSize:'clamp(12px, 1.05vw, 13.5px)', color:c.textMuted, marginTop:3 }}>{sub}</div>}
    </div>
  );
}

// Small embossed completion dial for the "loved" list rows.
function CompletionRing({ pct=0, size=42, hue=145, isDark }) {
  const c = useColors(isDark);
  const r=(size-7)/2, cc=size/2, len=2*Math.PI*r, p=Math.max(0,Math.min(1,pct));
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flex:'none' }}>
      <circle cx={cc} cy={cc} r={r} fill="none" stroke={isDark?'rgba(255,255,255,.1)':'rgba(0,0,0,.1)'} strokeWidth="3.5" />
      <circle cx={cc} cy={cc} r={r} fill="none" stroke={`oklch(70% 0.16 ${hue})`} strokeWidth="3.5" strokeLinecap="round"
        strokeDasharray={len} strokeDashoffset={len*(1-p)} transform={`rotate(-90 ${cc} ${cc})`}
        style={{ filter:`drop-shadow(0 0 3px oklch(70% 0.16 ${hue} / .5))` }} />
      <text x={cc} y={cc+4} textAnchor="middle" style={{ fontSize:11, fontWeight:700, fill:c.text }}>{Math.round(p*100)}</text>
    </svg>
  );
}

// "Guilty" rows show the honest number: how many seconds you typically hear
// before bailing, and how often you've done it.
function SkipTimeReadout({ seconds, count, hue=30, isDark, lang }) {
  const c = useColors(isDark);
  const s = seconds == null ? '—' : (Math.round(seconds * 10) / 10);
  const sU = lang==='ru' ? 'с' : 's';
  return (
    <div style={{ flex:'none', textAlign:'right', minWidth:58 }}>
      <div style={{ fontSize:'clamp(15px, 1.5vw, 18px)', fontWeight:700, lineHeight:1.05, whiteSpace:'nowrap', color:`oklch(74% 0.15 ${hue})` }}>≈{s}{sU}</div>
      <div className="mono" style={{ fontSize:'clamp(10px, 0.95vw, 12px)', color:c.textSubtle, marginTop:3, whiteSpace:'nowrap' }}>
        {count}× {lang==='ru'?'скип':'skip'}
      </div>
    </div>
  );
}

// One column of the honest-mirror panel (loved or guilty).
function EngagementColumn({ title, tracks, variant, isDark, lang, onPlayTrack, emptyText }) {
  const c = useColors(isDark);
  const lbl = { fontFamily:"'JetBrains Mono', monospace", fontSize:'clamp(10px, 1vw, 12px)', color:c.textSubtle, letterSpacing:'0.14em', textTransform:'uppercase', marginBottom:12 };
  const hue = variant==='loved' ? 145 : 30;
  return (
    <div style={{ minWidth:0 }}>
      <div style={lbl}>{title}</div>
      {tracks.length === 0
        ? <div style={{ fontSize:'clamp(12px, 1.1vw, 13.5px)', color:c.textMuted, fontStyle:'italic', padding:'4px 2px' }}>{emptyText || (lang==='ru'?'Статистика ещё не набралась':'Not enough data yet')}</div>
        : tracks.map((t)=>{
        const sub = variant==='loved'
          ? (lang==='ru'
              ? `${t.artist} · дослушано ${t.finish_count} ${plural(t.finish_count, lang, ['раз','раза','раз'], ['time','times'])}`
              : `${t.artist} · finished ${t.finish_count}×`)
          : t.artist;
        return (
        <div key={t.track_id}
          onClick={()=>{ if (onPlayTrack) onPlayTrack({ track:{ track_id:t.track_id, title:t.title, artist:t.artist, cover_art_path:t.cover_art_path }, score:1 }, []); }}
          style={{ display:'flex', alignItems:'center', gap:12, padding:'8px 9px', borderRadius:12, cursor:'pointer', transition:'background .15s ease' }}
          onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,.06)'}
          onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
          <AlbumCover title={t.title} artist={t.artist} size={44} isDark={isDark} coverPath={t.cover_art_path} radius={9} />
          <div style={{ minWidth:0, flex:1 }}>
            <div style={{ fontSize:'clamp(14px, 1.3vw, 15.5px)', color:c.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{t.title}</div>
            <div style={{ fontSize:'clamp(12px, 1vw, 13.5px)', color:c.textMuted, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{sub}</div>
          </div>
          {variant==='loved'
            ? <CompletionRing pct={t.completion} size={42} hue={hue} isDark={isDark} />
            : <SkipTimeReadout seconds={t.skip_seconds} count={t.skip_count} hue={hue} isDark={isDark} lang={lang} />}
        </div>
        );
      })}
    </div>
  );
}

// "Что ты дослушиваешь": overall completion VU + loved vs guilty columns.
function EngagementSection({ engagement, isDark, lang, onPlayTrack }) {
  const c = useColors(isDark);
  const reduced = usePrefersReducedMotion();
  const loved = engagement?.loved || [];
  const guilty = engagement?.guilty || [];
  const overall = engagement?.overall_completion || 0;
  if (!loved.length && !guilty.length && !(overall > 0)) {
    return (
      <div className={brushed(isDark)} style={{ borderRadius:18, padding:'28px 24px', textAlign:'center' }}>
        <div style={{ fontSize:'clamp(13px, 1.2vw, 15px)', color:c.textMuted, fontStyle:'italic' }}>
          {lang==='ru' ? 'Послушай ещё немного — и тут появится, что ты дослушиваешь до конца, а что бросаешь' : 'Listen a little more — your finish-vs-skip picture will appear here'}
        </div>
      </div>
    );
  }
  const pct = Math.round(overall*100);
  return (
    <div className={brushed(isDark)} style={{ borderRadius:18, padding:'24px 26px', display:'flex', flexDirection:'column', gap:24,
      animation: reduced?'none':'fadeIn 0.4s cubic-bezier(.22,.9,.3,1)' }}>
      <div style={{ display:'flex', alignItems:'center', gap:26, flexWrap:'wrap', justifyContent:'center' }}>
        <SkeuoArcGauge value={overall} hue={145} isDark={isDark} label={lang==='ru'?'дослушано':'completion'} />
        <div style={{ flex:'1 1 300px', minWidth:0 }}>
          <div style={{ fontSize:'clamp(18px, 2vw, 24px)', color:c.text, fontWeight:700, lineHeight:1.25 }}>
            {lang==='ru' ? `Ты дослушиваешь ${pct}% треков до конца` : `You finish ${pct}% of tracks`}
          </div>
          <div style={{ fontSize:'clamp(13px, 1.1vw, 15px)', color:c.textMuted, marginTop:8 }}>
            {lang==='ru' ? 'Правда, которую обычное число прослушиваний прячет.' : 'The truth a plain play count hides.'}
          </div>
        </div>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(260px, 1fr))', gap:'24px 32px' }}>
        <EngagementColumn title={lang==='ru'?'ЛЮБИШЬ ПО-НАСТОЯЩЕМУ':'TRULY LOVED'} tracks={loved} variant="loved" isDark={isDark} lang={lang} onPlayTrack={onPlayTrack} />
        <EngagementColumn title={lang==='ru'?'ЧАЩЕ ВСЕГО БРОСАЕШЬ':'OFTEN SKIPPED'} tracks={guilty} variant="guilty" isDark={isDark} lang={lang} onPlayTrack={onPlayTrack} />
      </div>
    </div>
  );
}

function DistributionsPanel({ stats, isDark, lang, navigateToArtist }) {
  const reduced = usePrefersReducedMotion();
  const c = useColors(isDark);
  const decades = stats?.decades || [];
  const genres = stats?.genres || [];
  const artists = (stats?.top_artists || []).slice(0, 5);
  const durations = stats?.duration_buckets || [];
  const formats = stats?.formats || [];
  const losslessPct = stats?.lossless_pct ?? 0;
  const colLbl = { fontFamily:"'JetBrains Mono', monospace", fontSize:'clamp(10px, 1vw, 12px)', color:c.textSubtle, letterSpacing:'0.2em', textTransform:'uppercase', marginBottom:'14px' };

  return (
    <div className={brushed(isDark)} style={{
      borderRadius:'18px', padding:'24px 26px',
      display:'flex', flexDirection:'column', gap:'30px',
      animation: reduced ? 'none' : 'fadeIn 0.4s cubic-bezier(.22,.9,.3,1)',
    }}>
      <EraBars decades={decades} isDark={isDark} lang={lang} reduced={reduced} labelStyle={colLbl} />
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(240px, 1fr))', gap:'30px 36px' }}>
        <GenreBars genres={genres} isDark={isDark} lang={lang} reduced={reduced} labelStyle={colLbl} />
        <ArtistMosaic artists={artists} isDark={isDark} lang={lang} labelStyle={colLbl} navigateToArtist={navigateToArtist} />
        <DurationBars buckets={durations} isDark={isDark} lang={lang} reduced={reduced} labelStyle={colLbl} />
        <FormatBars formats={formats} losslessPct={losslessPct} isDark={isDark} lang={lang} reduced={reduced} labelStyle={colLbl} />
      </div>
    </div>
  );
}

// NEW (Phase 5) — "quality": file format distribution (lossless vs lossy), the
// "I own these files" pride that streaming can't show. Derived from file
// extension server-side.
function FormatBars({ formats, losslessPct, isDark, lang, reduced, labelStyle }) {
  const c = useColors(isDark);
  if (!formats.length) return (
    <div><div style={labelStyle}>{lang==='ru'?'КАЧЕСТВО':'QUALITY'}</div><Empty lang={lang} /></div>
  );
  const max = formats.reduce((m,f)=>Math.max(m,f.count||0),0)||1;
  const LOSSLESS = ['FLAC','WAV','AIFF','ALAC','APE'];
  return (
    <div>
      <div style={labelStyle}>{lang==='ru'?'КАЧЕСТВО':'QUALITY'}</div>
      <div style={{ fontSize:'clamp(13px, 1.2vw, 15px)', color:c.text, marginBottom:13 }}>
        <b style={{ color:'oklch(68% 0.16 150)', fontWeight:700 }}>{losslessPct}%</b> {lang==='ru'?'без потерь':'lossless'}
      </div>
      {formats.map((f,i)=>{
        const w = Math.round((f.count/max)*100);
        const lossless = LOSSLESS.includes(f.format);
        const hue = lossless ? 150 : 50;
        return (
          <div key={f.format} style={{ marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5, gap:8 }}>
              <span style={{ color:c.textMuted, fontSize:'clamp(12px, 1.1vw, 14px)' }}>
                {f.format}{lossless && <span style={{ color:'oklch(68% 0.16 150)', marginLeft:6, fontSize:'0.85em' }}>✓</span>}
              </span>
              <span className="mono" style={{ color:c.textSubtle, fontSize:'clamp(11px, 1vw, 12px)' }}>{f.pct}%</span>
            </div>
            <div className={ske('inset', isDark)} style={{ height:11, borderRadius:6, overflow:'hidden' }}>
              <div style={{ height:'100%', width:`${w}%`, borderRadius:6, transformOrigin:'left',
                background:`linear-gradient(90deg, oklch(58% ${lossless?0.16:0.11} ${hue}), oklch(70% ${lossless?0.16:0.12} ${hue+12}))`,
                boxShadow:'inset 0 1px 1px rgba(255,255,255,.4)',
                animation: reduced?'none':'meterTick 0.55s cubic-bezier(.22,.9,.3,1) both',
                animationDelay: reduced?'0s':`${(i*0.06).toFixed(2)}s` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
function ListeningWidgetsRow({ data, rhythm, isDark, lang, onPlayTrack, navigateToArtist }) {
  const c = useColors(isDark);
  const reduced = usePrefersReducedMotion();
  const hU = lang==='ru' ? 'ч' : 'h', mU = lang==='ru' ? 'м' : 'm';
  const fmtDur = (sec) => {
    if (!sec) return '—';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return h > 0 ? `${h}${hU} ${m}${mU}` : `${m}${mU}`;
  };
  const since = data?.since ? new Date(data.since).toLocaleDateString(lang==='ru'?'ru-RU':'en-US', {month:'short', day:'numeric'}) : null;
  const top_track = data?.top_track;
  const top_artist = data?.top_artist;
  const animSec = useCountUp(data?.total_seconds_listened || 0, 700, reduced);
  const playsWord = ['плей','плея','плеев'], playsWordEn = ['play','plays'];

  const lbl = (t) => <div className="mono" style={{ fontSize:'clamp(10px, 0.95vw, 12px)', color:c.textSubtle, letterSpacing:'0.2em', textTransform:'uppercase' }}>{t}</div>;
  const card = { borderRadius:16, padding:'16px 18px', minHeight:104, display:'flex', flexDirection:'column', gap:8, border:`1px solid ${c.border}` };

  return (
    <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(220px, 1fr))', gap:14 }}>
      {/* total listened — label on the left, count-up readout pushed right */}
      <div className={ske('display', isDark)} style={{ ...card, flexDirection:'row', alignItems:'center', justifyContent:'space-between', gap:14 }}>
        <div style={{ minWidth:0, display:'flex', flexDirection:'column', gap:6 }}>
          <div className="mono" style={{ fontSize:'clamp(10px, 0.95vw, 12px)', color:c.textSubtle, letterSpacing:'0.12em', textTransform:'uppercase', lineHeight:1.3 }}>
            {lang==='ru'?'Суммарно прослушано':'Total listened'}
          </div>
          {since ? <span style={{ fontSize:'clamp(12px, 1vw, 13.5px)', color:c.textMuted }}>{lang==='ru'?`с ${since}`:`since ${since}`}</span> : null}
        </div>
        <div style={{ fontSize:'clamp(24px, 2.8vw, 32px)', color:'oklch(72% 0.18 145)', fontWeight:700, lineHeight:1.05, whiteSpace:'nowrap', flex:'none' }}>{fmtDur(animSec)}</div>
      </div>

      {/* ★ top track — click to play */}
      <button className={ske('display', isDark)} style={{ ...card, textAlign:'left', cursor: top_track?'pointer':'default' }}
        onClick={() => { if (top_track && onPlayTrack) onPlayTrack({ track:{ track_id:top_track.track_id, title:top_track.title, artist:top_track.artist }, score:1 }, []); }}>
        {lbl(lang==='ru'?'★ ТОП-ТРЕК':'★ TOP TRACK')}
        <div style={{ display:'flex', alignItems:'center', gap:12, flex:1 }}>
          <AlbumCover title={top_track?.title || ''} artist={top_track?.artist || ''} coverPath={top_track?.cover_art_path} size={44} radius={10} isDark={isDark} />
          <div style={{ minWidth:0 }}>
            <div style={{ fontSize:'clamp(15px, 1.5vw, 17px)', color:c.text, fontWeight:600, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{top_track?.title || '—'}</div>
            <div style={{ fontSize:'clamp(12px, 1vw, 13.5px)', color:c.textMuted, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
              {top_track ? `${top_track.artist} · ${top_track.play_count} ${plural(top_track.play_count, lang, playsWord, playsWordEn)}` : (lang==='ru'?'нет данных':'no data')}
            </div>
          </div>
        </div>
      </button>

      {/* ★ top artist — click to open */}
      <button className={ske('display', isDark)} style={{ ...card, textAlign:'left', cursor: top_artist?.slug?'pointer':'default' }}
        onClick={() => { if (top_artist?.slug && navigateToArtist) navigateToArtist(top_artist.slug); }}>
        {lbl(lang==='ru'?'★ ТОП-АРТИСТ':'★ TOP ARTIST')}
        <div style={{ display:'flex', alignItems:'center', gap:12, flex:1 }}>
          <AlbumCover title={top_artist?.name || ''} artist={top_artist?.name || ''} coverPath={top_artist?.image} size={44} radius={10} isDark={isDark} />
          <div style={{ minWidth:0 }}>
            <div style={{ fontSize:'clamp(15px, 1.6vw, 19px)', color:'oklch(75% 0.14 80)', fontWeight:700, lineHeight:1.15, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{top_artist?.name || '—'}</div>
            <div style={{ fontSize:'clamp(12px, 1vw, 13.5px)', color:c.textMuted }}>{top_artist ? `${top_artist.play_count} ${plural(top_artist.play_count, lang, playsWord, playsWordEn)}` : ' '}</div>
          </div>
        </div>
      </button>
    </div>
  );
}
function LibraryTabsStrip({ active, onChange, counts, lang, isDark }) {
  const c = useColors(isDark);
  // Mobile: collapse the wrapping text pills to an equal-width icon-only
  // segmented control. useIsMobile() is read in-component (not threaded as a
  // prop) to avoid the prop-drilling ReferenceError class of bug. Counts are
  // dropped here — each tab's content shows its own count header.
  const isMobile = useIsMobile();
  if (isMobile) {
    const labelFor = (id) => ((lang === 'ru')
      ? { albums:'Альбомы', recent:'Недавние', playlists:'Плейлисты', stats:'Статистика' }
      : { albums:'Albums', recent:'Recently', playlists:'Playlists', stats:'Statistics' })[id];
    const ICONS = {
      albums: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
      recent: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
      playlists: <><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3.5" cy="6" r="1.3" fill="currentColor" stroke="none"/><circle cx="3.5" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="3.5" cy="18" r="1.3" fill="currentColor" stroke="none"/></>,
      stats: <><line x1="6" y1="21" x2="6" y2="12"/><line x1="12" y1="21" x2="12" y2="4"/><line x1="18" y1="21" x2="18" y2="15"/></>,
    };
    const order = ['albums','recent','playlists','stats'];
    return (
      <div style={{ display:'flex', gap:6, padding:'2px 0' }}>
        {order.map(id => {
          const isActive = active === id;
          const label = labelFor(id);
          return (
            <button key={id} onClick={() => onChange(id)}
              title={label} aria-label={label} aria-pressed={isActive}
              style={{
                flex:1, minWidth:0, minHeight:44,
                display:'flex', alignItems:'center', justifyContent:'center',
                borderRadius:12, cursor:'pointer',
                background: isActive
                  ? 'linear-gradient(180deg, oklch(64% 0.18 272) 0%, oklch(53% 0.2 276) 100%)'
                  : 'rgba(255,255,255,.04)',
                color: isActive ? '#fff' : c.textMuted,
                border: `1px solid ${isActive ? 'rgba(124,91,255,.45)' : c.border}`,
                boxShadow: isActive
                  ? 'inset 0 1px 0 rgba(255,255,255,.28), 0 6px 18px rgba(124,91,255,.28)'
                  : 'inset 0 1px 0 rgba(255,255,255,.04)',
                transition:'all .25s cubic-bezier(.22,.9,.3,1)',
              }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                {ICONS[id]}
              </svg>
            </button>
          );
        })}
      </div>
    );
  }
  const tabs = lang === 'ru'
    ? [['albums','▦ Альбомы', counts.albums], ['recent','⟲ Недавние', counts.recent], ['playlists','♫ Плейлисты', counts.playlists], ['stats','◷ Статистика', null]]
    : [['albums','▦ Albums', counts.albums], ['recent','⟲ Recently', counts.recent], ['playlists','♫ Playlists', counts.playlists], ['stats','◷ Statistics', null]];
  return (
    <div style={{ display:'flex', gap:10, flexWrap:'wrap' }}>
      {tabs.map(([id, label, n]) => {
        const isActive = active === id;
        return (
          <button key={id} onClick={() => onChange(id)}
            style={{
              display:'flex', alignItems:'center', gap:8,
              padding:'10px 18px', borderRadius:12, fontSize:13, cursor:'pointer',
              background: isActive
                ? 'linear-gradient(180deg, oklch(64% 0.18 272) 0%, oklch(53% 0.2 276) 100%)'
                : 'rgba(255,255,255,.04)',
              color: isActive ? '#fff' : c.textMuted,
              border: `1px solid ${isActive ? 'rgba(124,91,255,.45)' : c.border}`,
              boxShadow: isActive
                ? 'inset 0 1px 0 rgba(255,255,255,.28), 0 6px 18px rgba(124,91,255,.28)'
                : 'inset 0 1px 0 rgba(255,255,255,.04)',
              transition:'all .25s cubic-bezier(.22,.9,.3,1)',
            }}
            onMouseEnter={e => { if (!isActive) { e.currentTarget.style.background = 'rgba(255,255,255,.08)'; e.currentTarget.style.color = c.text; e.currentTarget.style.transform = 'translateY(-1px)'; } }}
            onMouseLeave={e => { if (!isActive) { e.currentTarget.style.background = 'rgba(255,255,255,.04)'; e.currentTarget.style.color = c.textMuted; e.currentTarget.style.transform = 'translateY(0)'; } }}
          >
            <span>{label}</span>
            {n != null && <span className="mono" style={{
              fontSize:10, padding:'2px 8px', borderRadius:9, letterSpacing:'0.04em',
              background: isActive ? 'rgba(255,255,255,.2)' : 'rgba(255,255,255,.06)',
              color: isActive ? '#fff' : c.textSubtle,
              transition:'all .25s',
            }}>{n}</span>}
          </button>
        );
      })}
    </div>
  );
}
// ─── NEW PLAYLIST MODAL ───────────────────────────────────────────────────────
function NewPlaylistModal({ onCancel, onSubmit, lang }) {
  const [name, setName] = React.useState('');
  const [description, setDescription] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    const onEsc = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onEsc);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onEsc);
      document.body.style.overflow = '';
    };
  }, [onCancel]);

  const handleSubmit = async () => {
    const n = name.trim();
    if (!n) return;
    setBusy(true); setError(null);
    try {
      await onSubmit(n, description.trim() || null);
    } catch (e) {
      const msg = e?.detail || e?.message || String(e);
      if (/already exists|name already|409/.test(msg)) {
        setError(lang === 'ru' ? 'Плейлист с таким именем уже существует' : 'Playlist with this name already exists');
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(8,6,16,.62)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
        display: 'grid', placeItems: 'center',
      }}
    >
      <div style={{
        width: 'min(440px, 92vw)', padding: 28,
        background: 'linear-gradient(180deg, rgba(32,28,48,0.97) 0%, rgba(20,16,32,0.97) 60%)',
        border: '1px solid rgba(255,255,255,.08)',
        borderRadius: 18,
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,.14), 0 28px 70px rgba(0,0,0,.6), 0 0 0 1px rgba(124,91,255,.18)',
      }}>
        <div className="mono" style={{ fontSize: 10, letterSpacing: '0.24em', color: 'rgba(238,238,243,.5)', marginBottom: 18, textTransform: 'uppercase' }}>
          {lang === 'ru' ? 'Новый плейлист' : 'New playlist'}
        </div>
        <div style={{ fontSize: 12, color: 'rgba(238,238,243,.6)', margin: '14px 0 6px', letterSpacing: '0.02em' }}>
          {lang === 'ru' ? 'Название' : 'Name'}
        </div>
        <input
          autoFocus
          value={name} onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSubmit(); }}
          placeholder={lang === 'ru' ? 'Например: Late night ride' : 'e.g.: Late night ride'}
          style={{
            width: '100%', padding: '10px 12px',
            background: 'rgba(0,0,0,.32)', border: '1px solid rgba(255,255,255,.07)',
            borderRadius: 10, color: '#fff', fontSize: 14, outline: 'none',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,.3)',
          }}
        />
        <div style={{ fontSize: 12, color: 'rgba(238,238,243,.6)', margin: '14px 0 6px', letterSpacing: '0.02em' }}>
          {lang === 'ru' ? 'Описание (опционально)' : 'Description (optional)'}
        </div>
        <textarea
          value={description} onChange={(e) => setDescription(e.target.value)}
          rows={3}
          style={{
            width: '100%', padding: '10px 12px',
            background: 'rgba(0,0,0,.32)', border: '1px solid rgba(255,255,255,.07)',
            borderRadius: 10, color: '#fff',
            fontFamily: "'Noto Serif Display', Georgia, serif", fontStyle: 'italic', fontSize: 15,
            outline: 'none', boxShadow: 'inset 0 1px 2px rgba(0,0,0,.3)', resize: 'vertical', minHeight: 76,
          }}
        />
        {error && (
          <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8, fontSize: 12, color: '#ffc99a', background: 'rgba(255,180,80,.08)', border: '1px solid rgba(255,180,80,.25)' }}>
            {error}
          </div>
        )}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 22 }}>
          <button
            onClick={onCancel}
            style={{
              background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.07)',
              color: 'rgba(238,238,243,.7)', padding: '10px 18px', borderRadius: 10, fontSize: 13, cursor: 'pointer',
            }}>{lang === 'ru' ? 'Отмена' : 'Cancel'}</button>
          <button
            className="cta-v3"
            disabled={busy || !name.trim()}
            onClick={handleSubmit}
            style={{ opacity: busy || !name.trim() ? 0.5 : 1 }}>
            {busy ? '…' : (lang === 'ru' ? 'Создать ▶' : 'Create ▶')}
          </button>
        </div>
      </div>
    </div>
  );
}

function AlbumModal({ album, originRect, onClose, onPlayTrack, navigateToArtist, isDark, lang, onAddToPlaylist }) {
  const c = useColors(isDark);
  const [hoverRow, setHoverRow] = useState(-1);
  // 10+ feat-артистов заполоняли всю страницу — прячем хвост под "+N".
  const [showAllFeat, setShowAllFeat] = useState(false);
  const isMobile = useIsMobile();

  // ── Shared-element fly-in (FLIP): start at the clicked grid cover ────
  // originRect is the viewport rect of the cover the user clicked; the
  // stage starts translated+scaled onto it, then transitions to center
  // while the gatefold flip rotates. Falls back to flip-in-place if null.
  const stageRef = useRef(null);
  const originTransformRef = useRef(null);
  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el || !originRect || prefersReducedMotion()) return;
    const final = el.getBoundingClientRect();
    if (!final.width || !final.height) return;
    const sx = originRect.width / final.width;
    const sy = originRect.height / final.height;
    const dx = (originRect.left + originRect.width / 2) - (final.left + final.width / 2);
    const dy = (originRect.top + originRect.height / 2) - (final.top + final.height / 2);
    const atOrigin = `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`;
    originTransformRef.current = atOrigin;
    el.style.transform = atOrigin;
    // Double rAF: let the at-origin transform paint before transitioning out.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.transition = 'transform 0.7s cubic-bezier(.3,.75,.25,1)';
      el.style.transform = 'translate(0px, 0px) scale(1, 1)';
    }));
  }, []);

  // ── Gatefold close: reverse flip + fly back to the grid cover ────────
  const [closing, setClosing] = useState(false);
  const closingRef = useRef(false);
  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setClosing(true);
    const el = stageRef.current;
    if (el && originTransformRef.current && !prefersReducedMotion()) {
      el.style.transition = 'transform 0.45s cubic-bezier(.55,.06,.5,.9)';
      el.style.transform = originTransformRef.current;
    }
    setTimeout(onClose, 460);   // must outlive albumFlipOut (0.5s) start→visual close
  }, [onClose]);

  // Lock body scroll while open
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, []);

  // Esc to close
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') requestClose(); };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, [requestClose]);

  const playFromIdx = (idx) => {
    const t = album.tracks[idx];
    if (!t || !onPlayTrack) return;
    onPlayTrack({ track: t }, album.tracks.map(tt => ({ track: tt })));
    requestClose();
  };

  const fmtDur = (s) => {
    if (!s) return '—';
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
  };
  const totalDur = album.duration_seconds || 0;
  const totalDurFmt = `${Math.floor(totalDur/60)}:${String(Math.floor(totalDur%60)).padStart(2,'0')}`;

  // Cover URL + fallback hue (same scheme as AlbumCover)
  const hue = (((album.album_title||'?').charCodeAt(0)||65)*37 + ((album.primary_artist||'?').charCodeAt(0)||65)*17) % 360;
  const coverUrl = album.cover_art_path
    ? (album.cover_art_path.startsWith('http') ? album.cover_art_path : `${API}${album.cover_art_path}`)
    : null;

  return (
    <div
      onClick={requestClose}
      style={{
        position:'fixed', inset:0, zIndex:100,
        background:'rgba(0,0,0,.65)',
        // Mobile: the full-screen gatefold covers the overlay entirely — its
        // blur would be invisible yet stay composited the whole time.
        backdropFilter: isMobile ? 'none' : 'blur(8px)',
        WebkitBackdropFilter: isMobile ? 'none' : 'blur(8px)',
        display:'grid', placeItems:'center', padding: isMobile ? 0 : '24px',
        animation: closing ? 'fadeOverlayOut 0.35s ease 0.15s forwards' : 'fadeIn 0.25s ease',
      }}
    >
      <div
        ref={stageRef}
        className="album-flip-stage"
        onClick={e => e.stopPropagation()}
        // Mobile: the gatefold takes the whole screen (same flip mechanics);
        // desktop keeps the centered 700px sleeve.
        style={ isMobile
          ? { width:'100dvw', height:'100dvh', willChange:'transform' }
          : { width:'min(700px, 94vw)', height:'min(84vh, 700px)', willChange:'transform' } }
      >
        <div className={`album-flip${closing ? ' closing' : ''}`}>

          {/* ── FRONT: sleeve cover ─────────────────────────────── */}
          <div className="album-flip-face" style={{ background:'#0d0a12' }}>
            {coverUrl ? (
              <img src={coverUrl} alt="" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }} />
            ) : (
              <div style={{
                width:'100%', height:'100%', display:'grid', placeItems:'center',
                background:`linear-gradient(135deg, oklch(38% 0.13 ${hue}), oklch(52% 0.18 ${(hue+45)%360}))`,
                fontFamily:"'JetBrains Mono', monospace", fontSize:'52px', fontWeight:700, color:'rgba(255,255,255,.65)', letterSpacing:'0.04em',
              }}>{(album.album_title||'?').slice(0,2).toUpperCase()}</div>
            )}
            {/* sleeve sheen */}
            <div aria-hidden="true" style={{ position:'absolute', inset:0, background:'linear-gradient(115deg, rgba(255,255,255,.14) 0%, transparent 32%, transparent 68%, rgba(0,0,0,.28) 100%)' }} />
          </div>

          {/* ── BACK: blurred cover + tracklist ─────────────────── */}
          <div className="album-flip-face album-flip-back" style={{ background:'#0d0a12' }}>
            {coverUrl ? (
              // blur(64px) erases any detail beyond ~320px anyway — the thumb
              // is usually already in cache from the grid, so the back face
              // paints instantly instead of fetching the full-size art again.
              <img src={thumbCoverUrl(coverUrl)} alt="" aria-hidden="true" style={{
                position:'absolute', top:'-12%', left:'-12%', width:'124%', height:'124%',
                objectFit:'cover', filter:'blur(64px) saturate(1.35) brightness(.45)',
              }} />
            ) : (
              <div aria-hidden="true" style={{ position:'absolute', inset:0, background:`linear-gradient(135deg, oklch(28% 0.1 ${hue}), oklch(20% 0.08 ${(hue+45)%360}))` }} />
            )}
            <div aria-hidden="true" style={{ position:'absolute', inset:0, background:'linear-gradient(180deg, rgba(10,8,18,.32) 0%, rgba(10,8,18,.66) 100%)' }} />

            <div style={{
              position:'absolute', inset:0, display:'flex', flexDirection:'column',
              gap: isMobile ? '12px' : '16px',
              padding: isMobile
                ? 'calc(env(safe-area-inset-top, 0px) + 16px) 14px calc(env(safe-area-inset-bottom, 0px) + 14px)'
                : '26px 28px 24px',
            }}>

              {/* Hero: small cover + vinyl peeking out. On phones the user just
                  saw the full-screen front cover — skip the duplicate art and
                  give the width to title/artists/tracklist. */}
              <div className="album-back-rise" style={{ '--ab-d':'0.45s', display:'flex', gap:'22px', alignItems:'center' }}>
                {!isMobile && (
                <div style={{ position:'relative', width:'196px', height:'132px', flexShrink:0 }}>
                  <div className="album-vinyl" style={{ position:'absolute', left:'64px', top:'4px', width:'124px', height:'124px' }}>
                    <div style={{ position:'absolute', inset:0, display:'grid', placeItems:'center' }}>
                      <div style={{
                        width:'40px', height:'40px', borderRadius:'50%',
                        background:`linear-gradient(135deg, oklch(55% 0.16 ${hue}), oklch(40% 0.14 ${(hue+45)%360}))`,
                        display:'grid', placeItems:'center', boxShadow:'0 0 0 1px rgba(0,0,0,.6)',
                      }}>
                        <div style={{ width:'7px', height:'7px', borderRadius:'50%', background:'#0d0a12' }} />
                      </div>
                    </div>
                  </div>
                  <div style={{ position:'absolute', left:0, top:0, width:'132px', height:'132px', filter:'drop-shadow(6px 0 14px rgba(0,0,0,.45))' }}>
                    <AlbumCover title={album.album_title} artist={album.primary_artist} size={132} isDark={true} coverPath={album.cover_art_path} radius={12} />
                  </div>
                </div>
                )}
                <div style={{ flex:1, minWidth:0 }}>
                  <div className="mono" style={{ fontSize:'10px', color:'rgba(238,235,248,.5)', letterSpacing:'0.24em', textTransform:'uppercase' }}>{lang==='ru'?'АЛЬБОМ':'ALBUM'}</div>
                  <div style={{
                    fontFamily:"'Noto Serif Display', Georgia, serif", fontSize:'30px', lineHeight:1.08,
                    color:'#f5f3fa', margin:'6px 0 4px', letterSpacing:'-0.01em',
                    textShadow:'0 2px 18px rgba(0,0,0,.5)',
                    overflow:'hidden', display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical',
                  }}>{album.album_title}</div>
                  <div
                    onClick={() => { if (album.primary_artist_slug && navigateToArtist) { navigateToArtist(album.primary_artist_slug); requestClose(); } }}
                    style={{ fontSize:'14px', color:'#cdbcff', cursor:'pointer', display:'inline-block' }}
                  >{album.primary_artist} →</div>
                  {album.feat_artists?.length > 0 && (() => {
                    const FEAT_LIMIT = 6;
                    const feats = album.feat_artists;
                    const shown = showAllFeat ? feats : feats.slice(0, FEAT_LIMIT);
                    const hidden = feats.length - shown.length;
                    return (
                    <div style={{ marginTop:'7px', display:'flex', gap:'5px', flexWrap:'wrap' }}>
                      {shown.map(f => (
                        <span key={f.slug}
                          onClick={() => { if (navigateToArtist) { navigateToArtist(f.slug); requestClose(); } }}
                          style={{ padding:'2px 9px', borderRadius:'10px', background:'rgba(120,80,200,.25)', border:'1px solid rgba(160,130,255,.35)', color:'#d4c8ff', fontSize:'10px', cursor:'pointer' }}>
                          {f.name}
                        </span>
                      ))}
                      {hidden > 0 && (
                        <span
                          onClick={(e) => { e.stopPropagation(); setShowAllFeat(true); }}
                          style={{ padding:'2px 9px', borderRadius:'10px', background:'rgba(255,255,255,.08)', border:'1px solid rgba(255,255,255,.18)', color:'rgba(238,235,248,.75)', fontSize:'10px', cursor:'pointer' }}>
                          {lang==='ru' ? `+${hidden} ещё` : `+${hidden} more`}
                        </span>
                      )}
                      {showAllFeat && feats.length > FEAT_LIMIT && (
                        <span
                          onClick={(e) => { e.stopPropagation(); setShowAllFeat(false); }}
                          style={{ padding:'2px 9px', borderRadius:'10px', background:'rgba(255,255,255,.05)', border:'1px solid rgba(255,255,255,.12)', color:'rgba(238,235,248,.55)', fontSize:'10px', cursor:'pointer' }}>
                          {lang==='ru' ? 'Свернуть' : 'Collapse'}
                        </span>
                      )}
                    </div>
                    );
                  })()}
                  <div className="mono" style={{ display:'flex', gap:'14px', fontSize:'11px', color:'rgba(238,235,248,.55)', marginTop:'10px', letterSpacing:'0.06em' }}>
                    <span>{album.year_range || album.year || '—'}</span>
                    <span>·</span>
                    <span>{album.track_count} {lang==='ru'?'треков':'tracks'}</span>
                    <span>·</span>
                    <span>{totalDurFmt}</span>
                  </div>
                </div>
                <button
                  onClick={requestClose}
                  style={{ alignSelf:'flex-start', background:'rgba(255,255,255,.06)', color:'rgba(238,235,248,.7)', fontSize:'15px', cursor:'pointer', width:'34px', height:'34px', borderRadius:'50%', border:'1px solid rgba(255,255,255,.1)', display:'grid', placeItems:'center', transition:'all .2s' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,.14)'; e.currentTarget.style.color = '#fff'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,.06)'; e.currentTarget.style.color = 'rgba(238,235,248,.7)'; }}
                >✕</button>
              </div>

              {/* Actions */}
              <div className="album-back-rise" style={{ '--ab-d':'0.55s', display:'flex', gap:'8px' }}>
                <button onClick={() => playFromIdx(0)} style={{
                  padding:'9px 18px', borderRadius:'10px', fontSize:'12px', fontFamily:"'JetBrains Mono', monospace", letterSpacing:'0.06em',
                  background:'linear-gradient(180deg, oklch(62% 0.21 272), oklch(49% 0.22 283))', color:'#fff', border:'none', cursor:'pointer',
                  boxShadow:'inset 0 1px 0 rgba(255,255,255,.3), 0 6px 18px rgba(124,91,255,.35)',
                }}>▶ {lang==='ru'?'Играть всё':'Play All'}</button>
              </div>

              {/* Tracklist. Mobile: what's behind is the already-64px-blurred
                  cover image — a second live blur adds cost, not looks. */}
              <div style={{
                background: isMobile ? 'rgba(8,6,14,.72)' : 'rgba(8,6,14,.45)',
                border:'1px solid rgba(255,255,255,.08)', borderRadius:'14px',
                backdropFilter: isMobile ? 'none' : 'blur(14px)',
                WebkitBackdropFilter: isMobile ? 'none' : 'blur(14px)',
                overflow:'auto', flex:1, minHeight:0,
              }}>
                {album.tracks.map((t, i) => {
                  return (
                    <div
                      key={t.track_id}
                      className="album-row-in"
                      onClick={() => playFromIdx(i)}
                      onMouseEnter={() => setHoverRow(i)}
                      onMouseLeave={() => setHoverRow(-1)}
                      style={{
                        '--ar-i': Math.min(i, 20),
                        display:'grid', gridTemplateColumns:'28px 1fr 60px 30px',
                        gap:'10px', padding:'9px 14px', alignItems:'center',
                        borderBottom:'1px solid rgba(255,255,255,.06)',
                        fontSize:'13px', cursor:'pointer',
                        background: hoverRow === i ? 'rgba(255,255,255,.06)' : 'transparent',
                        transition:'background .15s',
                      }}
                    >
                      <span className="mono" style={{ color: hoverRow === i ? '#cdbcff' : 'rgba(238,235,248,.4)', fontSize:'11px', textAlign:'center' }}>{hoverRow === i ? '▶' : i+1}</span>
                      <span style={{ color:'#ece9f4', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{t.title}</span>
                      <span className="mono" style={{ color:'rgba(238,235,248,.4)', fontSize:'11px', textAlign:'right' }}>{fmtDur(t.duration)}</span>
                      {onAddToPlaylist ? (
                        <button
                          onClick={(e) => { e.stopPropagation(); onAddToPlaylist(t.track_id, e.currentTarget); }}
                          style={{ background:'transparent', border:'none', cursor:'pointer', fontSize:'17px', lineHeight:1, color:'rgba(238,235,248,.45)', padding:0 }}
                          title={lang==='ru'?'Добавить в плейлист':'Add to playlist'}
                        >＋</button>
                      ) : <span />}
                    </div>
                  );
                })}
              </div>

            </div>
          </div>

        </div>
      </div>
    </div>
  );
}


// ─── Processing-mode badge + Premium enrichment note ─────────────────────────
// Module-cached /instance/config fetch: the mode badge sits on several indexing
// surfaces at once (onboarding, modal, member wizard) — one request serves all.
let _instanceCfgPromise = null;
function fetchInstanceConfigCached() {
  if (!_instanceCfgPromise) {
    _instanceCfgPromise = apiFetch('/instance/config')
      .catch(err => { _instanceCfgPromise = null; throw err; });
  }
  return _instanceCfgPromise;
}

// Persistent "where does the processing run" indicator for every indexing
// surface. sharing == the whole stack lives on the user's machine → local
// processing (green); server == a hosted instance the member uploads to →
// processing on the operator's server (violet). Renders nothing until the
// config arrives so the wrong mode is never flashed.
function ProcessingModeBadge({ isDark, lang, style }) {
  const c = useColors(isDark);
  const ru = lang === 'ru';
  const [cfg, setCfg] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchInstanceConfigCached().then(r => { if (alive) setCfg(r); }).catch(() => {});
    return () => { alive = false; };
  }, []);
  if (!cfg) return null;

  const server = cfg.mode === 'server';
  const accent   = server ? 'oklch(62% 0.2 275)' : 'oklch(63% 0.17 142)';
  const accentBg = server
    ? `oklch(62% 0.2 275 / ${isDark ? 0.12 : 0.08})`
    : `oklch(63% 0.17 142 / ${isDark ? 0.12 : 0.08})`;
  const accentEdge = server ? 'oklch(62% 0.2 275 / 0.35)' : 'oklch(63% 0.17 142 / 0.35)';
  const label = server
    ? (ru ? 'ОБРАБОТКА НА СЕРВЕРЕ' : 'PROCESSING ON SERVER')
    : (ru ? 'ЛОКАЛЬНАЯ ОБРАБОТКА' : 'PROCESSING ON THIS DEVICE');
  let line = server
    ? (ru ? 'Треки обрабатываются на сервере MusiX — скорость зависит от его загрузки, возможна очередь.'
          : 'Your tracks are processed on the MusiX server — speed depends on server load; a queue is possible.')
    : (ru ? 'Всё происходит на этом компьютере — файлы никуда не отправляются.'
          : 'Everything runs on this computer — your files never leave it.');
  if (!server && cfg.ai_available === false) {
    line += ru
      ? ' ИИ-дополнения выключены — локальная модель не настроена.'
      : ' AI extras are off — no local model is configured.';
  }

  return (
    <div style={{
      display:'flex', alignItems:'flex-start', gap:'11px',
      padding:'11px 14px', borderRadius:'12px',
      background: accentBg,
      border: `1px solid ${accentEdge}`,
      ...style,
    }}>
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke={accent}
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        style={{ flexShrink:0, marginTop:'1px' }}>
        {server
          ? <path d="M17.5 19a4.5 4.5 0 1 0-.9-8.9A6 6 0 1 0 6 15.7 3.5 3.5 0 0 0 7.5 19Z" />
          : <>
              <rect x="3" y="4" width="18" height="12" rx="2" />
              <path d="M8 20h8M12 16v4" />
            </>}
      </svg>
      <div style={{ minWidth:0 }}>
        <div className="mono" style={{
          fontSize:'11px', fontWeight:'600', letterSpacing:'0.2em',
          color: accent, marginBottom:'3px',
        }}>
          {label}
        </div>
        <div style={{ fontSize:'12.5px', color:c.textMuted, lineHeight:1.5 }}>{line}</div>
      </div>
    </div>
  );
}

// ─── Premium edition gating ──────────────────────────────────────────────────
// MusiX ships in two editions. This single build-level flag is the ONLY switch:
// the "regular" edition sets it to false and every premium block below (Yandex
// import, metadata/cover enhancement, the header chip) disappears cleanly —
// each is wrapped in <PremiumGate>, so removing them never leaves a hole in the
// layout. Flip to false (or wire to an env/config at build) for the plain build.
const MUSIX_PREMIUM = true;

// Renders children only in the premium edition. Wrap every premium-only block in
// this so the regular edition is one flag-flip away, with no dangling markup.
function PremiumGate({ children }) {
  return MUSIX_PREMIUM ? <>{children}</> : null;
}

const PREMIUM_GOLD = 'oklch(76% 0.13 85)';

// Small gold "★ PREMIUM" pill — the shared premium marker (header chip, upload
// block, indexing stage). Pure/presentational; gate it with <PremiumGate> at the
// call site when it should vanish in the regular edition.
function PremiumBadge({ label = 'PREMIUM', style }) {
  return (
    <span className="mono" style={{
      display:'inline-flex', alignItems:'center', gap:'4px',
      padding:'2px 8px', borderRadius:'999px',
      fontSize:'9px', fontWeight:'700', letterSpacing:'0.22em',
      color: PREMIUM_GOLD,
      border:`1px solid ${PREMIUM_GOLD.replace(')', ' / 0.4)')}`,
      background: PREMIUM_GOLD.replace(')', ' / 0.08)'),
      whiteSpace:'nowrap',
      ...style,
    }}>
      ★ {label}
    </span>
  );
}

// Lightweight hover hint: a small circled "i" that reveals a positioned bubble
// on hover/focus. Matches the frontend's existing quiet-hint idiom without a
// heavyweight tooltip lib.
function InfoTip({ isDark, text, size = 15, style }) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position:'relative', display:'inline-flex', ...style }}
      onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} tabIndex={0}>
      <span aria-hidden style={{
        width:size, height:size, borderRadius:'50%', flexShrink:0, cursor:'help',
        display:'inline-flex', alignItems:'center', justifyContent:'center',
        fontSize: size*0.62, fontStyle:'italic', fontWeight:700, lineHeight:1,
        color: isDark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.6)',
        background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.07)',
      }}>i</span>
      {open && (
        <span style={{
          position:'absolute', bottom:'calc(100% + 8px)', left:'50%', transform:'translateX(-50%)',
          width:'250px', maxWidth:'70vw', padding:'10px 12px', borderRadius:'10px', zIndex:60,
          background: isDark ? '#23232b' : '#ffffff',
          color: isDark ? '#d8dbe3' : '#333',
          border:`1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
          boxShadow:'0 10px 34px rgba(0,0,0,0.28)',
          fontSize:'12px', lineHeight:1.5, fontWeight:400, letterSpacing:'normal',
          textTransform:'none', textAlign:'left', pointerEvents:'none',
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

// Header chip under the MusiX wordmark — the persistent edition marker. Present
// throughout the premium build; gone in the regular one.
function PremiumMark({ style }) {
  return (
    <PremiumGate>
      <PremiumBadge style={{ marginTop:'4px', ...style }} />
    </PremiumGate>
  );
}

// Compact premium marker next to the Facts indexing stage: a PREMIUM badge plus
// an info hint explaining that, thanks to the premium edition, album art and
// song info are sourced at higher quality. Vanishes in the regular edition.
function PremiumMetaHint({ isDark, lang, style }) {
  const ru = lang === 'ru';
  return (
    <PremiumGate>
      <span style={{ display:'inline-flex', alignItems:'center', gap:'7px', ...style }}>
        <PremiumBadge />
        <InfoTip isDark={isDark} text={ru
          ? 'У вас премиум-версия MusiX: обложки альбомов и информация о песнях подбираются из источников повышенного качества.'
          : 'You have the premium MusiX edition: album art and song info are sourced at higher quality.'} />
      </span>
    </PremiumGate>
  );
}

// ─── Indexing progress (compact, flat chronological list) ────────────────────
// `premiumNote` — show the compact PremiumMetaHint (badge + hover tip) beside the
// Facts stage. Only first-run (onboarding) indexing sets it; repeat indexing
// stays clean. Also gated by MUSIX_PREMIUM via PremiumGate inside the hint.
function IndexingProgress({ stepStatus, stageProgress, lang, c, isDark, premiumNote = false }) {
  // Flat, chronological order matching the real pipeline: lyrics fetch + CLAP
  // (Звучание) start together at t=0, dense (Текстовый поиск) runs after lyrics,
  // facts overlap, analysis last. `metadata` (MusicBrainz) is skipped server-side
  // so it is not shown — covers are still read during the lyrics/tag pass. Same
  // key order as WIZ_STAGE_LABELS so every indexing flow agrees.
  const stages = [
    { key:'lyrics',   icon:'♪', labelRu:'Тексты песен',          labelEn:'Lyrics' },
    { key:'audio',    icon:'♫', labelRu:'Анализ звучания',        labelEn:'Sound analysis' },
    { key:'dense',    icon:'◆', labelRu:'Подготовка поиска',  labelEn:'Search setup' },
    { key:'facts',    icon:'★', labelRu:'Факты о треках',           labelEn:'Track facts' },
    { key:'analysis', icon:'∿', labelRu:'Похожие треки',   labelEn:'Similar tracks' },
  ];

  const CheckIcon = ({ size=16, color }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12l6 6L20 6"/>
    </svg>
  );

  useEffect(() => {
    let styleEl = document.getElementById('indexing-pulse-style');
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = 'indexing-pulse-style';
      document.head.appendChild(styleEl);
    }
    styleEl.textContent = '@keyframes indexing-pulse { 0%,100%{opacity:0.4;transform:scale(0.8)} 50%{opacity:1;transform:scale(1)} }';
  }, [isDark]);

  const renderSection = (header, stages) => (
    <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
      {header && (
        <div className="mono" style={{
          fontSize:'12px', color: c.textSubtle, letterSpacing:'0.22em',
          paddingBottom:'6px', marginBottom:'2px',
          borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)'}`,
        }}>
          {header}
        </div>
      )}
      {stages.map(s => {
        const status = stepStatus[s.key] || 'pending';
        const active = status === 'running';
        const done = status === 'done' || status === 'completed';
        const fail = status === 'failed';
        const partial = status === 'partial';
        const accent = done ? c.green : fail ? c.red : partial ? c.amber : active ? c.amber : c.textSubtle;
        const prog = stageProgress && stageProgress[s.key];
        const pct = prog && prog.total && prog.total > 0 ? Math.max(0, Math.min(100, ((prog.current || 0) / prog.total) * 100)) : 0;
        const etaText = prog && prog.eta != null && prog.eta > 0 ? ` · ETA ${Math.round(prog.eta)}s` : '';
        // Single-step stages (analysis, total=1) carry no meaningful "X/Y" — show
        // the working indicator only. Counter appears once total > 1.
        const labelText = prog && prog.total != null && prog.total > 1
          ? `${prog.current || 0}/${prog.total}`
          : null;

        // Icon background
        let iconBg;
        if (done) iconBg = c.greenBg;
        else if (fail) iconBg = c.redBg;
        else if (active) iconBg = `oklch(72% 0.13 75 / 0.12)`;
        else iconBg = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';

        return (
          <Fragment key={s.key}>
          <div style={{ display:'flex', alignItems:'flex-start', gap:'12px' }}>
            <div style={{
              width:'35px', height:'35px', borderRadius:'10px', flexShrink:0, marginTop:'2px',
              display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'17px', color: accent, fontFamily:"'JetBrains Mono', monospace",
              background: iconBg,
              boxShadow: isDark ? 'inset 0 1px 0 rgba(255,255,255,0.05)' : 'inset 0 1px 0 rgba(255,255,255,0.85)',
            }}>
              {done ? <CheckIcon size={18} color={accent} />
                : fail ? '×'
                : active ? (
                    <span style={{
                      width:'10px', height:'10px', borderRadius:'50%',
                      background: accent,
                      animation: 'indexing-pulse 1.2s ease-in-out infinite',
                    }} />
                  )
                : s.icon}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ display:'flex', alignItems:'center', gap:'10px', flexWrap:'wrap' }}>
                <span style={{ fontSize:'17px', color: c.text, fontWeight: active?'600':'500' }}>
                  {lang==='ru' ? s.labelRu : s.labelEn}
                </span>
                {/* Facts is the metadata-enrichment stage — mark the premium
                    quality boost right beside its label. */}
                {premiumNote && s.key === 'facts' && (
                  <PremiumMetaHint isDark={isDark} lang={lang} />
                )}
              </div>
              {(active || done || partial || fail) && (
                <div style={{ display:'flex', flexDirection:'column', gap:'5px', marginTop:'6px' }}>
                  {active && (
                    <div>
                      <div style={{
                        height:'5px', borderRadius:'3px', overflow:'hidden', marginBottom:'4px',
                        background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                      }}>
                        <div style={{
                          height:'100%', width:`${pct}%`, borderRadius:'3px',
                          background: `linear-gradient(90deg, oklch(60% 0.18 270), oklch(68% 0.2 290))`,
                          transition: 'width 0.5s ease-out',
                          boxShadow: `0 0 6px oklch(60% 0.18 270 / 0.4)`,
                        }} />
                      </div>
                      {labelText && (
                        <div className="mono" style={{ fontSize:'12px', color:c.textMuted, letterSpacing:'0.06em' }}>
                          {labelText}{etaText}
                          {prog && prog.found != null && prog.not_found != null && (
                            <span>
                              {' '}· <span style={{ color: c.green }}>{prog.found} {lang==='ru'?'найдено':'found'}</span>
                              {prog.not_found > 0 && <span>, {prog.not_found} <span style={{ color: c.amber }}>{lang==='ru'?'не найдено':'not found'}</span></span>}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                  {done && (
                    <div>
                      <div className="mono" style={{ fontSize:'12px', color:c.green, letterSpacing:'0.06em' }}>
                        {prog && prog.message ? prog.message : (lang==='ru'?'Готово':'Done')}
                      </div>
                      {prog && prog.found != null && prog.not_found != null && (
                        <div className="mono" style={{ fontSize:'12px', color:c.textMuted, letterSpacing:'0.06em', marginTop:'2px' }}>
                          <span style={{ color: c.green }}>{prog.found} {lang==='ru'?'найдено':'found'}</span>
                          {prog.not_found > 0 && <span>, {prog.not_found} <span style={{ color: c.amber }}>{lang==='ru'?'не найдено':'not found'}</span></span>}
                        </div>
                      )}
                    </div>
                  )}
                  {partial && (
                    <div>
                      <div className="mono" style={{ fontSize:'12px', color:c.amber, letterSpacing:'0.06em' }}>
                        {prog && prog.message ? prog.message : ''}
                      </div>
                      {prog && prog.found != null && prog.not_found != null && (
                        <div className="mono" style={{ fontSize:'12px', color:c.textMuted, letterSpacing:'0.06em', marginTop:'2px' }}>
                          <span style={{ color: c.green }}>{prog.found} {lang==='ru'?'найдено':'found'}</span>
                          {prog.not_found > 0 && <span>, {prog.not_found} <span style={{ color: c.amber }}>{lang==='ru'?'не найдено':'not found'}</span></span>}
                        </div>
                      )}
                    </div>
                  )}
                  {fail && (
                    <div className="mono" style={{ fontSize:'12px', color:c.red, letterSpacing:'0.06em' }}>
                      {prog && prog.message ? prog.message : (lang==='ru'?'Ошибка':'Error')}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          </Fragment>
        );
      })}
    </div>
  );

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
      {renderSection(null, stages)}
    </div>
  );
}

// ─── Shared indexing-job tracking (spec 2026-07-10-spa-routing, phase 2) ─────
// One SSE consumer for /index/progress/{job_id} with the stage-merge semantics
// SettingsPanel and OnboardingScreen used to duplicate inline: event-specific
// fields (current/total/eta/message) overlay the stage snapshot, and merged
// values never regress to undefined. Returns a close function.
function openIndexProgressStream(jobId, { onProgress, onComplete, onError }) {
  const evt = new EventSource(`${API}/index/progress/${jobId}`);
  const statusMap = { completed: 'done', failed: 'failed', running: 'running', pending: 'pending' };
  let stepStatus = {};
  let stageProgress = {};
  let closed = false;
  const close = () => { closed = true; evt.close(); };
  evt.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error && !data.stages) { close(); onError(String(data.error)); return; }  // e.g. 'Job not found'
      if (data.stages) {
        if (data.stage && data.stages[data.stage]) {
          const ev = data.stages[data.stage];
          if (data.current !== undefined) ev.current = data.current;
          if (data.total !== undefined) ev.total = data.total;
          if (data.eta_seconds !== undefined) ev.eta_seconds = data.eta_seconds;
          if (data.message !== undefined) ev.message = data.message;
        }
        stepStatus = { ...stepStatus };
        stageProgress = { ...stageProgress };
        for (const [key, stage] of Object.entries(data.stages)) {
          stepStatus[key] = (statusMap[stage.status] || stage.status) || 'pending';
          const prev = stageProgress[key] || {};
          stageProgress[key] = {
            current: stage.current ?? prev.current ?? 0,
            total: stage.total ?? prev.total ?? 0,
            eta: stage.eta_seconds ?? prev.eta ?? null,
            message: stage.message ?? prev.message ?? null,
            found: stage.found ?? prev.found ?? null,
            not_found: stage.not_found ?? prev.not_found ?? null,
          };
        }
      }
      if (data.overall_status === 'completed') {
        close();
        // Anything still pending when the job completes is implicitly done.
        stepStatus = Object.fromEntries(Object.entries(stepStatus).map(([k, v]) => [k, v === 'pending' ? 'done' : v]));
        onProgress({ stepStatus, stageProgress });
        onComplete(data.stages?.lyrics?.current || data.stages?.metadata?.current || 0);
      } else if (data.overall_status === 'failed') {
        close();
        onProgress({ stepStatus, stageProgress });
        onError(data.error || data.message || 'failed');
      } else if (data.stages) {
        onProgress({ stepStatus, stageProgress });
      }
    } catch {}
  };
  evt.onerror = () => { if (!closed) { close(); onError('connection_lost'); } };
  return close;
}

// App-level indexing job state. Owns the EventSource so tracking survives
// closing the settings panel and section navigation; App re-attaches after a
// full reload by asking GET /library/status (the server keeps per-account job
// state, and the SSE stream replays a full snapshot to late subscribers).
// `error` may hold the sentinel 'connection_lost' — consumers localize it.
function useIndexingJob({ onCompleted } = {}) {
  const [status, setStatus] = useState('idle');   // idle | running | completed | failed
  const [jobInfo, setJobInfo] = useState(null);   // { jobId, resumed } | null
  const [stepStatus, setStepStatus] = useState({});
  const [stageProgress, setStageProgress] = useState({});
  const [error, setError] = useState(null);
  const [trackCount, setTrackCount] = useState(null);
  const closeRef = useRef(null);
  const onCompletedRef = useRef(onCompleted);
  onCompletedRef.current = onCompleted;

  // Seed the 'starting' UI before the POST /library/index round-trip returns.
  const begin = useCallback(() => {
    closeRef.current?.();
    closeRef.current = null;
    setJobInfo(null);
    setStatus('running');
    setError(null); setTrackCount(null);
    setStepStatus({ lyrics:'idle', facts:'idle', metadata:'idle', dense:'idle', audio:'idle', analysis:'idle' });
    setStageProgress({});
  }, []);

  const attach = useCallback((jobId, { resumed = false } = {}) => {
    if (!jobId) return;
    closeRef.current?.();
    setJobInfo({ jobId, resumed });
    setStatus('running');
    if (resumed) { setError(null); setTrackCount(null); setStepStatus({}); setStageProgress({}); }
    closeRef.current = openIndexProgressStream(jobId, {
      onProgress: (p) => { setStepStatus(p.stepStatus); setStageProgress(p.stageProgress); },
      onComplete: (count) => { setStatus('completed'); setTrackCount(count); onCompletedRef.current?.(count); },
      onError: (msg) => { setStatus('failed'); setError(msg); },
    });
  }, []);

  // Immediate-completion path: POST /library/index answered without a job_id.
  const completeSync = useCallback((count) => {
    setStepStatus({ lyrics:'done', facts:'done', metadata:'done', dense:'done', audio:'done', analysis:'done' });
    setTrackCount(count); setStatus('completed');
    onCompletedRef.current?.(count);
  }, []);

  const fail = useCallback((message) => { setStatus('failed'); setError(message); }, []);

  const reset = useCallback(() => {
    closeRef.current?.(); closeRef.current = null;
    setJobInfo(null); setStatus('idle'); setError(null); setTrackCount(null);
    setStepStatus({}); setStageProgress({});
  }, []);

  useEffect(() => () => { closeRef.current?.(); }, []);

  return { status, jobInfo, stepStatus, stageProgress, error, trackCount, begin, attach, completeSync, fail, reset };
}

// Floating indicator for an indexing job running while its origin UI (settings
// panel / onboarding modal) is closed. Click reopens Settings with the staged
// modal. Percent mirrors JobTracker.get_progress_summary's stage weights.
function IndexingStatusPill({ isDark, lang, stepStatus, stageProgress, onClick }) {
  const c = useColors(isDark);
  const isMobile = useIsMobile();
  const weights = { lyrics: 0.25, facts: 0.10, metadata: 0.05, dense: 0.20, audio: 0.25, analysis: 0.15 };
  let pct = 0;
  for (const [key, w] of Object.entries(weights)) {
    const sp = stageProgress[key];
    if (stepStatus[key] === 'done') pct += w * 100;
    else if (stepStatus[key] === 'running' && sp?.total) pct += w * (sp.current / sp.total) * 100;
  }
  pct = Math.min(100, Math.round(pct));
  return (
    <button onClick={onClick} className="mono" style={{
      position: 'fixed', right: 18,
      // Mobile: clear the bottom tab bar + mini player stack.
      bottom: isMobile ? 'calc(env(safe-area-inset-bottom, 0px) + 132px)' : 18,
      zIndex: 80,
      display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', borderRadius: 999,
      border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)'}`,
      background: isDark ? 'rgba(20,20,28,0.85)' : 'rgba(255,255,255,0.9)',
      backdropFilter: 'blur(10px)', WebkitBackdropFilter: 'blur(10px)',
      color: c.text, fontSize: 12, letterSpacing: '0.08em', cursor: 'pointer',
      boxShadow: '0 6px 24px rgba(0,0,0,0.25)', animation: 'fadeInUp 0.3s ease',
    }}>
      <Spinner size={13} />
      {(lang === 'ru' ? 'ИНДЕКСАЦИЯ' : 'INDEXING')} · {pct}%
    </button>
  );
}

// ─── Indexing Modal ──────────────────────────────────────────────────────────
function IndexingModal({
  isDark, lang, collectionName, stepStatus, trackCount, errorMessage, onClose, stageProgress,
  phase, onAiConfirm, onAiSkip, onAiBootstrapRun, onAiBootstrapLater, aiStatus,
  premiumNote = false,
}) {
  const c = useColors(isDark);
  const allDone = Object.values(stepStatus).length > 0 && Object.values(stepStatus).every(s => s === 'done');

  if (phase === 'ai-setup') {
    return (
      <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose?.(); }}>
        <div className={ske('panel', isDark)} style={{ maxWidth: 540, padding: 28, borderRadius: 20, animation: 'scaleIn 0.3s cubic-bezier(.22,.9,.3,1)' }}>
          <h2 className="serif" style={{ fontSize: 24, marginTop: 0, marginBottom: 8 }}>
            {lang==='ru'?'Возможности ИИ для этой библиотеки?':'AI features for this library?'}
          </h2>
          <ul style={{ paddingLeft: 18, fontSize: 13, lineHeight: 1.6, color: isDark?'#bbb':'#444' }}>
            <li>{lang==='ru'?'Чат-поиск в естественном языке':'Natural-language chat search'}</li>
            <li>{lang==='ru'?'Sonic Vibe (атмосферные подписи треков)':'Sonic Vibe (mood phrases per track)'}</li>
            <li>{lang==='ru'?'Refined Facts, Artist Bio':'Refined Facts, Artist Bio'}</li>
            <li>{lang==='ru'?'Ask AI, объяснение строк lyrics':'Ask AI, lyric explain'}</li>
          </ul>
          <p style={{ fontSize: 12, color: isDark?'#888':'#666', marginTop: 12 }}>
            {lang==='ru'
              ? 'Требуется ИИ-ассистент (LM Studio, Ollama, OpenAI-совместимый API).'
              : 'Requires an AI assistant endpoint (LM Studio, Ollama, or OpenAI-compatible API).'}
          </p>
          {aiStatus?.aiAvailable === true && (
            <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8,
              background: 'rgba(40,160,80,0.10)', fontSize: 12 }}>
              ✅ {lang==='ru'?'ИИ-ассистент подключён':'AI assistant connected'}{aiStatus.llmInfo?.model ? ` · ${aiStatus.llmInfo.model}` : ''}
            </div>
          )}
          {aiStatus?.aiAvailable === false && (
            <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8,
              background: 'rgba(239,68,68,0.10)', fontSize: 12 }}>
              ⚠ {lang==='ru'?'ИИ-ассистент не отвечает.':'AI assistant not responding.'} {lang==='ru'?'Можно включить и настроить позже.':'You can enable and configure later.'}
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
            <button
              onClick={() => onAiConfirm?.(true)}
              style={{
                flex: 1, padding: '10px 16px', borderRadius: 10, fontSize: 14, fontWeight: 600,
                background: 'linear-gradient(135deg, oklch(67% 0.18 270), oklch(52% 0.22 285))',
                color: '#fff', border: 'none', cursor: 'pointer',
              }}
            >{lang==='ru'?'Включить AI':'Enable AI'}</button>
            <button
              onClick={() => onAiConfirm?.(false)}
              style={{
                flex: 1, padding: '10px 16px', borderRadius: 10, fontSize: 14, fontWeight: 500,
                background: 'transparent', color: isDark?'#aaa':'#555',
                border: `1px solid ${isDark?'rgba(255,255,255,0.18)':'rgba(0,0,0,0.18)'}`,
                cursor: 'pointer',
              }}
            >{lang==='ru'?'Без AI':'No AI'}</button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === 'ai-bootstrap') {
    return (
      <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onAiBootstrapLater?.(); }}>
        <div className={ske('panel', isDark)} style={{ maxWidth: 540, padding: 28, borderRadius: 20, animation: 'scaleIn 0.3s cubic-bezier(.22,.9,.3,1)' }}>
          <h2 className="serif" style={{ fontSize: 24, marginTop: 0, marginBottom: 8 }}>
            ✅ {lang==='ru'?'Подготовка библиотеки завершена':'Library ready'}
          </h2>
          <p style={{ fontSize: 13, color: isDark?'#bbb':'#444', lineHeight: 1.5 }}>
            {trackCount ? `${trackCount} ${lang==='ru'?'треков добавлено.':'tracks added.'}` : ''}
          </p>
          <p style={{ fontSize: 14, color: isDark?'#ddd':'#222', marginTop: 16 }}>
            {lang==='ru'?'Запустить дополнительные ИИ-функции?':'Run extra AI features?'}
          </p>
          <ul style={{ paddingLeft: 18, fontSize: 12, color: isDark?'#aaa':'#555', lineHeight: 1.6 }}>
            <li>{lang==='ru'?'Sonic Vibe — атмосферная фраза для каждого трека':'Sonic Vibe — mood phrase per track'}</li>
            <li>{lang==='ru'?'Refined Facts — отфильтрованные/сжатые факты':'Refined Facts — filtered/shortened facts'}</li>
            <li>{lang==='ru'?'Artist Bio — биография для каждого артиста':'Artist Bio — bio per artist'}</li>
          </ul>
          <p style={{ fontSize: 11, color: isDark?'#777':'#888', marginTop: 12 }}>
            {lang==='ru'?'Это можно сделать позже в Настройках → ИИ-обогащение.':'You can run this later in Settings → AI enrichment.'}
          </p>
          <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
            <button
              onClick={() => onAiBootstrapRun?.()}
              style={{
                flex: 1, padding: '10px 16px', borderRadius: 10, fontSize: 14, fontWeight: 600,
                background: 'linear-gradient(135deg, oklch(67% 0.18 270), oklch(52% 0.22 285))',
                color: '#fff', border: 'none', cursor: 'pointer',
              }}
            >{lang==='ru'?'Запустить сейчас':'Run now'}</button>
            <button
              onClick={() => onAiBootstrapLater?.()}
              style={{
                flex: 1, padding: '10px 16px', borderRadius: 10, fontSize: 14, fontWeight: 500,
                background: 'transparent', color: isDark?'#aaa':'#555',
                border: `1px solid ${isDark?'rgba(255,255,255,0.18)':'rgba(0,0,0,0.18)'}`,
                cursor: 'pointer',
              }}
            >{lang==='ru'?'Позже':'Later'}</button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === 'ai-running') {
    return (
      <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onAiBootstrapLater?.(); }}>
        <div className={ske('panel', isDark)} style={{ maxWidth: 540, padding: 28, borderRadius: 20, animation: 'scaleIn 0.3s cubic-bezier(.22,.9,.3,1)' }}>
          <h2 className="serif" style={{ fontSize: 24, marginTop: 0, marginBottom: 6 }}>
            ✨ {lang==='ru'?'ИИ-обогащение':'AI enrichment'}
          </h2>
          <p style={{ fontSize: 13, color: isDark?'#bbb':'#444', lineHeight: 1.5, marginBottom: 18 }}>
            {lang==='ru'
              ? 'Звучание песен, факты и биографии. Можно закрыть — обработка продолжится в фоне, прогресс виден в Настройках.'
              : 'Song vibes, facts and bios. You can close — it keeps running in the background; progress stays in Settings.'}
          </p>
          <AiEnrichProgress ru={lang==='ru'} c={c} />
          <button onClick={() => onAiBootstrapLater?.()} className="ske-accent" style={{
            marginTop: 10, padding: '11px 22px', borderRadius: 12,
            fontSize: 15, fontWeight: 600, letterSpacing: '0.05em',
          }}>
            {lang==='ru'?'ЗАКРЫТЬ':'CLOSE'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      position:'fixed', inset:0, zIndex:100,
      display:'flex', alignItems:'center', justifyContent:'center',
      background: isDark ? 'rgba(0,0,0,0.75)' : 'rgba(40,30,60,0.4)',
      backdropFilter:'blur(8px)',
      animation:'fadeIn 0.2s ease',
    }} onClick={(allDone || errorMessage) ? onClose : undefined}>
      <div className={ske('panel', isDark)} onClick={e => e.stopPropagation()}
        style={{
          width:'580px', maxWidth:'90vw', borderRadius:'25px', padding:'40px 38px',
          animation:'scaleIn 0.3s cubic-bezier(.22,.9,.3,1)',
        }}>
        <div className="mono" style={{ fontSize:'17px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'12px' }}>
          {lang==='ru'?'ДОБАВЛЕНИЕ МУЗЫКИ':'ADDING MUSIC'} · {collectionName}
        </div>
        <div className="serif" style={{ fontSize:'37px', lineHeight:'1', letterSpacing:'-0.02em', color:c.text, marginBottom:'30px' }}>
          {errorMessage
            ? <i style={{ color: c.red }}>{lang==='ru'?'Ошибка':'Failed'}</i>
            : allDone
              ? <>{lang==='ru'?'Готово':'Done'} <i style={{ color: c.green }}>✓</i></>
              : <>{lang==='ru'?'Идёт':'Working'} <i style={{ color: c.amber }}>…</i></>}
        </div>
        <ProcessingModeBadge isDark={isDark} lang={lang} style={{ marginBottom:'22px' }} />
        {errorMessage ? (
          <div style={{ padding:'15px 18px', borderRadius:'12px',
            background: c.redBg, color: c.red, fontSize:'17px', marginBottom:'22px' }}>
            {errorMessage}
          </div>
        ) : (
          <IndexingProgress stepStatus={stepStatus} stageProgress={stageProgress} lang={lang} c={c} isDark={isDark} premiumNote={premiumNote} />
        )}
        {trackCount != null && (
          <div className="mono" style={{ marginTop:'22px', fontSize:'18px', color: c.textMuted, letterSpacing:'0.1em' }}>
            {trackCount} {lang==='ru'?'треков обработано':'tracks processed'}
          </div>
        )}
        {(allDone || errorMessage) && (
          <button onClick={onClose} className="ske-accent" style={{
            marginTop:'28px', padding:'13px 25px', borderRadius:'12px',
            fontSize:'17px', fontWeight:'600', letterSpacing:'0.06em',
          }}>
            {lang==='ru'?'ЗАКРЫТЬ':'CLOSE'}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── AI Indexing card (inside SettingsPanel) ──────────────────────────────────
// Pipeline order: tasks run back-to-back so a local LLM serves one job at a time.
const AI_ENRICH_TASKS = ['sonic_vibe', 'refined_facts', 'artist_bio'];

function AIIndexingCard({ isDark, lang, aiStatus }) {
  const c = useColors(isDark);
  const [status, setStatus] = useState({ sonic_vibe: null, refined_facts: null, artist_bio: null });
  const [pipelineTask, setPipelineTask] = useState(null);  // task currently driven by the run-all pipeline
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const cancelRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const data = await apiFetch(`/library/ai-index/status`);
      setStatus(data);
      return data;
    } catch { return null; /* swallow — surface via error state only if Run/Reset fails */ }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      cancelRef.current = true;
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [refresh]);

  const isRunning = (s) => s && s.status === 'running';
  const someRunning = isRunning(status.sonic_vibe) || isRunning(status.refined_facts) || isRunning(status.artist_bio);

  useEffect(() => {
    if (someRunning && !pollRef.current) {
      pollRef.current = setInterval(refresh, 3000);
    } else if (!someRunning && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
      // One last refresh to capture the final state.
      refresh();
    }
  }, [someRunning, refresh]);

  // One-button pipeline: POST each task in order, then poll its status until
  // it leaves 'running' before starting the next. This replaces the old
  // per-task Run buttons whose busy flags reset as soon as the POST returned
  // (the job kept running server-side), leaving buttons in wrong states.
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const runAll = async () => {
    setError(null);
    try {
      for (const taskType of AI_ENRICH_TASKS) {
        if (cancelRef.current) return;
        setPipelineTask(taskType);
        await apiFetch(`/library/ai-index/${taskType}`, {
          method: 'POST',
          body: JSON.stringify({
            lang,
            llm_base_url: localStorage.getItem('llm_base_url') || undefined,
            llm_model:    localStorage.getItem('llm_model')    || undefined,
            ...(taskType === 'artist_bio' && { bio_source: 'web' }),
          }),
        });
        let s;
        do {
          await sleep(3000);
          if (cancelRef.current) return;
          const data = await refresh();
          s = data && data[taskType];
        } while (s && s.status === 'running');
      }
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      if (!cancelRef.current) setPipelineTask(null);
    }
  };

  const resetCache = async (taskType) => {
    const confirmMsg = lang === 'ru'
      ? 'Сбросить кэш для этой задачи?'
      : 'Reset cache for this task?';
    if (!window.confirm(confirmMsg)) return;
    setError(null);
    try {
      await apiFetch(
        `/library/ai-index/${taskType}/cache`,
        { method: 'DELETE' },
      );
      refresh();
    } catch (e) {
      setError(e?.message || String(e));
    }
  };

  const fmtStatus = (s) => {
    if (!s) return lang === 'ru' ? 'Никогда не запускалась' : 'Never run';
    const skipped = s.n_skipped || 0;
    const parts = [
      `${lang === 'ru' ? 'Статус' : 'Status'}: ${s.status}`,
      `${s.n_done}/${s.n_total} ${lang === 'ru' ? 'обработано' : 'processed'}`,
    ];
    if (skipped) {
      parts.push(`${skipped} ${lang === 'ru' ? 'пропущено' : 'skipped'}`);
    }
    if (s.n_failed) {
      parts.push(`${s.n_failed} ${lang === 'ru' ? 'ошибок' : 'failed'}`);
    }
    if (s.lang) parts.push(s.lang.toUpperCase());
    return parts.join(' · ');
  };

  // Detect the "completed with zero real work" case so we can surface a
  // distinct note instead of letting "done · 0/N processed · N skipped"
  // get lost in the muted-mono line.
  const isEmptyDone = (s) =>
    s && s.status === 'done' && (s.n_done || 0) === 0 && (s.n_skipped || 0) > 0;

  const skipReasonHint = (taskType) => {
    if (taskType === 'sonic_vibe') {
      return lang === 'ru'
        ? 'Нет входных данных: ни sonic_tags, ни фактов. Сначала запусти SONIC PROMPT-PROBING и/или обогащение фактами в библиотеке.'
        : 'No inputs available: neither sonic_tags nor song facts. Run sonic prompt-probing and/or facts enrichment in the library first.';
    }
    if (taskType === 'artist_bio') {
      return lang === 'ru'
        ? 'Веб-поиск не вернул результатов. Проверь подключение к интернету и настройки ИИ-ассистента.'
        : 'Web search returned no results. Check your internet connection and AI assistant settings.';
    }
    return lang === 'ru'
      ? 'Нет фактов о треках/артистах для уточнения. Сначала обогати библиотеку фактами.'
      : 'No song or artist facts to refine. Enrich your library with facts first.';
  };

  const taskTitle = (t) => ({
    sonic_vibe:    'Sonic Vibe',
    refined_facts: lang === 'ru' ? 'Уточнённые факты' : 'Refined facts',
    artist_bio:    lang === 'ru' ? 'Биографии артистов' : 'Artist bios',
  })[t];

  const busyAll = !!pipelineTask || someRunning;
  const canRun = !busyAll && aiStatus?.aiAvailable && aiStatus?.aiEnabledForCollection !== false;

  // Aggregate line across tasks that have ever run.
  const agg = AI_ENRICH_TASKS.reduce((acc, t) => {
    const s = status[t];
    if (s) { acc.ran += 1; acc.done += s.n_done || 0; acc.total += s.n_total || 0; acc.failed += s.n_failed || 0; }
    return acc;
  }, { ran: 0, done: 0, total: 0, failed: 0 });

  return (
    <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px solid ${c.border}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <div className="mono-label" style={{ color: c.textMuted }}>
          {lang === 'ru' ? 'ИИ-ОБОГАЩЕНИЕ' : 'AI ENRICHMENT'}
        </div>
        <div style={{ fontSize: 11, color: c.textSubtle }}>
          {lang === 'ru' ? 'Язык' : 'Language'}: {lang.toUpperCase()}
        </div>
      </div>

      {aiStatus?.aiEnabledForCollection === false && (
        <div style={{
          margin: '0 0 12px', padding: '9px 12px', borderRadius: 10,
          fontSize: 12, lineHeight: 1.5,
          background: isDark ? 'rgba(255,160,40,0.10)' : 'rgba(255,160,40,0.08)',
          border: '1px solid rgba(255,160,40,0.35)',
          color: isDark ? '#f4c08a' : '#a06010',
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <span style={{ flex: 1 }}>
            ⚠ {lang==='ru'
              ? 'ИИ отключён для этой библиотеки.'
              : 'AI is disabled for this library.'}
          </span>
          <button
            onClick={async () => {
              try {
                await apiFetch(`/library/ai-enabled`, {
                  method: 'PATCH', body: JSON.stringify({ enabled: true }),
                });
                aiStatus?.setAiEnabledForCollection?.(true);
              } catch (e) { console.error(e); }
            }}
            style={{
              padding: '5px 12px', borderRadius: 999, fontSize: 12, fontWeight: 600,
              background: 'rgba(255,160,40,0.20)', color: 'inherit',
              border: '1px solid rgba(255,160,40,0.50)', cursor: 'pointer',
            }}
          >{lang==='ru'?'Включить':'Enable'}</button>
        </div>
      )}

      <div style={{ fontSize: 12, color: c.textMuted, lineHeight: 1.55, marginBottom: 12 }}>
        {lang === 'ru'
          ? 'Sonic vibe, уточнённые факты и биографии артистов. Три задачи выполняются по очереди с вашим ИИ-ассистентом.'
          : 'Sonic vibe, refined facts and artist bios. The three tasks run back to back with your AI assistant.'}
      </div>

      <button className="cta-v3" disabled={!canRun} onClick={runAll}
        title={!aiStatus?.aiAvailable ? (lang==='ru'?'ИИ-ассистент не подключён':'Connect AI assistant to enable') : undefined}
        style={{ width: '100%', opacity: canRun ? 1 : 0.45, cursor: canRun ? 'pointer' : 'not-allowed' }}>
        {busyAll
          ? `⏳ ${pipelineTask ? taskTitle(pipelineTask) : (lang==='ru'?'Идёт обработка':'Processing')}…`
          : (lang === 'ru' ? '▶ Запустить обогащение' : '▶ Run enrichment')}
      </button>

      <div style={{
        marginTop: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
        fontSize: 11, color: c.textMuted, fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>
          {agg.ran === 0
            ? (lang === 'ru' ? 'Ещё не запускалось' : 'Never run')
            : `${agg.done}/${agg.total} ${lang==='ru'?'обработано':'processed'}${agg.failed ? ` · ${agg.failed} ${lang==='ru'?'ошибок':'failed'}` : ''}`}
        </span>
        <button className="pill-v3" onClick={() => setDetailsOpen(o => !o)}
          style={{ padding: '4px 12px', fontSize: 11 }}>
          {lang === 'ru' ? 'Детали' : 'Details'} {detailsOpen ? '▴' : '▾'}
        </button>
      </div>

      {detailsOpen && AI_ENRICH_TASKS.map(t => {
        const s = status[t];
        return (
          <div key={t} style={{ padding: '10px 0 0', marginTop: 10, borderTop: `1px solid ${c.border}` }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
              <div style={{ fontWeight: 600, color: c.text, fontSize: 13 }}>
                {taskTitle(t)}
                {(pipelineTask === t || isRunning(s)) && <span style={{ color: c.textMuted, fontWeight: 400 }}> · {lang==='ru'?'идёт':'running'}…</span>}
              </div>
              <button onClick={() => resetCache(t)} disabled={busyAll}
                title={lang==='ru'?'Сбросить кэш задачи':'Reset task cache'}
                style={{
                  padding: '3px 10px', borderRadius: 999, fontSize: 11,
                  background: 'transparent', color: c.textMuted,
                  border: `1px solid ${c.border}`,
                  cursor: (busyAll) ? 'not-allowed' : 'pointer',
                  opacity: (busyAll) ? 0.45 : 1,
                }}>
                ↻ {lang === 'ru' ? 'Сброс' : 'Reset'}
              </button>
            </div>
            <div style={{ marginTop: 4, fontSize: 11, color: c.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>
              {fmtStatus(s)}
            </div>
            {isEmptyDone(s) && (
              <div style={{ marginTop: 4, fontSize: 11, lineHeight: 1.5, color: 'oklch(70% 0.13 75)' }}>
                ⚠ {lang === 'ru'
                  ? 'Завершено без реальной работы — все треки пропущены.'
                  : 'Completed with no real work — every track was skipped.'}
                <br />{skipReasonHint(t)}
              </div>
            )}
            {s && s.error && (
              <div style={{ marginTop: 4, fontSize: 11, lineHeight: 1.5, color: 'oklch(62% 0.18 25)' }}>
                ✗ {s.error}
              </div>
            )}
          </div>
        );
      })}

      {detailsOpen && (
        <div style={{ fontSize: 11, color: c.textSubtle, marginTop: 10, lineHeight: 1.5 }}>
          {lang === 'ru'
            ? 'Результат генерируется на текущем языке интерфейса. Смени язык и запусти повторно для другого.'
            : 'Output is generated in the current UI language. Switch language and re-run for another.'}
        </div>
      )}

      {error && (
        <div style={{ marginTop: 10, fontSize: 12, color: 'oklch(62% 0.18 25)' }}>
          {error}
        </div>
      )}
    </div>
  );
}

// ─── Settings overlay (slides in from right) ─────────────────────────────────

// Full registration URL for an invite code. The register form parses
// `#/register?invite=<12 chars>` from the hash, so a member can just open this.
function inviteLink(code) {
  return `${window.location.origin}/#/register?invite=${code}`;
}

// Human-friendly listening duration. No tech jargon — just hours/minutes.
function fmtListenDuration(sec, ru) {
  const s = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return ru ? `${h} ч ${m} мин` : `${h}h ${m}m`;
  if (m > 0) return ru ? `${m} мин` : `${m} min`;
  return ru ? `${s} сек` : `${s} sec`;
}

// Copy text to the clipboard, robust across browsing contexts. The async
// Clipboard API (navigator.clipboard) exists ONLY in a SECURE context — HTTPS or
// http://localhost — so it is undefined when the app is opened over plain HTTP at
// a LAN IP (e.g. http://192.168.0.168:8000), exactly how a self-hosted server is
// reached before a domain/HTTPS exists. Fall back to a hidden <textarea> +
// execCommand('copy'), which still works on non-secure origins. Returns a
// Promise<boolean> so callers can tell the user the truth (and offer the raw
// link) when copying genuinely isn't possible.
function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).then(() => true).catch(() => _legacyCopy(text));
  }
  return Promise.resolve(_legacyCopy(text));
}
function _legacyCopy(text) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-9999px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch (e) {
    return false;
  }
}

function InvitesPanel({ lang, showToast, hideHeader, reloadKey }) {
  const [invites, setInvites] = useState([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  const reload = useCallback(() => {
    setLoading(true);
    apiFetch('/auth/invites')
      .then(setInvites)
      .catch(err => showToast(String(err.message || err)))
      .finally(() => setLoading(false));
  }, [showToast]);

  useEffect(() => { reload(); }, [reload, reloadKey]);

  const copyLink = async (iv) => {
    // Prefer the server-built link (PUBLIC_BASE_URL / LAN-IP — works behind a
    // reverse proxy); fall back to the browser origin for older servers.
    const url = (iv && iv.link) || inviteLink(iv.code);
    const ok = await copyTextToClipboard(url);
    showToast(ok
      ? (lang === 'ru' ? 'Ссылка-приглашение скопирована' : 'Invite link copied')
      : (lang === 'ru' ? `Скопируйте ссылку вручную: ${url}` : `Copy the link manually: ${url}`));
  };

  const onCreate = async () => {
    setCreating(true);
    try {
      const inv = await apiFetch('/auth/invites', { method: 'POST' });
      copyLink(inv);   // copy the full link, not just the bare code
      reload();
    } catch (err) {
      showToast(String(err.message || err));
    } finally {
      setCreating(false);
    }
  };

  const onRevoke = async (code) => {
    try {
      await apiFetch(`/auth/invites/${encodeURIComponent(code)}`, { method: 'DELETE' });
      reload();
    } catch (err) {
      showToast(String(err.message || err));
    }
  };

  return (
    <div style={hideHeader ? { paddingTop: 2 } : { padding: '16px 0', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        {hideHeader ? <span /> : (
          <span className="mono" style={{ fontSize: 11, letterSpacing: '0.18em', color: 'rgba(238,238,243,.5)' }}>
            {lang === 'ru' ? 'ИНВАЙТЫ' : 'INVITES'}
          </span>
        )}
        <button onClick={onCreate} disabled={creating}
          style={{
            padding: '5px 12px', borderRadius: 8, fontSize: 12,
            background: 'rgba(124,91,255,0.2)', color: '#bba8ff',
            border: '1px solid rgba(124,91,255,0.3)',
          }}>
          {creating ? '…' : (lang === 'ru' ? '＋ Новый' : '＋ New')}
        </button>
      </div>
      {loading && <div style={{ fontSize: 12, color: '#666' }}>{lang === 'ru' ? 'Загрузка…' : 'Loading…'}</div>}
      {!loading && invites.length === 0 && (
        <div style={{ fontSize: 12, color: '#666' }}>
          {lang === 'ru' ? 'Открытых инвайтов нет' : 'No open invites'}
        </div>
      )}
      {invites.map(iv => {
        const expiresIn = Math.max(0, Math.floor((iv.expires_at * 1000 - Date.now()) / 86400000));
        return (
          <div key={iv.code} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.03)',
          }}>
            <div>
              <div className="mono" style={{ fontSize: 13, color: '#eee', letterSpacing: '0.05em' }}>{iv.code}</div>
              <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>
                {lang === 'ru' ? `истекает через ${expiresIn} дн.` : `expires in ${expiresIn} d`}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
              <button onClick={() => copyLink(iv)}
                style={{
                  fontSize: 11, padding: '4px 10px', borderRadius: 6,
                  background: 'rgba(124,91,255,0.12)', color: '#bba8ff',
                  border: '1px solid rgba(124,91,255,0.22)',
                }}>
                {lang === 'ru' ? 'Ссылка' : 'Link'}
              </button>
              <button onClick={() => onRevoke(iv.code)}
                style={{
                  fontSize: 11, padding: '4px 10px', borderRadius: 6,
                  background: 'transparent', color: '#888',
                  border: '1px solid rgba(255,255,255,0.06)',
                }}>
                {lang === 'ru' ? 'Удалить' : 'Revoke'}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Owner-only roster (server mode). Presentational: members + loading come from
// the parent (OwnerAdminDashboard owns the fetch so the stats summary can read
// the same data). Per-row stats + a guarded delete flow live here.
function MembersPanel({ members, loading, onReload, lang, showToast, hideHeader }) {
  const ru = lang === 'ru';
  const [confirmId, setConfirmId] = useState('');   // member row showing the confirm box
  const [confirmText, setConfirmText] = useState('');
  const [deletingId, setDeletingId] = useState('');

  const startConfirm = (id) => { setConfirmId(id); setConfirmText(''); };
  const cancelConfirm = () => { setConfirmId(''); setConfirmText(''); };

  const doDelete = async (m) => {
    if (deletingId) return;
    setDeletingId(m.id);
    try {
      await apiFetch(`/admin/accounts/${encodeURIComponent(m.id)}`, {
        method: 'DELETE',
        body: JSON.stringify({ confirm_email: confirmText.trim() }),
      });
      showToast(ru ? `Аккаунт ${m.email} удалён` : `Account ${m.email} deleted`);
      cancelConfirm();
      onReload && onReload();
    } catch (err) {
      showToast(String(err.message || err));
    } finally {
      setDeletingId('');
    }
  };

  return (
    <div style={hideHeader ? { paddingTop: 2 } : { padding: '16px 0', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      {!hideHeader && (
        <div className="mono" style={{ fontSize: 11, letterSpacing: '0.18em', color: 'rgba(238,238,243,.5)', marginBottom: 10 }}>
          {ru ? 'УЧАСТНИКИ' : 'MEMBERS'}{members.length > 0 ? ` · ${members.length}` : ''}
        </div>
      )}
      {loading && <div style={{ fontSize: 12, color: '#666' }}>{ru ? 'Загрузка…' : 'Loading…'}</div>}
      {!loading && members.length === 0 && (
        <div style={{ fontSize: 12, color: '#666' }}>{ru ? 'Пока только вы' : 'Just you so far'}</div>
      )}
      {members.map(m => {
        const joined = m.created_at ? new Date(m.created_at * 1000).toLocaleDateString() : '';
        const isOwner = m.role === 'owner';
        const confirming = confirmId === m.id;
        const matches = confirmText.trim().toLowerCase() === (m.email || '').toLowerCase();
        return (
          <div key={m.id} style={{ padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, color: '#eee', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.email}</div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>
                  {isOwner
                    ? (ru ? 'владелец' : 'owner')
                    : (ru ? `участник · с ${joined}` : `member · since ${joined}`)}
                </div>
                <div style={{ fontSize: 11, color: '#8a8a93', marginTop: 4, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  <span>♪ {m.songs || 0}</span>
                  <span>⏱ {fmtListenDuration(m.listened_sec, ru)}</span>
                  <span>♥ {m.likes || 0}</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
                {m.invite_code && (
                  <span className="mono" style={{ fontSize: 10, color: '#777' }}>{m.invite_code}</span>
                )}
                {!isOwner && !confirming && (
                  <button onClick={() => startConfirm(m.id)}
                    style={{ fontSize: 11, padding: '4px 10px', borderRadius: 6, background: 'transparent',
                      color: '#ff9090', border: '1px solid rgba(255,80,80,0.25)', cursor: 'pointer' }}>
                    {ru ? 'Удалить' : 'Delete'}
                  </button>
                )}
              </div>
            </div>
            {confirming && (
              <div style={{ marginTop: 8, padding: '10px 12px', borderRadius: 8,
                background: 'rgba(255,80,80,0.07)', border: '1px solid rgba(255,80,80,0.2)' }}>
                <div style={{ fontSize: 11.5, color: '#ffb0b0', lineHeight: 1.5, marginBottom: 8 }}>
                  {ru
                    ? 'Удалит всё безвозвратно: песни, прослушивания, лайки, загруженные файлы. Введите email для подтверждения:'
                    : 'Permanently removes everything: songs, plays, likes, uploaded files. Type the email to confirm:'}
                </div>
                <input value={confirmText} onChange={e => setConfirmText(e.target.value)}
                  placeholder={m.email} autoFocus
                  style={{ width: '100%', padding: '7px 10px', borderRadius: 7, fontSize: 12, boxSizing: 'border-box',
                    border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(0,0,0,0.3)', color: '#eee', outline: 'none', marginBottom: 8 }} />
                <div style={{ display: 'flex', gap: 8 }}>
                  <button disabled={!matches || deletingId === m.id} onClick={() => doDelete(m)}
                    style={{ fontSize: 12, padding: '6px 14px', borderRadius: 7, border: 'none',
                      cursor: (matches && deletingId !== m.id) ? 'pointer' : 'not-allowed',
                      background: matches ? 'oklch(58% 0.21 25)' : 'rgba(255,80,80,0.15)',
                      color: matches ? '#fff' : '#ff9090', opacity: deletingId === m.id ? 0.6 : 1 }}>
                    {deletingId === m.id ? (ru ? 'Удаление…' : 'Deleting…') : (ru ? 'Удалить навсегда' : 'Delete permanently')}
                  </button>
                  <button onClick={cancelConfirm}
                    style={{ fontSize: 12, padding: '6px 14px', borderRadius: 7, background: 'transparent',
                      color: '#999', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}>
                    {ru ? 'Отмена' : 'Cancel'}
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// Embedding tiers for the new-collection form — a two-stop slider replaces the
// raw model dropdown. Light = backend default (CPU-friendly), heavy = higher
// quality but really wants a GPU. Names must match ModelRegistry.TEXT_MODELS.
const HEAVY_TEXT_MODEL = 'Qwen/Qwen3-Embedding-0.6B';

// ─── Instance AI settings (owner) ─────────────────────────────────────────────
// Standalone editor for the server-side AI/embedding policy. Reads the
// authoritative resolver view from GET /instance/settings (key arrives masked:
// value=null + has_value), writes via PATCH. A blank API-key field leaves the
// stored secret untouched. Reused by OwnerAdminDashboard; kept separate from
// SettingsPanel's localStorage-coupled LLM editor (which still serves sharing).
function InstanceAISettings({ isDark, lang, showToast }) {
  const c = useColors(isDark);
  const ru = lang === 'ru';
  const [loaded, setLoaded] = useState(false);
  const [loadErr, setLoadErr] = useState(null);
  const [baseUrl, setBaseUrl] = useState('');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [hasKey, setHasKey] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [tier, setTier] = useState(0);   // index into WIZ_TIERS (Speed/Balance/Quality)
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let alive = true;
    apiFetch('/instance/settings').then(res => {
      if (!alive) return;
      const s = res.settings || {};
      const sval = (k) => (s[k] && s[k].value != null ? String(s[k].value) : '');
      const sbool = (k, dflt) => {
        const v = s[k] && s[k].value;
        if (v == null) return dflt;
        // Backend emits bool settings as '1'/'0' (not 'true'/'false'), so accept
        // the common truthy spellings — otherwise a saved-ON toggle reads OFF.
        return ['1', 'true', 'yes', 'on'].includes(String(v).trim().toLowerCase());
      };
      setBaseUrl(sval('LLM_BASE_URL'));
      setModel(sval('LLM_MODEL'));
      setHasKey(!!(s['LLM_API_KEY'] && s['LLM_API_KEY'].has_value));
      setAiEnabled(sbool('AI_ENABLED', false));
      const em = sval('EMBED_MODEL');
      const ti = WIZ_TIERS.findIndex(t => (t.model || '') === em);
      setTier(ti >= 0 ? ti : 0);
      setLoaded(true);
    }).catch(e => { if (alive) { setLoadErr(String(e.message || e)); setLoaded(true); } });
    return () => { alive = false; };
  }, []);

  const save = async () => {
    if (busy) return;
    setBusy(true);
    try {
      // Send the full policy; omit llm_api_key when blank so the secret is kept.
      // embed_model=null on the light tier clears the override (resolver default).
      const body = {
        llm_base_url: baseUrl.trim() || null,
        llm_model: model.trim() || null,
        embed_model: WIZ_TIERS[tier].model,
        clap_enabled: true,
        ai_enabled: aiEnabled,
        ...(apiKey.trim() ? { llm_api_key: apiKey.trim() } : {}),
      };
      const res = await apiFetch('/instance/settings', { method: 'PATCH', body: JSON.stringify(body) });
      const s = res.settings || {};
      setHasKey(!!(s['LLM_API_KEY'] && s['LLM_API_KEY'].has_value));
      setApiKey('');
      setSaved(true); setTimeout(() => setSaved(false), 2000);
    } catch (e) { showToast && showToast(String(e.message || e)); }
    finally { setBusy(false); }
  };

  const label = (t) => <div className="mono-label" style={{ color: c.textSubtle, marginBottom: 7 }}>{t}</div>;
  const inputStyle = {
    width: '100%', padding: '10px 13px', borderRadius: 10,
    border: `1px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.10)'}`,
    background: isDark ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.75)',
    color: c.text, fontSize: 14, outline: 'none', fontFamily: "'JetBrains Mono', monospace",
  };

  if (!loaded) return <div style={{ padding: '18px 0' }}><Spinner size={16} /></div>;

  return (
    <div>
      {loadErr && (
        <div style={{ padding: '9px 13px', marginBottom: 14, borderRadius: 10, background: c.redBg, color: c.red, fontSize: 13 }}>
          {loadErr}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13.5, fontWeight: 600, color: c.text }}>{ru ? 'Гуру (AI) — для всех участников' : 'Guru (AI) — for all members'}</span>
        <div style={{ display: 'inline-flex', borderRadius: 10, overflow: 'hidden', boxShadow: `inset 0 0 0 1px ${c.border}`, flexShrink: 0 }}>
          <button type="button" onClick={() => setAiEnabled(true)}
            style={{ padding: '7px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: 'pointer', transition: 'background .15s',
              background: aiEnabled ? 'oklch(63% 0.17 142)' : 'transparent', color: aiEnabled ? '#fff' : c.textMuted }}>
            {ru ? 'Вкл' : 'On'}
          </button>
          <button type="button" onClick={() => setAiEnabled(false)}
            style={{ padding: '7px 16px', fontSize: 13, fontWeight: 600, border: 'none', cursor: 'pointer', transition: 'background .15s',
              background: !aiEnabled ? 'oklch(58% 0.21 25)' : 'transparent', color: !aiEnabled ? '#fff' : c.textMuted }}>
            {ru ? 'Выкл' : 'Off'}
          </button>
        </div>
      </div>

      {aiEnabled ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12, marginBottom: 4 }}>
          <div style={{ gridColumn: '1 / -1' }}>
            {label('BASE URL')}
            <input value={baseUrl} onChange={e => setBaseUrl(e.target.value)}
              placeholder="http://localhost:1234/v1" style={inputStyle} />
          </div>
          <div>
            {label(ru ? 'МОДЕЛЬ' : 'MODEL')}
            <input value={model} onChange={e => setModel(e.target.value)}
              placeholder="gpt-4o-mini" style={inputStyle} />
          </div>
          <div>
            {label(ru ? 'API-КЛЮЧ · ПУСТО = НЕ МЕНЯТЬ' : 'API KEY · BLANK = KEEP')}
            <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
              placeholder={hasKey ? (ru ? 'ключ задан · ••••••' : 'key set · ••••••') : (ru ? 'обычно не нужен локально' : 'usually not needed locally')}
              style={inputStyle} />
          </div>
          <div className="mono-label" style={{ gridColumn: '1 / -1', color: c.textSubtle, letterSpacing: '0.14em' }}>
            LM STUDIO · OLLAMA · OPENAI
          </div>
        </div>
      ) : (
        <div style={{ fontSize: 12.5, color: c.textMuted, lineHeight: 1.5, marginBottom: 4 }}>
          {ru ? 'Чат о треках, биографии артистов, звуковые описания песен. Нужен OpenAI-совместимый сервер.'
              : 'Track chat, artist bios, sonic descriptions. Needs an OpenAI-compatible server.'}
        </div>
      )}

      <div style={{ height: 1, background: c.border, margin: '16px 0' }} />

      {label(ru ? 'КАЧЕСТВО ОБРАБОТКИ ТЕКСТОВ ПЕСЕН' : 'LYRICS-SEARCH QUALITY')}
      <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
        {WIZ_TIERS.map((t, i) => (
          <button key={i} type="button" disabled={busy}
            onClick={() => !busy && setTier(i)}
            className={`pill-v3${tier === i ? ' pill-v3-active' : ''}`}
            style={{ flex: 1, padding: '10px 6px', fontSize: 13, fontWeight: 600, cursor: busy ? 'not-allowed' : 'pointer' }}>
            {ru ? t.ru : t.en}
          </button>
        ))}
      </div>
      <div style={{ fontSize: 11, color: c.textMuted, lineHeight: 1.5, margin: '8px 2px 16px' }}>
        {ru
          ? 'Точность поиска по словам внутри песен. Применяется к новым библиотекам участников; уже добавленные не меняются.'
          : "How precisely members search words inside songs. Applied to new libraries; already-added libraries are unchanged."}
      </div>

      <button onClick={save} disabled={busy} className="cta-v3" style={{
        padding: '10px 20px', fontSize: 13,
        opacity: busy ? 0.55 : 1, cursor: busy ? 'not-allowed' : 'pointer',
        background: saved ? 'linear-gradient(180deg, oklch(63% 0.17 142), oklch(53% 0.18 138))' : undefined,
        boxShadow: saved ? 'inset 0 1px 0 rgba(255,255,255,0.25), 0 3px 10px oklch(63% 0.17 142 / 0.4)' : undefined,
      }}>
        {busy ? (ru ? 'Сохранение…' : 'Saving…') : saved ? (ru ? '✓ Сохранено' : '✓ Saved') : (ru ? 'Сохранить' : 'Save')}
      </button>
    </div>
  );
}

// ─── Owner admin dashboard (server mode) ──────────────────────────────────────
// Mounted by Root INSTEAD of <App> for the server-mode owner. The owner account
// runs the instance (AI policy, invites, members) and does not listen to music —
// "Log out" returns to LoginScreen to sign in as a regular member. Lives outside
// <App>, so it owns its own theme/lang state (mirrors SetupWizard).
function OwnerAdminDashboard({ onLogout }) {
  const [isDark, setDark] = useState(() => (localStorage.getItem('musix_theme') || 'dark') === 'dark');
  const [lang, setLang]   = useState(() => localStorage.getItem('musix_lang') || 'ru');
  const c = useColors(isDark);
  const ru = lang === 'ru';
  const [toast, setToast] = useState('');
  const showToast = (m) => { setToast(String(m)); setTimeout(() => setToast(''), 3200); };
  const handleTheme = () => setDark(d => { const n = !d; localStorage.setItem('musix_theme', n ? 'dark' : 'light'); return n; });
  const toggleLang = () => setLang(l => { const n = l === 'ru' ? 'en' : 'ru'; localStorage.setItem('musix_lang', n); return n; });
  const email = localStorage.getItem('musix_user_email') || '';
  const [invBump, setInvBump] = useState(0);   // bump → InvitesPanel reloads after a callout-created invite
  const [calloutBusy, setCalloutBusy] = useState(false);

  // Members are fetched here (not inside MembersPanel) so the stats-summary card
  // and the roster share one source. Reload via a bump counter — depending on a
  // (re-created-every-render) showToast in the effect would loop forever.
  const [members, setMembers] = useState([]);
  const [membersLoading, setMembersLoading] = useState(false);
  const [membersBump, setMembersBump] = useState(0);
  const reloadMembers = () => setMembersBump(b => b + 1);
  useEffect(() => {
    let alive = true;
    setMembersLoading(true);
    apiFetch('/admin/members')
      .then(d => { if (alive) setMembers(d); })
      .catch(e => { if (alive) setToast(String(e.message || e)); })
      .finally(() => { if (alive) setMembersLoading(false); });
    return () => { alive = false; };
  }, [membersBump]);
  const totals = members.reduce((a, m) => ({
    songs: a.songs + (m.songs || 0),
    listened: a.listened + (m.listened_sec || 0),
    likes: a.likes + (m.likes || 0),
  }), { songs: 0, listened: 0, likes: 0 });
  const createInviteFromCallout = async () => {
    if (calloutBusy) return;
    setCalloutBusy(true);
    try {
      const inv = await apiFetch('/auth/invites', { method: 'POST' });
      const url = inv.link || inviteLink(inv.code);
      const ok = await copyTextToClipboard(url);
      showToast(ok
        ? (ru ? 'Приглашение создано — ссылка скопирована' : 'Invite created — link copied')
        : (ru ? `Приглашение создано. Скопируйте ссылку вручную: ${url}` : `Invite created. Copy the link manually: ${url}`));
      setInvBump(b => b + 1);
    } catch (e) { showToast(String(e.message || e)); }
    finally { setCalloutBusy(false); }
  };

  return (
    <div className="grain ob-root" style={{
      '--ob-glass-bg': isDark ? 'rgba(255,255,255,.055)' : 'rgba(255,255,255,.62)',
      '--ob-glass-sheen': isDark ? 'rgba(255,255,255,.12)' : 'rgba(255,255,255,.9)',
      '--ob-glass-edge': isDark ? 'rgba(255,255,255,.18)' : 'rgba(0,0,0,.10)',
      '--ob-blob1':'#7d5cff', '--ob-blob2':'#3aa0ff', '--ob-blob3':'#c061ff',
      width: '100vw', height: '100vh', overflow: 'auto', position: 'relative',
      background: isDark
        ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
        : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
      color: c.text,
    }}>
      <DriftBackdrop />
      <div style={{ position:'relative', zIndex:1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '24px 32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <BrandMark size={36} isDark={isDark} />
          <span className="serif" style={{ fontSize: 28, letterSpacing: '-0.02em' }}>Musi<i style={{ color: 'oklch(62% 0.2 275)' }}>X</i></span>
          <span className="mono" style={{
            fontSize: 10, letterSpacing: '0.2em', padding: '3px 8px', borderRadius: 6,
            background: 'rgba(124,91,255,0.18)', color: '#bba8ff', border: '1px solid rgba(124,91,255,0.3)',
          }}>ADMIN</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button className="pill-v3" onClick={handleTheme} style={{ width: 34, height: 34, padding: 0, borderRadius: '50%', display: 'grid', placeItems: 'center' }}>
            {isDark ? '☀' : '☾'}
          </button>
          <button className="pill-v3" onClick={toggleLang} style={{ padding: '6px 12px' }}>{ru ? 'EN' : 'RU'}</button>
          <button onClick={onLogout} style={{
            padding: '7px 16px', borderRadius: 9, fontSize: 13, cursor: 'pointer',
            background: 'rgba(255,80,80,0.12)', color: '#ff9090', border: '1px solid rgba(255,80,80,0.22)',
          }}>
            {ru ? 'Выйти' : 'Log out'}
          </button>
        </div>
      </div>

      <div style={{ position:'relative', zIndex:1, maxWidth: 1080, margin: '8px auto 64px', padding: '0 32px' }}>
        {/* status + owner line */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:12, flexWrap:'wrap', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, fontSize:13, color:'#bfeccd',
            background:'rgba(95,208,138,.1)', borderRadius:12, padding:'10px 15px', boxShadow:'inset 0 0 0 1px rgba(95,208,138,.3)' }}>
            ✓ {ru ? 'Сервер настроен. Приглашайте участников — каждый принесёт свою музыку.'
                  : 'Server is set up. Invite members — each brings their own music.'}
          </div>
          {email && (
            <div className="mono-label" style={{ color: c.textSubtle, letterSpacing: '0.14em' }}>
              {ru ? 'ВЛАДЕЛЕЦ' : 'OWNER'} · {email}
            </div>
          )}
        </div>

        {/* prominent listen callout */}
        <div className="ob-glass" style={{ padding: '18px 22px', marginBottom: 18,
          background:'linear-gradient(180deg,rgba(154,133,255,.16),rgba(154,133,255,.06))',
          boxShadow:'inset 0 0 0 1px rgba(154,133,255,.38),0 0 30px rgba(124,92,255,.2)' }}>
          <div className="mono" style={{ fontSize: 11, letterSpacing: '0.22em', textTransform: 'uppercase', color: '#cfc4ff' }}>
            {ru ? 'Хотите слушать музыку?' : 'Want to listen?'}
          </div>
          <div className="serif" style={{ fontSize: 22, margin: '8px 0 0', letterSpacing:'-0.01em' }}>
            {ru ? 'Админ-аккаунт — только для управления' : 'The admin account is for management only'}
          </div>
          <div style={{ fontSize: 13.5, color: c.textMuted, marginTop: 6, lineHeight: 1.5 }}>
            {ru ? 'Чтобы слушать свою музыку, создайте приглашение и войдите как обычный участник — собственным аккаунтом.'
                : 'To listen to your music, create an invite and sign in as a regular member with your own account.'}
          </div>
          <button onClick={createInviteFromCallout} disabled={calloutBusy} className="cta-v3"
            style={{ marginTop: 14, padding: '10px 18px', fontSize: 13, opacity: calloutBusy ? 0.6 : 1, cursor: calloutBusy ? 'wait' : 'pointer' }}>
            {calloutBusy ? (ru ? 'Создаём…' : 'Creating…') : (ru ? '＋ Создать приглашение и скопировать ссылку' : '＋ Create invite & copy link')}
          </button>
        </div>

        {/* instance stats summary — totals across all accounts */}
        <div className="ob-glass" style={{ padding: '16px 20px', marginBottom: 14 }}>
          <div className="mono" style={{ fontSize: 11, letterSpacing: '0.18em', color: 'rgba(238,238,243,.5)', marginBottom: 12 }}>
            {ru ? 'СВОДКА' : 'OVERVIEW'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            {[
              { label: ru ? 'Аккаунтов' : 'Accounts', value: members.length },
              { label: ru ? 'Всего песен' : 'Total songs', value: totals.songs },
              { label: ru ? 'Прослушано' : 'Listened', value: fmtListenDuration(totals.listened, ru) },
              { label: ru ? 'Лайков' : 'Likes', value: totals.likes },
            ].map((s, i) => (
              <div key={i} style={{ minWidth: 0 }}>
                <div className="serif" style={{ fontSize: 24, fontWeight: 600, letterSpacing: '-0.01em',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.value}</div>
                <div style={{ fontSize: 11.5, color: c.textMuted, marginTop: 2 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* system settings — always open, full width */}
        <div className="ob-glass" style={{ padding: '18px 22px', marginBottom: 14 }}>
          <div className="serif" style={{ fontSize: 17, fontWeight: 600, marginBottom: 16 }}>
            {ru ? 'Настройки системы' : 'System settings'}
          </div>
          <InstanceAISettings isDark={isDark} lang={lang} showToast={showToast} />
        </div>

        {/* members + invites side by side */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(2, 1fr)', gap:14, alignItems:'start' }}>
          <div className="ob-glass" style={{ padding: '18px 20px' }}>
            <div className="serif" style={{ fontSize: 17, fontWeight: 600, marginBottom: 4 }}>
              {ru ? 'Участники' : 'Members'}
            </div>
            <MembersPanel members={members} loading={membersLoading} onReload={reloadMembers}
              lang={lang} showToast={showToast} hideHeader />
          </div>

          <div className="ob-glass" style={{ padding: '18px 20px' }}>
            <div className="serif" style={{ fontSize: 17, fontWeight: 600, marginBottom: 4 }}>
              {ru ? 'Приглашения' : 'Invites'}
            </div>
            <InvitesPanel lang={lang} showToast={showToast} hideHeader reloadKey={invBump} />
          </div>
        </div>

      </div>

      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 200,
          padding: '11px 18px', borderRadius: 12, fontSize: 13, maxWidth: '90vw',
          background: isDark ? 'rgba(20,20,28,0.95)' : 'rgba(255,255,255,0.96)',
          color: c.text, border: `1px solid ${c.border}`, boxShadow: '0 8px 30px rgba(0,0,0,0.3)',
        }}>
          {toast}
        </div>
      )}
    </div>
  );
}

function SettingsPanel({ isDark, lang, onClose, onCollectionsUpdate, aiStatus, onTheme, onLang, collections, userPoints, onLogout, instanceMode, showToast, indexingJob }) {
  const c = useColors(isDark);
  const isMobile = useIsMobile();  // full-screen panel + tighter gutters on phones
  const [closing, setClosing] = useState(false);
  const requestClose = () => { setClosing(true); setTimeout(onClose, 260); };
  // 'light' | 'heavy' — derived from the legacy text_model localStorage key so
  // an earlier explicit heavy pick survives the UI change.
  const [modelTier, setModelTier] = useState(() =>
    localStorage.getItem('text_model') === HEAVY_TEXT_MODEL ? 'heavy' : 'light');
  const [collName, setCollName] = useState('');
  const [folderPath, setFolderPath] = useState('');
  const [betterLyrics, setBetterLyrics] = useState(false);
  const [refineMetadata, setRefineMetadata] = useState(false);
  // Progress state comes from the App-level indexingJob hook (spec phase 2):
  // the SSE subscription lives above this panel, so closing it or navigating
  // sections no longer loses the running job. Opening the panel while a job
  // is already running (started earlier / resumed after F5) surfaces the
  // staged modal immediately, past the AI-setup step.
  const indexing = indexingJob.status === 'running';
  const stepStatus = indexingJob.stepStatus;
  const stageProgress = indexingJob.stageProgress;
  const modalTrackCount = indexingJob.trackCount;
  const modalError = indexingJob.error === 'connection_lost'
    ? (lang === 'ru' ? 'Соединение потеряно' : 'Connection lost')
    : indexingJob.error;
  const [showModal, setShowModal] = useState(() => indexingJob.status === 'running');
  const [indexPhase, setIndexPhase] = useState(() =>  // 'ai-setup' | 'indexing' | 'ai-bootstrap' | 'ai-running'
    indexingJob.status === 'running' ? 'indexing' : 'ai-setup');
  const [enabledForNewCollection, setEnabledForNewCollection] = useState(true);
  // Which AI choice the CURRENT run was started with — read on completion.
  // A ref (not state): the completion effect below must see the value the run
  // began with even if the component re-rendered in between.
  const aiEnabledRef = useRef(true);

  // Job completion → phase transition (mirrors the old inline SSE handler).
  // Gated on showModal so a background job that finishes while this panel
  // shows only the form doesn't flip phases behind the scenes.
  const prevJobStatusRef = useRef(indexingJob.status);
  useEffect(() => {
    const prev = prevJobStatusRef.current;
    prevJobStatusRef.current = indexingJob.status;
    if (prev !== 'running' || indexingJob.status !== 'completed' || !showModal) return;
    // (collections refresh happens in useIndexingJob's onCompleted at App level)
    if (aiStatus?.aiAvailable && aiEnabledRef.current) setIndexPhase('ai-bootstrap');
    else setIndexPhase('indexing');  // stay in 'indexing' phase showing Done UI
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexingJob.status, showModal]);

  const [llmBaseUrl, setLlmBaseUrl] = useState(() => localStorage.getItem('llm_base_url') || '');
  const [llmModel, setLlmModel] = useState(() => localStorage.getItem('llm_model') || '');
  const [llmKey, setLlmKey] = useState('');   // API key — secret, never read from localStorage
  const [llmSaved, setLlmSaved] = useState(false);
  const isOwner = (localStorage.getItem('musix_user_role') || '') === 'owner';
  // In server mode the SettingsPanel is only ever reached by a member (the owner
  // is routed to OwnerAdminDashboard). Members must not configure the LLM — the
  // admin's instance settings already override every request arg server-side
  // (llm_client.resolve_*), so the per-member endpoint/model form is dead UI.
  const serverMember = instanceMode === 'server';

  // Section UI state: collection switcher list + LLM advanced fields.
  const [llmOpen, setLlmOpen] = useState(false);

  // Persist the heavy pick under the legacy key; remove it for light so the
  // backend falls back to its default model (never store the string "null").
  useEffect(() => {
    if (modelTier === 'heavy') localStorage.setItem('text_model', HEAVY_TEXT_MODEL);
    else localStorage.removeItem('text_model');
  }, [modelTier]);

  // `aiEnabledArg` is passed explicitly by the caller because React state
  // updates batched in the same tick won't have flushed yet — reading
  // `enabledForNewCollection` from the closure here would see the *previous*
  // render's value (stale), causing the post-indexing phase transition to
  // misroute (e.g. "Skip AI" would still land on ai-bootstrap).
  const startIndexing = async (aiEnabledArg = enabledForNewCollection) => {
    aiEnabledRef.current = aiEnabledArg;
    indexingJob.begin();
    try {
      // Only send text_model for the heavy tier — omit otherwise so backend
      // falls back to its default (the light model).
      const validModel = modelTier === 'heavy' ? HEAVY_TEXT_MODEL : undefined;
      const res = await apiFetch('/library/index', { method:'POST',
        body: JSON.stringify({ folder_path:folderPath, better_lyrics_quality:betterLyrics, text_model:validModel, enhance_by_musicbrainz:refineMetadata }) });
      if (res.status === 'failed') { indexingJob.fail(res.message); return; }
      if (!res.job_id) {
        // Immediate completion — the completion effect above handles the
        // phase transition and the collections refresh.
        indexingJob.completeSync(res.count || 0);
        return;
      }
      indexingJob.attach(res.job_id);
    } catch (e) { indexingJob.fail(e.message); }
  };

  // Save + probe in one step: trim, sync state with what we persist (so a
  // later save can't desync from what was tested), then re-probe the LLM.
  const saveLLM = async () => {
    const url = llmBaseUrl.trim(), model = llmModel.trim();
    setLlmBaseUrl(url); setLlmModel(model);
    localStorage.setItem('llm_base_url', url);
    localStorage.setItem('llm_model', model);
    // Owner: persist to instance settings so the choice is server-side (members
    // inherit, API key is storable) and authoritative over per-browser values.
    // Saving an LLM config implies the owner wants AI on.
    if (isOwner) {
      try {
        await apiFetch('/instance/settings', { method: 'PATCH', body: JSON.stringify({
          llm_base_url: url,
          llm_model: model,
          ...(llmKey.trim() ? { llm_api_key: llmKey.trim() } : {}),
          ai_enabled: true,
        }) });
        setLlmKey('');   // don't keep the secret in component state after save
      } catch (e) { showToast?.(String(e.message || e)); }
    }
    setLlmSaved(true); setTimeout(() => setLlmSaved(false), 2000);
    await aiStatus?.refresh?.();
  };

  const fieldLabel = (text) => (
    <div className="mono-label" style={{ color:c.textSubtle, marginBottom:7 }}>{text}</div>
  );
  const inputStyle = {
    width:'100%', minHeight:44, padding:'10px 13px', borderRadius:10,
    border:`1px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.10)'}`,
    background: isDark ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.75)',
    color: c.text, fontSize:14, outline:'none', fontFamily:"'JetBrains Mono', monospace",
  };

  return (
    <>
      <div onClick={requestClose} style={{
        position:'fixed', inset:0, zIndex:90,
        background: isDark ? 'rgba(0,0,0,0.5)' : 'rgba(40,30,60,0.25)',
        backdropFilter:'blur(4px)',
        animation: closing ? 'fadeOut 0.22s ease forwards' : 'fadeIn 0.2s ease',
      }} />
      <div className="grain" style={{
        position:'fixed', top:0, right:0, bottom:0, zIndex:91,
        width: isMobile ? '100vw' : 'min(540px, 92vw)', display:'flex', flexDirection:'column',
        borderRadius:0, overflow:'hidden',
        background: isDark
          ? 'linear-gradient(180deg, rgba(24,24,32,0.92) 0%, rgba(14,14,20,0.95) 100%)'
          : 'linear-gradient(180deg, rgba(250,249,253,0.94) 0%, rgba(238,237,245,0.96) 100%)',
        backdropFilter:'blur(28px) saturate(1.3)',
        WebkitBackdropFilter:'blur(28px) saturate(1.3)',
        borderLeft:`1px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'}`,
        boxShadow: isDark ? '-18px 0 50px rgba(0,0,0,0.5)' : '-18px 0 50px rgba(40,30,80,0.14)',
        animation: closing ? 'slideRightOut 0.26s cubic-bezier(.22,.9,.3,1) forwards' : 'slideRight 0.32s cubic-bezier(.22,.9,.3,1)',
      }}>
        <div style={{ padding:'24px clamp(16px,5vw,28px) 18px', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
          <div>
            <div className="mono-label" style={{ color:c.textSubtle, marginBottom:6 }}>
              {lang==='ru'?'НАСТРОЙКИ':'SETTINGS'}
            </div>
            <div className="serif" style={{ fontSize:'clamp(22px,6vw,30px)', lineHeight:'1', letterSpacing:'-0.02em', color:c.text }}>
              {lang==='ru'?'Студия':'Studio'} <i style={{ color:'oklch(62% 0.2 275)' }}>{lang==='ru'?'настроек':'tools'}</i>
            </div>
          </div>
          <button onClick={requestClose} className="pill-v3" style={{
            width:34, height:34, padding:0, borderRadius:'50%',
            display:'grid', placeItems:'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>

        <div style={{ flex:1, overflowY:'auto', padding:'18px clamp(16px,5vw,28px) 32px', display:'flex', flexDirection:'column', gap:18 }}>

          {/* ─── Account (Phase D — one account = one library; collection hidden) ─── */}
          <section className="panel-v3" style={{ padding:'18px 20px' }}>
            <div className="mono-label" style={{ color:c.textSubtle, marginBottom:14 }}>
              {lang==='ru'?'АККАУНТ':'ACCOUNT'}
            </div>
            <div style={{ display:'flex', alignItems:'center', gap:14 }}>
              <div style={{
                width:46, height:46, borderRadius:'50%', flexShrink:0,
                display:'grid', placeItems:'center',
                background:'linear-gradient(135deg, oklch(66% 0.19 280) 0%, oklch(50% 0.21 265) 100%)',
                boxShadow:'inset 0 1px 0 rgba(255,255,255,0.35), 0 4px 14px oklch(60% 0.18 270 / 0.35)',
              }}>
                <span className="serif-display" style={{ color:'#fff', fontSize:20, fontStyle:'normal' }}>
                  {(localStorage.getItem('musix_user_email') || '?')[0].toUpperCase()}
                </span>
              </div>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ fontSize:15, fontWeight:600, color:c.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                  {localStorage.getItem('musix_user_email') || (lang==='ru'?'не вошёл':'not logged in')}
                </div>
                <div className="mono-label" style={{ color:c.textMuted, marginTop:3, letterSpacing:'0.14em' }}>
                  {(localStorage.getItem('musix_user_role')||'').toUpperCase()} · {(userPoints ?? 0)} {lang==='ru'?'ТРЕКОВ':'TRACKS'}
                </div>
              </div>
              <button onClick={onLogout}
                style={{ padding:'6px 14px', borderRadius:8, fontSize:13, cursor:'pointer',
                  background:'rgba(255,80,80,0.12)', color:'#ff9090',
                  border:'1px solid rgba(255,80,80,0.2)' }}>
                {lang==='ru'?'Выйти':'Log out'}
              </button>
            </div>
          </section>

          {/* ─── Intelligence: LLM status + advanced fields + one-button enrichment ─── */}
          <section className="panel-v3" style={{ padding:'18px 20px' }}>
            <div className="mono-label" style={{ color:c.textSubtle, marginBottom:14 }}>
              {lang==='ru'?'ИНТЕЛЛЕКТ':'INTELLIGENCE'}
            </div>

            <div style={{ display:'flex', alignItems:'center', gap:10, fontSize:12, color:c.text }}>
              <span style={{
                display:'inline-block', width:8, height:8, borderRadius:'50%', flexShrink:0,
                background: aiStatus?.aiAvailable === true ? 'oklch(65% 0.18 145)'
                  : aiStatus?.aiAvailable === false ? 'oklch(60% 0.20 25)'
                  : 'oklch(72% 0.16 60)',
                boxShadow: aiStatus?.aiAvailable === true ? '0 0 8px oklch(65% 0.18 145 / 0.6)' : 'none',
              }} />
              <span style={{ flex:1, minWidth:0, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                {aiStatus?.aiAvailable === true && (lang === 'ru'
                  ? `ИИ-ассистент подключён${aiStatus.llmInfo?.model ? ` · ${aiStatus.llmInfo.model}` : ''}`
                  : `AI assistant connected${aiStatus.llmInfo?.model ? ` · ${aiStatus.llmInfo.model}` : ''}`)}
                {aiStatus?.aiAvailable === false && (lang === 'ru'
                  ? `ИИ-ассистент офлайн · ${aiStatus.llmError || 'не отвечает'}`
                  : `AI assistant offline · ${aiStatus.llmError || 'no response'}`)}
                {aiStatus?.aiAvailable === null && (lang === 'ru' ? 'Проверка ИИ-ассистента…' : 'Probing AI assistant…')}
              </span>
              <button className="pill-v3" style={{ padding:'4px 12px', fontSize:11 }}
                onClick={() => aiStatus?.refresh?.()}>
                ↻ {lang==='ru'?'Проверить':'Re-check'}
              </button>
              {!serverMember && (
                <button className={`pill-v3${llmOpen ? ' pill-v3-active' : ''}`} style={{ padding:'4px 12px', fontSize:11 }}
                  onClick={() => setLlmOpen(o => !o)}>
                  {lang==='ru'?'Настроить':'Configure'} {llmOpen ? '▴' : '▾'}
                </button>
              )}
            </div>

            {!serverMember && llmOpen && (
              <div style={{ marginTop:14 }}>
                {fieldLabel('BASE URL')}
                <input value={llmBaseUrl} onChange={e=>setLlmBaseUrl(e.target.value)}
                  placeholder="http://localhost:1234/v1" style={{ ...inputStyle, marginBottom:12 }} />
                {fieldLabel(lang==='ru'?'МОДЕЛЬ':'MODEL')}
                <input value={llmModel} onChange={e=>setLlmModel(e.target.value)}
                  placeholder="gpt-4o-mini" style={{ ...inputStyle, marginBottom:14 }} />
                {isOwner && (
                  <>
                    {fieldLabel(lang==='ru'?'API-КЛЮЧ · ОСТАВЬТЕ ПУСТЫМ, ЧТОБЫ НЕ МЕНЯТЬ':'API KEY · LEAVE BLANK TO KEEP')}
                    <input type="password" value={llmKey} onChange={e=>setLlmKey(e.target.value)}
                      placeholder={lang==='ru'?'для локального сервера обычно не нужен':'usually not needed for a local server'}
                      style={{ ...inputStyle, marginBottom:14 }} />
                  </>
                )}
                <div style={{ display:'flex', alignItems:'center', gap:12 }}>
                  <button onClick={saveLLM} className="cta-v3" style={{
                    padding:'9px 18px', fontSize:13,
                    background: llmSaved ? 'linear-gradient(180deg, oklch(63% 0.17 142), oklch(53% 0.18 138))' : undefined,
                    boxShadow: llmSaved ? 'inset 0 1px 0 rgba(255,255,255,0.25), 0 3px 10px oklch(63% 0.17 142 / 0.4)' : undefined,
                  }}>
                    {llmSaved ? (lang==='ru'?'✓ Сохранено':'✓ Saved') : (lang==='ru'?'Сохранить и проверить':'Save & test')}
                  </button>
                  <span className="mono-label" style={{ color:c.textSubtle, letterSpacing:'0.14em' }}>
                    LM STUDIO · OLLAMA · OPENAI
                  </span>
                </div>
              </div>
            )}

            <AIIndexingCard isDark={isDark} lang={lang} aiStatus={aiStatus} />
          </section>

          {/* ─── Appearance ─── */}
          <section className="panel-v3" style={{ padding:'18px 20px' }}>
            <div className="mono-label" style={{ color:c.textSubtle, marginBottom:14 }}>
              {lang === 'ru' ? 'ВНЕШНИЙ ВИД' : 'APPEARANCE'}
            </div>
            <div style={{ display:'flex', gap:8, alignItems:'center', flexWrap:'wrap' }}>
              <button className="pill-v3" onClick={onTheme}
                style={{ display:'flex', alignItems:'center', gap:8 }}>
                {isDark
                  ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                  : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>}
                <span>{lang === 'ru' ? (isDark ? 'Светлая тема' : 'Тёмная тема') : (isDark ? 'Light theme' : 'Dark theme')}</span>
              </button>
              <button className="pill-v3" onClick={() => onLang && onLang(lang === 'ru' ? 'en' : 'ru')}>
                {lang === 'ru' ? 'EN' : 'RU'}
              </button>
            </div>
          </section>

          {/* ─── About ─── */}
          <div style={{ padding:'6px 22px', textAlign:'center' }}>
            <div className="serif" style={{ fontSize:16, color:c.textMuted, fontStyle:'italic', lineHeight:1.4 }}>
              {lang==='ru'?'«Слушай смысл, не названия.»':'"Listen to the meaning, not the titles."'}
            </div>
            <div className="mono-label" style={{ color:c.textSubtle, marginTop:8 }}>
              MUSIX · STUDIO CONSOLE · v2
            </div>
          </div>
        </div>


        {showModal && (
          <IndexingModal isDark={isDark} lang={lang} collectionName={collName||'my_collection'}
            stepStatus={stepStatus} trackCount={modalTrackCount} errorMessage={modalError}
            onClose={() => { setShowModal(false); indexingJob.reset(); }}
            stageProgress={stageProgress}
            phase={indexPhase}
            aiStatus={aiStatus}
            onAiConfirm={async (enabled) => {
              setEnabledForNewCollection(enabled);
              const colName = collName.trim() || 'my_collection';
              try {
                await apiFetch(`/library/ai-enabled`, {
                  method: 'PATCH', body: JSON.stringify({ enabled }),
                });
              } catch (e) { console.error('failed to persist ai_enabled', e); }
              if (enabled) {
                localStorage.setItem('llm_base_url', llmBaseUrl.trim());
                localStorage.setItem('llm_model', llmModel.trim());
                await aiStatus?.refresh?.();
              }
              setIndexPhase('indexing');
              await startIndexing(enabled);  // pass explicitly — state hasn't flushed
            }}
            onAiSkip={() => { setEnabledForNewCollection(false); setIndexPhase('indexing'); startIndexing(false); }}
            onAiBootstrapRun={async () => {
              const lang2 = lang || 'en';
              const tasks = ['sonic_vibe', 'refined_facts', 'artist_bio'];
              const llmBaseUrl = localStorage.getItem('llm_base_url') || undefined;
              const llmModel   = localStorage.getItem('llm_model')    || undefined;
              // Switch to the live AI-progress view first, then fire the tasks.
              // allSettled (not all) so one task's failure doesn't abort the others;
              // AiEnrichProgress then polls /library/ai-index/status for the counts.
              setIndexPhase('ai-running');
              const results = await Promise.allSettled(tasks.map(t =>
                apiFetch(`/library/ai-index/${t}`, { method:'POST', body: JSON.stringify({
                  lang: lang2, llm_base_url: llmBaseUrl, llm_model: llmModel,
                  ...(t === 'artist_bio' && { bio_source: 'web' }),
                }) })
              ));
              results.forEach((r, i) => {
                if (r.status === 'rejected') console.error(`AI indexing task '${tasks[i]}' failed:`, r.reason);
              });
            }}
            onAiBootstrapLater={() => setShowModal(false)}
          />
        )}
      </div>
    </>
  );
}

// ─── Onboarding (compact, themed) ─────────────────────────────────────────────
// ─── Server-mode upload onboarding (Phase C) ─────────────────────────────────
// ── Stage ETA helpers (shared by MemberIndexing + SetupWizard) ───────────────
// fmtEta turns a millisecond estimate into a short "~Xs" / "~Xm Ys" string.
function fmtEta(ms, ru) {
  const s = Math.max(1, Math.round(ms / 1000));
  if (s < 60) return ru ? `~${s} с` : `~${s}s`;
  const m = Math.floor(s / 60), r = s % 60;
  return ru ? `~${m} мин ${r} с` : `~${m}m ${r}s`;
}
// computeStageEtas estimates remaining time per stage from the SSE current/total.
// `store` is a mutable { key: {t0, c0} } object (a ref's .current) that anchors
// each stage's start, so the rate is averaged over the whole stage rather than
// jumping frame-to-frame. Returns a delta { key: "~Xs" | null } to merge into the
// eta state. `dense` (text search) is skipped on purpose — it's an indeterminate
// bar with no granular progress, so no ETA is computed for it.
function computeStageEtas(stages, store, statusMap, ru, now) {
  const delta = {};
  for (const [k, v] of Object.entries(stages)) {
    if (k === 'dense') continue;
    const status = statusMap[v.status] || v.status;
    const total = v.total ?? 0, current = v.current ?? 0;
    if (status === 'running' && total > 0 && current > 0 && current < total) {
      const m = store[k] || (store[k] = { t0: now, c0: current });
      const dc = current - m.c0, dt = now - m.t0;
      delta[k] = (dc > 0 && dt > 800) ? fmtEta((total - current) * dt / dc, ru) : null;
    } else {
      delete store[k];
      delta[k] = null;
    }
  }
  return delta;
}

// Reusable stage bar (member onboarding). `indeterminate` shows a moving shimmer
// for stages with no granular progress (text search + the awaited AI phase);
// `eta` is a short remaining-time string shown next to the … while running.
function OBStageBar({ c, label, state, pct, indeterminate, eta, count, trailing }) {
  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
        <span style={{ display:'flex', alignItems:'center', gap:8, minWidth:0 }}>
          <span className="mono" style={{ fontSize:12, letterSpacing:'0.12em', color: state==='running' ? c.text : c.textSubtle }}>{label}</span>
          {trailing}
        </span>
        <span style={{ display:'flex', alignItems:'center', gap:8, fontSize:11, color: state==='failed' ? c.red : c.textSubtle }}>
          {count && state==='running' && <span className="mono" style={{ fontVariantNumeric:'tabular-nums', opacity:.95 }}>{count}</span>}
          {eta && state==='running' && <span style={{ fontVariantNumeric:'tabular-nums', opacity:.85 }}>{eta}</span>}
          <span>{state==='done' ? '✓' : state==='failed' ? '✗' : state==='running' ? '…' : '·'}</span>
        </span>
      </div>
      <div style={{ height:5, borderRadius:3, background:c.border, overflow:'hidden', position:'relative' }}>
        {indeterminate && state==='running'
          ? <div className="ob-indet" />
          : <div style={{ height:'100%', width:`${pct||0}%`, transition:'width 0.4s', background: state==='done' ? 'oklch(63% 0.17 142)' : 'linear-gradient(90deg, oklch(65% 0.18 270), oklch(75% 0.17 280))' }} />}
      </div>
    </div>
  );
}

// Live AI-enrichment progress, shared by every in-flow AI phase (member onboarding,
// owner upload, folder "Run now"). Polls /library/ai-index/status — the same jobs
// the auto AI tasks (_run_ai_tasks) and the manual POSTs register — and shows
// n_done/n_total per task. The 3 tasks run back-to-back server-side, so at any
// moment one is 'running' and the rest are 'queued'/'done'; each task's latest
// status is rendered honestly. Polls continuously until unmounted — the parent
// (SSE 'completed' or a Close button) controls the lifetime.
const AI_ENRICH_STAGES = [
  { key:'sonic_vibe',    ru:'Звучание песен',     en:'Song vibes' },
  { key:'refined_facts', ru:'Углубление фактов',  en:'Deeper facts' },
  { key:'artist_bio',    ru:'Биографии артистов', en:'Artist bios' },
];

// Live guru progress driven by the indexing SSE stream itself (`ai_stages` key,
// published by the backend's awaited AI phase). Unlike AiEnrichProgress (below),
// this can't confuse the current run with a previous run's rows and needs no
// extra polling — preferred wherever the SSE stream carries ai_stages.
function GuruStagesFromSse({ ru, c, aiStages }) {
  return (
    <>
      {AI_ENRICH_STAGES.map(s => {
        const st = (aiStages && aiStages[s.key]) || { status: 'pending', n_done: 0, n_total: 0 };
        const raw = st.status;
        const state = (raw === 'done' || raw === 'skipped') ? 'done'
                    : (raw === 'failed' || raw === 'cancelled') ? 'failed'
                    : (raw === 'running' || raw === 'queued') ? 'running' : 'pending';
        const total = st.n_total || 0, done = st.n_done || 0;
        const pct = state === 'done' ? 100 : total > 0 ? Math.min(100, Math.round(100 * done / total)) : 0;
        const count = (state === 'running' && total > 0) ? `${done}/${total}` : null;
        const indeterminate = state === 'running' && total <= 0;
        return (
          <OBStageBar key={s.key} c={c} label={ru ? s.ru : s.en}
            state={state} pct={pct} count={count} indeterminate={indeterminate} />
        );
      })}
    </>
  );
}

function AiEnrichProgress({ ru, c }) {
  const [status, setStatus] = useState({});
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const data = await apiFetch('/library/ai-index/status');
        if (!cancelled) setStatus(data || {});
      } catch { /* transient — keep polling */ }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  return (
    <>
      {AI_ENRICH_STAGES.map(s => {
        const st = status[s.key] || null;
        const raw = st && st.status;
        const state = raw === 'done' ? 'done' : raw === 'failed' ? 'failed' : 'running';
        const total = (st && st.n_total) || 0;
        const done = (st && st.n_done) || 0;
        const pct = total > 0 ? Math.min(100, Math.round(100 * done / total)) : 0;
        const count = (state === 'running' && total > 0) ? `${done}/${total}` : null;
        const indeterminate = state === 'running' && total <= 0;
        return (
          <OBStageBar key={s.key} c={c} label={ru ? s.ru : s.en}
            state={state} pct={pct} count={count} indeterminate={indeterminate} />
        );
      })}
    </>
  );
}

// Inline member indexing: core stages from SSE + the 3 guru AI stages shown as
// indeterminate bars while the backend runs them (awaited, after FACTS) — only
// when the instance LLM is online (otherwise the job completes with no AI phase
// and aiWaiting never trips). Replaces the modal UploadIndexingWizard in the
// member onboarding so it sits inside the SetupRail layout.
function MemberIndexing({ ru, c, isDark, jobId, onDone }) {
  // The job we actually stream. A Yandex import starts as a DOWNLOAD job and
  // hands off to the real indexing job on completion (indexing_job_id) — after
  // a page reload we may attach to either, so the switch lives here.
  const [activeJobId, setActiveJobId] = useState(jobId);
  useEffect(() => { setActiveJobId(jobId); }, [jobId]);
  const [stepStatus, setStepStatus] = useState({});
  const [stageProgress, setStageProgress] = useState({});
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);
  const [aiStages, setAiStages] = useState(null);   // live guru progress from SSE
  const [dl, setDl] = useState(null);               // Yandex download phase {current,total,done}
  const [etas, setEtas] = useState({});
  const etaRef = useRef({});

  useEffect(() => {
    if (!activeJobId) return;
    const evt = new EventSource(`${API}/index/progress/${activeJobId}`);
    evt.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.stages) {
          const statusMap = { completed:'done', failed:'failed', running:'running', pending:'pending' };
          setStepStatus(prev => { const next = { ...prev }; for (const [k, v] of Object.entries(data.stages)) next[k] = statusMap[v.status] || v.status || 'pending'; return next; });
          setStageProgress(prev => { const next = { ...prev }; for (const [k, v] of Object.entries(data.stages)) next[k] = { current: v.current ?? prev[k]?.current ?? 0, total: v.total ?? prev[k]?.total ?? 0 }; return next; });
          setEtas(prev => ({ ...prev, ...computeStageEtas(data.stages, etaRef.current, statusMap, ru, Date.now()) }));
        }
        if (data.stage === 'download') {
          setDl({ current: data.current ?? 0, total: data.total ?? 0, done: false });
        }
        if (data.ai_stages) setAiStages(data.ai_stages);
        if (data.overall_status === 'completed') {
          evt.close();
          if (data.indexing_job_id && data.indexing_job_id !== activeJobId) {
            // Download finished → follow the indexing job it spawned.
            setDl(d => d ? { ...d, done: true } : d);
            setActiveJobId(data.indexing_job_id);
          } else {
            setDone(true); setTimeout(onDone, 1400);
          }
        }
        else if (data.overall_status === 'failed') { evt.close(); setError(data.error || data.message || 'failed'); }
      } catch (err) {}
    };
    return () => evt.close();
  }, [activeJobId]);

  if (done) {
    return (
      <div className="ob-glass" style={{ padding:'30px 28px', textAlign:'center', boxShadow:'inset 0 1px 0 var(--ob-glass-sheen),0 12px 50px rgba(0,0,0,.28),0 0 70px rgba(95,208,138,.22)' }}>
        <div style={{ fontSize:'60px', lineHeight:1, marginBottom:'12px', filter:'drop-shadow(0 0 26px rgba(95,208,138,.9)) drop-shadow(0 0 8px rgba(95,208,138,.75))' }}>✨</div>
        <div className="mono" style={{ fontSize:'11px', letterSpacing:'0.24em', textTransform:'uppercase', color:'#a9ecc4' }}>{ru ? 'Всё готово' : 'All set'}</div>
        <h2 className="serif" style={{ fontSize:'28px', margin:'8px 0' }}>{ru ? 'Библиотека готова' : 'Your library is ready'}</h2>
        <p style={{ fontSize:'13.5px', color:c.textMuted, lineHeight:1.6 }}>{ru ? 'Открываем плеер…' : 'Opening the player…'}</p>
      </div>
    );
  }

  // The awaited guru phase starts once every core stage is done but the job is
  // still running; ai_stages from the SSE stream is the authoritative signal.
  const coreDone = Object.keys(WIZ_STAGE_LABELS).every(k => stepStatus[k] === 'done');
  return (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'14px' }}>
        {error ? (ru ? 'Что-то пошло не так' : 'Something went wrong') : (ru ? 'Готовим вашу музыку…' : 'Preparing your music…')}
      </h2>
      <ProcessingModeBadge isDark={isDark} lang={ru ? 'ru' : 'en'} style={{ marginBottom:'18px' }} />
      {error ? (
        <div style={{ padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{error}</div>
      ) : (
        <>
          {dl && (
            <OBStageBar c={c} label={ru ? 'Скачивание из Яндекса' : 'Downloading from Yandex'}
              state={dl.done ? 'done' : 'running'}
              pct={dl.done ? 100 : (dl.total > 0 ? Math.round(100 * dl.current / dl.total) : 0)}
              count={!dl.done && dl.total > 0 ? `${dl.current}/${dl.total}` : null}
              indeterminate={!dl.done && dl.total === 0} />
          )}
          {Object.keys(WIZ_STAGE_LABELS).map(k => {
            const st = stepStatus[k] || 'pending';
            const pr = stageProgress[k] || { current:0, total:0 };
            const pct = st === 'done' ? 100 : pr.total > 0 ? Math.round(100 * pr.current / pr.total) : 0;
            const count = (st === 'running' && pr.total > 1) ? `${pr.current || 0}/${pr.total}` : null;
            return (
              <Fragment key={k}>
                <OBStageBar c={c} label={ru ? WIZ_STAGE_LABELS[k].ru : WIZ_STAGE_LABELS[k].en} state={st} pct={pct} count={count} eta={etas[k]}
                  trailing={k === 'facts' ? <PremiumMetaHint isDark={isDark} lang={ru ? 'ru' : 'en'} /> : null} />
              </Fragment>
            );
          })}
          {(aiStages || coreDone) && (
            <>
              <div className="mono" style={{ display:'flex', alignItems:'center', gap:'9px', margin:'16px 0 10px',
                fontSize:'11px', letterSpacing:'0.2em', textTransform:'uppercase', color:'#c3b8ff' }}>
                ✨ {ru ? 'С помощью гуру' : 'With the guru'}
              </div>
              <GuruStagesFromSse ru={ru} c={c} aiStages={aiStages} />
            </>
          )}
          <div style={{ display:'flex', alignItems:'center', gap:'8px', marginTop:'16px', fontSize:'12px', color:c.textSubtle, lineHeight:1.5 }}>
            <span aria-hidden>🌙</span>
            {ru ? 'Можно уйти со страницы или закрыть вкладку — подготовка продолжится, а прогресс будет здесь, когда вернётесь.'
                : 'Feel free to leave this page or close the tab — preparation continues, and the progress will be here when you return.'}
          </div>
        </>
      )}
    </div>
  );
}

// Compact Yandex account link used INSIDE the "Your files" upload block: linking
// the account lets indexing enrich manual uploads with higher-quality album art
// and metadata from the Yandex catalog (the backend enrichment uses the account
// token once linked). Premium-only — the caller wraps it in <PremiumGate>. This
// only links the account; it does NOT import anything (that's YandexImportFlow).
function YandexEnhanceLink({ isDark, lang }) {
  const c = useColors(isDark);
  const ru = lang === 'ru';
  const [linked, setLinked] = useState(null);     // null = loading, else bool
  const [session, setSession] = useState(null);   // {session_id, user_code, verification_url}
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    apiFetch('/import/yandex')
      .then(r => { if (alive) setLinked(!!(r && r.linked)); })
      .catch(() => { if (alive) setLinked(false); });
    return () => { alive = false; };
  }, []);

  const startAuth = async () => {
    setErr(null); setBusy(true);
    try {
      const res = await apiFetch('/import/yandex/auth/start', { method: 'POST' });
      if (res.status === 'authorized') { setLinked(true); setBusy(false); return; }
      setSession(res);
    } catch (e) { setErr(e.message); setBusy(false); }
  };

  // Poll the device session until authorized / expired / error.
  useEffect(() => {
    if (!session?.session_id) return;
    let stop = false;
    const tick = async () => {
      try {
        const s = await apiFetch(`/import/yandex/auth/status?session_id=${encodeURIComponent(session.session_id)}`);
        if (stop) return;
        if (s.status === 'authorized') { setLinked(true); setSession(null); setBusy(false); return; }
        if (s.status === 'expired' || s.status === 'error') {
          setErr(s.error || (s.status === 'expired' ? (ru ? 'Код истёк' : 'Code expired') : (ru ? 'Ошибка входа' : 'Login error')));
          setSession(null); setBusy(false); return;
        }
        if (s.user_code && !session.user_code) setSession(prev => ({ ...prev, ...s }));
      } catch (e) { /* transient — keep polling */ }
    };
    const id = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(id); };
  }, [session?.session_id, session?.user_code, ru]);

  const gold = PREMIUM_GOLD;
  const wrap = { marginTop:'14px', padding:'14px 18px', borderRadius:'14px' };

  if (linked === null) return null;  // avoid flashing the CTA before status loads

  if (linked) {
    return (
      <div className="ob-glass" style={wrap}>
        <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
          <span style={{ color:'oklch(70% 0.18 145)', fontSize:'16px' }}>✓</span>
          <div style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.5 }}>
            {ru
              ? 'Яндекс подключён — обложки и метаданные загруженных файлов будут качественнее.'
              : 'Yandex connected — album art and metadata for your uploads will be higher quality.'}
          </div>
        </div>
      </div>
    );
  }

  // Device-flow in progress: show the code + verification link.
  if (session) {
    return (
      <div className="ob-glass" style={wrap}>
        <div style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.55, marginBottom:'10px' }}>
          {ru ? 'Откройте страницу Яндекса и введите код:' : 'Open the Yandex page and enter the code:'}
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:'12px', flexWrap:'wrap' }}>
          <div className="mono" style={{ fontSize:'22px', fontWeight:'700', letterSpacing:'0.18em', color:c.text }}>
            {session.user_code || '····'}
          </div>
          {session.verification_url && (
            <a href={session.verification_url} target="_blank" rel="noreferrer"
              className={ske('btn', isDark)}
              style={{ padding:'8px 14px', borderRadius:'9px', fontSize:'13px', color:c.text, textDecoration:'none' }}>
              {ru ? 'Открыть Яндекс ↗' : 'Open Yandex ↗'}
            </a>
          )}
          <span style={{ fontSize:'12px', color:c.textSubtle, display:'inline-flex', alignItems:'center', gap:'6px' }}>
            <Spinner size={12} /> {ru ? 'Ожидание входа…' : 'Waiting for sign-in…'}
          </span>
        </div>
        {err && <div style={{ marginTop:'10px', fontSize:'12px', color:c.red }}>{err}</div>}
      </div>
    );
  }

  // Not linked: the CTA row.
  return (
    <div className="ob-glass" style={{ ...wrap, borderLeft:`2px solid ${gold}` }}>
      <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'6px' }}>
        <PremiumBadge />
        <span className="mono" style={{ fontSize:'11px', letterSpacing:'0.16em', color:gold }}>
          {ru ? 'КАЧЕСТВЕННЕЕ ОБЛОЖКИ И МЕТАДАННЫЕ' : 'BETTER COVERS & METADATA'}
        </span>
      </div>
      <div style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.55, marginBottom:'12px' }}>
        {ru
          ? 'Войдите в Яндекс Музыку — MusiX подтянет обложки альбомов и точные метаданные для ваших файлов из каталога Яндекса.'
          : 'Sign in to Yandex Music — MusiX will pull album art and accurate metadata for your files from the Yandex catalog.'}
      </div>
      <button onClick={startAuth} disabled={busy} className={ske('btn', isDark)}
        style={{ padding:'10px 18px', borderRadius:'10px', fontSize:'13px', fontWeight:'600',
          cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.6 : 1,
          display:'inline-flex', alignItems:'center', gap:'8px' }}>
        <span style={{ width:'20px', height:'20px', borderRadius:'6px', flexShrink:0,
          background:'linear-gradient(135deg, #ffcc00, #ff5c5c)', color:'#1a1a1a',
          display:'inline-flex', alignItems:'center', justifyContent:'center', fontSize:'13px', fontWeight:'800' }}>Я</span>
        {busy ? (ru ? 'Подключение…' : 'Connecting…') : (ru ? 'Войти в Яндекс' : 'Sign in to Yandex')}
      </button>
      {err && <div style={{ marginTop:'10px', fontSize:'12px', color:c.red }}>{err}</div>}
    </div>
  );
}

// Yandex Music import, embedded in the onboarding "Music" step. Self-contained
// state machine: auth (device flow) → sources (multi-select) → progress (download
// then the normal indexing wizard). Talks to the /import/yandex/* endpoints and,
// for the indexing phase, reuses MemberIndexing on the indexing_job_id the backend
// hands back in the download job's completion event.
function YandexImportFlow({ isDark, lang, onDone, onBack, onPhase }) {
  const c = useColors(isDark);
  const ru = lang === 'ru';
  const [step, setStep] = useState('auth');     // 'auth' | 'sources' | 'progress'
  const [authErr, setAuthErr] = useState(null);
  const [session, setSession] = useState(null); // {session_id, user_code, verification_url}

  // ── Auth: device flow ────────────────────────────────────────────────────
  const startAuth = useCallback(async () => {
    setAuthErr(null);
    setSession(null);
    try {
      const res = await apiFetch('/import/yandex/auth/start', { method: 'POST' });
      if (res.status === 'authorized') { setStep('sources'); return; }
      setSession(res);
    } catch (e) {
      setAuthErr(e.message);
    }
  }, []);

  // On mount: skip auth if already linked, else kick off the device flow.
  useEffect(() => {
    let cancelled = false;
    apiFetch('/import/yandex')
      .then(res => { if (!cancelled) { if (res && res.linked) setStep('sources'); else startAuth(); } })
      .catch(() => { if (!cancelled) startAuth(); });
    return () => { cancelled = true; };
  }, [startAuth]);

  // Poll the device session until authorized / expired / error.
  useEffect(() => {
    if (step !== 'auth' || !session?.session_id) return;
    let stop = false;
    const tick = async () => {
      try {
        const s = await apiFetch(`/import/yandex/auth/status?session_id=${encodeURIComponent(session.session_id)}`);
        if (stop) return;
        if (s.status === 'authorized') { setStep('sources'); return; }
        if (s.status === 'expired' || s.status === 'error') {
          setAuthErr(s.error || (s.status === 'expired' ? (ru ? 'Код истёк' : 'Code expired') : (ru ? 'Ошибка входа' : 'Login error')));
          return;
        }
        if (s.user_code && !session.user_code) setSession(prev => ({ ...prev, ...s }));
      } catch (e) { /* transient — keep polling */ }
    };
    const id = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(id); };
  }, [step, session?.session_id, session?.user_code, ru]);

  // ── Sources: list + multi-select ─────────────────────────────────────────
  const [sources, setSources] = useState(null);  // null = loading
  const [srcErr, setSrcErr] = useState(null);
  const [selected, setSelected] = useState(() => new Set());

  useEffect(() => {
    if (step !== 'sources') return;
    let cancelled = false;
    setSources(null);
    setSrcErr(null);
    apiFetch('/import/yandex/playlists')
      .then(res => { if (!cancelled) setSources(res.sources || []); })
      .catch(e => {
        if (cancelled) return;
        if (/not linked/i.test(e.message) || /401/.test(e.message)) { setStep('auth'); startAuth(); }
        else setSrcErr(e.message);
      });
    return () => { cancelled = true; };
  }, [step, startAuth]);

  const srcKey = (s) => (s === 'likes' ? 'likes' : `kind:${s.kind}`);
  const toggle = (key) => setSelected(prev => {
    const next = new Set(prev);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  });

  // ── Progress: start import + follow the download job ──────────────────────
  const [downloadJobId, setDownloadJobId] = useState(null);
  const [dl, setDl] = useState({ current: 0, total: 0, message: '' });
  const [indexingJobId, setIndexingJobId] = useState(null);
  const [report, setReport] = useState(null);
  const [progErr, setProgErr] = useState(null);
  const [noIndex, setNoIndex] = useState(false);  // download done, nothing to index

  const startImport = async () => {
    const chosen = (sources || []).filter(s => selected.has(srcKey(s.source)));
    const body = chosen.map(s => (s.source === 'likes' ? { source: 'likes' } : { kind: s.source.kind }));
    if (!body.length) return;
    setProgErr(null);
    setStep('progress');
    if (onPhase) onPhase('progress');  // parent flips to the dedicated progress page
    try {
      const res = await apiFetch('/import/yandex/start', {
        method: 'POST',
        body: JSON.stringify({ sources: body, lang }),
      });
      setDownloadJobId(res.job_id);
    } catch (e) {
      setProgErr(e.message);
    }
  };

  useEffect(() => {
    if (!downloadJobId) return;
    const evt = new EventSource(`${API}/index/progress/${downloadJobId}`);
    evt.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.stage === 'download') {
          setDl({ current: data.current ?? 0, total: data.total ?? 0, message: data.message || '' });
        }
        if (data.yandex_report) setReport(data.yandex_report);
        if (data.overall_status === 'completed') {
          evt.close();
          if (data.indexing_job_id) {
            setIndexingJobId(data.indexing_job_id);
          } else {
            // Race fallback: if we subscribed after the download already finished,
            // the live handoff event may have been missed (and the backend snapshot
            // didn't carry it on an older build). Ask the account's current job
            // directly — after download it's the indexing job that's now running.
            apiFetch('/import/yandex/status').then(s => {
              const jid = s && s.job_id;
              if (jid && jid !== downloadJobId && s.overall_status !== 'completed') {
                setIndexingJobId(jid);
              } else {
                setNoIndex(true); setTimeout(onDone, 1800);
              }
            }).catch(() => { setNoIndex(true); setTimeout(onDone, 1800); });
          }
        } else if (data.overall_status === 'failed') {
          evt.close();
          setProgErr(data.error || data.message || (ru ? 'Импорт не удался' : 'Import failed'));
        }
      } catch (err) {}
    };
    evt.onerror = () => { /* EventSource auto-reconnects; ignore transient drops */ };
    return () => evt.close();
  }, [downloadJobId, onDone, ru]);

  // ── Render ────────────────────────────────────────────────────────────────
  const kicker = (txt) => (
    <div className="mono" style={{ fontSize:'11px', color:c.textSubtle, letterSpacing:'0.24em', textTransform:'uppercase', marginBottom:'8px' }}>{txt}</div>
  );
  const backLink = onBack && (
    <button onClick={onBack} className={ske('btn', isDark)}
      style={{ marginTop:'16px', padding:'8px 16px', borderRadius:'9px', fontSize:'13px', color:c.textMuted, cursor:'pointer' }}>
      ← {ru ? 'Назад' : 'Back'}
    </button>
  );

  if (step === 'progress' && indexingJobId) {
    return <MemberIndexing ru={ru} c={c} isDark={isDark} jobId={indexingJobId} onDone={onDone} />;
  }

  if (step === 'progress') {
    const pct = dl.total > 0 ? Math.round(100 * dl.current / dl.total) : 0;
    return (
      <div className="ob-glass" style={{ padding:'26px 28px' }}>
        {kicker(ru ? 'Яндекс · Импорт' : 'Yandex · Import')}
        <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'16px' }}>
          {progErr ? (ru ? 'Что-то пошло не так' : 'Something went wrong')
            : noIndex ? (ru ? 'Готово' : 'Done')
            : (ru ? 'Скачиваем из Яндекса…' : 'Downloading from Yandex…')}
        </h2>
        {progErr ? (
          <>
            <div style={{ padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{progErr}</div>
            {backLink}
          </>
        ) : (
          <>
            <OBStageBar c={c} label={ru ? 'Скачивание' : 'Download'}
              state={noIndex ? 'done' : 'running'} pct={noIndex ? 100 : pct}
              indeterminate={dl.total === 0 && !noIndex} />
            {dl.message && !noIndex && (
              <div style={{ fontSize:'12.5px', color:c.textMuted, marginTop:'8px', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{dl.message}</div>
            )}
            {report && (
              <div style={{ marginTop:'14px', fontSize:'13px', color:c.textMuted }}>
                {ru ? 'Скачано' : 'Downloaded'}: <b style={{ color:c.text }}>{report.downloaded}</b>
                {report.already > 0 && <> · {ru ? 'уже было' : 'already'}: {report.already}</>}
                {report.skipped?.length > 0 && (
                  <details style={{ marginTop:'8px' }}>
                    <summary style={{ cursor:'pointer', color:c.textSubtle }}>
                      {ru ? 'Пропущено' : 'Skipped'}: {report.skipped.length}
                    </summary>
                    <div style={{ maxHeight:'140px', overflow:'auto', marginTop:'6px' }}>
                      {report.skipped.map((s, i) => (
                        <div key={i} style={{ fontSize:'12px', padding:'3px 0', color:c.textSubtle }}>
                          {s.artist} — {s.title}: {s.reason}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </>
        )}
      </div>
    );
  }

  if (step === 'sources') {
    const count = selected.size;
    return (
      <div className="ob-glass" style={{ padding:'26px 28px' }}>
        {kicker(ru ? 'Яндекс · Шаг 2' : 'Yandex · Step 2')}
        <h2 className="serif" style={{ fontSize:'28px', letterSpacing:'-0.02em', marginBottom:'6px' }}>
          {ru ? <>Что <i style={{ color:'oklch(62% 0.2 275)' }}>импортировать</i></> : <>What to <i style={{ color:'oklch(62% 0.2 275)' }}>import</i></>}
        </h2>
        <p style={{ fontSize:'13.5px', color:c.textMuted, lineHeight:1.6, marginBottom:'18px' }}>
          {ru ? 'Выберите один или несколько источников.' : 'Pick one or more sources.'}
        </p>
        {sources === null && !srcErr && (
          <div style={{ textAlign:'center', padding:'30px', color:c.textMuted }}><Spinner size={16} /> {ru ? 'Загрузка…' : 'Loading…'}</div>
        )}
        {srcErr && (
          <div style={{ padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{srcErr}</div>
        )}
        {sources && sources.length > 0 && (
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(200px, 1fr))', gap:'12px' }}>
            {sources.map((s) => {
              const key = srcKey(s.source);
              const on = selected.has(key);
              return (
                <button key={key} onClick={() => toggle(key)}
                  className="ob-glass"
                  style={{ display:'flex', alignItems:'center', gap:'12px', padding:'12px', borderRadius:'14px',
                    cursor:'pointer', textAlign:'left',
                    border:`2px solid ${on ? 'oklch(62% 0.2 275)' : c.border}`,
                    boxShadow: on ? '0 0 0 3px oklch(62% 0.2 275 / 0.18)' : 'none' }}>
                  {s.cover
                    ? <img src={s.cover} alt="" style={{ width:'48px', height:'48px', borderRadius:'8px', objectFit:'cover', flexShrink:0 }} />
                    : <div style={{ width:'48px', height:'48px', borderRadius:'8px', flexShrink:0,
                        background:'linear-gradient(135deg, oklch(62% 0.2 275), oklch(55% 0.2 320))',
                        display:'flex', alignItems:'center', justifyContent:'center', fontSize:'22px' }}>{s.source === 'likes' ? '♥' : '♪'}</div>}
                  <div style={{ minWidth:0, flex:1 }}>
                    <div style={{ fontSize:'13.5px', fontWeight:'600', color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{s.title}</div>
                    <div style={{ fontSize:'12px', color:c.textSubtle, marginTop:'2px' }}>{s.track_count} {ru ? 'треков' : 'tracks'}</div>
                  </div>
                  <div style={{ width:'20px', height:'20px', borderRadius:'6px', flexShrink:0,
                    border:`2px solid ${on ? 'oklch(62% 0.2 275)' : c.border}`,
                    background: on ? 'oklch(62% 0.2 275)' : 'transparent',
                    display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontSize:'13px' }}>{on ? '✓' : ''}</div>
                </button>
              );
            })}
          </div>
        )}
        {sources && sources.length === 0 && (
          <div style={{ fontSize:'13px', color:c.textMuted }}>{ru ? 'Источники не найдены.' : 'No sources found.'}</div>
        )}
        <div style={{ display:'flex', gap:'10px', alignItems:'center', marginTop:'18px' }}>
          <button onClick={startImport} disabled={count === 0} className="ske-accent"
            style={{ padding:'12px 22px', borderRadius:'12px', fontSize:'14px', fontWeight:'600', letterSpacing:'0.06em',
              cursor: count === 0 ? 'not-allowed' : 'pointer', opacity: count === 0 ? 0.5 : 1 }}>
            {ru ? `▶ Импортировать ${count || ''}` : `▶ Import ${count || ''}`}
          </button>
          {onBack && (
            <button onClick={onBack} className={ske('btn', isDark)}
              style={{ padding:'12px 18px', borderRadius:'12px', fontSize:'13px', color:c.textMuted, cursor:'pointer' }}>
              {ru ? 'Назад' : 'Back'}
            </button>
          )}
        </div>
      </div>
    );
  }

  // step === 'auth'
  return (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      {kicker(ru ? 'Яндекс · Шаг 1' : 'Yandex · Step 1')}
      <h2 className="serif" style={{ fontSize:'28px', letterSpacing:'-0.02em', marginBottom:'6px' }}>
        {ru ? <>Войдите в <i style={{ color:'oklch(62% 0.2 275)' }}>Яндекс</i></> : <>Sign in to <i style={{ color:'oklch(62% 0.2 275)' }}>Yandex</i></>}
      </h2>
      {authErr ? (
        <>
          <div style={{ padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px', margin:'10px 0' }}>{authErr}</div>
          <button onClick={startAuth} className="ske-accent"
            style={{ padding:'11px 22px', borderRadius:'10px', fontSize:'14px', fontWeight:'600', cursor:'pointer' }}>
            {ru ? 'Попробовать снова' : 'Try again'}
          </button>
        </>
      ) : !session?.user_code ? (
        <div style={{ textAlign:'center', padding:'24px', color:c.textMuted }}><Spinner size={16} /> {ru ? 'Получаем код…' : 'Getting a code…'}</div>
      ) : (
        <>
          <p style={{ fontSize:'13.5px', color:c.textMuted, lineHeight:1.6, margin:'8px 0 16px' }}>
            {ru ? 'Откройте страницу Яндекса и введите код:' : 'Open the Yandex page and enter the code:'}
          </p>
          <div className="mono" style={{ fontSize:'34px', fontWeight:'700', letterSpacing:'0.18em', textAlign:'center',
            padding:'16px', borderRadius:'14px', background: isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.04)', color:c.text, marginBottom:'16px' }}>
            {session.user_code}
          </div>
          <a href={session.verification_url} target="_blank" rel="noreferrer" className="ske-accent"
            style={{ display:'inline-block', padding:'12px 24px', borderRadius:'12px', fontSize:'14px', fontWeight:'600', cursor:'pointer', textDecoration:'none' }}>
            {ru ? 'Открыть Яндекс ↗' : 'Open Yandex ↗'}
          </a>
          <div style={{ display:'flex', alignItems:'center', gap:'8px', marginTop:'16px', fontSize:'12.5px', color:c.textSubtle }}>
            <Spinner size={13} /> {ru ? 'Ждём подтверждения…' : 'Waiting for confirmation…'}
          </div>
        </>
      )}
      {backLink}
    </div>
  );
}

// Grid-rows expand/collapse wrapper (paired with .ob-expand in index.css).
// Content mounts lazily on open and unmounts ~0.5s after close (once the 0fr
// collapse settles) — YandexImportFlow must not stay mounted while hidden:
// its mount effect starts a device-flow auth session and polling.
function ObExpand({ open, children }) {
  const [mounted, setMounted] = useState(open);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (open) {
      setMounted(true);
      // Double-rAF: let the 0fr initial state paint before flipping to 1fr,
      // otherwise the browser coalesces both states and skips the transition.
      let cancelled = false;
      const raf = requestAnimationFrame(() => requestAnimationFrame(() => { if (!cancelled) setShown(true); }));
      return () => { cancelled = true; cancelAnimationFrame(raf); };
    }
    setShown(false);
    // Unmount on a timer (not transitionend) so it also fires under
    // prefers-reduced-motion, where no transition event ever comes.
    const t = setTimeout(() => setMounted(false), 500);
    return () => clearTimeout(t);
  }, [open]);
  if (!mounted) return null;
  return (
    <div className={`ob-expand${shown ? ' ob-expand-open' : ''}`}>
      <div className="ob-expand-inner">{children}</div>
    </div>
  );
}

function ServerOnboardingScreen({ isDark, lang, onDone, onLang, onTheme }) {
  const c = useColors(isDark);
  const [files, setFiles] = useState([]);
  // Per-file: { name, size, status: 'queued'|'uploading'|'done'|'failed', upload_id?, error? }
  const [progress, setProgress] = useState([]);
  const [uploadStarted, setUploadStarted] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [showWizard, setShowWizard] = useState(false);
  const [memberIndexRoot, setMemberIndexRoot] = useState(null);
  const [indexingFolder, setIndexingFolder] = useState(false);
  const [mode, setMode] = useState('pick');   // kept for compat — now only used internally
  const [uploadExpanded, setUploadExpanded] = useState(false);  // expand upload section
  const [yandexExpanded, setYandexExpanded] = useState(false);  // expand Yandex inline section
  const [yandexImporting, setYandexImporting] = useState(false); // Yandex flow entered its progress phase
  const [premiumHint, setPremiumHint] = useState(false);  // inline hint for non-premium Yandex click
  const premiumHintDismissed = (() => {
    try { return localStorage.getItem('musix_premium_hint_dismissed') === '1'; } catch { return false; }
  })();
  const dismissPremiumHint = () => {
    try { localStorage.setItem('musix_premium_hint_dismissed', '1'); } catch {}
    setPremiumHint(false);
  };
  const ru = lang === 'ru';
  const uid = (typeof localStorage !== 'undefined' && localStorage.getItem('musix_user_id')) || '';
  // Marketing welcome shows once per account on first onboarding, then never.
  const [welcomed, setWelcomed] = useState(() => {
    try { return localStorage.getItem('musix_welcome_seen_' + uid) === '1'; } catch (e) { return false; }
  });
  const dismissWelcome = () => {
    try { localStorage.setItem('musix_welcome_seen_' + uid, '1'); } catch (e) {}
    setWelcomed(true);
  };
  // Folder selection happens IN THE BROWSER (webkitdirectory), then the files
  // are uploaded like any other — server-mode members may upload but may NOT
  // point the host indexer at a path (/library/index is owner-only). React drops
  // the unknown JSX prop, so it's set on the DOM node directly — via a CALLBACK
  // REF, not a mount effect: this input lives behind the welcome-slide gate and
  // mounts only after dismiss, so a useEffect([]) would fire once while the node
  // is still null and never re-run, leaving "PICK FOLDER" as a plain file picker
  // on first onboarding. A callback ref fires exactly when the node attaches.
  const attachFolderInput = useCallback((node) => {
    if (node) {
      node.setAttribute('webkitdirectory', '');
      node.setAttribute('directory', '');
    }
  }, []);

  // Picked a whole folder: keep only audio files (a directory carries covers,
  // playlists, etc.), then funnel into the same upload pipeline as PICK FILES.
  const onPickFolder = (e) => {
    const picked = Array.from(e.target.files || []).filter(
      f => /\.(flac|mp3|m4a|aac|ogg|wav|opus)$/i.test(f.name),
    );
    setFiles(picked);
    setProgress(picked.map(f => ({ name: f.name, size: f.size, status: 'queued' })));
  };

  const onPick = (e) => {
    const picked = Array.from(e.target.files || []);
    setFiles(picked);
    setProgress(picked.map(f => ({ name: f.name, size: f.size, status: 'queued' })));
  };
  const onDrop = (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files || []).filter(
      f => /\.(flac|mp3|m4a|aac|ogg|wav|opus)$/i.test(f.name),
    );
    setFiles(dropped);
    setProgress(dropped.map(f => ({ name: f.name, size: f.size, status: 'queued' })));
  };

  const uploadOne = async (file, idx) => {
    setProgress(p => p.map((row, i) => i === idx ? { ...row, status: 'uploading' } : row));
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      const res = await apiFetch('/library/upload', { method: 'POST', body: fd });
      setProgress(p => p.map((row, i) => i === idx
        ? { ...row, status: 'done', upload_id: res.upload_id } : row));
      return res.upload_id;
    } catch (e) {
      setProgress(p => p.map((row, i) => i === idx
        ? { ...row, status: 'failed', error: e.message } : row));
      return null;
    }
  };

  const startUpload = async () => {
    if (!files.length) return;
    // Entering the upload phase flips the screen to the dedicated progress page
    // (uploadStarted covers the brief gaps between sequential files).
    setUploadStarted(true);
    // Sequential to keep server memory bounded — parallel would need a pool.
    const ids = [];
    for (let i = 0; i < files.length; i++) {
      const id = await uploadOne(files[i], i);
      if (id) ids.push(id);
    }
    if (!ids.length) { setUploadStarted(false); return; }
    setCommitting(true);
    try {
      const res = await apiFetch('/library/upload/batch-commit', {
        method: 'POST',
        body: JSON.stringify({ upload_ids: ids, lang }),
      });
      setJobId(res.job_id);
      setShowWizard(true);
    } catch (e) {
      setCommitting(false);
      setUploadStarted(false);
      alert(`Batch commit failed: ${e.message}`);
    }
  };

  const allDone = progress.length > 0 && progress.every(p => p.status === 'done' || p.status === 'failed');
  const anyUploading = progress.some(p => p.status === 'uploading');
  const successCount = progress.filter(p => p.status === 'done').length;

  // Page phase: the source picker gives way to a dedicated progress page the
  // moment work starts — uploading files, importing from Yandex, or indexing.
  const uploadBusy = uploadStarted || anyUploading || committing;
  const busy = (showWizard && !!jobId) || uploadBusy || yandexImporting;

  // Server-mode opt-in (MEMBER_INDEX_ROOT): if the operator mounted a trusted
  // folder and exposed it via /instance/config, members may index it in place
  // instead of uploading. Fetch once; the button below renders only when set.
  useEffect(() => {
    apiFetch('/instance/config')
      .then(cfg => { if (cfg && cfg.mode === 'server' && cfg.member_index_root) setMemberIndexRoot(cfg.member_index_root); })
      .catch(() => {});
  }, []);

  // Reload / navigation away mid-indexing: the server keeps the per-account job
  // slot (same contract App uses at /library/status), so ask it on mount and
  // jump straight back to the progress page instead of the source picker.
  useEffect(() => {
    let alive = true;
    apiFetch('/library/status')
      .then(st => {
        if (!alive || !st || !st.job_id) return;
        if (st.overall_status === 'running' || st.overall_status === 'pending') {
          setWelcomed(true);   // they're mid-flow — the marketing slide is behind them
          setJobId(st.job_id);
          setShowWizard(true);
        }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  // Index the mounted root in place, then hand the returned job to the same
  // MemberIndexing progress view the upload flow uses.
  const startFolderIndex = async () => {
    if (indexingFolder || anyUploading || committing) return;
    setIndexingFolder(true);
    try {
      const res = await apiFetch('/library/index', {
        method: 'POST',
        body: JSON.stringify({ folder_path: memberIndexRoot }),
      });
      if (res && res.job_id) { setJobId(res.job_id); setShowWizard(true); }
      else throw new Error((res && res.message) || 'no job_id');
    } catch (e) {
      setIndexingFolder(false);
      alert(`${ru ? 'Не удалось добавить музыку' : 'Failed to add music'}: ${e.message}`);
    }
  };

  const OB_VARS = {
    '--ob-glass-bg': isDark ? 'rgba(255,255,255,.055)' : 'rgba(255,255,255,.62)',
    '--ob-glass-sheen': isDark ? 'rgba(255,255,255,.12)' : 'rgba(255,255,255,.9)',
    '--ob-glass-edge': isDark ? 'rgba(255,255,255,.18)' : 'rgba(0,0,0,.10)',
    '--ob-card-bg': isDark ? 'linear-gradient(180deg,#1d1d23,#131318)' : 'linear-gradient(180deg,#ffffff,#eef0f5)',
    '--ob-card-edge': isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.06)',
    '--ob-blob1':'#7d5cff', '--ob-blob2':'#3aa0ff', '--ob-blob3':'#c061ff',
  };

  const header = (
    <div style={{ display:'flex', justifyContent:'space-between', padding:'24px 32px' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
        <BrandMark size={36} isDark={isDark} />
        <span className="serif" style={{ fontSize:'28px', letterSpacing:'-0.02em' }}>Musi<i style={{ color:'oklch(62% 0.2 275)' }}>X</i></span>
      </div>
      <TopRightControls isDark={isDark} lang={lang} onLang={onLang} onTheme={onTheme}
        showTheme={!showWizard} showSettings={false} />
    </div>
  );

  if (!welcomed) {
    return (
      <div className="grain ob-root" style={{ ...OB_VARS,
        width:'100vw', height:'100vh', overflow:'auto', position:'relative',
        background: isDark
          ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
          : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
        color: c.text }}>
        {header}
        <div style={{ maxWidth:'min(1100px, 94vw)', margin:'24px auto 48px', padding:'0 32px', position:'relative' }}>
          <DriftBackdrop />
          <GlassCard style={{ padding:'clamp(30px, 3vw, 60px) clamp(28px, 3vw, 56px)' }}>
            <WelcomeSlide ru={ru} c={c} onStart={dismissWelcome} />
          </GlassCard>
        </div>
      </div>
    );
  }

  return (
    <div className="grain ob-root" style={{ ...OB_VARS,
      width:'100vw', height:'100vh', overflow:'auto', position:'relative',
      background: isDark
        ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
        : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
      color: c.text,
    }}>
      {header}

      <div style={{ maxWidth:'900px', margin:'28px auto 64px', padding:'0 32px', position:'relative' }}>
        <div style={{ display:'flex', gap:24, position:'relative' }}>
          <DriftBackdrop />
          <SetupRail ru={ru}
            steps={[
              { key:'source',   label: ru?'Откуда музыка?':'Music source' },
              { key:'indexing', label: ru?'Подготовка музыки':'Preparing music' },
              { key:'done',     label: ru?'Готово!':'Done!' },
            ]}
            currentKey={busy ? 'indexing' : 'source'} />
          <div style={{ flex:1, position:'relative', zIndex:1, minWidth:0 }}>
        {/* ═══ Source picker — HIDDEN (not unmounted) once the flow is busy, so
            uploads driven from this component's state keep running. ═══ */}
        <div style={{ display: busy ? 'none' : undefined }}>
        <div className="mono" style={{ fontSize:'11px', color:c.textSubtle, letterSpacing:'0.24em', textTransform:'uppercase', marginBottom:'8px' }}>
          {ru?'Шаг 1 · Откуда музыка?':'Step 1 · Music source'}
        </div>
        <h2 className="serif" style={{ fontSize:'30px', lineHeight:'1.04', letterSpacing:'-0.02em', marginBottom:'22px' }}>
          {ru ? <>Откуда <i style={{ color:'oklch(62% 0.2 275)' }}>музыка</i>?</> : <>Where is your <i style={{ color:'oklch(62% 0.2 275)' }}>music</i> from?</>}
        </h2>

        {/* ═══ Premium hint (inline, one-time) ═══════════════════════════════ */}
        {premiumHint && !premiumHintDismissed && (
          <div className="ob-glass ob-listin" style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:'12px', padding:'14px 20px', borderRadius:'14px' }}>
            <div style={{ display:'flex', alignItems:'center', gap:'10px', flex:1, minWidth:0 }}>
              <span style={{ width:'32px', height:'32px', borderRadius:'8px', background:'linear-gradient(135deg, #ffcc00, #ff5c5c)', color:'#1a1a1a', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'15px', fontWeight:'800', flexShrink:0 }}>?</span>
              <span style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.5 }}>
                {ru
                  ? 'Импорт из Яндекс Музыки доступен в PRO-версии MusiX — подписка открывает доступ к плейлистам и «Мне нравится».'
                  : 'Yandex Music import is available in MusiX PRO — a subscription unlocks playlists and liked tracks.'}
              </span>
            </div>
            <button onClick={dismissPremiumHint} style={{ fontSize:'18px', color:c.textSubtle, cursor:'pointer', background:'none', border:'none', padding:'4px', lineHeight:1, flexShrink:0 }}>✕</button>
          </div>
        )}

        {/* ═══ Два кликабельных блока «или-или» ════════════════════════════ */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(260px, 1fr))', gap:'16px', marginBottom: (uploadExpanded || yandexExpanded) ? '0' : '16px', marginTop: (premiumHint && !premiumHintDismissed) ? '16px' : 0 }}>
          {/* ── Card 1: Yandex Music ── */}
          <button onClick={() => {
            if (anyUploading || committing) return;
            if (MUSIX_PREMIUM) {
              setYandexExpanded(!yandexExpanded);
              if (!yandexExpanded) setUploadExpanded(false);  // аккордеон: открыли Яндекс → файлы закрылись
            }
            else if (!premiumHintDismissed) setPremiumHint(true);
          }}
            disabled={anyUploading || committing}
            className="ob-glass"
            style={{
              display:'flex', flexDirection:'column', alignItems:'center', gap:'14px',
              padding:'32px 24px', borderRadius:'18px', cursor: MUSIX_PREMIUM && !anyUploading && !committing ? 'pointer' : 'default',
              opacity: MUSIX_PREMIUM && !anyUploading && !committing ? 1 : (MUSIX_PREMIUM ? 0.5 : 0.65),
              textAlign:'center', transition: 'transform 0.2s, box-shadow 0.2s',
              border: `1px solid ${yandexExpanded && MUSIX_PREMIUM ? 'oklch(62% 0.2 275)' : (MUSIX_PREMIUM && !anyUploading && !committing ? 'rgba(255,255,255,0.12)' : c.border)}`,
              boxShadow: yandexExpanded && MUSIX_PREMIUM ? '0 0 0 3px oklch(62% 0.2 275 / 0.18)' : 'none',
            }}
            onMouseEnter={MUSIX_PREMIUM && !anyUploading && !committing ? (e) => { e.currentTarget.style.transform = 'translateY(-2px)'; } : undefined}
            onMouseLeave={MUSIX_PREMIUM && !anyUploading && !committing ? (e) => { e.currentTarget.style.transform = ''; } : undefined}>
            <div style={{
              width:'56px', height:'56px', borderRadius:'16px',
              background:'linear-gradient(135deg, #ffcc00, #ff5c5c)',
              color:'#1a1a1a', display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'28px', fontWeight:'800', flexShrink:0,
            }}>Я</div>
            <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
              <span className="serif" style={{ fontSize:'17px', fontWeight:'600', color:c.text }}>
                {ru ? 'Яндекс Музыка' : 'Yandex Music'}
              </span>
              {MUSIX_PREMIUM ? <PremiumBadge /> : (
                <span style={{
                  display:'inline-flex', alignItems:'center', gap:'3px',
                  padding:'2px 7px', borderRadius:'999px',
                  fontSize:'9px', fontWeight:'700', letterSpacing:'0.22em',
                  color: isDark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.25)',
                  border:`1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  background:'transparent',
                }}>🔒 {ru?'ПРО':'PRO'}</span>
              )}
            </div>
            <span style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.5 }}>
              {MUSIX_PREMIUM
                ? (ru ? 'Импортировать плейлисты и «Мне нравится»' : 'Import playlists and liked tracks')
                : (ru ? 'Доступно в PRO-версии' : 'Available in PRO')}
            </span>
          </button>

          {/* ── Card 2: Local Files ── */}
          <button onClick={() => {
            setUploadExpanded(!uploadExpanded);
            if (!uploadExpanded) setYandexExpanded(false);  // аккордеон: открыли файлы → Яндекс закрылся
          }}
            className="ob-glass"
            style={{
              display:'flex', flexDirection:'column', alignItems:'center', gap:'14px',
              padding:'32px 24px', borderRadius:'18px', cursor:'pointer',
              textAlign:'center', transition: 'transform 0.2s, box-shadow 0.2s',
              border: `1px solid ${uploadExpanded ? 'oklch(62% 0.2 275)' : 'rgba(255,255,255,0.12)'}`,
              boxShadow: uploadExpanded ? '0 0 0 3px oklch(62% 0.2 275 / 0.18)' : 'none',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = ''; }}>
            <div style={{
              width:'56px', height:'56px', borderRadius:'16px',
              background: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.05)',
              display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'28px', flexShrink:0,
            }}>📁</div>
            <span className="serif" style={{ fontSize:'17px', fontWeight:'600', color:c.text }}>
              {ru ? 'Свои файлы' : 'My files'}
            </span>
            <span style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.5 }}>
              {ru ? 'Загрузить FLAC, MP3, M4A с компьютера' : 'Upload FLAC, MP3, M4A from your computer'}
            </span>
          </button>
        </div>

        {/* ═══ Развёрнутый контент: локальные файлы ═══════════════════════════ */}
        <ObExpand open={uploadExpanded}>
            <div>
              <ProcessingModeBadge isDark={isDark} lang={lang} style={{ marginBottom:'16px' }} />

              <div className="ob-glass"
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                style={{ padding:'44px 28px', borderRadius:'18px', border:`2px dashed ${c.border}`, textAlign:'center' }}>
                <div style={{ fontSize:'40px', marginBottom:'10px' }}>🎵</div>
                <p style={{ fontSize:'14px', color:c.textMuted, marginBottom:'16px' }}>
                  {lang==='ru'?'Перетащи аудиофайлы сюда, или':'Drag audio files here, or'}
                </p>
                <input ref={attachFolderInput} type="file" multiple
                  onChange={onPickFolder}
                  disabled={anyUploading||committing}
                  style={{ display:'none' }}
                  id="onboard-folder-picker" />
                <input type="file" multiple
                  accept=".flac,.mp3,.m4a,.aac,.ogg,.wav,.opus"
                  onChange={onPick}
                  disabled={anyUploading||committing}
                  style={{ display:'none' }}
                  id="onboard-picker" />
                <div style={{ display:'flex', gap:'10px', justifyContent:'center', flexWrap:'wrap' }}>
                  <label htmlFor="onboard-folder-picker" className="ske-accent"
                    style={{ display:'inline-block', padding:'11px 22px', borderRadius:'10px', fontSize:'14px',
                      cursor:'pointer', opacity: anyUploading||committing ? 0.5 : 1,
                      pointerEvents: anyUploading||committing ? 'none' : 'auto' }}>
                    {lang==='ru'?'ВЫБРАТЬ ПАПКУ':'PICK FOLDER'}
                  </label>
                  <label htmlFor="onboard-picker" className={ske('btn', isDark)}
                    style={{ display:'inline-block', padding:'11px 22px', borderRadius:'10px', fontSize:'14px',
                      color:c.textMuted,
                      cursor:'pointer', opacity: anyUploading||committing ? 0.5 : 1,
                      pointerEvents: anyUploading||committing ? 'none' : 'auto' }}>
                    {lang==='ru'?'ИЛИ ФАЙЛЫ':'OR FILES'}
                  </label>
                </div>
              </div>

              {/* Premium: link Yandex account for better metadata on uploads */}
              {MUSIX_PREMIUM && (
                <YandexEnhanceLink isDark={isDark} lang={lang} />
              )}
            </div>
        </ObExpand>
        </div>

        {/* ═══ Яндекс Музыка — OUTSIDE the hidden source wrapper: the flow must
            stay mounted (and visible) when its import turns the page into the
            dedicated progress view. ═══ */}
        <div style={{ display: (busy && !yandexImporting) ? 'none' : undefined }}>
          <ObExpand open={yandexExpanded}>
            <YandexImportFlow isDark={isDark} lang={lang} onDone={onDone}
              onPhase={(p) => setYandexImporting(p === 'progress')}
              onBack={() => { setYandexExpanded(false); setYandexImporting(false); }} />
          </ObExpand>
        </div>

        {/* ═══ Отдельная страница: загрузка файлов на сервер ═══════════════════ */}
        {uploadBusy && !showWizard && (
          <div className="ob-listin">
            <div className="mono" style={{ fontSize:'11px', color:c.textSubtle, letterSpacing:'0.24em', textTransform:'uppercase', marginBottom:'8px' }}>
              {ru?'Шаг 2 · Подготовка музыки':'Step 2 · Preparing music'}
            </div>
            <h2 className="serif" style={{ fontSize:'30px', lineHeight:'1.04', letterSpacing:'-0.02em', marginBottom:'18px' }}>
              {committing
                ? (ru ? <>Запускаем <i style={{ color:'oklch(62% 0.2 275)' }}>обработку</i>…</> : <>Starting <i style={{ color:'oklch(62% 0.2 275)' }}>processing</i>…</>)
                : (ru ? <>Загружаем <i style={{ color:'oklch(62% 0.2 275)' }}>файлы</i>…</> : <>Uploading your <i style={{ color:'oklch(62% 0.2 275)' }}>files</i>…</>)}
            </h2>
            <div className="ob-glass" style={{ padding:'18px 20px', borderRadius:'14px', marginBottom:'14px' }}>
              <OBStageBar c={c} label={ru ? 'Загрузка на сервер' : 'Upload to server'}
                state={committing ? 'done' : 'running'}
                pct={progress.length > 0 ? Math.round(100 * progress.filter(p => p.status === 'done' || p.status === 'failed').length / progress.length) : 0}
                count={`${progress.filter(p => p.status === 'done').length}/${progress.length}`} />
              <div style={{ display:'flex', alignItems:'center', gap:'8px', fontSize:'12px', color:'#d9a76a', lineHeight:1.5 }}>
                <span aria-hidden>⚠</span>
                {ru ? 'Пока файлы загружаются с этого устройства — не закрывайте страницу. Дальше всё сделает сервер, и страницу можно будет закрыть.'
                    : 'While files are uploading from this device, keep the page open. After that the server takes over and you may leave.'}
              </div>
            </div>
          </div>
        )}

        {/* ═══ Прогресс загрузки файлов (список) — виден при выборе и во время
            загрузки; на странице индексации уже не нужен ═══ */}
        <div style={{ display: showWizard || yandexImporting ? 'none' : undefined }}>
        {progress.length > 0 && (
          <div className="ob-glass ob-listin" style={{ marginTop:'14px', padding:'16px 20px', borderRadius:'14px' }}>
            <div className="mono" style={{ fontSize:'13px', color:c.textSubtle, marginBottom:'10px' }}>
              {progress.length} {lang==='ru'?'ФАЙЛОВ':'FILES'}
              {successCount > 0 && ` · ${successCount} ${lang==='ru'?'ГОТОВО':'DONE'}`}
            </div>
            <div style={{ maxHeight:'240px', overflow:'auto' }}>
              {progress.map((p, i) => (
                <div key={i} style={{ display:'flex', justifyContent:'space-between',
                  padding:'6px 0', borderBottom:`1px solid ${c.border}`, fontSize:'13px' }}>
                  <span style={{ overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{p.name}</span>
                  <span style={{ color:
                    p.status === 'done' ? 'oklch(70% 0.18 145)' :
                    p.status === 'failed' ? c.red :
                    p.status === 'uploading' ? c.textMuted : c.textSubtle,
                    flexShrink:0, marginLeft:'12px' }}>
                    {p.status === 'done' ? '✓' :
                     p.status === 'failed' ? `✗ ${p.error || 'failed'}` :
                     p.status === 'uploading' ? '…' : '·'}
                  </span>
                </div>
              ))}
            </div>

            {!allDone && !anyUploading && !uploadStarted && (
              <button onClick={startUpload} className="ske-accent"
                style={{ width:'100%', marginTop:'14px', padding:'12px 20px', borderRadius:'12px',
                  fontSize:'14px', fontWeight:'600', letterSpacing:'0.08em', cursor:'pointer' }}>
                {lang==='ru'?'▶ ЗАГРУЗИТЬ И ДОБАВИТЬ':'▶ Upload & add'}
              </button>
            )}
            {anyUploading && (
              <div style={{ marginTop:'14px', textAlign:'center', color:c.textMuted, fontSize:'13px' }}>
                <Spinner size={14} /> {lang==='ru'?'Загрузка…':'Uploading…'}
              </div>
            )}
          </div>
        )}
        </div>

        {/* ═══ Dev section: mounted folder (hidden under spoiler) ════════════ */}
        <div style={{ display: busy ? 'none' : undefined }}>
        {memberIndexRoot && (
          <details style={{ marginTop:'18px', opacity:0.6 }}>
            <summary style={{ cursor:'pointer', fontSize:'12px', color:c.textSubtle, letterSpacing:'0.12em', textTransform:'uppercase', userSelect:'none' }}>
              {ru ? '🛠 Для разработчиков — примонтированная папка' : '🛠 Developer — mounted folder'}
            </summary>
            <div className="ob-glass" style={{ marginTop:'10px', padding:'16px 20px', borderRadius:'14px' }}>
              <div style={{ fontSize:'13px', color:c.textMuted, lineHeight:1.55, marginBottom:'12px' }}>
                {ru
                  ? <>Музыка примонтирована на сервере: <code style={{ color:c.text }}>{memberIndexRoot}</code>. Заменит текущую библиотеку.</>
                  : <>Music mounted on server: <code style={{ color:c.text }}>{memberIndexRoot}</code>. Replaces your current library.</>}
              </div>
              <button onClick={startFolderIndex} disabled={indexingFolder||anyUploading||committing}
                className={ske('btn', isDark)}
                style={{ padding:'10px 18px', borderRadius:'10px', fontSize:'13px', fontWeight:'600',
                  cursor: indexingFolder||anyUploading||committing ? 'not-allowed' : 'pointer',
                  opacity: indexingFolder||anyUploading||committing ? 0.5 : 1 }}>
                {indexingFolder
                  ? (ru?'Запуск…':'Starting…')
                  : (ru?`Индексировать: ${memberIndexRoot}`:`Index: ${memberIndexRoot}`)}
              </button>
            </div>
          </details>
        )}
        </div>
        {showWizard && jobId && (
          <MemberIndexing ru={ru} c={c} isDark={isDark} jobId={jobId} onDone={onDone} />
        )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Thin wrapper that reuses IndexingModal but subscribes to the SSE stream for the
// job_id returned by /library/upload/batch-commit. The job lives in the same
// shared JobTracker as the folder-scan flow, so the /index/progress stream works.
function UploadIndexingWizard({ isDark, lang, jobId, onDone }) {
  const c = useColors(isDark);
  const [stageProgress, setStageProgress] = useState({});
  const [stepStatus, setStepStatus] = useState({
    lyrics:'idle', facts:'idle', metadata:'idle', dense:'idle', audio:'idle', analysis:'idle',
  });
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);
  // Server-mode: after FACTS the backend runs the awaited AI tasks (bios/facts)
  // before flipping overall_status to 'completed'. We stay on this screen until
  // then; aiWaiting drives a "finishing" note so the wait is explained.
  const [aiWaiting, setAiWaiting] = useState(false);
  const [aiStages, setAiStages] = useState(null);  // live guru progress from SSE

  useEffect(() => {
    const evt = new EventSource(`${API}/index/progress/${jobId}`);
    evt.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.stages) {
          const statusMap = { completed:'done', failed:'failed', running:'running', pending:'pending' };
          setStepStatus(prev => {
            const next = { ...prev };
            for (const [k, v] of Object.entries(data.stages)) {
              next[k] = statusMap[v.status] || v.status || 'pending';
            }
            return next;
          });
          setStageProgress(prev => {
            const next = { ...prev };
            for (const [k, v] of Object.entries(data.stages)) {
              next[k] = {
                current: v.current ?? prev[k]?.current ?? 0,
                total: v.total ?? prev[k]?.total ?? 0,
                eta: v.eta_seconds ?? prev[k]?.eta ?? null,
                message: v.message ?? prev[k]?.message ?? null,
                found: v.found ?? prev[k]?.found ?? null,
                not_found: v.not_found ?? prev[k]?.not_found ?? null,
              };
            }
            return next;
          });
        }
        if (data.ai_stages) { setAiStages(data.ai_stages); setAiWaiting(true); }
        // Fallback for streams without ai_stages: once EVERY core stage is done
        // but the job isn't 'completed', the backend is in the awaited AI phase
        // (facts alone completes early — it runs in parallel with encoding).
        if (data.overall_status !== 'completed' && data.stages
            && Object.values(data.stages).every(s => s.status === 'completed')) {
          setAiWaiting(true);
        }
        if (data.overall_status === 'completed') {
          evt.close(); setDone(true);
          setTimeout(onDone, 1200);
        } else if (data.overall_status === 'failed') {
          evt.close(); setError(data.error || data.message);
        }
      } catch {}
    };
    evt.onerror = () => { evt.close(); if (!done) setError(lang==='ru'?'Соединение потеряно':'Connection lost'); };
    return () => evt.close();
  }, [jobId]);

  return (
    <>
      <IndexingModal
        isDark={isDark} lang={lang} collectionName=""
        stepStatus={stepStatus} trackCount={done ? (stageProgress.lyrics?.current ?? 0) : null}
        errorMessage={error}
        onClose={() => {}}
        stageProgress={stageProgress}
      />
      {aiWaiting && !done && !error && (
        <div style={{
          position:'fixed', left:0, right:0, bottom:0, zIndex:10000,
          padding:'12px 16px', backdropFilter:'blur(6px)',
          background: isDark ? 'rgba(20,20,28,0.92)' : 'rgba(250,250,255,0.95)',
          color: isDark ? '#cdd5e0' : '#334',
          borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'}`,
        }}>
          <div style={{ maxWidth:'620px', margin:'0 auto' }}>
            <div className="mono" style={{ display:'flex', alignItems:'center', gap:'8px', fontSize:'10px',
              letterSpacing:'0.2em', textTransform:'uppercase', color:'#c3b8ff', marginBottom:'10px' }}>
              <Spinner size={12} /> ✨ {lang==='ru' ? 'С помощью гуру' : 'With the guru'}
            </div>
            {aiStages
              ? <GuruStagesFromSse ru={lang==='ru'} c={c} aiStages={aiStages} />
              : <AiEnrichProgress ru={lang==='ru'} c={c} />}
            <div style={{ fontSize:'11px', color: isDark ? 'rgba(255,255,255,.45)' : 'rgba(0,0,0,.45)', marginTop:'4px' }}>
              {lang==='ru' ? 'Плеер откроется автоматически, как только всё будет готово.' : 'The player opens automatically once everything is ready.'}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function OnboardingScreen({ isDark, lang, onDone, onLang, onTheme, indexingJob }) {
  const c = useColors(isDark);
  // Phase C: branch on instance mode. Sharing keeps the folder-input UI below;
  // server replaces it with a drag-drop uploader. Fetch /instance/config once on
  // mount (public route) so the right UI shows on the very first visit.
  const [mode, setMode] = useState(null);   // null while loading
  const [modeLoadError, setModeLoadError] = useState(null);
  useEffect(() => {
    let alive = true;
    apiFetch('/instance/config')
      .then(cfg => { if (alive) setMode(cfg?.mode || 'sharing'); })
      .catch(e => { if (alive) { setModeLoadError(e.message); setMode('sharing'); } });
    return () => { alive = false; };
  }, []);
  const [folderPath, setFolderPath] = useState('');
  const [collName, setCollName] = useState('my_collection');
  const [betterLyrics, setBetterLyrics] = useState(false);
  const [refineMetadata, setRefineMetadata] = useState(false);
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState('');
  // Progress comes from the App-level indexingJob hook (spec phase 2) — a
  // reload mid-indexing resumes tracking via GET /library/status in App, and
  // the effects below re-open the modal and finish onboarding on completion.
  const indexing = indexingJob.status === 'running';
  const stepStatus = indexingJob.stepStatus;
  const stageProgress = indexingJob.stageProgress;
  const modalTrackCount = indexingJob.trackCount;
  const modalError = indexingJob.error === 'connection_lost'
    ? (lang === 'ru' ? 'Соединение потеряно' : 'Connection lost')
    : indexingJob.error;
  const [showModal, setShowModal] = useState(() => indexingJob.status === 'running');

  // Resumed job attached after mount (App's /library/status answer races this
  // screen's first render) — surface the modal as soon as it starts reporting.
  useEffect(() => {
    if (indexingJob.status === 'running') setShowModal(true);
  }, [indexingJob.status]);

  // Job ran to completion (fresh or resumed) → finish onboarding.
  const prevJobStatusRef = useRef(indexingJob.status);
  useEffect(() => {
    const prev = prevJobStatusRef.current;
    prevJobStatusRef.current = indexingJob.status;
    if (prev !== 'running' || indexingJob.status !== 'completed') return;
    const t = setTimeout(onDone, 800);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexingJob.status]);

  const handlePick = async () => {
    setPicking(true); setError('');
    try { const res = await apiFetch('/library/pick-folder'); if (res.path) setFolderPath(res.path); else setError(lang==='ru'?'Папка не выбрана':'No folder picked'); }
    catch (e) { setError(e.message); } finally { setPicking(false); }
  };

  const handleIndex = async () => {
    if (!folderPath || indexing) return;
    setShowModal(true);
    indexingJob.begin();
    try {
      const res = await apiFetch('/library/index', { method:'POST',
        body: JSON.stringify({ folder_path:folderPath, better_lyrics_quality:betterLyrics, enhance_by_musicbrainz:refineMetadata }) });
      if (res.status === 'failed') { indexingJob.fail(res.message); return; }
      if (!res.job_id) {
        // Immediate completion — the effect above fires onDone.
        indexingJob.completeSync(res.count || 0);
        return;
      }
      indexingJob.attach(res.job_id);
    } catch (e) { indexingJob.fail(e.message); }
  };

  // Phase C mode gate — runs before the sharing-mode UI below.
  if (mode === null) {
    return (
      <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh',
        background: isDark ? '#0a0a0e' : '#ececf3' }}>
        <Spinner size={20} />
      </div>
    );
  }
  if (mode === 'server') {
    return <ServerOnboardingScreen
      isDark={isDark} lang={lang} onDone={onDone}
      onLang={onLang} onTheme={onTheme}
    />;
  }
  // Sharing mode — existing folder-index UI unchanged below.

  return (
    <div className="grain ob-root" style={{
      '--ob-glass-bg': isDark ? 'rgba(255,255,255,.055)' : 'rgba(255,255,255,.62)',
      '--ob-glass-sheen': isDark ? 'rgba(255,255,255,.12)' : 'rgba(255,255,255,.9)',
      '--ob-glass-edge': isDark ? 'rgba(255,255,255,.18)' : 'rgba(0,0,0,.10)',
      '--ob-blob1':'#7d5cff', '--ob-blob2':'#3aa0ff', '--ob-blob3':'#c061ff',
      width:'100vw', height:'100vh', overflow:'auto', position:'relative',
      background: isDark
        ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
        : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
      color: c.text,
    }}>
      <div style={{ display:'flex', justifyContent:'space-between', padding:'24px 32px' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
          <BrandMark size={36} isDark={isDark} />
          <span className="serif" style={{ fontSize:'28px', letterSpacing:'-0.02em' }}>Musi<i style={{ color:'oklch(62% 0.2 275)' }}>X</i></span>
        </div>
        <TopRightControls isDark={isDark} lang={lang} onLang={onLang} onTheme={onTheme}
          showTheme={!indexing} showSettings={false} />
      </div>

      <div style={{ maxWidth:'900px', margin:'28px auto 64px', padding:'0 32px', position:'relative' }}>
        <div style={{ display:'flex', gap:24, position:'relative' }}>
          <DriftBackdrop />
          <SetupRail ru={lang==='ru'}
            steps={[
              { key:'source',   label: lang==='ru'?'Откуда музыка?':'Music source' },
              { key:'indexing', label: lang==='ru'?'Подготовка библиотеки':'Preparing library' },
              { key:'done',     label: lang==='ru'?'Готово!':'Done!' },
            ]}
            currentKey={indexing ? 'indexing' : 'source'} />
          <div style={{ flex:1, position:'relative', zIndex:1, minWidth:0 }}>
        <div className="mono" style={{ fontSize:'14px', color:c.textSubtle, letterSpacing:'0.32em', marginBottom:'14px' }}>
          — {lang==='ru'?'ПЕРВЫЙ ЗАПУСК':'FIRST RUN'}
        </div>
        <h1 className="serif" style={{ fontSize:'clamp(38px,5vw,56px)', lineHeight:'1.02', letterSpacing:'-0.025em', marginBottom:'18px' }}>
          {lang==='ru' ? <>Добро <i style={{ color:'oklch(62% 0.2 275)' }}>пожаловать</i>.</> : <>Welcome <i style={{ color:'oklch(62% 0.2 275)' }}>aboard</i>.</>}
        </h1>
        <p style={{ fontSize:'15px', color:c.textMuted, lineHeight:'1.6', marginBottom:'34px' }}>
          {lang==='ru'
            ? 'MusiX добавляет музыку локально и позволяет искать треки по смыслу — текстам и звуку. Ничего не уходит в облако.'
            : 'MusiX adds your music locally so you can search tracks by meaning — lyrics and audio. Nothing leaves your machine.'}
        </p>

        <ProcessingModeBadge isDark={isDark} lang={lang} style={{ marginBottom:'16px' }} />

        <div className={ske('panel', isDark)} style={{ padding:'26px 28px', borderRadius:'18px' }}>
          <div className="mono" style={{ fontSize:'14px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>
            {lang==='ru'?'ИМЯ БИБЛИОТЕКИ':'LIBRARY NAME'}
          </div>
          <input value={collName} onChange={e=>setCollName(e.target.value)} disabled={indexing}
            className={ske('inset', isDark)} style={{ width:'100%', padding:'10px 13px', borderRadius:'10px', border:'none',
              color:c.text, fontSize:'15px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'18px' }} />

          <div className="mono" style={{ fontSize:'14px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>
            {lang==='ru'?'ПАПКА С МУЗЫКОЙ':'MUSIC FOLDER'}
          </div>
          <div style={{ display:'flex', gap:'8px', marginBottom:'14px' }}>
            <button onClick={handlePick} disabled={picking||indexing}
              className={ske('btn', isDark)} style={{ padding:'11px 16px', borderRadius:'10px', fontSize:'14px',
                color:c.textMuted, display:'flex', alignItems:'center', gap:'6px', flexShrink:0,
                cursor: picking||indexing?'not-allowed':'pointer' }}>
              {picking ? <Spinner size={12} /> : '📁'}
              {picking ? '…' : (lang==='ru'?'Выбрать':'Pick')}
            </button>
            <input value={folderPath} onChange={e=>setFolderPath(e.target.value)} disabled={indexing}
              placeholder="/path/to/music"
              className={ske('inset', isDark)} style={{ flex:1, padding:'11px 13px', borderRadius:'10px', border:'none',
                color:c.text, fontSize:'15px', outline:'none', fontFamily:"'JetBrains Mono', monospace" }} />
          </div>

          <label style={{ display:'flex', alignItems:'center', gap:'8px', fontSize:'14px', color:c.textMuted, marginBottom:'8px' }}>
            <ToggleSwitch checked={betterLyrics} onChange={v=>setBetterLyrics(v)} isDark={isDark} />
            {lang==='ru'?'Лучшее качество текстов (Musixmatch — медленнее)':'Better lyrics (Musixmatch — slower)'}
          </label>
          <label style={{ display:'flex', alignItems:'center', gap:'8px', fontSize:'14px', color:c.textMuted, marginBottom:'18px' }}>
            <ToggleSwitch checked={refineMetadata} onChange={v=>setRefineMetadata(v)} isDark={isDark} />
            {lang==='ru'?'Дополнить пропущенные метаданные из интернета':'Refine metadata with online lookup'}
          </label>

          <button onClick={handleIndex} disabled={indexing||!folderPath.trim()}
            className="ske-accent" style={{
              width:'100%', padding:'13px 20px', borderRadius:'12px',
              fontSize:'15px', fontWeight:'600', letterSpacing:'0.08em',
              opacity: indexing||!folderPath.trim() ? 0.5 : 1,
              cursor: indexing||!folderPath.trim() ? 'not-allowed' : 'pointer',
              display:'flex', alignItems:'center', justifyContent:'center', gap:'8px',
            }}>
            {indexing ? <><Spinner size={14} color="white" /> {lang==='ru'?'ИДЁТ ОБРАБОТКА':'Adding music…'}</> : (lang==='ru'?'▶ ДОБАВИТЬ МУЗЫКУ':'▶ Add music')}
          </button>

          {error && <div style={{ marginTop:'14px', padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'14px' }}>{error}</div>}
        </div>
          </div>
        </div>
      </div>

      {showModal && (
        <IndexingModal isDark={isDark} lang={lang} collectionName={collName}
          stepStatus={stepStatus} trackCount={modalTrackCount} errorMessage={modalError}
          onClose={() => { setShowModal(false); indexingJob.reset(); }}
          stageProgress={stageProgress} premiumNote />
      )}
    </div>
  );
}

// ─── No-Qdrant screen ─────────────────────────────────────────────────────────
function NoQdrantScreen({ isDark, lang }) {
  const c = useColors(isDark);
  return (
    <div className="grain" style={{
      width:'100vw', height:'100vh', display:'flex', alignItems:'center', justifyContent:'center',
      background: isDark ? 'radial-gradient(ellipse at top, #15151b 0%, #07070a 100%)' : 'radial-gradient(ellipse at top, #fafaff 0%, #e3e2e8 100%)',
      color: c.text, padding:'24px',
    }}>
      <div className={ske('panel', isDark)} style={{ maxWidth:'480px', padding:'40px 36px', borderRadius:'22px', animation:'fadeInUp 0.4s ease' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'14px', marginBottom:'24px' }}>
          <BrandMark size={42} isDark={isDark} />
          <span className="serif" style={{ fontSize:'34px', letterSpacing:'-0.02em' }}>Musi<i style={{ color:'oklch(62% 0.2 275)' }}>X</i></span>
        </div>
        <div className="mono" style={{ fontSize:'14px', color:c.red, letterSpacing:'0.22em', marginBottom:'10px' }}>
          ● {lang==='ru'?'ОТКЛЮЧЕНО':'OFFLINE'}
        </div>
        <h2 className="serif" style={{ fontSize:'34px', lineHeight:'1', letterSpacing:'-0.02em', marginBottom:'14px' }}>
          {lang==='ru'?<><i style={{ color: c.red }}>Поиск временно недоступен</i>.</> : <><i style={{ color: c.red }}>Search temporarily unavailable</i>.</>}
        </h2>
        <p style={{ fontSize:'14px', color:c.textMuted, lineHeight:'1.6', marginBottom:'18px' }}>
          {lang==='ru'?'Поиск временно недоступен. Запусти его одной командой:':'Search is temporarily unavailable. Boot it with one command:'}
        </p>
        <div className={ske('display', isDark)} style={{
          padding:'14px 18px', borderRadius:'12px', marginBottom:'22px',
          fontFamily:"'JetBrains Mono', monospace", fontSize:'14px', color: c.green,
          textShadow: `0 0 8px ${c.green.replace(')', ' / 0.4)')}`,
          userSelect:'all',
        }}>docker-compose up -d</div>
        <button onClick={() => window.location.reload()} className="ske-accent" style={{
          width:'100%', padding:'12px 20px', borderRadius:'12px',
          fontSize:'15px', fontWeight:'600', letterSpacing:'0.06em',
        }}>
          ↻ {lang==='ru'?'ПЕРЕПРОВЕРИТЬ':'RE-CHECK'}
        </button>
      </div>
    </div>
  );
}

// ─── Delete confirm modal ─────────────────────────────────────────────────────
function DeleteConfirm({ isDark, lang, name, onConfirm, onCancel }) {
  const c = useColors(isDark);
  return (
    <div onClick={onCancel} style={{
      position:'fixed', inset:0, zIndex:120,
      background:'rgba(0,0,0,0.55)', backdropFilter:'blur(6px)',
      display:'flex', alignItems:'center', justifyContent:'center',
      animation:'fadeIn 0.2s ease',
    }}>
      <div className={ske('panel', isDark)} onClick={e=>e.stopPropagation()} style={{
        width:'400px', maxWidth:'90vw', padding:'28px 30px', borderRadius:'18px',
        animation:'scaleIn 0.25s cubic-bezier(.22,.9,.3,1)',
      }}>
        <div className="mono" style={{ fontSize:'14px', color:c.red, letterSpacing:'0.22em', marginBottom:'10px' }}>
          ● {lang==='ru'?'УДАЛЕНИЕ':'DELETE'}
        </div>
        <div className="serif" style={{ fontSize:'26px', lineHeight:'1.05', letterSpacing:'-0.01em', marginBottom:'12px', color:c.text }}>
          {lang==='ru'?<>Удалить <i>{name}</i>?</> : <>Delete <i>{name}</i>?</>}
        </div>
        <p style={{ fontSize:'15px', color:c.textMuted, marginBottom:'22px', lineHeight:'1.5' }}>
          {lang==='ru'?'Библиотека и все данные будут удалены безвозвратно.':'Your library and all its data will be permanently deleted.'}
        </p>
        <div style={{ display:'flex', gap:'10px' }}>
          <button onClick={onCancel} className={ske('btn', isDark)} style={{
            flex:1, padding:'11px 18px', borderRadius:'10px', fontSize:'14px', fontWeight:'600',
            color: c.textMuted, letterSpacing:'0.06em',
          }}>{lang==='ru'?'ОТМЕНА':'CANCEL'}</button>
          <button onClick={onConfirm} style={{
            flex:1, padding:'11px 18px', borderRadius:'10px', fontSize:'14px', fontWeight:'600',
            color:'white', letterSpacing:'0.06em',
            background:'linear-gradient(180deg, oklch(60% 0.21 25), oklch(48% 0.22 25))',
            boxShadow:'inset 0 1px 0 rgba(255,255,255,0.2), inset 0 -1px 0 rgba(0,0,0,0.25), 0 3px 10px oklch(58% 0.21 25 / 0.4)',
          }}>{lang==='ru'?'УДАЛИТЬ':'DELETE'}</button>
        </div>
      </div>
    </div>
  );
}

// ─── Song facts helper (hardcoded, track-aware) ─────────────────────────────
function getSongFacts(track, lang) {
  if (!track) return [];
  const t = lang === 'ru';
  const hue = ((track.title?.charCodeAt(0)||65)*37 + (track.artist?.charCodeAt(0)||65)*17) % 360;
  const keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  const modes = ['Major','Minor'];
  const key = keys[Math.abs(track.title?.charCodeAt(0)||0) % keys.length];
  const mode = modes[Math.abs(track.artist?.charCodeAt(0)||0) % 2];
  const bpm = 80 + Math.abs(track.title?.charCodeAt(1)||0) % 60;
  const moods = ['Melancholic','Energetic','Dreamy','Dark','Euphoric','Nostalgic','Intimate','Cinematic'];
  const mood = moods[Math.abs(track.title?.charCodeAt(2)||0) % moods.length];
  const lang2 = ['English','Russian','Japanese','Korean','French'];
  const langVal = lang2[Math.abs(track.artist?.charCodeAt(1)||0) % lang2.length];

  return [
    { icon: '♪', label: t?'Тональность':'Key', value: `${key} ${mode}` },
    { icon: '⏱', label: t?'Темп':'BPM', value: `${bpm} BPM` },
    { icon: '◐', label: t?'Настроение':'Mood', value: mood },
    { icon: 'μ', label: t?'Жанр':'Genre', value: track.genre || 'Indie' },
    { icon: '📅', label: t?'Год':'Year', value: track.year || '—' },
    { icon: '🌐', label: t?'Язык':'Lang', value: langVal },
    { icon: '⏱', label: t?'Длительность':'Length', value: track.duration ? secsToMMSS(track.duration) : '—' },
    { icon: '♫', label: t?'Альбом':'Album', value: track.album || '—' },
  ];
}

// ─── LYRICS BACK FACE ───────────────────────────────────────────────────────
function LyricsBackFace({
  track, isVisible, isDark, lang,
  expandedLines, explainStates, onToggleLyricExplain, aiActive,
}) {
  const scrollRef = useRef(null);

  // Reset scroll position when track changes — start every new lyric at the top.
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [track?.track_id]);

  const lyrics = track?.lyrics || null;
  const title = track?.title || '';

  const headingColor = isDark ? '#666' : '#8a8275';
  const bodyColor    = isDark ? '#d8d4c8' : '#2a2620';

  return (
    <div
      ref={scrollRef}
      className="player-scroll"
      style={{
        width: '100%',
        height: '100%',
        overflowY: 'auto',
        padding: '22px 24px',
        fontFamily: "Georgia, 'Noto Serif Display', serif",
        fontSize: 14,
        lineHeight: 1.7,
        color: bodyColor,
        whiteSpace: 'pre-wrap',
      }}
    >
      <div style={{
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        fontSize: 9,
        letterSpacing: '0.18em',
        color: headingColor,
        marginBottom: 14,
        textTransform: 'uppercase',
      }}>
        {title} · {lang === 'ru' ? 'ТЕКСТ' : 'LYRICS'}
      </div>

      {lyrics ? (
        <div style={{ whiteSpace: 'pre-wrap' }}>
          {lyrics.split('\n').map((line, i) => (
            line.trim() === '' ? (
              <div key={i} aria-hidden="true" style={{ height: 16 }} />
            ) : (
              <InlineLyricExplain
                key={i}
                line={line}
                lineIdx={i}
                expandedLines={expandedLines}
                explainStates={explainStates}
                onToggle={onToggleLyricExplain}
                isDark={isDark}
                aiActive={aiActive}
                lang={lang}
              />
            )
          ))}
        </div>
      ) : (
        <div style={{ color: headingColor, fontStyle: 'italic' }}>
          {lang === 'ru' ? 'тексты ещё не добавлены' : 'no lyrics added yet'}
        </div>
      )}
    </div>
  );
}

// ─── VIBE LINE ───────────────────────────────────────────────────────────────
function VibeLine({ trackId, lang, isDark }) {
  const [phrase, setPhrase] = useState(null);

  useEffect(() => {
    if (!trackId) { setPhrase(null); return; }
    let cancelled = false;
    apiFetch(
      `/metadata/tracks/${encodeURIComponent(trackId)}/sonic-vibe` +
      `?lang=${encodeURIComponent(lang || 'en')}`
    )
      .then(res => { if (!cancelled) setPhrase(res?.phrase || null); })
      .catch(() => { if (!cancelled) setPhrase(null); });
    return () => { cancelled = true; };
  }, [trackId, lang]);

  if (!phrase) return null;

  return (
    <div
      key={phrase}
      style={{
        fontFamily: "Georgia, 'Noto Serif Display', serif",
        fontStyle: 'italic',
        fontSize: 14,
        lineHeight: 1.45,
        textAlign: 'center',
        color: 'oklch(70% 0.10 75)',
        maxWidth: 420,
        margin: '8px auto 0',
        animation: 'vibeSlideIn 240ms cubic-bezier(0.22, 0.9, 0.3, 1)',
      }}
    >
      ❝ {phrase} ❞
    </div>
  );
}

// ─── PLAYER SCORE BARS ──────────────────────────────────────────────────────
function PlayerScoreBars({ breakdown, isDark }) {
  if (!breakdown) return null;

  const trackBg = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(22,22,32,0.10)';
  const fillBg = 'oklch(60% 0.18 270)';
  const labelColor = isDark ? '#888' : '#5a5a66';

  const row = (label, value) => {
    const present = value !== null && value !== undefined;
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 9,
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        opacity: present ? 1 : 0.35,
        marginBottom: 4,
      }}>
        <span style={{ width: 36, color: labelColor, letterSpacing: '0.1em' }}>{label}</span>
        <div style={{ flex: 1, height: 3, background: trackBg, borderRadius: 2, overflow: 'hidden' }}>
          {present && (
            <div style={{
              height: '100%',
              width: `${Math.max(0, Math.min(100, value * 100))}%`,
              background: fillBg,
              borderRadius: 2,
            }} />
          )}
        </div>
        <span style={{ width: 24, textAlign: 'right', color: labelColor }}>
          {present ? Math.round(value * 100) : '—'}
        </span>
      </div>
    );
  };

  return (
    <div style={{
      position: 'absolute',
      right: 12,
      top: '50%',
      transform: 'translateY(-50%)',
      width: 168,
      padding: '10px 12px',
      background: isDark
        ? 'linear-gradient(180deg, rgba(30,30,38,0.96), rgba(22,22,28,0.96))'
        : 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,244,250,0.96))',
      border: `1px solid ${isDark ? 'rgba(255,255,255,0.10)' : 'rgba(22,22,32,0.10)'}`,
      borderRadius: 8,
      backdropFilter: 'blur(12px) saturate(1.1)',
      WebkitBackdropFilter: 'blur(12px) saturate(1.1)',
      boxShadow: isDark
        ? '0 8px 26px rgba(0,0,0,0.5)'
        : '0 8px 22px rgba(40,30,60,0.18)',
      zIndex: 5,
      pointerEvents: 'none',
      animation: 'scoreBarsFade 120ms ease-out',
    }}>
      {row('TEXT',  breakdown.text_dense_score)}
      {row('AUDIO', breakdown.audio_score)}
    </div>
  );
}

// ─── AUDIO ANALYSIS + COVER COLOR + SPECTRUM BARS ───────────────────────────

// RGB → HSL (0-360 / 0-100 / 0-100). Used for theming the spectrum bars by
// the cover's dominant color in a way that survives lightness/saturation
// tweaks without going neon.
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0, s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if      (max === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / d + 2;
    else                h = (r - g) / d + 4;
    h /= 6;
  }
  return { h: h * 360, s: s * 100, l: l * 100 };
}

// Module-level singleton: AudioContext + AnalyserNode survive PlayerSection
// unmount/remount (navigation away → back). The browser only allows ONE
// MediaElementSource per <audio> element ever — recreating the analyser on
// each mount would throw "already connected" and leave the spectrum dead.
const _spectrumState = {
  ctx: null,
  analyser: null,
  dataArray: null,
  setupAttempted: false,
};

// Mobile browsers (Android Chrome in particular) suspend a background page's
// AudioContext when the screen locks or the tab is hidden. Once the <audio>
// element is routed through createMediaElementSource its output goes ONLY
// through that context — a suspended context means the element keeps
// "playing" (currentTime advances, the buffer downloads) while producing
// pure silence, and Chrome then drops the media notification because the tab
// stopped being audible. There is no way to un-route a MediaElementSource,
// so on mobile we never create it: no spectrum wave, but screen-off playback
// survives track changes. (iPadOS masquerades as Mac — detect via touch.)
const _IS_MOBILE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.platform));

// Idempotent: wire <audio> into AnalyserNode. MUST be invoked **synchronously
// inside a user gesture** (click/touch handler) so the AudioContext is born
// in 'running' state. If we defer this to a useEffect, the ctx ends up
// suspended and the spectrum stays dead on the first track — async setup
// leaves the resume() call outside the gesture window in some browsers.
function _setupSpectrumAnalyser(el) {
  if (_spectrumState.setupAttempted || !el) return;
  _spectrumState.setupAttempted = true;
  if (_IS_MOBILE) return; // audio stays wired directly to the output
  let ctx = null;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    ctx = new Ctx();
    const source = ctx.createMediaElementSource(el);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;             // 128 frequency bins
    analyser.smoothingTimeConstant = 0.62;
    source.connect(analyser);
    analyser.connect(ctx.destination);  // pass-through so audio still plays
    _spectrumState.ctx = ctx;
    _spectrumState.analyser = analyser;
    _spectrumState.dataArray = new Uint8Array(analyser.frequencyBinCount);
    // Safety net: the browser may suspend the context while the page is
    // hidden (sound dies but the element keeps advancing — see _IS_MOBILE
    // note above). Resume whenever media actually starts flowing or the page
    // becomes visible again; a prior user gesture makes resume() legal here.
    // Attached once — setup is guarded by setupAttempted, so no leak.
    const resume = () => {
      if (_spectrumState.ctx && _spectrumState.ctx.state === 'suspended') {
        _spectrumState.ctx.resume().catch(() => {});
      }
    };
    el.addEventListener('playing', resume);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) resume();
    });
  } catch (e) {
    if (ctx && typeof ctx.close === 'function') ctx.close().catch(() => {});
  }
}

// Hook: exposes the singleton refs and ensures setup eventually happens even
// when playback starts via paths that bypass togglePlay (track-click in the
// queue → playTrackAt, auto-play from initialTrack effect, etc.). Setup is
// idempotent — togglePlay still gets first crack at it from inside a user
// gesture for the "born-running" ctx case.
function useAudioAnalyser(audioRef, isPlaying) {
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);

  useEffect(() => {
    const el = audioRef?.current;
    if (el) _setupSpectrumAnalyser(el);  // no-op if already attempted
    analyserRef.current = _spectrumState.analyser;
    dataArrayRef.current = _spectrumState.dataArray;
    const ctx = _spectrumState.ctx;
    if (isPlaying && ctx && ctx.state === 'suspended') {
      ctx.resume().catch(() => {});
    }
  }, [audioRef, isPlaying]);

  return { analyserRef, dataArrayRef };
}

// Pull a "best" representative color out of the cover image. Strategy:
//   1) Downscale to 32×32 on an offscreen canvas
//   2) For each opaque pixel in the mid-luminance band, score = saturation
//      × proximity-to-ideal-luminance, pick the max
//   3) If no pixel qualifies (very monochrome cover), average all opaque
//      pixels as a fallback
// Cached per URL in a ref so track-toggle doesn't re-decode.
function useCoverColor(coverUrl) {
  const [color, setColor] = useState(null);
  const cacheRef = useRef({});
  useEffect(() => {
    if (!coverUrl) { setColor(null); return; }
    if (cacheRef.current[coverUrl]) { setColor(cacheRef.current[coverUrl]); return; }
    let cancelled = false;
    const img = new Image();
    // Covers are served same-origin (/api/v1/covers/...), where canvas reads
    // are always permitted. Do NOT set crossOrigin for them: the ambient layer
    // loads the very same cover first as a no-cors <img>/background-image, and a
    // later crossOrigin='anonymous' request for that URL can't reuse the cached
    // no-cors response — the browser raises a CORS error, the canvas is tainted,
    // getImageData throws, and the hook falls back to null → the purple default
    // (hsl(270,...)). Opt into CORS only for remote covers (incl. our own API
    // when the page runs from file:// — every cover URL is absolute then).
    const isRemote = /^https?:\/\//i.test(coverUrl) &&
      !coverUrl.startsWith(window.location.origin);
    if (isRemote) img.crossOrigin = 'anonymous';
    // Remote sampling gets its OWN cache key (?cors=1): cover responses are
    // cached immutable for a year, and a copy cached by a no-cors <img> before
    // the server started sending unconditional ACAO (or by a host that omits
    // it on no-cors requests) would be reused here and rejected as a CORS
    // error. The param never reaches the display path — only this hidden
    // sampling image. Color cache below stays keyed on the original URL.
    const fetchUrl = isRemote
      ? coverUrl + (coverUrl.includes('?') ? '&' : '?') + 'cors=1'
      : coverUrl;
    img.onload = () => {
      if (cancelled) return;
      try {
        const W = 32, H = 32;
        const canvas = document.createElement('canvas');
        canvas.width = W; canvas.height = H;
        const cx = canvas.getContext('2d');
        cx.drawImage(img, 0, 0, W, H);
        const data = cx.getImageData(0, 0, W, H).data;
        let bestScore = -1;
        let best = null;
        let sr = 0, sg = 0, sb = 0, n = 0;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i], g = data[i+1], b = data[i+2], a = data[i+3];
          if (a < 200) continue;
          sr += r; sg += g; sb += b; n++;
          const max = Math.max(r,g,b), min = Math.min(r,g,b);
          const lum = (max + min) / 2;
          const sat = max === 0 ? 0 : (max - min) / 255;
          if (lum > 50 && lum < 220) {
            // Centered around mid-luminance ~135 — penalize edges
            const score = sat * (1 - Math.abs(lum - 135) / 135);
            if (score > bestScore) {
              bestScore = score;
              best = { r, g, b };
            }
          }
        }
        if (!best) best = n ? { r: sr/n, g: sg/n, b: sb/n } : { r: 124, g: 91, b: 255 };
        const hsl = rgbToHsl(best.r, best.g, best.b);
        cacheRef.current[coverUrl] = hsl;
        setColor(hsl);
      } catch (e) {
        // CORS-tainted canvas (server didn't send ACAO) or decode failure
        setColor(null);
      }
    };
    img.onerror = () => { if (!cancelled) setColor(null); };
    img.src = fetchUrl;
    return () => { cancelled = true; };
  }, [coverUrl]);
  return color;
}

// ── Artist-hero aurora palette ───────────────────────────────────────────────
// Album covers are sampled for their dominant colours to paint the artist hero's
// "no photo" state (a drifting For-You-style gradient). Deliberately separate from
// useCoverColor above, which sits on the player's singleton sampling path — we keep
// that blast radius at zero.
function _clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
// Shortest distance between two hues on the 0-360 colour wheel.
function _hueDist(a, b) { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; }

// Load a cover and hand back its W×H pixel buffer, mirroring useCoverColor's CORS
// dance: same-origin reads must OMIT crossOrigin (a cached no-cors copy would taint
// the canvas otherwise); remote reads opt into CORS on a ?cors=1 sampling URL.
// Resolves null on taint/decode/error — never rejects.
function loadCoverPixels(url, W, H) {
  W = W || 32; H = H || 32;
  return new Promise((resolve) => {
    if (!url) { resolve(null); return; }
    const img = new Image();
    const isRemote = /^https?:\/\//i.test(url) && !url.startsWith(window.location.origin);
    if (isRemote) img.crossOrigin = 'anonymous';
    const fetchUrl = isRemote ? url + (url.includes('?') ? '&' : '?') + 'cors=1' : url;
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = W; canvas.height = H;
        const cx = canvas.getContext('2d', { willReadFrequently: true });
        cx.drawImage(img, 0, 0, W, H);
        resolve(cx.getImageData(0, 0, W, H).data);
      } catch (e) {
        resolve(null);  // CORS-tainted canvas / decode failure
      }
    };
    img.onerror = () => resolve(null);
    img.src = fetchUrl;
  });
}

// Quantise an RGBA pixel buffer into up to k vivid, hue-separated colours. Vivid
// pixels (mid-luminance, saturated) are binned into 12 hue buckets weighted by
// saturation; the heaviest buckets win, each ≥40° apart so we never return
// near-identical stops. Monochrome/sparse covers fall back to spreading the
// strongest colour across hue+lightness so one album still yields 2-3 stops.
function pixelsToPalette(data, k) {
  if (!data) return [];
  const BINS = 12;
  const bins = [];
  for (let i = 0; i < BINS; i++) bins.push({ w: 0, h: 0, s: 0, l: 0, n: 0 });
  let fallback = null, fbScore = -1;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 200) continue;
    const hsl = rgbToHsl(data[i], data[i + 1], data[i + 2]);
    const h = hsl.h, s = hsl.s, l = hsl.l;
    if (l > 22 && l < 82) {
      const score = (s / 100) * (1 - Math.abs(l - 55) / 55);
      if (score > fbScore) { fbScore = score; fallback = { h, s, l }; }
    }
    if (l <= 22 || l >= 82 || s < 18) continue;
    const bi = Math.min(BINS - 1, Math.floor((h / 360) * BINS));
    const bn = bins[bi];
    bn.w += s; bn.h += h * s; bn.s += s * s; bn.l += l * s; bn.n++;
  }
  const cand = bins
    .filter(bn => bn.n > 0 && bn.w > 0)
    .map(bn => ({ h: bn.h / bn.w, s: bn.s / bn.w, l: bn.l / bn.w, w: bn.w }))
    .sort((x, y) => y.w - x.w);
  const out = [];
  for (let i = 0; i < cand.length && out.length < k; i++) {
    const cc = cand[i];
    if (out.some(o => _hueDist(o.h, cc.h) < 40)) continue;
    out.push({ h: cc.h, s: cc.s, l: cc.l });
  }
  if (out.length < k) {
    const base = out[0] || fallback || { h: 270, s: 50, l: 55 };
    const spread = [0, 22, -26, 12], lift = [0, 10, -8, 6];
    let si = out.length === 0 ? 0 : 1;
    while (out.length < k && si < spread.length) {
      out.push({
        h: ((base.h + spread[si]) % 360 + 360) % 360,
        s: _clamp(base.s, 30, 80),
        l: _clamp(base.l + lift[si], 20, 80),
      });
      si++;
    }
  }
  return out.slice(0, k);
}

// Theme-safe CSS stops: clamp saturation/lightness so the artist name stays legible
// over the gradient (brighter band in light theme, deeper in dark).
function auroraStops(colors, isDark) {
  return colors.map(({ h, s, l }) => {
    const sc = _clamp(s, 38, 72);
    const lc = isDark ? _clamp(l, 38, 56) : _clamp(l, 58, 74);
    return `hsl(${h.toFixed(0)}, ${sc.toFixed(0)}%, ${lc.toFixed(0)}%)`;
  });
}

// Synthetic 3-colour palette from a name-derived hue — used when an artist has no
// album covers to sample, so the aurora still has something to shimmer.
function nameHueColors(hue) {
  return [
    { h: hue, s: 60, l: 55 },
    { h: (hue + 45) % 360, s: 52, l: 50 },
    { h: ((hue + 330) % 360), s: 46, l: 60 },
  ];
}

// Guarantee at least `want` stops so the base linear-gradient always has ≥2 colour
// stops (a 1-stop gradient is invalid CSS). Pads by spreading the first colour
// across hue+lightness; an empty input falls back to the name-derived hue.
function padPalette(colors, hue, want) {
  want = want || 3;
  const out = (colors && colors.length) ? colors.slice() : nameHueColors(hue);
  if (out.length >= want) return out;
  const base = out[0];
  const spread = [18, -24, 30, -14], lift = [8, -10, 5, -6];
  for (let i = 0; out.length < want && i < spread.length; i++) {
    out.push({
      h: ((base.h + spread[i]) % 360 + 360) % 360,
      s: _clamp(base.s, 30, 80),
      l: _clamp(base.l + lift[i], 20, 80),
    });
  }
  return out;
}

// Sample N album covers into a 3-4 colour aurora palette. One cover → its 2-3
// dominant colours; many covers → one dominant per cover (hue-deduped, capped 4),
// echoing how the player tints its equalizer per track. A single useEffect keyed on
// the joined URL list keeps this rules-of-hooks-safe (no per-cover hook calls).
function useCoverPalette(urls) {
  const key = (urls || []).join('|');
  const [state, setState] = useState({ colors: [], loading: !!key });
  const cacheRef = useRef({});
  useEffect(() => {
    if (!key) { setState({ colors: [], loading: false }); return; }
    if (cacheRef.current[key]) { setState({ colors: cacheRef.current[key], loading: false }); return; }
    let cancelled = false;
    setState(s => ({ colors: s.colors, loading: true }));
    const list = key.split('|');
    Promise.all(list.map(u => loadCoverPixels(u, 32, 32))).then(pixels => {
      if (cancelled) return;
      const valid = pixels.filter(Boolean);
      let colors = [];
      if (valid.length === 1) {
        colors = pixelsToPalette(valid[0], 3);
      } else if (valid.length >= 2) {
        for (let i = 0; i < valid.length && colors.length < 4; i++) {
          const cc = pixelsToPalette(valid[i], 1)[0];
          if (cc && !colors.some(o => _hueDist(o.h, cc.h) < 28)) colors.push(cc);
        }
      }
      cacheRef.current[key] = colors;
      setState({ colors, loading: false });
    });
    return () => { cancelled = true; };
  }, [key]);
  return state;
}

// Mirrored spectrum strip. Reads frequency bins from the shared analyser
// once per rAF and animates child bar elements via scaleY transform. Side
// determines bin-to-bar mapping: BOTH sides anchor low-freq near the cover
// so a beat thunders inward from both directions, and high-freq decays
// outward toward the column edge.
function SpectrumBars({ side, analyserRef, dataArrayRef, color, isPlaying, barCount = 40 }) {
  const pathRef = useRef(null);
  const rawRef = useRef(null);
  const targetsRef = useRef(null);
  const currentsRef = useRef(null);
  const fadeRef = useRef(0);
  const peakRef = useRef(0.12);   // smoothed running peak for the AGC
  const N = Math.max(8, barCount);

  if (!targetsRef.current || targetsRef.current.length !== N) {
    rawRef.current = new Float32Array(N);
    targetsRef.current = new Float32Array(N);
    currentsRef.current = new Float32Array(N);
  }

  useEffect(() => {
    let frameId;
    const A = 46;   // amplitude in viewBox units (half of the 100-tall strip)
    // Smooth a polyline into a quadratic path (curve through segment midpoints).
    const smooth = (pts) => {
      let d = '';
      for (let i = 1; i < pts.length; i++) {
        const mx = (pts[i - 1][0] + pts[i][0]) / 2;
        const my = (pts[i - 1][1] + pts[i][1]) / 2;
        d += `Q ${pts[i - 1][0].toFixed(2)} ${pts[i - 1][1].toFixed(2)} ${mx.toFixed(2)} ${my.toFixed(2)} `;
      }
      const last = pts[pts.length - 1];
      d += `L ${last[0].toFixed(2)} ${last[1].toFixed(2)} `;
      return d;
    };
    // Symmetric filled ribbon: top edge + mirrored bottom edge, closed.
    const buildPath = () => {
      const cur = currentsRef.current;
      const top = [], bot = [];
      for (let i = 0; i < N; i++) {
        const x = (i / (N - 1)) * 100;
        const a = Math.max(0, Math.min(1, cur[i])) * A * fadeRef.current;
        top.push([x, 50 - a]);
        bot.push([x, 50 + a]);
      }
      const botRev = bot.reverse();
      return `M ${top[0][0].toFixed(2)} ${top[0][1].toFixed(2)} ` + smooth(top) +
             `L ${botRev[0][0].toFixed(2)} ${botRev[0][1].toFixed(2)} ` + smooth(botRev) + 'Z';
    };
    const render = () => {
      const data = dataArrayRef?.current;
      const analyser = analyserRef?.current;
      const raw = rawRef.current;
      if (analyser && data) {
        analyser.getByteFrequencyData(data);
        // Lower ~45% of bins hold vocal/melody/bass; spreading them across the
        // strip keeps the outer end alive instead of permanently dead.
        const usefulBins = Math.floor(data.length * 0.45);
        let frameMax = 0;
        for (let i = 0; i < N; i++) {
          const start = Math.floor(i * usefulBins / N);
          const end   = Math.floor((i + 1) * usefulBins / N);
          let sum = 0;
          for (let j = start; j < end; j++) sum += data[j];
          const v = sum / Math.max(1, end - start) / 255;
          raw[i] = v;
          if (v > frameMax) frameMax = v;
        }
        // AGC: a smoothed peak (fast attack, slow release) normalises the SHAPE
        // so loud passages no longer flat-line at max; the overall size still
        // tracks loudness (quiet → smaller), keeping the wave informative.
        const pk = peakRef.current;
        peakRef.current = pk + (frameMax - pk) * (frameMax > pk ? 0.45 : 0.02);
        const denom = Math.max(peakRef.current, 0.05);
        const loud = Math.min(1, peakRef.current * 1.5 + 0.1);
        for (let i = 0; i < N; i++) {
          targetsRef.current[i] = Math.min(1, (raw[i] / denom) * loud);
        }
      } else {
        for (let i = 0; i < N; i++) targetsRef.current[i] = 0;
        peakRef.current *= 0.9;
      }
      // Global fade so the wave eases in/out when play state flips.
      const targetFade = isPlaying ? 1 : 0;
      fadeRef.current += (targetFade - fadeRef.current) * 0.06;
      // DOM index `d` → data-bin lookup so left/right mirror around the cover.
      for (let d = 0; d < N; d++) {
        const dataBin = side === 'left' ? (N - 1 - d) : d;
        const target = targetsRef.current[dataBin] || 0;
        currentsRef.current[d] += (target - currentsRef.current[d]) * 0.28;
      }
      const el = pathRef.current;
      if (el) el.setAttribute('d', buildPath());
      frameId = requestAnimationFrame(render);
    };
    frameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frameId);
  }, [analyserRef, dataArrayRef, isPlaying, N, side]);

  // Cover-color → wave tint (HSL). Clamp saturation/lightness so very dark or
  // very pale covers still produce a readable tint.
  const hue = color ? color.h : 270;
  const sat = color ? Math.min(85, Math.max(45, color.s)) : 60;
  const lit = color ? Math.min(72, Math.max(52, color.l)) : 62;
  const barColor = `hsl(${hue.toFixed(0)}, ${sat.toFixed(0)}%, ${lit.toFixed(0)}%)`;
  const gradId = `spec-grad-${side}`;
  // Horizontal fill ramp: bright at the cover-side edge, dim at the outer edge,
  // so the wave reads as anchored on the album. (left strip → cover on the right)
  const coverAtRight = side === 'left';

  return (
    <div className={`player-spectrum player-spectrum--${side}`} aria-hidden="true">
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id={gradId}
            x1={coverAtRight ? '0' : '1'} y1="0" x2={coverAtRight ? '1' : '0'} y2="0">
            <stop offset="0%"   stopColor={barColor} stopOpacity="0.10" />
            <stop offset="55%"  stopColor={barColor} stopOpacity="0.55" />
            <stop offset="100%" stopColor={barColor} stopOpacity="0.92" />
          </linearGradient>
        </defs>
        <path ref={pathRef} d="" fill={`url(#${gradId})`} />
      </svg>
    </div>
  );
}

// ─── AIChatDrawer ────────────────────────────────────────────────────────────
// Each chip stays SMALL — only `label` is shown on the pill. Clicking it drops the
// full 1-2 sentence `prompt` into the input, where the user can read/edit before sending.
const TRACK_CHAT_SUGGESTED_PROMPTS = [
  {
    icon: '💭',
    label: { ru: 'О чём песня?', en: "What's it about?" },
    prompt: {
      ru: 'О чём эта песня на самом деле? Расскажи простыми словами, как будто объясняешь другу.',
      en: 'What is this song really about? Explain it in plain words, like you would to a friend.',
    },
  },
  {
    icon: '📖',
    label: { ru: 'История', en: 'Backstory' },
    prompt: {
      ru: 'Расскажи историю создания этой песни и насколько популярной она была, когда вышла.',
      en: 'Tell me the story behind this song and how big it was when it first came out.',
    },
  },
  {
    icon: '✍️',
    label: { ru: 'Сильные строчки', en: 'Key lines' },
    prompt: {
      ru: 'Разбери пару самых сильных строчек: есть ли в них отсылки или скрытый смысл?',
      en: 'Break down a couple of the strongest lines — any references or hidden meaning in them?',
    },
  },
  {
    icon: '💿',
    label: { ru: 'Семплы', en: 'Samples' },
    prompt: {
      ru: 'Какие песни семплировались в этом треке? Расскажи, откуда взяты семплы.',
      en: 'What songs are sampled in this track? Tell me where the samples come from.',
    },
  },
];

// Humanized labels for the streaming status stages. `label` = present tense
// (current step, shimmering); `done` = past tense (collapsed ✓ row).
const TRACK_CHAT_STAGES = {
  thinking: {
    icon: 'orb',
    label: { ru: 'Думаю', en: 'Thinking' },
    done:  { ru: 'Подумал', en: 'Thought it through' },
  },
  web_search: {
    icon: '🌐',
    label: { ru: 'Ищу в интернете', en: 'Searching the web' },
    done:  { ru: 'Поискал в интернете', en: 'Searched the web' },
  },
  reading: {
    icon: '📖',
    label: { ru: 'Читаю найденное', en: 'Reading what I found' },
    done:  { ru: 'Прочитал найденное', en: 'Read the results' },
  },
};

// ─── TrackChatActivity — live "what am I doing" ticker inside the loading
// message. The last stage is current (animated icon + shimmer text); earlier
// stages collapse into muted ✓ rows. Web-search rows show the query.
function TrackChatActivity({ activity, lang }) {
  const items = (activity && activity.length) ? activity : [{ stage: 'thinking' }];
  const L = (o) => (lang === 'ru' ? o.ru : o.en);
  return (
    <div className="ai-activity">
      {items.map((a, i) => {
        const meta = TRACK_CHAT_STAGES[a.stage] || TRACK_CHAT_STAGES.thinking;
        const isCurrent = i === items.length - 1;
        if (!isCurrent) {
          return (
            <div key={i} className="ai-activity-row ai-activity-done">
              <span className="ai-tick" aria-hidden>✓</span>
              <span>{L(meta.done)}</span>
              {a.query && <span className="ai-activity-q">«{a.query}»</span>}
            </div>
          );
        }
        return (
          <div key={i} className="ai-activity-row">
            {meta.icon === 'orb'
              ? <span className="ai-orb" aria-hidden />
              : <span className="ai-glyph" aria-hidden>{meta.icon}</span>}
            <span className="ai-activity-current">{L(meta.label)}…</span>
            {a.query && <span className="ai-activity-q">«{a.query}»</span>}
          </div>
        );
      })}
    </div>
  );
}

function AIChatDrawer({ isOpen, onClose, track, lang, isDark, showToast }) {
  const c = useColors(isDark);
  const trackId = track?.track_id || null;
  const { messages, sendMessage, clearChat } = useTrackChat(trackId, localStorage.getItem('musix_user_id'), lang);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (endRef.current) endRef.current.scrollTop = endRef.current.scrollHeight;
  }, [messages, isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose && onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!track) return null;

  const buildTrackContext = () => ({
    title: track.title || '',
    artist: track.artist || '',
    album: track.album || null,
    year: track.year || null,
    genre: track.genre || null,
    full_lyrics: track.lyrics || '',
  });

  const getLlmKw = () => ({
    llm_base_url: localStorage.getItem('llm_base_url') || undefined,
    llm_model: localStorage.getItem('llm_model') || undefined,
    });

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput('');
    try {
      await sendMessage(text, buildTrackContext(), getLlmKw());
    } finally {
      setSending(false);
    }
  };

  const handlePromptClick = (p) => {
    setInput(lang === 'ru' ? p.prompt.ru : p.prompt.en);
    if (inputRef.current) inputRef.current.focus();
  };

  const handleClearChat = () => {
    clearChat();
    if (showToast) {
      showToast(lang === 'ru' ? 'Чат очищен' : 'Chat cleared');
    }
  };

  return (
    <>
      {/* Drawer panel — anchored absolute inside .queue-chat-area.
          Slides UP from beneath the queue (translateY 100%↔0), so FactsRail
          stays visible above. Glass surface (blur/saturate/tint + top
          hairline highlight) now lives in .track-chat-glass so the theme
          override is a plain CSS swap instead of inline JS branching. */}
      <div
        className="track-chat-glass"
        style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          transform: isOpen ? 'translateY(0)' : 'translateY(100%)',
          opacity: isOpen ? 1 : 0,
          transition: 'transform 280ms cubic-bezier(0.16, 1, 0.3, 1), opacity 280ms cubic-bezier(0.16, 1, 0.3, 1)',
          zIndex: 50,
          borderRadius: 14,
          pointerEvents: isOpen ? 'auto' : 'none',
        }}
      >
        {/* Slim header — 32px row with title label + close. Cover/title/
            artist/year intentionally dropped — visible in the left column. */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          padding: '8px 14px', flexShrink: 0,
          borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'}`,
        }}>
          <span style={{ fontSize: '14px' }} aria-hidden>✨</span>
          <span style={{ flex: 1, fontSize: '13px', fontWeight: 600, color: c.text, letterSpacing: '-0.01em' }}>
            {lang === 'ru' ? 'Чат по треку' : 'Track chat'}
          </span>
          {messages.length > 0 && (
            <button onClick={handleClearChat}
              className="pill-v3"
              aria-label={lang === 'ru' ? 'Новый чат' : 'New chat'}
              title={lang === 'ru' ? 'Новый чат' : 'New chat'}
              style={{ padding: '3px 9px', fontSize: '13px', cursor: 'pointer' }}>↺</button>
          )}
          <button onClick={onClose}
            className="pill-v3"
            aria-label={lang === 'ru' ? 'Закрыть' : 'Close'}
            title={lang === 'ru' ? 'Закрыть' : 'Close'}
            style={{ padding: '3px 9px', fontSize: '13px', cursor: 'pointer' }}>✕</button>
        </div>

        {/* Messages — or, when the chat is empty, a hero state with a
            breathing orb + invitation, so the first open doesn't greet the
            user with a blank void. */}
        {messages.length === 0 ? (
          <div className="tc-hero">
            <span className="tc-hero-orb" aria-hidden />
            <div style={{ fontSize: '14px', fontWeight: 600, color: c.text, letterSpacing: '-0.01em' }}>
              {lang === 'ru' ? 'Спросите про этот трек' : 'Ask about this track'}
            </div>
            <div style={{ fontSize: '12px', color: c.textMuted, maxWidth: '260px', lineHeight: 1.45 }}>
              {lang === 'ru'
                ? 'Смысл, история, семплы, отсылки — или выберите подсказку ниже.'
                : 'Meaning, backstory, samples, references — or pick a chip below.'}
            </div>
          </div>
        ) : (
          <div ref={endRef} style={{
            flex: 1, overflowY: 'auto', padding: '12px 18px',
            display: 'flex', flexDirection: 'column', gap: '10px',
          }}>
            {messages.map((m, i) => (
              <div key={i} style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '88%',
                padding: m.loading ? '10px 14px' : '9px 13px',
                borderRadius: m.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                background: m.role === 'user' ? c.userBubble : c.aiBubble,
                color: m.role === 'user' ? 'white' : c.text,
                fontSize: '14px', lineHeight: '1.5',
              }}>
                {m.loading
                  ? <TrackChatActivity activity={m.activity} lang={lang} />
                  : m.role === 'assistant'
                    ? <MarkdownText text={m.text} />
                    : m.text}
                {m.web_search_used && (
                  <div style={{ marginTop: '4px', fontSize: '10px', color: c.textSubtle, fontStyle: 'italic' }}>
                    🌐 {lang === 'ru' ? 'по данным из интернета' : 'web search'}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Suggested prompts — single scrollable rail of icon chips just
            above the input (Perplexity / ChatGPT quick-actions pattern).
            Staggered entrance only while the chat is still empty. */}
        <div className="tc-chip-rail" style={{
          display: 'flex', gap: '6px',
          padding: '8px 18px',
          overflowX: 'auto',
          borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'}`,
          flexShrink: 0,
          minHeight: '38px',
        }}>
          {TRACK_CHAT_SUGGESTED_PROMPTS.map((p, i) => (
            <button key={i} onClick={() => handlePromptClick(p)}
              className={messages.length === 0 ? 'tc-chip tc-chip-in' : 'tc-chip'}
              title={lang === 'ru' ? p.prompt.ru : p.prompt.en}
              style={messages.length === 0 ? { animationDelay: `${i * 60}ms` } : undefined}>
              <span className="tc-chip-icon" aria-hidden>{p.icon}</span>
              {lang === 'ru' ? p.label.ru : p.label.en}
            </button>
          ))}
        </div>

        {/* Input */}
        <div style={{
          display: 'flex', gap: '8px', alignItems: 'center',
          padding: '10px 18px 12px', flexShrink: 0,
        }}>
          <input
            ref={inputRef}
            className="tc-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder={lang === 'ru' ? 'Спросить про этот трек…' : 'Ask about this track…'}
            disabled={sending}
            style={{
              flex: 1, padding: '8px 12px', borderRadius: '10px',
              background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)',
              border: `1px solid ${c.border}`,
              color: c.text, fontSize: '14px', outline: 'none',
              fontFamily: "'Geist', sans-serif",
            }}
          />
          <button onClick={handleSend} disabled={sending || !input.trim()}
            className={sending || !input.trim() ? '' : 'cta-v3'}
            aria-label={lang === 'ru' ? 'Отправить' : 'Send'}
            style={{
              width: '36px', height: '36px', borderRadius: '10px', padding: 0, flexShrink: 0,
              background: (sending || !input.trim()) ? (isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)') : undefined,
              color: (sending || !input.trim()) ? c.textSubtle : 'white',
              cursor: (sending || !input.trim()) ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </>
  );
}

// ─── InlineLyricExplain — per-line draw-under explanation panel ──────────────
function InlineLyricExplain({
  line, lineIdx, expandedLines, explainStates,
  onToggle, isDark, aiActive, lang,
}) {
  const c = useColors(isDark);
  const [hovered, setHovered] = useState(false);
  const isExpanded = expandedLines.has(lineIdx);
  const state = explainStates.get(lineIdx);

  const handleClick = (e) => {
    e.stopPropagation();
    onToggle(lineIdx, line);
  };

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ position: 'relative', padding: '2px 0' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ flex: 1 }}>{line}</span>
        {aiActive && line.trim() && (hovered || isExpanded) && (
          <button
            onClick={handleClick}
            title={isExpanded ? (lang === 'ru' ? 'Свернуть' : 'Collapse') : (lang === 'ru' ? 'Объяснить эту строку' : 'Explain this line')}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: isExpanded ? c.accent : c.textSubtle,
              fontSize: '14px', padding: '2px 6px',
              opacity: isExpanded ? 1 : 0.7,
              transition: 'opacity 120ms',
            }}
          >✨</button>
        )}
      </div>
      {isExpanded && state && (
        <div
          className="panel-v3"
          style={{
            margin: '6px 0 10px',
            padding: '10px 14px',
            fontSize: '13px',
            color: c.text,
            borderLeft: `2px solid ${c.accent}`,
          }}
        >
          {state.loading
            ? (() => {
                const barBase = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
                const barShine = isDark ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.60)';
                const barStyle = {
                  height: 9, borderRadius: 5,
                  backgroundColor: barBase,
                  backgroundImage: `linear-gradient(90deg, transparent 0%, ${barShine} 50%, transparent 100%)`,
                  backgroundSize: '200% 100%',
                  backgroundRepeat: 'no-repeat',
                };
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: c.textSubtle, fontSize: 12 }}>
                      <span style={{ display: 'inline-block', animation: 'pulse 1.4s ease-in-out infinite' }}>✨</span>
                      <span>{lang === 'ru' ? 'Думаю…' : 'Thinking…'}</span>
                    </div>
                    <div style={{ ...barStyle, animation: 'shimmer 1.6s ease-in-out infinite' }} />
                    <div style={{ ...barStyle, width: '62%', animation: 'shimmer 1.6s ease-in-out 0.3s infinite' }} />
                  </div>
                );
              })()
            : state.error
              ? <span style={{ color: c.red }}>{lang === 'ru' ? 'Ошибка' : 'Error'}: {state.error}</span>
              : <>
                  <MarkdownText text={state.message} />
                  {state.web_search_used && (
                    <div style={{ marginTop: '4px', fontSize: '10px', color: c.textSubtle, fontStyle: 'italic' }}>
                      🌐 web search
                    </div>
                  )}
                </>}
        </div>
      )}
    </div>
  );
}

// ─── PLAYER SECTION ─────────────────────────────────────────────────────────
// Volume control for the player toolbar: a Windows-style speaker icon whose
// glyph reflects the current level (muted / low / mid / high). Click toggles
// mute (remembering the prior level). The slider stays collapsed into the icon
// and slides out to the right on hover, folding back when the pointer leaves —
// unless a drag is in progress, so dragging past the icon edge doesn't snap it
// shut. Positioned absolutely so the centered toolbar row doesn't reflow.
function VolumeControl({ volume, onChange, isDark, lang }) {
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const lastVolRef = useRef(volume > 0 ? volume : 0.85);

  useEffect(() => { if (volume > 0) lastVolRef.current = volume; }, [volume]);
  useEffect(() => {
    if (!dragging) return;
    const stop = () => setDragging(false);
    window.addEventListener('mouseup', stop);
    window.addEventListener('pointerup', stop);
    return () => { window.removeEventListener('mouseup', stop); window.removeEventListener('pointerup', stop); };
  }, [dragging]);

  const expanded = hovered || dragging;
  const pct = Math.round((volume || 0) * 100);
  const fill = isDark ? 'rgba(255,255,255,0.62)' : 'rgba(0,0,0,0.42)';
  const trackBg = isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)';
  const muted = volume <= 0;
  const lvl = muted ? 0 : volume <= 0.33 ? 1 : volume <= 0.66 ? 2 : 3;

  const toggleMute = () => {
    if (volume > 0) { lastVolRef.current = volume; onChange(0); }
    else { onChange(lastVolRef.current > 0 ? lastVolRef.current : 0.85); }
  };

  return (
    <div
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button" className="player-icon-btn"
        onClick={toggleMute}
        aria-label={lang === 'ru' ? 'Громкость' : 'Volume'}
        title={muted ? (lang === 'ru' ? 'Включить звук' : 'Unmute') : (lang === 'ru' ? 'Выключить звук' : 'Mute')}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="currentColor" stroke="none" />
          {muted ? (
            <g><line x1="16" y1="9.5" x2="21.5" y2="14.5" /><line x1="21.5" y1="9.5" x2="16" y2="14.5" /></g>
          ) : (
            <g>
              {lvl >= 1 && <path d="M14.5 9.5a3 3 0 0 1 0 5" />}
              {lvl >= 2 && <path d="M16.6 7.8a6 6 0 0 1 0 8.4" />}
              {lvl >= 3 && <path d="M18.7 6a9 9 0 0 1 0 12" />}
            </g>
          )}
        </svg>
      </button>
      <div style={{
        position: 'absolute', left: 'calc(100% - 6px)', top: '50%', transform: 'translateY(-50%)',
        display: 'flex', alignItems: 'center',
        paddingTop: 10, paddingBottom: 10,
        width: expanded ? 100 : 0, opacity: expanded ? 1 : 0,
        overflow: 'hidden', pointerEvents: expanded ? 'auto' : 'none',
        transition: 'width 0.22s cubic-bezier(0.22,0.9,0.3,1), opacity 0.18s ease',
      }}>
        <input
          type="range" min={0} max={1} step={0.01}
          value={volume}
          onChange={e => onChange(Number(e.target.value))}
          onMouseDown={() => setDragging(true)}
          onPointerDown={() => setDragging(true)}
          className="player-volume"
          aria-label={lang === 'ru' ? 'Уровень громкости' : 'Volume level'}
          style={{
            width: 84, height: 4, borderRadius: 2, margin: '0 5px',
            background: `linear-gradient(to right, ${fill} ${pct}%, ${trackBg} ${pct}%)`,
          }}
        />
      </div>
    </div>
  );
}

// ─── PlayerAmbient — full-bleed ambient wash behind the whole shell ──────────
// Lifted out of PlayerSection so the blurred-cover wash spans the ENTIRE
// viewport and flows *behind* the transparent floating nav, instead of
// stopping at the section's left edge (which left a hard black strip under the
// rail). PlayerSection renders transparent and sits on top of this layer.
// Driven by App.playerTrack, which stays in lockstep with the player's current
// track via handleTrackChange.
function PlayerAmbient({ track, isDark }) {
  const coverUrl = track?.cover_art_path
    ? (track.cover_art_path.startsWith('http') ? track.cover_art_path : `${API}${track.cover_art_path}`)
    : null;
  const pBg    = isDark ? '#0B0E14' : '#f4f5fa';
  const pBgEnd = isDark ? '#0F111A' : '#eaeaf0';
  // Two-layer cross-fade on track change. Swapping backgroundImage in place
  // drops the wash to the bare gradient while the next cover decodes; instead
  // the incoming cover is preloaded off-screen, mounted ON TOP at opacity 0,
  // faded in (playerBgCoverIn), and the old layer is pruned once the fade
  // lands. Max two layers — rapid skips replace the incoming one.
  const [bgStack, setBgStack] = useState(() => (coverUrl ? [{ url: coverUrl, key: 0 }] : []));
  useEffect(() => {
    const top = bgStack[bgStack.length - 1];
    if ((top?.url || null) === (coverUrl || null)) return;
    if (!coverUrl) { setBgStack([]); return; }
    let dead = false;
    const img = new Image();
    const push = () => {
      if (dead) return;
      setBgStack(s => [...s.slice(-1), { url: coverUrl, key: (s[s.length - 1]?.key ?? -1) + 1 }]);
    };
    img.onload = push;
    img.onerror = push; // a failed cover paints nothing; the base gradient shows
    img.src = coverUrl;
    return () => { dead = true; };
  }, [coverUrl, bgStack]);
  return (
    <div style={{
      position:'absolute', inset:0, zIndex:0, overflow:'hidden', pointerEvents:'none',
      background:`linear-gradient(180deg, ${pBg} 0%, ${pBgEnd} 100%)`,
    }}>
      {bgStack.map((layer, i) => (
        <div key={layer.key} className="player-bg-cover player-bg-cover--fade"
          style={{ backgroundImage:`url(${layer.url})`, '--bg-cover-op': isDark ? 0.48 : 0.26 }}
          onAnimationEnd={i === bgStack.length - 1 && bgStack.length > 1
            ? () => setBgStack(s => s.slice(-1))
            : undefined} />
      ))}
      <div className="player-bg-shade" style={{
        background: isDark
          ? `linear-gradient(180deg, ${pBg}66 0%, ${pBg}b3 70%, ${pBg}e6 100%)`
          : `linear-gradient(180deg, ${pBg}8c 0%, ${pBg}d0 100%)`,
      }} />
      <div className="player-bg-grain" />
      <div className="player-bg-vignette" />
    </div>
  );
}

// ── Search/chat ambient — full-bleed colour field behind BOTH the transparent
// nav rail and the section content, so the background is continuous and the
// rail floats above it (no vertical seam at the rail's edge). Subtle enough
// not to fight the results grid of the regular search tab.
function SearchAmbient({ isDark }) {
  return (
    <div aria-hidden="true" style={{
      position:'absolute', inset:0, zIndex:0, overflow:'hidden', pointerEvents:'none',
      background: isDark
        ? `radial-gradient(ellipse 75% 90% at 22% -8%, oklch(60% 0.18 275 / 0.13), transparent 58%),
           radial-gradient(ellipse 60% 70% at 92% 108%, oklch(70% 0.13 60 / 0.06), transparent 55%)`
        : `radial-gradient(ellipse 75% 90% at 22% -8%, oklch(60% 0.18 275 / 0.09), transparent 58%),
           radial-gradient(ellipse 60% 70% at 92% 108%, oklch(70% 0.13 60 / 0.07), transparent 55%)`,
    }} />
  );
}

// ── Player Similarity Rail (CLAP «похожие / контраст» under the queue) ───────
// Fetches the current track's top-3 similar + top-3 contrasting neighbours from
// the memoized top-pairs cache. Renders nothing when there is no data (the
// analysis hasn't run, or the track isn't in the cache) or while the AI chat
// drawer is open. A click queues the track to play NEXT (not replace, not end).
function SimilarityColumn({ accent, label, items, tint, glow, isDark, lang, onQueueNext }) {
  const text   = isDark ? '#FFFFFF' : '#161620';
  const muted  = isDark ? '#9BA1B0' : 'rgba(22,22,32,0.62)';
  const subtle = isDark ? 'rgba(255,255,255,0.30)' : 'rgba(22,22,32,0.42)';
  const queueTip = lang === 'ru' ? 'Нажмите, чтобы добавить в очередь' : 'Click to add to the queue';
  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{
        fontSize: 10, letterSpacing: '0.18em',
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        color: accent, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: accent }} />
        {label}
      </div>
      {items.length === 0 ? (
        <div style={{ fontSize: 12, color: subtle, padding: '6px 4px' }}>—</div>
      ) : items.map((t, i) => (
        <div
          key={t.track_id || i}
          className="sim-card"
          onClick={() => onQueueNext(t)}
          aria-label={(t.title || '') + ' — ' + (t.artist || '') + ' · ' + queueTip}
          style={{
            '--sim-glow': glow,
            display: 'flex', alignItems: 'center', gap: 10, padding: '6px 8px',
            cursor: 'pointer', background: tint, borderLeft: `2px solid ${accent}`,
          }}>
          <span className="q-tip">{queueTip}</span>
          <AlbumCover title={t.title} artist={t.artist} size={38} isDark={isDark} coverPath={t.cover_art_path} radius={8} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 13, fontWeight: 500, color: text, letterSpacing: '-0.01em',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>{t.title || '—'}</div>
            <div style={{
              fontSize: 12, color: muted, marginTop: 1,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>{t.artist || '—'}</div>
          </div>
          {typeof t.score === 'number' && (
            <span style={{
              fontSize: 10, color: subtle, flexShrink: 0,
              fontFamily: "'JetBrains Mono', ui-monospace, monospace",
            }}>{Math.round(t.score)}%</span>
          )}
        </div>
      ))}
    </div>
  );
}

function SimilarityRail({ trackId, lang, isDark, drawerOpen, onQueueNext }) {
  const [data, setData] = useState(null); // { similar:[], dissimilar:[] } | null
  useEffect(() => {
    if (!trackId) { setData(null); return; }
    let alive = true;
    setData(null);
    apiFetch('/library/top-pairs/' + encodeURIComponent(trackId))
      .then(r => {
        if (!alive) return;
        const has = r && r.available && (((r.similar || []).length) || ((r.dissimilar || []).length));
        setData(has ? { similar: r.similar || [], dissimilar: r.dissimilar || [] } : null);
      })
      .catch(() => { if (alive) setData(null); });
    return () => { alive = false; };
  }, [trackId]);

  if (drawerOpen || !data) return null;

  return (
    <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ borderTop: `1px solid ${isDark ? '#2a2a32' : 'rgba(22,22,32,0.08)'}` }} />
      <div style={{ display: 'flex', gap: 12, paddingTop: 2 }}>
        <SimilarityColumn
          accent="#34D399"
          label={lang === 'ru' ? 'ПОХОЖИЕ' : 'SIMILAR'}
          items={data.similar}
          tint={isDark ? 'rgba(52,211,153,0.06)' : 'rgba(52,211,153,0.08)'}
          glow="rgba(52,211,153,0.38)"
          isDark={isDark}
          lang={lang}
          onQueueNext={onQueueNext}
        />
        <SimilarityColumn
          accent="#F59E0B"
          label={lang === 'ru' ? 'КОНТРАСТ' : 'CONTRAST'}
          items={data.dissimilar}
          tint={isDark ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.08)'}
          glow="rgba(245,158,11,0.40)"
          isDark={isDark}
          lang={lang}
          onQueueNext={onQueueNext}
        />
      </div>
    </div>
  );
}

// ── Queue drag-to-reorder hook ──────────────────────────────────────────────
// Pointer-events drag for the player QUEUE. Every row is draggable EXCEPT the
// now-playing one (lockedIndex) — entering an album mid-way leaves the earlier,
// never-actually-played rows above the current track, and those must stay
// movable (drag one below the current row to hear it). Rows above the current
// track are the history zone: they won't auto-play. Stream dedup keys on
// track_id, not position, so crossing the now-playing boundary is safe
// (design: 2026-07-10-queue-reorder-unlock).
//   • Mouse: 5px move threshold, so a plain click still plays the track.
//   • Touch: ~200ms long-press, so a vertical scroll still works (scroll wins
//     unless you hold still — a pointercancel from native scroll aborts it).
// Commits via onCommit(from, over): `over` is the destination index in ORIGINAL
// array coordinates; the caller splices the with-`from`-removed array at `over`.
//
// Stability: window listeners are memoized ([]) and read ALL mutable state from
// the S.current session (incl. a snapshot of onCommit taken at drag start), so
// a setDrag re-render can't strand a stale closure on window.
function useQueueReorder({ lockedIndex, count, onCommit, scrollRef }) {
  const reduced = useMemo(() => {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
    catch (e) { return false; }
  }, []);
  const [drag, setDrag] = useState(null);   // {from, over, dy, height} | null — drives the render
  const S = useRef(null);                    // mutable drag session
  const rafRef = useRef(0);
  const clickGuard = useRef(false);          // swallow the click that trails a real drag

  const stopAuto = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; }
  }, []);

  // Untransformed geometry of every draggable row currently in the DOM.
  const measure = useCallback((container) =>
    Array.from(container.querySelectorAll('[data-qrow]')).map((n) => {
      const r = n.getBoundingClientRect();
      return { index: Number(n.dataset.qrow), mid: r.top + r.height / 2, height: r.height };
    }), []);

  // Destination index in ORIGINAL coordinates from the live pointer Y.
  const computeOver = useCallback((s, clientY) => {
    let over = s.from;
    for (let i = 0; i < s.rects.length; i++) {
      const rc = s.rects[i];
      if (rc.index === s.from) continue;
      if (rc.index < s.from && clientY < rc.mid) over = Math.min(over, rc.index);
      if (rc.index > s.from && clientY > rc.mid) over = Math.max(over, rc.index);
    }
    return over;
  }, []);

  const update = useCallback((clientY) => {
    const s = S.current;
    if (!s || !s.active) return;
    s.lastY = clientY;
    const sc = scrollRef && scrollRef.current;
    const scrollDelta = sc ? sc.scrollTop - s.scroll0 : 0;
    const dy = (clientY - s.grabY) + scrollDelta;   // +scrollDelta keeps the row under the finger while auto-scrolling
    s.over = computeOver(s, clientY);
    setDrag({ from: s.from, over: s.over, dy, height: s.height });
  }, [scrollRef, computeOver]);

  const tickAuto = useCallback(() => {
    const s = S.current;
    const sc = scrollRef && scrollRef.current;
    if (!s || !s.active || !sc) { stopAuto(); return; }
    const r = sc.getBoundingClientRect();
    const EDGE = 52, MAX = 15;
    let v = 0;
    if (s.lastY < r.top + EDGE) v = -MAX * Math.min(1, (r.top + EDGE - s.lastY) / EDGE);
    else if (s.lastY > r.bottom - EDGE) v = MAX * Math.min(1, (s.lastY - (r.bottom - EDGE)) / EDGE);
    if (v !== 0) {
      const before = sc.scrollTop;
      sc.scrollTop = sc.scrollTop + v;
      const applied = sc.scrollTop - before;
      if (applied !== 0) {
        // Shift the cached (untransformed) midpoints by the scroll delta instead
        // of re-measuring — the rows now carry drag transforms, so a fresh
        // getBoundingClientRect would read shifted positions and skew the slot calc.
        for (let i = 0; i < s.rects.length; i++) s.rects[i].mid -= applied;
        update(s.lastY);
      }
    }
    rafRef.current = requestAnimationFrame(tickAuto);
  }, [scrollRef, stopAuto, update]);

  const begin = useCallback((clientY) => {
    const s = S.current;
    const sc = scrollRef && scrollRef.current;
    if (!s || !sc) return;
    s.active = true;
    s.grabY = clientY;
    s.scroll0 = sc.scrollTop;
    s.rects = measure(sc);
    const own = s.rects.find((rc) => rc.index === s.from);
    s.height = own ? own.height : 64;
    clickGuard.current = true;            // a drag started → suppress the trailing click
    update(clientY);
    stopAuto();
    rafRef.current = requestAnimationFrame(tickAuto);
  }, [scrollRef, measure, update, stopAuto, tickAuto]);

  // The drag logic lives in refs (reassigned each render with the latest stable
  // closures). The functions ACTUALLY bound to window are the stable dispatchers
  // below — so add/remove always use the SAME identity and a setDrag re-render
  // can never strand a listener (removeEventListener would otherwise miss).
  const moveRef = useRef(null), upRef = useRef(null), cancelRef = useRef(null);
  const winMove = useCallback((e) => { if (moveRef.current) moveRef.current(e); }, []);
  const winUp = useCallback(() => { if (upRef.current) upRef.current(); }, []);
  const winCancel = useCallback(() => { if (cancelRef.current) cancelRef.current(); }, []);

  const detach = useCallback(() => {
    stopAuto();
    window.removeEventListener('pointermove', winMove);
    window.removeEventListener('pointerup', winUp);
    window.removeEventListener('pointercancel', winCancel);
  }, [stopAuto, winMove, winUp, winCancel]);

  const finish = useCallback((commit) => {
    const s = S.current;
    detach();
    if (s && s.longPress) clearTimeout(s.longPress);
    if (s && s.active && commit && s.over !== s.from && typeof s.onCommit === 'function') {
      s.onCommit(s.from, s.over);
    }
    // The trailing click (fired right after pointerup) is swallowed via clickGuard.
    // If a drag ended over a different element no click may fire at all, so self-
    // clear shortly after to avoid eating the NEXT legitimate click.
    if (s && s.active) setTimeout(() => { clickGuard.current = false; }, 350);
    S.current = null;
    setDrag(null);
  }, [detach]);

  moveRef.current = (e) => {
    const s = S.current;
    if (!s) return;
    if (!s.active) {
      const moved = Math.abs(e.clientY - s.downY) + Math.abs(e.clientX - s.downX);
      if (s.isTouch) { if (moved > 8) finish(false); return; }   // pre-long-press move = scroll → abort
      if (moved > 5) begin(e.clientY);                            // mouse drag threshold
      return;
    }
    e.preventDefault();    // gesture is ours now — block native scroll / text selection
    update(e.clientY);
  };
  upRef.current = () => finish(true);
  cancelRef.current = () => finish(false);

  // Begin a press session. Recreated each render so it snapshots the CURRENT
  // onCommit into the session for the stable handlers to read.
  // `immediate` (grip handle) lifts the row right on pointerdown — no
  // long-press wait and no scroll-vs-drag ambiguity (the grip carries
  // touch-action:none, so the browser never contests the gesture).
  // Movable rows = everything except the pinned now-playing one.
  const movable = count - (lockedIndex >= 0 && lockedIndex < count ? 1 : 0);

  const startSession = (index, e, immediate = false) => {
    if (index === lockedIndex) return;            // now-playing row — pinned
    if (movable < 2) return;                      // nothing to reorder
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (S.current) finish(false);                 // defensive: never run two sessions
    const isTouch = e.pointerType !== 'mouse';
    S.current = {
      from: index, active: false, isTouch,
      downX: e.clientX, downY: e.clientY, grabY: e.clientY,
      scroll0: 0, rects: [], height: 0, over: index, lastY: e.clientY, longPress: 0,
      onCommit,
    };
    window.addEventListener('pointermove', winMove, { passive: false });
    window.addEventListener('pointerup', winUp);
    window.addEventListener('pointercancel', winCancel);
    if (immediate) {
      begin(e.clientY);
    } else if (isTouch) {
      S.current.longPress = setTimeout(() => {
        if (S.current && !S.current.active) begin(S.current.downY);
      }, 200);
    }
  };

  // Cleanup on unmount — never strand listeners or an animation frame.
  useEffect(() => () => { detach(); }, [detach]);

  // Per-row props: data tag, drag/shift/locked classes, transform offset, and
  // (for draggable rows) the pointerdown that opens a session.
  const getRowProps = (index) => {
    const draggable = index !== lockedIndex && movable >= 2;
    const cls = [];
    const style = {};
    if (draggable) cls.push('q-draggable');
    if (drag) {
      // Row order in the DOM never changes mid-drag — only transforms move; the
      // array commit happens on drop. (The lifted row's opacity is forced full
      // and the pinned now-playing row dimmed inline, since the row carries an
      // inline opacity that would otherwise outrank the q-* classes.)
      if (index === drag.from) {
        cls.push('q-dragging');
        style.transform = `translateY(${drag.dy}px) scale(${reduced ? 1 : 1.025})`;
        style.opacity = 1;
        style.cursor = 'grabbing';
      } else {
        // The pinned row still SHIFTS with the flow (a drag may cross the
        // now-playing boundary) — it just can't be grabbed, and dims to say so.
        cls.push('q-shift');
        if (index === lockedIndex) {
          cls.push('q-locked');
          style.opacity = 0.4;
        }
        let ty = 0;
        if (drag.over > drag.from && index > drag.from && index <= drag.over) ty = -drag.height;
        else if (drag.over < drag.from && index >= drag.over && index < drag.from) ty = drag.height;
        style.transform = `translateY(${ty}px)`;
      }
    }
    return {
      'data-qrow': index,
      className: cls.join(' '),
      style,
      onPointerDown: draggable ? (e) => startSession(index, e) : undefined,
    };
  };

  // Grip-handle props: pointerdown here grabs the row instantly (stops
  // propagation so the row's own long-press session doesn't double-start).
  const getHandleProps = (index) => {
    const draggable = index !== lockedIndex && movable >= 2;
    if (!draggable) return {};
    return {
      onPointerDown: (e) => { e.stopPropagation(); startSession(index, e, true); },
    };
  };

  // True (and self-resets) when the click event trailing a real drag should be
  // swallowed so the drop doesn't also fire the row's play-on-click.
  const consumeClick = useCallback(() => {
    if (clickGuard.current) { clickGuard.current = false; return true; }
    return false;
  }, []);

  return { getRowProps, getHandleProps, consumeClick, dragging: !!drag };
}

// ─── Огонёк/Вода: cover combustion effect ────────────────────────────────────
// One-shot canvas particle burst that wraps the album cover in 3D: a BACK
// canvas (z:-1) behind the art and a FRONT canvas (z:3) over its rim share one
// rAF loop. Fire: soft additive flame particles (hot white-yellow core cooling
// to deep red at the tip) rise from the bottom behind the cover and climb its
// side edges, with drifting embers and faint smoke above the crest. Water: a
// splash crown off the top edge, rivulets running down the front rims that
// stretch with speed and burst into droplets at the bottom, plus a cool mist
// behind. Sprites are pre-rendered radial gradients (no per-frame filters),
// DPR is capped at 2, and everything dies within FX_MS. `playKey` remounts
// the component so each press replays with fresh randomness.
function CoverCombustion({ kind = 'fire', playKey = 0 }) {
  const backRef = React.useRef(null);
  const frontRef = React.useRef(null);

  React.useEffect(() => {
    // Reduced motion: the CSS aura alone acknowledges the gesture.
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const backCv = backRef.current, frontCv = frontRef.current;
    const wrap = frontCv && frontCv.closest('.player-art-wrap');
    if (!backCv || !frontCv || !wrap) return;

    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const rect = frontCv.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W < 10 || H < 10) return;
    const setup = (cv) => {
      cv.width = Math.max(1, Math.round(W * DPR));
      cv.height = Math.max(1, Math.round(H * DPR));
      const ctx = cv.getContext('2d');
      ctx.scale(DPR, DPR);
      return ctx;
    };
    const bctx = setup(backCv), fctx = setup(frontCv);

    // Cover box in canvas coordinates (the canvases overflow the cover on all
    // sides — see .cover-fx insets). All speeds/radii scale off cover size S.
    const wr = wrap.getBoundingClientRect();
    const CL = wr.left - rect.left, CT = wr.top - rect.top;
    const CR = CL + wr.width, CB = CT + wr.height;
    const S = wr.width;

    const rnd = (a, b) => a + Math.random() * (b - a);
    const TAU = Math.PI * 2;
    const sprite = (stops) => {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const x = c.getContext('2d');
      const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
      for (const [o, col] of stops) g.addColorStop(o, col);
      x.fillStyle = g;
      x.fillRect(0, 0, 64, 64);
      return c;
    };
    // Color ramp: index 0 = hottest/freshest, 2 = coolest/oldest.
    const SPR = kind === 'fire' ? {
      ramp: [
        sprite([[0, 'rgba(255,251,214,0.95)'], [0.30, 'rgba(255,206,84,0.60)'], [1, 'rgba(255,140,0,0)']]),
        sprite([[0, 'rgba(255,168,60,0.85)'], [0.35, 'rgba(255,104,0,0.50)'], [1, 'rgba(255,60,0,0)']]),
        sprite([[0, 'rgba(255,92,24,0.80)'], [0.40, 'rgba(219,42,0,0.45)'], [1, 'rgba(150,20,0,0)']]),
      ],
      ember: sprite([[0, 'rgba(255,244,200,1)'], [0.30, 'rgba(255,178,70,0.90)'], [1, 'rgba(255,120,0,0)']]),
      haze: sprite([[0, 'rgba(46,40,38,0.32)'], [1, 'rgba(46,40,38,0)']]),
    } : {
      ramp: [
        sprite([[0, 'rgba(236,252,255,0.95)'], [0.30, 'rgba(158,224,255,0.60)'], [1, 'rgba(80,170,255,0)']]),
        sprite([[0, 'rgba(148,214,255,0.85)'], [0.35, 'rgba(70,158,255,0.50)'], [1, 'rgba(30,100,255,0)']]),
        sprite([[0, 'rgba(84,160,255,0.72)'], [0.40, 'rgba(32,92,230,0.42)'], [1, 'rgba(12,52,180,0)']]),
      ],
      ember: null,
      haze: sprite([[0, 'rgba(142,202,255,0.22)'], [1, 'rgba(142,202,255,0)']]),
    };

    // Particles: bx = base x (sway is added at draw time so it never drifts).
    const parts = [];
    const push = (p) => { if (parts.length < 420) parts.push(p); };

    const spawnFlame = (front) => {
      let bx, tall = false;
      if (front) {
        // Front tongues hug the side rims; rare short licks at the bottom centre.
        const r = Math.random();
        bx = r < 0.42 ? rnd(CL - 0.02 * S, CL + 0.09 * S)
          : r < 0.84 ? rnd(CR - 0.09 * S, CR + 0.02 * S)
          : rnd(CL + 0.28 * S, CR - 0.28 * S);
      } else {
        // Back: bottom wall (under-glow + occasional crest above the top edge)
        // plus columns rising just outside the left/right rims — the "engulfed
        // in 3D" read comes from these side walls behind the art.
        const r = Math.random();
        bx = r < 0.55 ? rnd(CL + 0.04 * S, CR - 0.04 * S)
          : r < 0.78 ? rnd(CL - 0.07 * S, CL + 0.02 * S)
          : rnd(CR - 0.02 * S, CR + 0.07 * S);
        tall = Math.random() < 0.28;
      }
      const centre = front && bx > CL + 0.2 * S && bx < CR - 0.2 * S;
      const k = (front ? 0.72 : 1) * (centre ? 0.55 : 1);
      push({
        type: 'flame', front, bx,
        y: rnd(CB - 0.02 * S, CB + 0.05 * S),
        vx: 0,
        vy: -rnd(0.30, 0.62) * S * k * (tall ? 1.7 : 1),
        r: rnd(0.065, 0.125) * S * (front ? 0.75 : 1) * (centre ? 0.6 : 1),
        life: rnd(0.65, 1.2) * (tall ? 1.35 : 1) * (centre ? 0.6 : 1),
        age: 0, swayF: rnd(6, 11), swayA: rnd(0.03, 0.09) * S, ph: rnd(0, TAU),
      });
    };
    const spawnEmber = () => push({
      type: 'ember', front: Math.random() < 0.5,
      bx: rnd(CL + 0.05 * S, CR - 0.05 * S), y: rnd(CB - 0.15 * S, CB),
      vx: rnd(-0.06, 0.06) * S, vy: -rnd(0.5, 0.95) * S,
      r: rnd(1.6, 3.2) * (S / 280),
      life: rnd(1.0, 1.7), age: 0, swayF: rnd(5, 9), swayA: rnd(0.02, 0.05) * S, ph: rnd(0, TAU),
    });
    const spawnSmoke = () => push({
      type: 'haze', front: false,
      bx: rnd(CL + 0.1 * S, CR - 0.1 * S), y: rnd(CT - 0.05 * S, CT + 0.35 * S),
      vx: rnd(-0.03, 0.03) * S, vy: -rnd(0.10, 0.22) * S, grow: 0.10 * S,
      r: rnd(0.06, 0.11) * S, life: rnd(0.9, 1.5), age: 0,
      swayF: rnd(2, 4), swayA: rnd(0.01, 0.03) * S, ph: rnd(0, TAU),
    });

    const spawnCrown = (n) => {
      // The initial "wave hits the top edge" splash — droplets arc up and out,
      // then gravity pulls them down past the sides. Centre ones go BEHIND the
      // art (they crest over the top rim), edge ones fall in front of it.
      for (let i = 0; i < n; i++) {
        const bx = rnd(CL, CR);
        const edge = bx < CL + 0.22 * S || bx > CR - 0.22 * S;
        push({
          type: 'drop', front: edge, bx,
          y: rnd(CT - 0.05 * S, CT + 0.01 * S),
          vx: rnd(-0.30, 0.30) * S * (edge ? 1 : 0.55),
          vy: -rnd(0.15, 0.55) * S,
          r: rnd(0.014, 0.038) * S, life: 3, age: 0,
          swayF: 0, swayA: 0, ph: 0,
        });
      }
    };
    const spawnRivulet = () => {
      // Streams running down the FRONT face, clinging to the left/right rims
      // so the art stays readable; a few sheets down the middle.
      const r = Math.random();
      const bx = r < 0.38 ? rnd(CL - 0.01 * S, CL + 0.06 * S)
        : r < 0.76 ? rnd(CR - 0.06 * S, CR + 0.01 * S)
        : rnd(CL + 0.1 * S, CR - 0.1 * S);
      push({
        type: 'drop', front: true, bx,
        y: rnd(CT - 0.06 * S, CT + 0.02 * S),
        vx: rnd(-0.02, 0.02) * S, vy: rnd(0.05, 0.25) * S,
        r: rnd(0.016, 0.042) * S, life: 3, age: 0,
        swayF: rnd(7, 13), swayA: rnd(0.006, 0.018) * S, ph: rnd(0, TAU),
      });
    };
    const spawnSplash = (x, y) => {
      const n = 2 + ((Math.random() * 3) | 0);
      for (let i = 0; i < n; i++) push({
        type: 'splash', front: true, bx: x, y,
        vx: rnd(-0.35, 0.35) * S, vy: -rnd(0.05, 0.30) * S,
        r: rnd(0.007, 0.016) * S, life: rnd(0.25, 0.45), age: 0,
        swayF: 0, swayA: 0, ph: 0,
      });
    };
    const spawnMist = (n) => {
      for (let i = 0; i < n; i++) push({
        type: 'haze', front: false,
        bx: rnd(CL, CR), y: rnd(CT + 0.15 * S, CB), grow: 0.04 * S,
        vx: rnd(-0.02, 0.02) * S, vy: -rnd(0.02, 0.06) * S,
        r: rnd(0.14, 0.28) * S, life: rnd(1.4, 2.0), age: 0,
        swayF: rnd(1, 3), swayA: rnd(0.01, 0.02) * S, ph: rnd(0, TAU),
      });
    };

    const TOTAL = 2.7, EMIT = kind === 'fire' ? 1.5 : 1.15;
    const G = 2.6 * S, BUOY = 1.05 * S;
    let raf = 0, last = performance.now(), elapsed = 0;
    let accB = 0, accF = 0, accE = 0, accS = 0;
    if (kind === 'water') { spawnCrown(26); spawnMist(8); }

    const step = (now) => {
      const dt = Math.min(0.04, (now - last) / 1000);
      last = now; elapsed += dt;
      // Global envelope: full strength, then a long cool-down fade.
      const env = elapsed < 1.85 ? 1 : Math.max(0, 1 - (elapsed - 1.85) / 0.8);

      if (elapsed < EMIT) {
        const ease = 1 - elapsed / EMIT;
        if (kind === 'fire') {
          accB += 95 * ease * dt; while (accB > 1) { spawnFlame(false); accB--; }
          accF += 48 * ease * dt; while (accF > 1) { spawnFlame(true); accF--; }
          accE += 7 * dt; while (accE > 1) { spawnEmber(); accE--; }
          if (elapsed > 0.45) { accS += 6 * dt; while (accS > 1) { spawnSmoke(); accS--; } }
        } else {
          accF += 55 * ease * dt; while (accF > 1) { spawnRivulet(); accF--; }
        }
      }

      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        p.age += dt;
        if (p.age >= p.life) { parts.splice(i, 1); continue; }
        if (p.type === 'flame') {
          p.vy = Math.max(p.vy - BUOY * dt, -1.5 * S);          // buoyancy
          p.vx = (p.vx + rnd(-1, 1) * 0.25 * S * dt) * 0.95;    // turbulent wander
        } else if (p.type === 'ember') {
          p.vy -= 0.5 * S * dt;
        } else if (p.type === 'haze') {
          p.r += p.grow * dt;                                    // smoke/mist billows
        } else if (p.type === 'drop') {
          p.vy = Math.min(p.vy + G * dt, 1.6 * S);              // gravity w/ terminal v
          if (p.y > CB + 0.04 * S) {                             // hit the bottom rim
            spawnSplash(p.bx, CB + 0.03 * S);
            parts.splice(i, 1);
            continue;
          }
        } else if (p.type === 'splash') {
          p.vy += G * dt;
        }
        p.bx += p.vx * dt;
        p.y += p.vy * dt;
      }

      bctx.clearRect(0, 0, W, H);
      fctx.clearRect(0, 0, W, H);
      for (const p of parts) {
        const ctx = p.front ? fctx : bctx;
        const t = p.age / p.life;
        let img, a, rr = p.r, elong = 1, swayK = 1;
        if (p.type === 'flame') {
          img = SPR.ramp[t < 0.32 ? 0 : t < 0.68 ? 1 : 2];
          a = Math.min(1, t * 5) * Math.pow(1 - t, 1.15);
          rr = p.r * (0.6 + 0.4 * (1 - t));                     // taper toward the tip
          elong = 1.65;
          swayK = 0.35 + t;                                      // tips sway more than roots
        } else if (p.type === 'ember') {
          img = SPR.ember;
          a = (0.55 + 0.45 * Math.sin(p.age * 26 + p.ph)) * Math.pow(1 - t, 0.8);
        } else if (p.type === 'haze') {
          img = SPR.haze;
          a = Math.min(1, t * 4) * Math.pow(1 - t, 1.4);
        } else if (p.type === 'drop') {
          const sp = Math.min(1, Math.abs(p.vy) / (1.2 * S));   // fast water = white foam
          img = SPR.ramp[sp > 0.55 ? 0 : sp > 0.25 ? 1 : 2];
          a = 0.9;
          elong = 1 + 1.6 * sp;                                  // stretch with speed
        } else {
          img = SPR.ramp[0];
          a = 0.9 * (1 - t);
        }
        ctx.globalAlpha = Math.max(0, Math.min(1, a * env));
        // Additive glow for the hot/wet bodies; normal alpha for smoke & mist.
        ctx.globalCompositeOperation = p.type === 'haze' ? 'source-over' : 'lighter';
        const dx = p.bx + (p.swayA ? Math.sin(p.age * p.swayF + p.ph) * p.swayA * swayK : 0);
        ctx.drawImage(img, dx - rr, p.y - rr * elong, rr * 2, rr * 2 * elong);
      }

      if (elapsed < TOTAL && (elapsed < EMIT || parts.length)) {
        raf = requestAnimationFrame(step);
      } else {
        bctx.clearRect(0, 0, W, H);
        fctx.clearRect(0, 0, W, H);
      }
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playKey, kind]);

  return (
    <>
      <div className={`cover-fx cover-fx--back cover-fx--${kind}`} aria-hidden="true">
        <span className="cover-fx__aura" />
        <canvas ref={backRef} className="cover-fx__canvas" />
      </div>
      <div className={`cover-fx cover-fx--front cover-fx--${kind}`} aria-hidden="true">
        <canvas ref={frontRef} className="cover-fx__canvas" />
      </div>
    </>
  );
}

function PlayerSection({ isDark, lang, initialPlaylist, initialTrack, onPlayTrack, onTrackChange, onRequestAutoplay, onStreamSignal, audio, visible, lyricsMode, onToggleLyrics, onCloseLyrics, showToast, navigateToArtist, aiStatus, onAddToPlaylist, onQueueNext, onReorderQueue, shuffleOn, onToggleShuffle, streamActive, streamAdapt }) {
  const [playlist, setPlaylist] = useState(initialPlaylist || []);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [hoveredQueueIdx, setHoveredQueueIdx] = useState(-1);
  const isMobile = useIsMobile();             // mobile player: full-width cover, queue drawer
  const [queueOpen, setQueueOpen] = useState(false);  // mobile queue slide-up drawer

  // Derived: current track from playlist + index. Declared HERE (not later)
  // so useEffects below see the real value, not the hoisted-var undefined.
  // @babel/standalone uses es2015 preset which transpiles `const` to `var`;
  // accessing currentTrack before declaration yields `undefined`, which made
  // effect deps `[currentTrack?.track_id]` always evaluate to `[undefined]`
  // and silently skip — i.e. lyric-explain state never reset on track change.
  const currentTrack = currentIndex >= 0 && currentIndex < playlist.length ? playlist[currentIndex].track : null;

  // ── Queue drag-to-reorder ───────────────────────────────────────────────
  // Freshest playlist mirrored into a ref so the commit (snapshotted by the
  // drag hook at grab time) reorders the LATEST array — a stream append landing
  // mid-drag (always at the tail) won't get dropped.
  const playlistRef = useRef(playlist);
  playlistRef.current = playlist;
  // Now-playing id mirrored the same way: the commit below recomputes
  // currentIndex from it, and the track may advance mid-drag (song ended).
  const currentTrackIdRef = useRef(null);
  currentTrackIdRef.current = currentTrack ? currentTrack.track_id : null;
  const queueScrollRef = useRef(null);
  const [droppedId, setDroppedId] = useState(null);   // track id to flash after a drop
  const droppedTimerRef = useRef(null);

  // Commit a reorder: move item `from` → `over` (original coords), push the new
  // order UP to App.playerPlaylist (source of truth) so the next wave append
  // can't clobber it, and briefly glow the moved row. Only the now-playing row
  // is pinned by the hook, so a drag may cross the now-playing boundary —
  // recompute currentIndex from the track id (ref: fresh even if the track
  // advanced mid-drag).
  const reorderQueue = (from, over) => {
    const cur = playlistRef.current || [];
    if (from < 0 || from >= cur.length) return;
    const dest = Math.max(0, Math.min(over, cur.length - 1));
    const next = cur.slice();
    const moved = next.splice(from, 1)[0];
    next.splice(dest, 0, moved);
    const curId = currentTrackIdRef.current;
    if (curId != null) {
      const ni = next.findIndex(h => ((h && h.track) ? h.track : h)?.track_id === curId);
      if (ni >= 0) setCurrentIndex(ni);
    }
    setPlaylist(next);
    if (onReorderQueue) onReorderQueue(next);
    const id = moved && moved.track && moved.track.track_id;
    if (id) {
      setDroppedId(id);
      if (droppedTimerRef.current) clearTimeout(droppedTimerRef.current);
      droppedTimerRef.current = setTimeout(() => setDroppedId(null), 560);
    }
  };

  const queueReorder = useQueueReorder({
    lockedIndex: currentIndex,   // pin ONLY the now-playing row; history stays movable
    count: playlist.length,
    onCommit: reorderQueue,
    scrollRef: queueScrollRef,
  });

  // ── Shuffle FLIP ────────────────────────────────────────────────────────
  // The new order arrives asynchronously (click → App.toggleShuffle →
  // initialPlaylist → setPlaylist), so the "first" rects must be captured at
  // click time; the layout effect below plays first→last once the reordered
  // rows are in the DOM. Skipped under prefers-reduced-motion (no capture →
  // no animation, instant reorder).
  const shuffleRectsRef = useRef(null);   // Map<track_id, DOMRect> | null
  const handleShuffleClick = () => {
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const root = queueScrollRef.current;
    if (!reduced && root) {
      const m = new Map();
      root.querySelectorAll('[data-tid]').forEach(el => m.set(el.dataset.tid, el.getBoundingClientRect()));
      shuffleRectsRef.current = m;
    }
    if (onToggleShuffle) onToggleShuffle();
  };
  useLayoutEffect(() => {
    const prev = shuffleRectsRef.current;
    if (!prev) return;
    shuffleRectsRef.current = null;
    const root = queueScrollRef.current;
    if (!root || !root.querySelectorAll) return;
    const view = root.getBoundingClientRect();
    const inView = (r) => r.bottom > view.top && r.top < view.bottom;
    let k = 0;   // stagger counter over rows that actually move
    root.querySelectorAll('[data-tid]').forEach(el => {
      const old = prev.get(el.dataset.tid);
      if (!old || !el.animate) return;
      const now = el.getBoundingClientRect();
      const dy = old.top - now.top;
      // Rows that never cross the visible scroll area just snap into place.
      if (Math.abs(dy) < 2 || (!inView(old) && !inView(now))) return;
      el.animate([
        { transform: `translateY(${dy}px)` },
        { transform: `translateY(${dy * 0.5}px) scale(0.97)`, offset: 0.55 },
        { transform: 'translateY(0) scale(1)' },
      ], { duration: 460, delay: Math.min(k * 16, 160), easing: 'cubic-bezier(0.22, 0.9, 0.24, 1)', fill: 'backwards' });
      k++;
    });
  }, [playlist]);

  // Derive playback state from shared audio hook
  const isPlaying = audio?.isPlaying ?? false;
  const isBuffering = audio?.isBuffering ?? false;
  // Subscribe to external time store — PlayerSection's progress UI rebuilds
  // ~4×/sec, but the rest of the app no longer re-renders on each tick.
  const currentTime = useCurrentTime();
  const duration = audio?.duration ?? 0;
  const [volume, setVolume] = useState(0.85);

  // Sync volume to audio
  useEffect(() => { if (audio) audio.setVolume(volume); }, [volume]);

  // 3D tilt state for album art (Spatial concept)
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const artWrapRef = useRef(null);

  // Tilt magnitude. The same ±1 normalized pointer offset barely reads on a
  // small screen (the cover is physically tiny there), so bump the rotation
  // ~30% (10°→13°) below ~820px to keep the 3D parallax legible. Computed once
  // per mount — the section remounts on nav, which covers resizes in practice
  // (same rationale as spectrumBarCount above).
  const tiltGain = useMemo(() => ((window.innerWidth || 1280) <= 820 ? 13 : 10), []);

  // Audio-reactive spectrum: hook into the shared <audio> via AnalyserNode
  // and extract a dominant color from the current cover. Both feed the
  // mirrored SpectrumBars that flank the cover row during playback.
  const { analyserRef, dataArrayRef } = useAudioAnalyser(audio?.audioRef, isPlaying);

  // More bars on wide screens so the strips don't look sparse when the
  // cover row stretches (27"+ monitors). Computed once per mount — the
  // section remounts on every nav, which covers window resizes in practice.
  const spectrumBarCount = useMemo(() => {
    const w = window.innerWidth || 1280;
    if (w >= 2200) return 56;
    if (w >= 1700) return 44;
    return 32;
  }, []);

  // Adaptive cover size. On short viewports the fixed 60vh cover + title + sonic
  // vibe + controls overflow the column (overflow:hidden + center), pushing the
  // Lossless badge off-screen. Measure the column and shrink the cover so ≥10%
  // of the column height stays free — but never grow it past the current
  // clamp(220px,60vh,640px) cap. Sized in a layout effect so it's set pre-paint.
  const playerColRef = useRef(null);
  const playerHintRef = useRef(null);
  const playerMetaRef = useRef(null);
  const [coverPx, setCoverPx] = useState(null);
  useLayoutEffect(() => {
    if (!visible) return;
    const col = playerColRef.current;
    if (!col) return;
    const measure = () => {
      const colH = col.clientHeight;
      const colW = col.clientWidth;
      if (colH <= 0) return;
      const cs = getComputedStyle(col);
      const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
      const gap = parseFloat(cs.rowGap) || parseFloat(cs.gap) || 0;
      const nGaps = Math.max(0, col.children.length - 1);
      const hintH = playerHintRef.current ? playerHintRef.current.offsetHeight : 0;
      const metaH = playerMetaRef.current ? playerMetaRef.current.offsetHeight : 0;
      const winH = window.innerHeight || colH;
      const capPx = Math.min(760, Math.max(220, 0.72 * winH));  // height cap (raised for a larger cover)
      const reserve = 0.07 * colH;                               // ≥7% free space (logo is shorter now)
      const avail = colH - padY - gap * nGaps - reserve - hintH - metaH;
      // Horizontal guard: the cover is square + sized only by height, so on a
      // tall-but-narrow viewport it could exceed the column width and clip
      // (col has overflow:hidden). Cap by width minus room for the flanking
      // prev/next buttons + their gaps (~150px).
      const availW = colW - 150;
      const px = Math.round(Math.max(140, Math.min(capPx, avail, availW)));
      setCoverPx(prev => (prev !== null && Math.abs(prev - px) < 2) ? prev : px);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(col);
    if (playerMetaRef.current) ro.observe(playerMetaRef.current);
    window.addEventListener('resize', measure);
    return () => { ro.disconnect(); window.removeEventListener('resize', measure); };
  }, [visible, currentTrack?.track_id, lyricsMode]);

  // Vinyl-stack transition: snapshot of outgoing track + direction, plus a
  // nonce that bumps on every track change to re-trigger the CSS entry
  // animation (CSS animations only replay when the keyed element remounts).
  const [outgoingTrack, setOutgoingTrack] = useState(null);
  const [transitionDir, setTransitionDir] = useState('next');
  const [entryNonce, setEntryNonce] = useState(0);
  const transitionTimerRef = useRef(null);

  // AI Chat drawer open/close
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Close the drawer whenever the playing track changes
  useEffect(() => { setDrawerOpen(false); }, [currentTrack?.track_id]);

  // Inline lyric explain state — per-line expansion + fetched explanations
  const [expandedLines, setExpandedLines] = useState(new Set());
  const [explainStates, setExplainStates] = useState(new Map());

  // On-demand lyrics. Only search hits carry `lyrics` inline; autoplay-queue,
  // recently-played, liked and stream tracks don't, so the player fetches the
  // full text by track_id and every source shows lyrics uniformly.
  const [fetchedLyrics, setFetchedLyrics] = useState(null);
  useEffect(() => {
    const id = currentTrack?.track_id;
    if (!id) { setFetchedLyrics(null); return; }
    // Inline lyrics already present (search path) — no fetch needed.
    if (currentTrack?.lyrics) { setFetchedLyrics(currentTrack.lyrics); return; }
    let alive = true;
    setFetchedLyrics(null);
    apiFetch(`/search/tracks/${encodeURIComponent(id)}/lyrics`)
      .then(r => { if (alive) setFetchedLyrics((r && r.lyrics) || null); })
      .catch(() => { if (alive) setFetchedLyrics(null); });
    return () => { alive = false; };
  }, [currentTrack?.track_id]);

  // Track object the lyrics view consumes — currentTrack with lyrics filled in
  // from whichever source had them (inline or fetched).
  const lyricsTrack = useMemo(
    () => (currentTrack ? { ...currentTrack, lyrics: currentTrack.lyrics || fetchedLyrics || null } : null),
    [currentTrack, fetchedLyrics]
  );

  // Reset when track changes — different lyrics, different state
  useEffect(() => {
    setExpandedLines(new Set());
    setExplainStates(new Map());
  }, [currentTrack?.track_id]);

  const handleToggleLyricExplain = useCallback(async (lineIdx, line) => {
    // Capture whether we're collapsing or expanding BEFORE the state update
    const wasExpanded = expandedLines.has(lineIdx);
    const hasResult = explainStates.has(lineIdx) && !explainStates.get(lineIdx)?.error;

    // Toggle expansion
    setExpandedLines(prev => {
      const next = new Set(prev);
      if (next.has(lineIdx)) {
        next.delete(lineIdx);
      } else {
        next.add(lineIdx);
      }
      return next;
    });

    // If we were collapsing, no fetch needed
    if (wasExpanded) return;

    // If we already have a cached result, no fetch needed
    if (hasResult) return;

    // Mark loading
    setExplainStates(prev => {
      const next = new Map(prev);
      next.set(lineIdx, { loading: true });
      return next;
    });

    // Fetch
    try {
      const res = await apiFetch('/chat/track-chat', {
        method: 'POST',
        body: JSON.stringify({
          track_context: {
            title: currentTrack?.title || '',
            artist: currentTrack?.artist || '',
            album: currentTrack?.album || null,
            year: currentTrack?.year || null,
            genre: currentTrack?.genre || null,
            full_lyrics: currentTrack?.lyrics || fetchedLyrics || '',
          },
          mode: 'lyric_explain',
          selected_line: line,
          message: lang === 'ru' ? 'Объясни эту строчку' : 'Explain this line',
          // Explicit answer language (backend forces the reply language from this).
          lang,
          llm_base_url: localStorage.getItem('llm_base_url') || undefined,
          llm_model: localStorage.getItem('llm_model') || undefined,
          }),
      });
      setExplainStates(prev => {
        const next = new Map(prev);
        next.set(lineIdx, {
          loading: false,
          message: res.message,
          web_search_used: res.web_search_used,
        });
        return next;
      });
    } catch (e) {
      setExplainStates(prev => {
        const next = new Map(prev);
        next.set(lineIdx, { loading: false, error: e.message });
        return next;
      });
    }
  }, [currentTrack, fetchedLyrics, expandedLines, explainStates]);

  // Burst feedback for fire/water icon clicks (single active burst at a time)
  const [burstFor, setBurstFor] = useState(null);
  const burstTimerRef = useRef(null);

  // Огонёк/Вода cover combustion: kind drives fire vs water; the nonce remounts
  // <CoverCombustion> so each press replays cleanly even mid-burn.
  const [fxKind, setFxKind] = useState(null);
  const [fxKey, setFxKey] = useState(0);
  const fxTimerRef = useRef(null);
  const FX_MS = 2800;

  // Play/pause glassy indicator: 'play' | 'pause' | null. Keyed nonce
  // re-mounts the element so the CSS animation replays on every toggle.
  const [playFeedback, setPlayFeedback] = useState(null);
  const [playFeedbackKey, setPlayFeedbackKey] = useState(0);
  const playFeedbackTimerRef = useRef(null);
  const PLAY_FEEDBACK_MS = 1200;

  // First-3-clicks hint, persisted across page loads via localStorage so
  // returning users don't see it forever. Counter is incremented on each
  // cover-click toggle, not on every play/pause from other surfaces.
  const HINT_LIMIT = 3;
  const HINT_STORAGE_KEY = 'musix_cover_play_hint_count';
  const [hintVisible, setHintVisible] = useState(() => {
    try {
      const n = parseInt(localStorage.getItem(HINT_STORAGE_KEY) || '0', 10);
      return n < HINT_LIMIT;
    } catch { return true; }
  });
  const hintHideTimerRef = useRef(null);

  // Cleanup transient timers on unmount
  useEffect(() => () => {
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    if (burstTimerRef.current) clearTimeout(burstTimerRef.current);
    if (fxTimerRef.current) clearTimeout(fxTimerRef.current);
    if (playFeedbackTimerRef.current) clearTimeout(playFeedbackTimerRef.current);
    if (hintHideTimerRef.current) clearTimeout(hintHideTimerRef.current);
  }, []);

  const flashPlayFeedback = (kind) => {
    setPlayFeedback(kind);
    setPlayFeedbackKey(k => k + 1);
    if (playFeedbackTimerRef.current) clearTimeout(playFeedbackTimerRef.current);
    playFeedbackTimerRef.current = setTimeout(() => setPlayFeedback(null), PLAY_FEEDBACK_MS);
  };

  // Cover-specific click handler: triggers togglePlay (which also flashes the
  // glassy indicator) and ticks the hint counter. Hint is cover-specific —
  // Space-bar users don't need to be reminded which surface they just used.
  const handleCoverClick = () => {
    // A horizontal swipe just ended — the trailing click is not a tap.
    if (swipeClickGuardRef.current) { swipeClickGuardRef.current = false; return; }
    if (!currentTrack) return;
    togglePlay();
    if (hintVisible) {
      try {
        const cur = parseInt(localStorage.getItem(HINT_STORAGE_KEY) || '0', 10);
        const next = cur + 1;
        localStorage.setItem(HINT_STORAGE_KEY, String(next));
        if (next >= HINT_LIMIT) {
          // Let user see the hint during this final click, then fade.
          if (hintHideTimerRef.current) clearTimeout(hintHideTimerRef.current);
          hintHideTimerRef.current = setTimeout(() => setHintVisible(false), 700);
        }
      } catch { /* ignore localStorage failures (private mode etc.) */ }
    }
  };

  const handleArtMouseMove = useCallback((e) => {
    if (!artWrapRef.current) return;
    const r = artWrapRef.current.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width - 0.5) * 2;  // -1..+1
    const ny = ((e.clientY - r.top) / r.height - 0.5) * 2;
    setTilt({ x: ny, y: nx });
  }, []);
  const handleArtMouseLeave = useCallback(() => {
    setTilt({ x: 0, y: 0 });
    setIsHovering(false);
  }, []);
  const handleArtMouseEnter = useCallback(() => {
    setIsHovering(true);
  }, []);

  // ── Mobile swipe-to-skip on the cover ────────────────────────────────────
  // The finger drags the cover horizontally (transform written imperatively —
  // no per-move re-renders); releasing past the threshold commits next/prev.
  // The dragged offset is handed to the mobile exit keyframes via --swipe-x
  // so the fly-out continues from under the finger instead of snapping back.
  const SWIPE_COMMIT_PX = 70;
  const swipeRef = useRef(null);             // live gesture session
  const swipeClickGuardRef = useRef(false);  // swallow the click trailing a swipe

  const handleArtTouchStart = (e) => {
    if (!currentTrack || lyricsMode || outgoingTrack) return;
    const t = e.touches[0];
    swipeRef.current = { x0: t.clientX, y0: t.clientY, dx: 0, horiz: null };
  };
  const handleArtTouchMove = (e) => {
    const s = swipeRef.current;
    if (!s) return;
    const t = e.touches[0];
    const dx = t.clientX - s.x0, dy = t.clientY - s.y0;
    // Lock gesture axis on the first decisive move; vertical = native scroll.
    if (s.horiz === null && (Math.abs(dx) > 8 || Math.abs(dy) > 8)) {
      s.horiz = Math.abs(dx) > Math.abs(dy);
    }
    if (!s.horiz) return;
    s.dx = dx;
    const el = artWrapRef.current;
    if (el) {
      el.style.transition = 'none';
      el.style.transform = `translateX(${(dx * 0.85).toFixed(1)}px) rotate(${(dx * 0.02).toFixed(2)}deg)`;
    }
  };
  const handleArtTouchEnd = (e, cancelled = false) => {
    const s = swipeRef.current;
    swipeRef.current = null;
    if (!s) return;
    const el = artWrapRef.current;
    const commitNext = !cancelled && s.horiz && s.dx <= -SWIPE_COMMIT_PX && currentIndex < playlist.length - 1;
    const commitPrev = !cancelled && s.horiz && s.dx >= SWIPE_COMMIT_PX && currentIndex > 0;
    if (s.horiz && Math.abs(s.dx) > 8) {
      // The click trailing this touchend is swipe residue, not a tap. Some
      // browsers skip that click entirely after a real drag — self-reset so
      // the guard can't eat the user's NEXT legitimate tap.
      swipeClickGuardRef.current = true;
      setTimeout(() => { swipeClickGuardRef.current = false; }, 400);
    }
    if (el) {
      if (commitNext || commitPrev) {
        // Hand the offset to the exit keyframes, reset the wrap instantly —
        // the outgoing snapshot picks the motion up from this exact position.
        el.style.setProperty('--swipe-x', `${Math.round(s.dx * 0.85)}px`);
        el.style.transition = 'none';
        el.style.transform = '';
      } else {
        el.style.transition = 'transform 240ms cubic-bezier(0.22, 0.9, 0.3, 1)';
        el.style.transform = '';
      }
    }
    if (commitNext) nextTrack();
    else if (commitPrev) prevTrack();
  };

  // Cover URL for color sampling — matches what AlbumCover renders internally.
  // useCoverColor caches per URL so flipping back to a known track is free.
  const coverUrl = currentTrack?.cover_art_path
    ? (currentTrack.cover_art_path.startsWith('http')
        ? currentTrack.cover_art_path
        : `${API}${currentTrack.cover_art_path}`)
    : null;
  const coverColor = useCoverColor(coverUrl);

  // Premium color tokens — theme-aware so the player's deep "console" feel
  // adapts to light theme instead of staying jet-black everywhere.
  const pBg          = isDark ? '#0B0E14' : '#f4f5fa';
  const pBgEnd       = isDark ? '#0F111A' : '#eaeaf0';
  const pSurface     = isDark ? '#1A1D27' : '#ebeaf2';
  // Dynamic accent — the cover's dominant HSL, clamped exactly like the
  // spectrum wave so the progress bar / queue / lyric glow speak the same hue
  // as the wave. Light theme sits darker so the tint holds contrast on pale
  // surfaces. Consumers reference var(--player-accent): it's a REGISTERED
  // custom property (see @property in index.css), so track changes glide to
  // the new hue instead of snapping.
  const accentColor = coverColor
    ? `hsl(${coverColor.h.toFixed(0)}, ${Math.min(85, Math.max(45, coverColor.s)).toFixed(0)}%, ${(isDark
        ? Math.min(72, Math.max(52, coverColor.l))
        : Math.min(52, Math.max(38, coverColor.l))).toFixed(0)}%)`
    : '#7C5BFF';
  const pAccent      = 'var(--player-accent, #7C5BFF)';
  const pText        = isDark ? '#FFFFFF' : '#161620';
  const pTextMuted   = isDark ? '#9BA1B0' : 'rgba(22,22,32,0.62)';
  const pTextSubtle  = isDark ? 'rgba(255,255,255,0.30)' : 'rgba(22,22,32,0.42)';
  const pUnfilled    = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(22,22,32,0.10)';
  const pBorder      = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(22,22,32,0.14)';
  const pBorderSubtle= isDark ? 'rgba(255,255,255,0.06)' : 'rgba(22,22,32,0.08)';
  const pPillBg      = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(22,22,32,0.04)';

  // Sync playlist + currentIndex from parent props in a single effect so the
  // findIndex below always reads the *fresh* playlist. The previous split into
  // two effects had a race: when both props changed in the same render (e.g.
  // user picks a track from a new search result set), the [initialTrack] effect
  // read the still-stale local `playlist` state and findIndex returned -1, so
  // autoplay silently failed.
  useEffect(() => {
    if (!initialPlaylist) return;
    setPlaylist(initialPlaylist);
    if (!initialTrack || initialPlaylist.length === 0) return;
    const idx = initialPlaylist.findIndex(h => h.track?.track_id === initialTrack.track_id);
    if (idx < 0) return;
    setCurrentIndex(idx);
    const track = initialPlaylist[idx].track;
    const url = buildStreamUrl(track.track_id);
    if (audio) {
      // Home unmounts every section, so the first nav away from home REMOUNTS
      // this section and re-runs this effect with the persisted playlist+track.
      // Re-pointing the src must NOT resurrect a paused track: read the LIVE
      // element state BEFORE setSrc (which synchronously rewrites
      // dataset.playbackTrackId). Autoplay only when the section is visible AND
      // this isn't just a remount of the already-loaded, paused track. A real
      // user pick always carries a different track id, so it still autoplays.
      const el = audio.audioRef?.current;
      const reloadingPausedTrack = !!(el && el.dataset.playbackTrackId === String(track.track_id) && el.paused);
      audio.setSrc(url, { trackId: track.track_id }, { autoplay: visible && !reloadingPausedTrack });
    }
  }, [initialPlaylist, initialTrack]);

  // Sync audio source when current track changes
  useEffect(() => {
    if (!currentTrack || !audio) return;
    const url = buildStreamUrl(currentTrack.track_id);
    // Only auto-play when section is visible
    audio.setSrc(url, { trackId: currentTrack.track_id }, { autoplay: visible && isPlaying });
  }, [currentTrack?.track_id]);

  // Audio continues in background across nav. On nav-back, resume ONLY if
  // the audio element itself isn't paused. Read the LIVE el.paused state
  // (not React isPlaying) — closure over isPlaying was racey: a stale
  // isPlaying=true from a previous render would call audio.play() even after
  // the user explicitly paused, restarting playback on the next nav.
  useEffect(() => {
    if (!audio || !currentTrack) return;
    const el = audio.audioRef?.current;
    if (visible && el && !el.paused) {
      setTimeout(() => audio.play(), 50);
    }
  }, [visible]);

  const playTrackAt = (index) => {
    if (index < 0 || index >= playlist.length) return;
    // Bootstrap spectrum analyser inside this (user-gesture) sync path so the
    // AudioContext is born in 'running' state — see togglePlay for the why.
    // Track-clicks in the queue go through here, not togglePlay.
    const audioEl = audio?.audioRef?.current;
    if (audioEl) _setupSpectrumAnalyser(audioEl);
    if (_spectrumState.ctx && _spectrumState.ctx.state === 'suspended') {
      _spectrumState.ctx.resume().catch(() => {});
    }

    setCurrentIndex(index);
    const track = playlist[index].track;
    const url = buildStreamUrl(track.track_id);
    if (audio) audio.setSrc(url, { trackId: track.track_id, noInfluence: !!playlist[index]._noInfluence }, { autoplay: true });
    // Propagate the new track up so App.playerTrack stays in sync. Without
    // this, LandingScreen / NowPlayingPebble / MiniPlaybackPopout (which all
    // read playerTrack at App level) would keep showing the *initial* track
    // even after the user skipped to a different one inside the playlist.
    if (onTrackChange) onTrackChange(track);
  };

  // When audio ends: advance within the current playlist if there's a next
  // track; otherwise request the autoplay queue from App (which fetches
  // similar tracks from the backend and appends them).
  //
  // Dep choice: audio?.audioRef (a stable useRef) rather than audio
  // (a new object literal each useAudioPlayer render). Without this,
  // listener teardown+re-add would fire on every timeupdate (~4Hz),
  // briefly leaving the element without any 'ended' listener.
  // Trigger the vinyl transition: snapshot current track for the outgoing
  // animation, bump entryNonce to remount the entry wrapper, and schedule
  // cleanup once the longest animation (720ms entry) finishes.
  const triggerVinyl = (dir) => {
    if (currentTrack) {
      setOutgoingTrack(currentTrack);
      setTransitionDir(dir);
    }
    setEntryNonce(n => n + 1);
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    // 320ms delay + 600ms bounce = 920ms total; small buffer for jitter.
    transitionTimerRef.current = setTimeout(() => {
      setOutgoingTrack(null);
      // Drop the swipe offset so a later non-swipe transition (auto-advance,
      // arrow tap) doesn't start its exit from a stale dragged position.
      if (artWrapRef.current) artWrapRef.current.style.removeProperty('--swipe-x');
    }, 960);
  };

  const triggerBurst = (kind) => {
    setBurstFor(kind);
    if (burstTimerRef.current) clearTimeout(burstTimerRef.current);
    burstTimerRef.current = setTimeout(() => setBurstFor(null), 640);
  };

  const triggerCoverFx = (kind) => {
    setFxKind(kind);
    setFxKey(k => k + 1);
    if (fxTimerRef.current) clearTimeout(fxTimerRef.current);
    fxTimerRef.current = setTimeout(() => setFxKind(null), FX_MS);
  };

  // Огонёк/Вода gesture: light the cover effect, flash the icon, record the
  // ephemeral signal, and ask App to rebuild the wave (it's a strong signal).
  const sendTaste = (kind) => {
    if (!currentTrack?.track_id) return;
    markPlaybackInteracted(audio?.audioRef?.current);
    triggerCoverFx(kind);
    triggerBurst(kind);
    postTasteSignal(currentTrack.track_id, kind);
    if (onStreamSignal) onStreamSignal('reaction', currentTrack);
  };

  useEffect(() => {
    const el = audio?.audioRef?.current;
    if (!el) return;
    const onEnded = () => {
      const lastIdx = playlist.length - 1;
      if (currentIndex >= 0 && currentIndex < lastIdx) {
        // Plain auto-advance — vinyl transition for visual continuity
        triggerVinyl('next');
        playTrackAt(currentIndex + 1);
      } else if (onRequestAutoplay && currentTrack) {
        // Exhausted current playlist — ask App to fetch more, then START the
        // first fresh track ourselves (same as the off-player advance path).
        // The prop-sync effect deliberately never resurrects an already-loaded
        // paused track, so without this the refill only landed in the queue
        // while the audio stayed silent — with the screen off the media
        // notification died with it and playback never came back.
        Promise.resolve(onRequestAutoplay(currentTrack)).then(fresh => {
          const first = Array.isArray(fresh) ? fresh[0] : null;
          if (!first) return;
          const t = first.track ? first.track : first;
          audio.setSrc(buildStreamUrl(t.track_id),
            { trackId: t.track_id, noInfluence: !!first._noInfluence },
            { autoplay: true });
          if (onTrackChange) onTrackChange(t);
        }).catch(() => {});
      }
    };
    el.addEventListener('ended', onEnded);
    return () => el.removeEventListener('ended', onEnded);
  }, [currentIndex, playlist.length, audio?.audioRef, currentTrack, onRequestAutoplay]);

  const togglePlay = () => {
    if (!audio) return;
    // Bootstrap the spectrum analyser INSIDE the user gesture. Doing this
    // synchronously here (instead of in a useEffect) ensures the AudioContext
    // is created with state='running' rather than 'suspended'; otherwise the
    // spectrum stays dark on the first track because resume() called from
    // an async effect is treated as out-of-gesture by some browsers.
    const audioEl = audio.audioRef?.current;
    if (audioEl) _setupSpectrumAnalyser(audioEl);
    if (_spectrumState.ctx && _spectrumState.ctx.state === 'suspended') {
      _spectrumState.ctx.resume().catch(() => {});
    }

    // Capture current isPlaying BEFORE toggling so we know which icon to
    // flash. Fired only from intentional user toggles (cover click, Space)
    // — auto-events on the audio element bypass this and don't flash.
    const willPause = isPlaying;
    audio.togglePlay();
    flashPlayFeedback(willPause ? 'pause' : 'play');
  };
  const nextTrack = () => {
    // Manual skip = strong stream signal: mark the abandoned listen as
    // interacted BEFORE playTrackAt flushes it via setSrc, then let App drop
    // the prefetched stream buffer and refetch with the fresh profile.
    markPlaybackInteracted(audio?.audioRef?.current);
    if (currentIndex < playlist.length - 1) {
      triggerVinyl('next');
      playTrackAt(currentIndex + 1);
    }
    if (onStreamSignal && currentTrack) onStreamSignal('skip', currentTrack);
  };
  const prevTrack = () => {
    markPlaybackInteracted(audio?.audioRef?.current);
    if (currentIndex > 0) {
      triggerVinyl('prev');
      playTrackAt(currentIndex - 1);
    }
  };
  const seek = (time) => { if (audio) audio.seek(time); };

  // Expose now-playing to the OS media controls (Windows SMTC / media keys).
  // Wired to the same handlers as the on-screen player buttons.
  useMediaSession({
    currentTrack,
    isPlaying,
    audioRef: audio?.audioRef,
    onPlay: () => audio?.play?.(),
    onPause: () => audio?.pause?.(),
    onNext: nextTrack,
    onPrev: prevTrack,
    onSeek: seek,
  });

  // Claim Space ownership while this section is mounted so the global shortcut
  // handler defers to the local handler below (which adds the cover flash).
  // Mount-only ([] deps) so the flag never flickers on a re-render — a flicker
  // would briefly let both handlers fire and re-introduce the double-toggle.
  useEffect(() => {
    _playerOwnsSpace = true;
    return () => { _playerOwnsSpace = false; };
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      // Read time live from the audio element to avoid stale closures.
      // (Listing currentTime in deps would re-attach this listener ~4×/sec
      // and was the original cause of the playback re-render storm.)
      const now = audio?.audioRef?.current?.currentTime || 0;
      if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
      if (e.code === 'ArrowLeft') seek(Math.max(0, now - 10));
      if (e.code === 'ArrowRight') seek(Math.min(duration, now + 10));
      if (e.code === 'ArrowUp') { e.preventDefault(); setVolume(v => Math.min(1, v + 0.05)); }
      if (e.code === 'ArrowDown') { e.preventDefault(); setVolume(v => Math.max(0, v - 0.05)); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [duration, audio]);

  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const progressPct = duration ? (currentTime / duration * 100) : 0;

  return (
    <div className="player-accent-scope" style={{
      flex:1, display:'flex', flexDirection:'column', overflow:'hidden',
      // The ambient wash (blurred cover + shade + grain + vignette) now lives
      // at the App shell level (PlayerAmbient) as a full-bleed layer so it
      // flows behind the floating nav. This section stays transparent and rides
      // on top of it — no more hard black strip under the rail.
      background: 'transparent',
      position: 'relative',
      '--player-accent': accentColor,
    }}>

      {/* Main content area — top padding gives the cover breathing room from
          the (now collapsed) section header bar. */}
      <div style={{ flex:1, display:'flex', flexDirection: isMobile ? 'column' : 'row', overflow: isMobile ? 'auto' : 'hidden', padding: isMobile ? '10px 12px 24px' : 'clamp(24px, 5vh, 56px) 32px 24px', gap: isMobile ? 14 : 28, position:'relative' }}>

        {/* ════════════════ LEFT: Player ════════════════
            justifyContent:center keeps the cover+controls visually centered
            within the column when there's vertical slack. The cover-row sits
            outside the constrained-width inner wrapper so prev/next side
            buttons can flank the cover in the column's empty horizontal slack. */}
        <div className="player-fade-in" ref={playerColRef} style={{
          flex: isMobile ? '0 0 auto' : 1, minWidth:0,
          display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent: isMobile ? 'flex-start' : 'center',
          gap:'clamp(14px, 2vh, 24px)', position:'relative', zIndex:1,
          overflow: isMobile ? 'visible' : 'hidden', padding:'clamp(8px, 2vh, 24px) 0',
        }}>

          {/* First-3-clicks hint above the cover row. Stays in the DOM so the
              fade-out animates; collapses to a near-zero footprint when hidden
              so it doesn't push the cover down for returning users. */}
          {currentTrack && (
            <div
              ref={playerHintRef}
              className={`player-art-hint${hintVisible ? '' : ' player-art-hint--hidden'}`}
              style={{ height: hintVisible ? 'auto' : 0, marginBottom: hintVisible ? 4 : 0 }}
              aria-hidden={!hintVisible}
            >
              {lang === 'ru'
                ? 'Нажми на обложку, чтобы поставить на паузу'
                : 'Tap the cover to play or pause'}
            </div>
          )}

          {/* Cover row: [◄] [cover] [►] with two absolute spectrum strips
              underneath. position:relative anchors the absolute strips.
              The cover (.player-art-wrap, z:1) and side buttons (z:1) sit
              on top of the strips (z:0); the strips' blur halo bleeds onto
              the cover's edges for the "wave entering the album" feel. */}
          <div className="player-cover-row" style={{
            position:'relative',
            display:'flex', alignItems:'center', justifyContent:'center',
            gap:'clamp(12px, 2.5vw, 36px)', width:'100%', flexShrink:0,
          }}>
            {/* No spectrum on phones: the analyser is never wired there
                (_IS_MOBILE), but the mounted components still burned a rAF
                loop each — pure battery drain with zero pixels moving. */}
            {!isMobile && (
              <SpectrumBars
                side="left"
                analyserRef={analyserRef}
                dataArrayRef={dataArrayRef}
                color={coverColor}
                isPlaying={isPlaying}
                barCount={spectrumBarCount}
              />
            )}
            {!isMobile && (
              <SpectrumBars
                side="right"
                analyserRef={analyserRef}
                dataArrayRef={dataArrayRef}
                color={coverColor}
                isPlaying={isPlaying}
                barCount={spectrumBarCount}
              />
            )}
            {!isMobile && (
            <button
              type="button" className="player-side-btn"
              onClick={prevTrack} disabled={currentIndex <= 0}
              title={lang==='ru'?'Предыдущий':'Previous'}
              aria-label={lang==='ru'?'Предыдущий':'Previous'}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
            </button>
            )}

            {/* Album art: flip card + vinyl-stack transition. Clicking toggles
                play and flashes the glassy feedback indicator. */}
            <div
              ref={artWrapRef}
              className="player-art-wrap"
              onMouseMove={lyricsMode || outgoingTrack ? undefined : handleArtMouseMove}
              onMouseEnter={lyricsMode ? undefined : handleArtMouseEnter}
              onMouseLeave={handleArtMouseLeave}
              onClick={currentTrack ? handleCoverClick : undefined}
              onTouchStart={isMobile ? handleArtTouchStart : undefined}
              onTouchMove={isMobile ? handleArtTouchMove : undefined}
              onTouchEnd={isMobile ? handleArtTouchEnd : undefined}
              onTouchCancel={isMobile ? (e) => handleArtTouchEnd(e, true) : undefined}
              style={{
                cursor: currentTrack ? 'pointer' : 'default',
                ...(coverPx ? { width: `${coverPx}px`, maxWidth: `${coverPx}px` } : null),
              }}
            >
              {/* Огонёк/Вода combustion — a back layer (behind the art) + a
                  front layer (over its rim) wrap the cover in 3D. Mounted only
                  while burning; keyed on fxKey so each press replays. */}
              {fxKind && <CoverCombustion key={fxKey} kind={fxKind} playKey={fxKey} />}

              {/* Outgoing cover snapshot — animates "into the stack". Keyed on
                  entryNonce so rapid next-spam restarts the exit cleanly. */}
              {outgoingTrack && (
                <div key={`outgoing-${entryNonce}`}
                  className={`player-art-outgoing player-art-outgoing--${transitionDir}`}>
                  <div className="player-art-front">
                    <AlbumCover
                      title={outgoingTrack.title || ''}
                      artist={outgoingTrack.artist || ''}
                      fluid
                      eager
                      isDark={isDark}
                      coverPath={outgoingTrack.cover_art_path}
                      radius={20}
                    />
                  </div>
                </div>
              )}

              {/* Press wrapper: own transform layer for the physical-button
                  scale-down on click. Sits between wrap and entry so its
                  scale doesn't fight with the entry/tilt/flip transforms. */}
              <div className="player-art-press">
                {/* Incoming cover — keyed on entryNonce so the CSS entry
                    animation replays on every track change. Direction class
                    is applied only after a real transition has been triggered;
                    the initial mount stays still so the cover doesn't fly in
                    on first page load. */}
                <div
                  key={entryNonce}
                  className={`player-art-entry${entryNonce > 0 ? ` player-art-entry--${transitionDir}` : ''}`}
                >
                  <div
                    className="player-art-tilt"
                    style={{
                      transform: lyricsMode || outgoingTrack
                        ? 'none'
                        : `rotateY(${tilt.y * tiltGain}deg) rotateX(${-tilt.x * tiltGain}deg) scale(${isHovering ? 1.04 : 1})`,
                    }}
                  >
                    <div
                      className="player-art-flipper"
                      style={{ transform: lyricsMode ? 'rotateY(180deg)' : 'none' }}
                    >
                      <div className="player-art-front">
                        <AlbumCover
                          title={currentTrack?.title || ''}
                          artist={currentTrack?.artist || ''}
                          fluid
                          eager
                          isDark={isDark}
                          coverPath={currentTrack?.cover_art_path}
                          radius={20}
                          spinning={isPlaying}
                        />
                        {/* Frosted blur overlay during play/pause feedback.
                            Keyed on playFeedbackKey to restart animation. */}
                        {playFeedback && (
                          <div key={`blur-${playFeedbackKey}`}
                            className="player-art-blur-overlay" />
                        )}
                        {/* Buffering veil — blurs the cover behind a spinner
                            while audio is still loading. Always mounted (when a
                            track is up) so it can fade BOTH in and out; the
                            .is-on class drives the opacity. */}
                        {currentTrack && (
                          <div className={`player-art-buffering${isBuffering ? ' is-on' : ''}`}
                            aria-hidden={!isBuffering}>
                            <span className="player-art-spinner" />
                          </div>
                        )}
                        {/* Shine — halved alpha (was 0.22 → 0.11) per request */}
                        <div
                          className="player-art-shine"
                          style={{
                            background: isHovering
                              ? `linear-gradient(${135 + tilt.y * 20}deg, rgba(255,255,255,0) 30%, rgba(255,255,255,0.11) ${48 + tilt.y * 8}%, rgba(255,255,255,0) 65%)`
                              : 'linear-gradient(135deg, rgba(255,255,255,0) 35%, rgba(255,255,255,0) 65%)',
                          }}
                        />
                      </div>
                      <div className="player-art-back">
                        <LyricsBackFace
                          track={lyricsTrack}
                          isVisible={lyricsMode}

                          isDark={isDark}
                          lang={lang}
                          expandedLines={expandedLines}
                          explainStates={explainStates}
                          onToggleLyricExplain={handleToggleLyricExplain}
                          aiActive={!!(aiStatus && aiStatus.aiActive)}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Glassy play/pause indicator — appears on cover click, fades
                  out over ~1200ms. Outside tilt/flip so it stays upright.
                  The icon shape itself is the glass surface (CSS mask + backdrop-
                  filter); no surrounding circle. */}
              {playFeedback && (
                <div key={`fb-${playFeedbackKey}`} className="player-art-feedback">
                  <div className={`player-art-feedback-icon player-art-feedback-icon--${playFeedback}`} />
                </div>
              )}
            </div>

            {!isMobile && (
            <button
              type="button" className="player-side-btn"
              onClick={nextTrack} disabled={currentIndex >= playlist.length - 1}
              title={lang==='ru'?'Следующий':'Next'}
              aria-label={lang==='ru'?'Следующий':'Next'}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6"/>
              </svg>
            </button>
            )}

            {/* Mobile: compact prev/next arrows overlay the row's side gutters
                (the near-full-width cover leaves no room for in-flow buttons). */}
            {isMobile && currentTrack && (
              <>
                <button
                  type="button" className="player-side-btn player-side-btn--flank player-side-btn--flank-left"
                  onClick={prevTrack} disabled={currentIndex <= 0}
                  aria-label={lang==='ru'?'Предыдущий':'Previous'}
                >
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="15 18 9 12 15 6"/>
                  </svg>
                </button>
                <button
                  type="button" className="player-side-btn player-side-btn--flank player-side-btn--flank-right"
                  onClick={nextTrack} disabled={currentIndex >= playlist.length - 1}
                  aria-label={lang==='ru'?'Следующий':'Next'}
                >
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="9 18 15 12 9 6"/>
                  </svg>
                </button>
              </>
            )}
          </div>

          {/* Controls + meta column, constrained to cover width. The cover-row
              above is the only element allowed to spill outside this clamp. */}
          <div ref={playerMetaRef} style={{
            width:'100%',
            maxWidth: coverPx ? `min(${Math.round(coverPx * 1.35)}px, 100%)` : 'min(clamp(300px, 80vh, 860px), 100%)',
            display:'flex', flexDirection:'column',
            gap:'clamp(10px, 1.6vh, 18px)',
            minHeight:0,
          }}>

            {/* Title centered — action icons moved below seek bar */}
            {currentTrack ? (
              <div style={{ textAlign:'center', padding:'0 8px' }}>
                <div style={{
                  fontSize:'clamp(20px, 2.6vh, 32px)', fontWeight:'700', color:pText,
                  letterSpacing:'-0.02em', lineHeight:'1.15',
                  whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                }}>
                  {currentTrack.title}
                </div>
                <div
                  style={{
                    fontSize:'clamp(14px, 1.75vh, 20px)', color:pTextMuted, marginTop:'6px', fontWeight:'400',
                    display: 'inline-block',
                  }}
                >
                  <ArtistCredit track={currentTrack} navigateToArtist={navigateToArtist} lang={lang} color={pTextMuted} />
                </div>
                {currentTrack.album && (
                  <div style={{ fontSize:'clamp(12px, 1.4vh, 16px)', color:pTextSubtle, marginTop:'3px' }}>
                    {currentTrack.album}{currentTrack.year ? ` · ${currentTrack.year}` : ''}
                  </div>
                )}
                <VibeLine
                  trackId={currentTrack?.track_id}
                  lang={lang}
                  isDark={isDark}
                />
              </div>
            ) : (
              <div style={{ textAlign:'center', padding:'30px 0' }}>
                <div style={{ fontSize:'14px', color:pTextSubtle, fontFamily:"'JetBrains Mono', monospace", letterSpacing:'0.15em' }}>
                  {lang==='ru'?'ВЫБЕРИТЕ ТРЕК':'SELECT A TRACK'}
                </div>
              </div>
            )}

            {/* Progress bar */}
            <div className="player-progress-row">
              <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
                <span style={{ color:pTextSubtle, fontFamily:"'JetBrains Mono', monospace", minWidth:'38px', textAlign:'right', fontSize:'11px' }}>
                  {fmt(currentTime)}
                </span>
                <input
                  type="range" min={0} max={duration || 0} step={0.5}
                  value={currentTime}
                  onChange={e => seek(Number(e.target.value))}
                  className="player-progress"
                  style={{
                    flex:1, height:'6px', borderRadius:'3px',
                    background: `linear-gradient(to right, ${pAccent} ${progressPct}%, ${pUnfilled} ${progressPct}%)`,
                  }}
                />
                <span style={{ color:pTextSubtle, fontFamily:"'JetBrains Mono', monospace", minWidth:'38px', fontSize:'11px' }}>
                  {fmt(duration)}
                </span>
              </div>
            </div>

            {/* Action icons — like / dislike / lyrics / ask AI.
                (No transport row on phones: play/pause = cover tap, prev/next =
                the flank arrows beside the cover + swipe.) */}
            <div className="player-actions-row" style={{ display:'flex', justifyContent:'center', gap:4 }}>
              <button
                type="button" className="player-icon-btn"
                onClick={() => sendTaste('fire')}
                disabled={!currentTrack?.track_id}
                title={lang === 'ru' ? 'Огонёк — больше такого в волне' : 'Fire — more like this in the wave'}
                aria-label={lang === 'ru' ? 'Огонёк' : 'Fire'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2"
                  strokeLinecap="round" strokeLinejoin="round">
                  <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
                </svg>
                {burstFor === 'fire' && <span className="player-icon-burst player-icon-burst--fire" />}
              </button>
              <button
                type="button" className="player-icon-btn"
                onClick={() => sendTaste('water')}
                disabled={!currentTrack?.track_id}
                title={lang === 'ru' ? 'Вода — остудить волну' : 'Water — cool the wave'}
                aria-label={lang === 'ru' ? 'Вода' : 'Water'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2"
                  strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>
                </svg>
                {burstFor === 'water' && <span className="player-icon-burst player-icon-burst--water" />}
              </button>
              <button
                type="button" className="player-icon-btn"
                onClick={(e) => onAddToPlaylist && onAddToPlaylist(currentTrack?.track_id, e.currentTarget)}
                disabled={!currentTrack?.track_id}
                title={lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
              <button
                type="button" className="player-icon-btn"
                data-active={lyricsMode ? 'lyrics' : ''}
                aria-pressed={lyricsMode}
                onClick={onToggleLyrics}
                title={lang === 'ru' ? 'Текст песни' : 'Lyrics'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="4" y1="7"  x2="20" y2="7"/>
                  <line x1="4" y1="12" x2="20" y2="12"/>
                  <line x1="4" y1="17" x2="14" y2="17"/>
                </svg>
              </button>
              {aiStatus?.aiActive && (
                <button
                  type="button" className="player-icon-btn"
                  onClick={() => setDrawerOpen(true)}
                  title={lang === 'ru' ? 'Спросить AI' : 'Ask AI'}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2.5l1.6 4.4L18 8.5l-4.4 1.6L12 14.5l-1.6-4.4L6 8.5l4.4-1.6L12 2.5zM18.5 13l.95 2.6L22 16.5l-2.55.9L18.5 20l-.95-2.6L15 16.5l2.55-.9L18.5 13zM5.5 13l.95 2.6L9 16.5l-2.55.9L5.5 20l-.95-2.6L2 16.5l2.55-.9L5.5 13z"/>
                  </svg>
                </button>
              )}
              <VolumeControl volume={volume} onChange={setVolume} isDark={isDark} lang={lang} />
              {/* Shuffle — hidden in stream mode: the wave picks the next track
                  itself, so a hand-shuffled order would be a lie there. */}
              {!streamActive && (
                <button
                  type="button" className="player-icon-btn"
                  data-active={shuffleOn ? 'shuffle' : ''}
                  aria-pressed={!!shuffleOn}
                  onClick={handleShuffleClick}
                  disabled={playlist.length === 0}
                  title={lang === 'ru'
                    ? (shuffleOn ? 'Выключить перемешивание — вернуть порядок' : 'Перемешать очередь')
                    : (shuffleOn ? 'Turn shuffle off — restore order' : 'Shuffle the queue')}
                  aria-label={lang === 'ru' ? 'Перемешать очередь' : 'Shuffle the queue'}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 3 21 3 21 8" />
                    <line x1="4" y1="20" x2="21" y2="3" />
                    <polyline points="21 16 21 21 16 21" />
                    <line x1="15" y1="15" x2="21" y2="21" />
                    <line x1="4" y1="4" x2="9" y2="9" />
                  </svg>
                </button>
              )}
            </div>

            {/* Lossless badge — shown for FLAC always, M4A when bitrate > 600 kbps */}
            {currentTrack && (() => {
              const fp = (currentTrack.file_path || '').toLowerCase();
              const isFlac = fp.endsWith('.flac');
              const isLosslessM4a = fp.endsWith('.m4a') && (currentTrack.bitrate_kbps || 0) > 600;
              if (!isFlac && !isLosslessM4a) return null;
              // Paint the black PNG icon to a flat color via CSS filter.
              // brightness(0) normalises all pixels to pure black (removing
              // any anti-alias grey), invert(1) flips to pure white.
              // opacity() then sets the final perceived lightness.
              // Dark mode:  white at 65% → soft light grey on dark background.
              // Light mode: pure black at 50% → medium grey on light background.
              const iconFilter = isDark
                ? 'brightness(0) invert(1) opacity(0.65)'
                : 'brightness(0) opacity(0.5)';
              const losslessBase = coverPx ? Math.max(80, Math.min(150, Math.round(coverPx * 0.26))) : 128;
              // Phones: the badge read too small next to a near-full-width cover.
              const losslessW = isMobile ? Math.round(losslessBase * 1.4) : losslessBase;
              // Badge rendered height ≈ width × 64/215. Size the hint dot to
              // ~64% of that so the "i" reads as a balanced companion to the mark.
              const hintSize = Math.max(16, Math.round(losslessW * 0.19));
              const losslessLabel = lang === 'ru'
                ? <span><strong>Lossless</strong> — звук без потерь: трек играет в точности как студийный файл, без сжатия и без потери деталей.</span>
                : <span><strong>Lossless</strong> — bit-perfect audio: this track plays exactly like the studio file, with no compression and no lost detail.</span>;
              return (
                <div style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:9, padding:'0 20px' }}>
                  <img
                    src={LOSSLESS_LOGO_SRC}
                    alt="Lossless"
                    title={isFlac ? 'FLAC · Lossless' : `ALAC · ${currentTrack.bitrate_kbps} kbps`}
                    style={{
                      height: 'auto',
                      width: losslessW,
                      filter: iconFilter,
                      userSelect: 'none',
                      pointerEvents: 'none',
                    }}
                  />
                  <HintBadge size={hintSize} label={losslessLabel}
                    ariaLabel={lang === 'ru' ? 'Что такое Lossless' : 'What Lossless means'} />
                </div>
              );
            })()}

          </div>
        </div>

        {/* ════════════════ RIGHT: Facts + Queue ════════════════
            Capped at 40% width, but never wider than 620px — on large
            monitors (27"+) the rail doesn't need to keep growing, so the
            freed space goes to the cover row and its spectrum strips.
            Top padding matches the left column's, so the
            "О ПЕСНЕ" header aligns with the cover's vertical start. When the
            queue is empty, the column centers its content vertically so the
            facts+queue cards don't sit awkwardly at the top with empty space
            beneath. When the queue has items, it grows + scrolls normally. */}
        <div style={{
          flex: isMobile ? '0 0 auto' : '0 0 min(40%, 620px)',
          maxWidth: isMobile ? '100%' : 'min(40%, 620px)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: playlist.length === 0 ? 'center' : 'flex-start',
          gap: isMobile ? 12 : 18,
          overflow: isMobile ? 'visible' : 'hidden',
          position: 'relative',
          zIndex: 1,
          padding: isMobile ? '4px 2px 8px' : 'clamp(24px, 5vh, 56px) 4px 16px',
          minHeight: 0,
        }}>

          {/* FACTS rail — player variant from T4 */}
          <FactsRail
            trackId={currentTrack?.track_id}

            isDark={isDark}
            lang={lang}
            variant="player"
          />

          {/* Separator between facts and queue (desktop only) */}
          {!isMobile && <div style={{ borderTop: `1px solid ${isDark ? '#2a2a32' : 'rgba(22,22,32,0.08)'}` }} />}

          {/* Mobile: facts stay above; this button opens the queue full-screen */}
          {isMobile && (
            <button type="button" onClick={() => setQueueOpen(true)}
              style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:10, width:'100%', minHeight:48, padding:'0 16px', borderRadius:14, cursor:'pointer', color:pText,
                background: isDark ? 'rgba(255,255,255,.05)' : 'rgba(22,22,32,.04)',
                border:`1px solid ${isDark ? 'rgba(255,255,255,.08)' : 'rgba(22,22,32,.10)'}` }}>
              <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontSize:11, letterSpacing:'0.16em' }}>{lang==='ru'?'ОЧЕРЕДЬ':'QUEUE'} · {playlist.length}</span>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
            </button>
          )}

          {/* Queue + Similarity: inline on desktop (display:contents = zero layout
              change); a full-screen slide-up drawer on mobile (translateY toggle). */}
          <div style={isMobile ? {
            position:'fixed', inset:0, zIndex:80,
            display:'flex', flexDirection:'column', gap:14,
            padding:'10px 14px calc(14px + env(safe-area-inset-bottom, 0px))',
            background: isDark ? 'rgba(14,14,20,0.98)' : 'rgba(248,247,252,0.98)',
            backdropFilter:'blur(20px)', WebkitBackdropFilter:'blur(20px)',
            transform: queueOpen ? 'translateY(0)' : 'translateY(100%)',
            transition:'transform .3s cubic-bezier(.22,.9,.3,1)', overflow:'hidden',
          } : { display:'contents' }}>
            {isMobile && (
              <div style={{ display:'flex', justifyContent:'flex-end', flexShrink:0 }}>
                <button type="button" onClick={() => setQueueOpen(false)} aria-label={lang==='ru'?'Закрыть':'Close'}
                  style={{ width:40, height:40, borderRadius:'50%', display:'grid', placeItems:'center', background:'transparent', border:0, cursor:'pointer', color:pText }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
              </div>
            )}

          {/* Queue + Chat overlay area — anchors AIChatDrawer's absolute
              positioning so the drawer overlays ONLY this region, leaving
              FactsRail above untouched. overflow:hidden clips the drawer
              during its slide-up animation. */}
          <div className="queue-chat-area" style={{
            flex: playlist.length === 0 ? '0 0 auto' : 1,
            position: 'relative',
            overflow: 'hidden',
            // Floor at ~3 queue rows + header so the queue never collapses below
            // 3 visible tracks on short viewports (the SimilarityRail below takes
            // its natural height, leaving the queue the rest).
            minHeight: playlist.length === 0 ? 0 : 236,
            display: 'flex',
            flexDirection: 'column',
          }}>
          {/* QUEUE block (inner) — fills the wrapper. */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
          }}>
            {/* «Подстроились под твой вайб» — only in stream mode and only when
                the server named distinguishable contributors (fire/replay). */}
            {streamActive && streamAdapt && streamAdapt.active && (streamAdapt.tracks || []).length > 0 && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 9, flexShrink: 0,
                marginBottom: 10, padding: '7px 12px', borderRadius: 12,
                background: isDark ? 'rgba(124,92,255,.10)' : 'rgba(124,92,255,.07)',
                border: `1px solid ${isDark ? 'rgba(124,92,255,.26)' : 'rgba(124,92,255,.20)'}`,
              }}>
                <span aria-hidden="true" style={{ fontSize: 13, lineHeight: 1 }}>✨</span>
                <span style={{ fontSize: 12.5, fontWeight: 600, color: pText, minWidth: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {lang === 'ru' ? 'Подстроились под твой вайб' : 'Tuned to your vibe'}
                </span>
                <span style={{ display: 'flex', gap: 5, marginLeft: 'auto', flexShrink: 0 }}>
                  {(streamAdapt.tracks || []).slice(0, 2).map((t, j) => (
                    <LazyCover key={t.track_id || j} url={homeCoverUrl(t.cover_art_path)}
                               style={{ width: 24, height: 24, borderRadius: 6 }}
                               fallback="linear-gradient(135deg,#7c5cff,#b06bff)" />
                  ))}
                </span>
              </div>
            )}
            <div style={{
              fontSize: 10,
              letterSpacing: '0.20em',
              color: isDark ? '#888' : '#5a5a66',
              fontFamily: "'JetBrains Mono', ui-monospace, monospace",
              marginBottom: 10,
              flexShrink: 0,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <span>{lang === 'ru' ? 'ОЧЕРЕДЬ' : 'QUEUE'} · {playlist.length} {lang === 'ru' ? 'ТРЕКОВ' : 'TRACKS'}</span>
              <HintBadge
                size={15}
                label={lang === 'ru'
                  ? <span>Треки в очереди можно двигать — <strong>тяните за ⋮⋮ сбоку строки</strong> (или зажмите саму строку). А ещё тут видно следующие треки в рекомендациях: смотрите, как они меняются от ваших действий.</span>
                  : <span>Reorder the queue — <strong>drag the ⋮⋮ handle</strong> at the side of a row (or press and hold the row). You can also see what's coming up in recommendations, and how it shifts as you listen.</span>}
                placement="down"
                ariaLabel={lang === 'ru' ? 'Как работает очередь' : 'How the queue works'}
              />
            </div>
            <div ref={queueScrollRef} className="player-scroll" style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
              {playlist.length === 0 ? (
                <div style={{
                  textAlign: 'center',
                  padding: '30px 20px',
                  color: isDark ? '#888' : '#5a5a66',
                  fontSize: 13,
                }}>
                  {lang === 'ru' ? 'Очередь пуста' : 'Queue empty'}
                </div>
              ) : (() => {
                const firstAutoplayIdx = playlist.findIndex(h => h && h._autoplay);
                return playlist.map((hit, i) => {
                  const t = hit.track || {};
                  const active = i === currentIndex;
                  const rp = queueReorder.getRowProps(i);
                  const dropped = !!droppedId && t.track_id === droppedId;
                  return (
                    <Fragment key={t.track_id || i}>
                      {/* Shuffle intermixes hand-picked and autoplay tracks, so
                          the divider would lie — hide it while shuffled. */}
                      {i === firstAutoplayIdx && firstAutoplayIdx !== -1 && !shuffleOn && (
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          margin: '6px 0 4px',
                          color: isDark ? '#666' : '#8a8275',
                          fontSize: 9,
                          fontFamily: "'JetBrains Mono', ui-monospace, monospace",
                          letterSpacing: '0.18em',
                        }}>
                          <span style={{ flex: 1, height: 1, background: isDark ? '#2a2a32' : 'rgba(22,22,32,0.10)' }} />
                          <span>{lang === 'ru' ? 'АВТОПОДБОР' : 'AUTOPLAY'}</span>
                          <span style={{ flex: 1, height: 1, background: isDark ? '#2a2a32' : 'rgba(22,22,32,0.10)' }} />
                        </div>
                      )}
                      <div
                        data-qrow={rp['data-qrow']}
                        data-tid={t.track_id || ''}
                        className={`player-playlist-item${active ? ' player-active' : ''}${rp.className ? ' ' + rp.className : ''}${dropped ? ' q-dropped' : ''}`}
                        onPointerDown={rp.onPointerDown}
                        onClick={() => {
                          if (queueReorder.consumeClick()) return;   // swallow the click that trails a drag
                          markPlaybackInteracted(audio?.audioRef?.current); playTrackAt(i);
                        }}
                        onMouseEnter={() => setHoveredQueueIdx(i)}
                        onMouseLeave={() => setHoveredQueueIdx(-1)}
                        style={{
                          display:'flex', alignItems:'center', gap:'14px',
                          padding:'10px 14px', cursor:'pointer', marginBottom:'2px',
                          position: 'relative',
                          opacity: hit._autoplay && !active ? 0.75 : 1,
                          ...rp.style,
                        }}>
                        {/* Index or equalizer */}
                        <div style={{ width:'22px', textAlign:'center', flexShrink:0 }}>
                          {active && isPlaying ? (
                            <div style={{ display:'inline-flex', alignItems:'flex-end', height:'16px', gap:'0' }}>
                              <span className="player-eq-bar" style={{ height:'4px' }} />
                              <span className="player-eq-bar" style={{ height:'8px' }} />
                              <span className="player-eq-bar" style={{ height:'6px' }} />
                            </div>
                          ) : (
                            <span style={{
                              fontSize:'12px', fontFamily:"'JetBrains Mono', monospace",
                              color: active ? pAccent : pTextSubtle,
                            }}>{i + 1}</span>
                          )}
                        </div>

                        <AlbumCover title={t.title} artist={t.artist} size={46} isDark={isDark} coverPath={t.cover_art_path} radius={10} />

                        <div style={{ flex:1, minWidth:0 }}>
                          <div style={{
                            fontSize:'14px', fontWeight: active ? '600' : '400',
                            color: active ? pText : pTextMuted,
                            whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                            letterSpacing:'-0.01em',
                          }}>{t.title || '—'}</div>
                          <div style={{ fontSize:'13px', color:pTextSubtle, marginTop:'1px' }}>{t.artist || '—'}</div>
                        </div>

                        {t.genre && (
                          <span style={{
                            fontSize:'11px', color:pTextSubtle, fontFamily:"'JetBrains Mono', monospace",
                            padding:'3px 8px', borderRadius:'9999px',
                            background: pPillBg,
                            border: `1px solid ${pBorderSubtle}`,
                          }}>{t.genre}</span>
                        )}
                        {i === hoveredQueueIdx && (
                          <button
                            type="button" className="player-icon-btn"
                            onClick={(e) => { e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(t.track_id, e.currentTarget); }}
                            title={lang === 'ru' ? 'Добавить в плейлист' : 'Add to playlist'}
                            style={{ width: 30, height: 30 }}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                              <line x1="12" y1="5" x2="12" y2="19" />
                              <line x1="5" y1="12" x2="19" y2="12" />
                            </svg>
                          </button>
                        )}
                        {i === hoveredQueueIdx && hit.score_breakdown && (
                          <PlayerScoreBars breakdown={hit.score_breakdown} isDark={isDark} />
                        )}
                        {rp.className.split(' ').indexOf('q-draggable') >= 0 && (
                          <span
                            className="q-grip"
                            title={lang === 'ru' ? 'Перетащите, чтобы переставить' : 'Drag to reorder'}
                            onClick={(e) => e.stopPropagation()}
                            {...queueReorder.getHandleProps(i)}
                          >⋮⋮</span>
                        )}
                      </div>
                    </Fragment>
                  );
                });
              })()}
            </div>
          </div>

          {/* AI Chat drawer (desktop) — overlays ONLY the queue-chat-area
              wrapper above. Slides up from bottom; FactsRail stays visible.
              Pass lyricsTrack (not raw currentTrack) so the chat's track
              context carries the on-demand fetched lyrics for non-search
              sources — stream/autoplay/home strip lyrics from the payload,
              so currentTrack.lyrics is empty there and the chat would
              otherwise send no lyrics to the model.
              On phones the drawer lives OUTSIDE this container (below) —
              anchored in here it was trapped behind the queue's translateY,
              so opening the chat forced the queue open with it. */}
          {!isMobile && (
            <AIChatDrawer
              isOpen={drawerOpen}
              onClose={() => setDrawerOpen(false)}
              track={lyricsTrack || currentTrack}
              lang={lang}
              isDark={isDark}

              showToast={showToast}
            />
          )}
          </div>

          <SimilarityRail
            trackId={currentTrack?.track_id}
            lang={lang}
            isDark={isDark}
            drawerOpen={drawerOpen}
            onQueueNext={onQueueNext}
          />
          </div>

          {/* AI Chat (mobile) — its own fixed overlay above the player, so the
              chat opens WITHOUT dragging the queue drawer up behind it. The
              scrim closes on tap; the glass card slides via the drawer's own
              translateY. */}
          {isMobile && (
            <div
              onClick={() => setDrawerOpen(false)}
              style={{
                position:'fixed', inset:0, zIndex:90,
                background:'rgba(8,8,14,0.38)',
                opacity: drawerOpen ? 1 : 0,
                pointerEvents: drawerOpen ? 'auto' : 'none',
                transition:'opacity 240ms ease',
              }}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                style={{
                  position:'absolute', left:8, right:8,
                  top:'max(12px, env(safe-area-inset-top, 0px))',
                  bottom:'calc(10px + env(safe-area-inset-bottom, 0px))',
                }}
              >
                <AIChatDrawer
                  isOpen={drawerOpen}
                  onClose={() => setDrawerOpen(false)}
                  track={lyricsTrack || currentTrack}
                  lang={lang}
                  isDark={isDark}
                  showToast={showToast}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Audio is managed at App level via useAudioPlayer hook */}
    </div>
  );
}

// ─── ARTIST ATLAS SECTION (v2 — immersive hero + glass dock) ─────────────────
// Single-scroll page: hero (AudioDB artist photo) → liquid-glass anchor dock →
// bio → facts shelf → albums timeline. Tracks never spill onto the page —
// album covers open the shared AlbumModal gatefold (same as Library).

const atlasImgUrl = (p) => (p ? (p.startsWith('http') ? p : `${API}${p}`) : null);

// True artist cutouts are PNGs with transparent margins, but AudioDB
// sometimes ships a plain rectangular photo in the cutout slot. Probe the
// alpha channel on a tiny canvas: null = probing, true = has transparency
// (render as a free-standing cutout), false = solid photo (render with the
// soft-mask treatment so its edges dissolve into the blurred backdrop).
function useImageHasAlpha(url) {
  const [hasAlpha, setHasAlpha] = useState(null);
  useEffect(() => {
    setHasAlpha(null);
    if (!url) return;
    if (/\.jpe?g(\?|#|$)/i.test(url)) { setHasAlpha(false); return; }  // JPEG has no alpha channel
    let dead = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (dead) return;
      try {
        const cv = document.createElement('canvas');
        cv.width = cv.height = 24;
        const ctx = cv.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, 24, 24);
        const px = ctx.getImageData(0, 0, 24, 24).data;
        let transparent = false;
        for (let i = 3; i < px.length; i += 4) {
          if (px[i] < 250) { transparent = true; break; }
        }
        setHasAlpha(transparent);
      } catch (e) {
        setHasAlpha(false);  // tainted canvas / decode failure → treat as photo
      }
    };
    img.onerror = () => { if (!dead) setHasAlpha(false); };
    img.src = url;
    return () => { dead = true; };
  }, [url]);
  return hasAlpha;
}

// "GB" → 🇬🇧 via regional-indicator pair; null for anything not 2 ASCII letters.
const atlasFlag = (cc) => {
  if (!cc || !/^[A-Za-z]{2}$/.test(cc)) return null;
  return String.fromCodePoint(...[...cc.toUpperCase()].map(ch => 0x1F1E6 + ch.charCodeAt(0) - 65));
};

const atlasRuPlural = (n, one, few, many) => {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
};

// Same hue-from-text family as the AlbumCover / AlbumModal fallbacks.
const atlasHue = (a, b) => ((((a || '?').charCodeAt(0) || 65) * 37 + ((b || a || '?').charCodeAt(0) || 65) * 17) % 360);

const atlasGlass = (isDark) => ({
  background: isDark ? 'rgba(21,18,33,0.55)' : 'rgba(255,255,255,0.62)',
  backdropFilter: 'blur(18px) saturate(1.35)',
  WebkitBackdropFilter: 'blur(18px) saturate(1.35)',
  border: `1px solid ${isDark ? 'rgba(255,255,255,0.13)' : 'rgba(255,255,255,0.7)'}`,
  boxShadow: isDark
    ? 'inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.25), 0 14px 38px rgba(0,0,0,0.42)'
    : 'inset 0 1px 0 rgba(255,255,255,0.95), inset 0 -1px 0 rgba(0,0,0,0.05), 0 14px 30px rgba(40,30,70,0.14)',
});

function AtlasAnchorPills({ sections, activeId, onAnchor, isDark, compact }) {
  const c = useColors(isDark);
  return (
    <div style={{ display:'flex', gap:6, minWidth:0 }}>
      {sections.map(s => {
        const active = activeId === s.id;
        return (
          <button key={s.id} onClick={() => onAnchor(s.id)} className="mono"
            style={{
              padding: compact ? '4px 11px' : '6px 14px', borderRadius:99,
              fontSize: compact ? 9 : 10, letterSpacing:'0.18em', cursor:'pointer',
              border:`1px solid ${active ? 'oklch(60% 0.18 270 / 0.55)' : (isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)')}`,
              background: active ? 'oklch(60% 0.18 270 / 0.16)' : 'transparent',
              color: active ? c.text : c.textMuted,
              transition:'all 180ms ease', whiteSpace:'nowrap',
            }}>{s.label}</button>
        );
      })}
    </div>
  );
}

// Writes the cursor position (px) as CSS vars for the .atlas-glare highlight —
// the liquid-glass glint that follows the pointer on the docks and play button.
const atlasGlareMove = (e) => {
  const el = e.currentTarget;
  const r = el.getBoundingClientRect();
  el.style.setProperty('--gx', `${e.clientX - r.left}px`);
  el.style.setProperty('--gy', `${e.clientY - r.top}px`);
};

function AtlasSpinButton({ onSpin, lang, compact }) {
  return (
    <button onClick={onSpin} className="atlas-glare atlas-play-btn" onMouseMove={atlasGlareMove}
      title={lang==='ru'?'Включить артиста':'Play artist'}
      style={{
        padding: compact ? '7px 13px' : '9px 18px', borderRadius:99, flexShrink:0,
        background:'linear-gradient(135deg, oklch(67% 0.18 270), oklch(52% 0.22 285))',
        color:'#fff', fontSize: compact ? 11 : 13, fontWeight:600, letterSpacing:'0.04em',
        border:'1px solid oklch(50% 0.2 275 / 0.4)', cursor:'pointer',
        // box-shadow lives in .atlas-play-btn (inline would beat the :hover lift)
      }}>
      ▶{compact ? '' : ` ${lang==='ru'?'Включить артиста':'Play artist'}`}
    </button>
  );
}

function AtlasHero({ data, isDark, lang, onNav, heroRef, playingHere }) {
  const c = useColors(isDark);
  const hue = atlasHue(data.name, data.name.slice(1));
  // Only a real artist photo counts as a backdrop. Album covers are NO LONGER
  // promoted to a full-bleed image — they feed the aurora gradient instead (below).
  const backdrop = atlasImgUrl(data.thumb_path);
  const cutout = atlasImgUrl(data.cutout_path);
  const flag = atlasFlag(data.country_code);

  // Identity line: genre + where the artist is from.
  const originLine = [
    data.genre,
    flag
      ? `${flag} ${(data.country || data.country_code || '').toUpperCase()}`
      : (data.country ? data.country.toUpperCase() : null),
  ].filter(Boolean).join(' · ');

  // Library-derived lines: which decades this listener owns + the catalogue size.
  const decadeLabel = lang==='ru' ? 'Десятилетия в твоей библиотеке' : 'Decades in your library';
  const countsLine = [
    data.album_count ? `${data.album_count} ${lang==='ru' ? atlasRuPlural(data.album_count,'АЛЬБОМ','АЛЬБОМА','АЛЬБОМОВ') : (data.album_count===1?'ALBUM':'ALBUMS')}` : null,
    data.track_count ? `${data.track_count} ${lang==='ru' ? atlasRuPlural(data.track_count,'ТРЕК','ТРЕКА','ТРЕКОВ') : (data.track_count===1?'TRACK':'TRACKS')}` : null,
  ].filter(Boolean).join(' · ');

  // Three exclusive hero treatments:
  //  • cutout — transparent PNG of the artist, raised into the upper band (no backdrop)
  //  • photo  — a real artist photo exists: show it un-blurred, centred, fading to facts
  //  • aurora — no artist photo: a drifting gradient painted from album-cover colours
  //             (or a name-derived hue when there are no covers to sample)
  const mode = cutout ? 'cutout' : (backdrop ? 'photo' : 'aurora');

  // Album covers → aurora palette (sampled only in aurora mode; [] is a hook no-op).
  // One album yields 2-3 colours; several yield one dominant each (hue-deduped).
  const coverUrls = mode === 'aurora'
    ? Array.from(new Set((data.albums || [])
        .map(a => thumbCoverUrl(atlasImgUrl(a.cover_art_path)))
        .filter(Boolean)))
    : [];
  const { colors: sampledColors } = useCoverPalette(coverUrls);
  const auroraCss = auroraStops(padPalette(sampledColors, hue, 3), isDark);
  const isMobile = useIsMobile();

  // Cursor parallax — same refraction move as the home orb: --hx/--hy ∈
  // [-0.5, 0.5] are written straight on the hero node (no state → no
  // re-renders at mousemove rate); .atlas-plx-* classes pick their depth.
  const heroMove = (e) => {
    const el = heroRef.current;
    if (!el || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--hx', (((e.clientX - r.left) / r.width) - 0.5).toFixed(3));
    el.style.setProperty('--hy', (((e.clientY - r.top) / r.height) - 0.5).toFixed(3));
  };
  const heroLeave = () => {
    const el = heroRef.current;
    if (el) { el.style.setProperty('--hx', '0'); el.style.setProperty('--hy', '0'); }
  };

  return (
    <div ref={heroRef} onMouseMove={heroMove} onMouseLeave={heroLeave}
      style={{ position:'relative', height: mode==='aurora' ? (isMobile ? 'clamp(190px, 26vh, 240px)' : 'clamp(250px, 32vh, 320px)') : (isMobile ? 'clamp(280px, 42vh, 360px)' : 'clamp(420px, 56vh, 600px)'), overflow:'visible' }}>
      {/* Light field: every colour layer lives here, not in the hero. The field
          runs ~58% past the hero's bottom and dissolves with one long mask
          (.atlas-field), so no mode has a horizontal colour cutoff — the dock
          (z6) and the content column (z2) render over its fading tail. */}
      <div className="atlas-field" aria-hidden="true">
        {mode === 'photo' ? (
          // blurred backdrop — fills the whole field so the photo's glow melts
          // into the content instead of stopping at the hero's edge
          <img src={backdrop} alt="" className="atlas-plx-far" style={{
            position:'absolute', top:'-4%', left:'-5%', width:'110%', height:'108%',
            objectFit:'cover', objectPosition:'50% 40%',
            filter:`blur(28px) ${isDark ? 'brightness(0.7) saturate(1.1)' : 'brightness(0.85) saturate(1.05)'}`,
          }} />
        ) : mode === 'cutout' ? (
          <div className="atlas-plx-far" style={{ position:'absolute', inset:0 }}>
            {/* near-neutral deep base — spans the full field; the mask, not the
                gradient, decides where the colour ends */}
            <div style={{
              position:'absolute', inset:0,
              background: isDark
                ? `radial-gradient(ellipse 92% 71% at 73% 32%, oklch(21% 0.015 ${hue}) 0%, oklch(13% 0.01 ${hue}) 55%, ${c.bg} 100%)`
                : `radial-gradient(ellipse 92% 71% at 73% 32%, oklch(96% 0.012 ${hue}) 0%, oklch(92% 0.01 ${hue}) 55%, ${c.bg} 100%)`,
            }} />
            {/* Light-burst inside the centred box (63% ≈ the hero's share of the
                field) so --burst-x keeps tracking the cutout; rays/flare spill
                past the box and get clipped by the field, not the hero. */}
            <div style={{ position:'relative', maxWidth:1120, margin:'0 auto', height:'63%', padding:'0 32px' }}>
              <div className="atlas-burst" style={{
                '--burst-x': '80%',
                '--burst-hue': hue,
                '--ray-color': isDark ? 'rgba(255,255,255,0.14)' : 'rgba(120,92,220,0.18)',
                '--flare-inner': isDark ? 'rgba(255,250,240,0.68)' : 'rgba(255,255,255,0.85)',
                '--flare-mid': isDark ? `oklch(68% 0.14 ${hue} / 0.42)` : `oklch(80% 0.12 ${hue} / 0.45)`,
                '--spark': isDark ? 'rgba(255,238,205,0.95)' : `oklch(58% 0.16 ${hue} / 0.7)`,
              }}>
                <div className="atlas-particles atlas-plx-dust" />
              </div>
            </div>
          </div>
        ) : (
          // aurora — base gradient + drifting album-colour blobs span the whole
          // field, so the blobs sink below the dock before the mask melts them
          <div className="atlas-plx-far" style={{ position:'absolute', inset:0 }}>
            <div style={{
              position:'absolute', inset:0,
              background:`linear-gradient(150deg, ${auroraCss.join(', ')})`,
              transition:'background 600ms ease',
            }} />
            <div className="atlas-aurora">
              {[0,1,2,3].map(i => (
                <div key={i} className={`aurora-blob aurora-blob-${i}`}
                     style={{ background: auroraCss[i % auroraCss.length] }} />
              ))}
            </div>
          </div>
        )}
        {/* depth accent */}
        <div style={{
          position:'absolute', inset:0,
          background:`radial-gradient(ellipse 90% 130% at 18% 0%, ${isDark?'rgba(124,91,255,0.16)':'rgba(124,91,255,0.10)'}, transparent 55%)`,
        }} />
        <div className="atlas-grain" />
        {/* Soft left fade toward the floating nav rail — spans the whole field
            so the rail floats above the bleed, too */}
        <div style={{
          position:'absolute', inset:0,
          background:`linear-gradient(90deg, ${c.bg} 0px, ${c.bg}00 120px)`,
        }} />
      </div>

      {/* main photo (photo mode) — crisp, height-scaled, 4-edge mask. Clipped by
          its own wrapper (the hero is overflow:visible now) so a wide portrait
          can't leak outside; rides the mid parallax layer. */}
      {mode === 'photo' && (
        <div aria-hidden="true" style={{ position:'absolute', inset:0, overflow:'hidden', pointerEvents:'none' }}>
          <img src={backdrop} alt="" className="atlas-plx-mid" style={{
            position:'absolute', top:0, left:'50%', transform:'translateX(-50%)',
            height:'100%', width:'auto', maxWidth:'none',
            filter: isDark ? 'brightness(0.92) saturate(1.04)' : 'saturate(1.02)',
            WebkitMaskImage:
              'linear-gradient(180deg, transparent 0%, #000 10%, #000 84%, transparent 100%), ' +
              'linear-gradient(90deg, transparent 0%, #000 12%, #000 88%, transparent 100%)',
            WebkitMaskComposite: 'source-in intersect',
            maskImage:
              'linear-gradient(180deg, transparent 0%, #000 10%, #000 84%, transparent 100%), ' +
              'linear-gradient(90deg, transparent 0%, #000 12%, #000 88%, transparent 100%)',
            maskComposite: 'intersect',
          }} />
        </div>
      )}

      {/* top readability veil for the breadcrumb (the old bottom wash to c.bg is
          gone — the field's mask owns the dissolve now) */}
      <div aria-hidden="true" style={{
        position:'absolute', left:0, right:0, top:0, height:'40%', pointerEvents:'none',
        background:`linear-gradient(180deg, ${isDark?'rgba(13,13,16,0.2)':'rgba(242,241,246,0.14)'} 0%, transparent 100%)`,
      }} />

      <div style={{ position:'relative', maxWidth:1120, margin:'0 auto', height:'100%', padding:'0 32px' }}>
        {/* Light-burst (cutout only) — inside the container so --burst-x is measured
            against the same centred box as the figure, keeping the glow on the cutout. */}
        {mode === 'cutout' && (
          <div className="atlas-burst" aria-hidden="true" style={{
            '--burst-x': '80%',
            '--burst-hue': hue,
            '--ray-color': isDark ? 'rgba(255,255,255,0.14)' : 'rgba(120,92,220,0.18)',
            '--flare-inner': isDark ? 'rgba(255,250,240,0.68)' : 'rgba(255,255,255,0.85)',
            '--flare-mid': isDark ? `oklch(68% 0.14 ${hue} / 0.42)` : `oklch(80% 0.12 ${hue} / 0.45)`,
            '--spark': isDark ? 'rgba(255,238,205,0.95)' : `oklch(58% 0.16 ${hue} / 0.7)`,
          }}>
            <div className="atlas-particles" />
          </div>
        )}

        {/* Breadcrumb */}
        <div className="mono" style={{
          position:'absolute', top:20, left:32, zIndex:3, fontSize:10,
          letterSpacing:'0.22em', textTransform:'uppercase',
          color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(20,18,32,0.55)',
          textShadow: isDark ? '0 1px 8px rgba(0,0,0,0.5)' : 'none',
        }}>
          <span style={{ cursor:'pointer' }} onClick={() => onNav?.('library')}>{lang==='ru'?'БИБЛИОТЕКА':'LIBRARY'}</span>
          {' / '}
          <span>{lang==='ru'?'АРТИСТЫ':'ARTISTS'}</span>
          {' / '}
          <span style={{ color: isDark ? '#fff' : '#161620' }}>{data.name.toUpperCase()}</span>
          {/* mini-EQ: this artist is playing right now */}
          {playingHere && (
            <span className="hero-eq" aria-hidden="true" style={{ marginLeft:10, marginRight:0, height:10, verticalAlign:'middle' }}>
              <span /><span /><span /><span /><span />
            </span>
          )}
        </div>

        {/* Cutout figure — cutout mode only. A blurred, brightened echo of the
            same PNG glows behind the figure, and the main image gets a 4-edge
            mask, so AudioDB's hard crop lines dissolve into light instead of
            ending at a rectangle. Both ride the mid parallax layer. */}
        {mode === 'cutout' && (
          <img src={cutout} alt="" aria-hidden="true" className="atlas-cutout-echo atlas-plx-mid" style={{
            position:'absolute', right:36, top:44, bottom:34, maxWidth:'44%',
            objectFit:'contain', objectPosition:'top right', zIndex:1,
            filter:`blur(26px) saturate(1.35) ${isDark ? 'brightness(1.5)' : 'brightness(1.1)'}`,
            opacity: isDark ? 0.5 : 0.42,
            transform:'scale(1.07)',
          }} />
        )}
        {mode === 'cutout' && (
          <img src={cutout} alt="" className="atlas-cutout atlas-plx-mid" style={{
            position:'absolute', right:36, top:44, bottom:34, maxWidth:'44%',
            objectFit:'contain', objectPosition:'top right', zIndex:1,
            // No drop-shadow: it would paint a blurred black silhouette into the
            // cutout's transparent margins, graying the otherwise-clear backdrop.
            // Depth comes from the light-burst and the blurred echo instead.
            WebkitMaskImage:
              'linear-gradient(180deg, transparent 0%, #000 4%, #000 90%, transparent 100%), ' +
              'linear-gradient(90deg, transparent 0%, #000 6%, #000 94%, transparent 100%)',
            WebkitMaskComposite: 'source-in intersect',
            maskImage:
              'linear-gradient(180deg, transparent 0%, #000 4%, #000 90%, transparent 100%), ' +
              'linear-gradient(90deg, transparent 0%, #000 6%, #000 94%, transparent 100%)',
            maskComposite: 'intersect',
          }} />
        )}

        {/* Name + context lines. In cutout mode the name sits on the left,
            vertically centred against the figure; otherwise it anchors bottom-left. */}
        <div className="atlas-plx-near" style={{
          position:'absolute', left:32, right: mode==='cutout' ? '46%' : (mode==='aurora' ? 32 : '44%'), zIndex:2, minWidth:0,
          ...(mode==='cutout' ? { top:'50%', transform:'translateY(-50%)' } : mode==='aurora' ? { top:52 } : { bottom:58 }),
        }}>
          <h1 className="serif" style={{
            fontSize: mode==='cutout' ? 'clamp(46px, 5.6vw, 78px)' : 'clamp(38px, 4.6vw, 60px)',
            fontWeight:300, lineHeight:1.02,
            color:c.text, letterSpacing:'-0.025em', margin:0,
            textShadow: isDark ? '0 2px 26px rgba(0,0,0,0.6)' : '0 2px 22px rgba(255,255,255,0.8)',
          }}>{data.name}</h1>
          {/* Identity — genre · origin (accent). 'Noto Color Emoji' supplies the flag
              glyph Windows' system emoji font lacks, so it stops rendering as bare "US". */}
          {originLine && (
            <div className="mono" style={{
              marginTop:14, fontSize:15, fontWeight:600, letterSpacing:'0.16em', textTransform:'uppercase',
              fontFamily: "'JetBrains Mono', 'Noto Color Emoji', ui-monospace, monospace",
              color: isDark ? 'rgba(228,219,255,0.96)' : 'oklch(34% 0.16 282)',
              textShadow: isDark ? '0 1px 12px rgba(0,0,0,0.6)' : 'none',
              whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
            }}>{originLine}</div>
          )}

          {/* Library-derived block — decades you own + catalogue counts (muted) */}
          {(data.decade_range || countsLine) && (
            <div style={{ marginTop:12, display:'flex', flexDirection:'column', gap:4 }}>
              {data.decade_range && (
                <div className="mono" style={{
                  fontSize:13, letterSpacing:'0.12em',
                  color: isDark ? 'rgba(210,202,228,0.7)' : 'oklch(46% 0.05 282)',
                  textShadow: isDark ? '0 1px 9px rgba(0,0,0,0.5)' : 'none',
                  whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                }}>
                  <span style={{ opacity:0.78 }}>{decadeLabel}</span>
                  {' · '}
                  <span style={{ textTransform:'uppercase' }}>{data.decade_range}</span>
                </div>
              )}
              {countsLine && (
                <div className="mono" style={{
                  fontSize:13, letterSpacing:'0.14em', textTransform:'uppercase',
                  color: isDark ? 'rgba(210,202,228,0.7)' : 'oklch(46% 0.05 282)',
                  textShadow: isDark ? '0 1px 9px rgba(0,0,0,0.5)' : 'none',
                  whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                }}>{countsLine}</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AtlasBio({ bio, isDark, lang }) {
  const c = useColors(isDark);
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const boxRef = useRef(null);
  const COLLAPSED_PX = 178;
  useEffect(() => {
    const el = boxRef.current;
    if (el) setOverflowing(el.scrollHeight > COLLAPSED_PX + 24);
  }, [bio]);
  const clamped = overflowing && !expanded;
  return (
    <div>
      <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:14 }}>
        <span className="mono" style={{ fontSize:10, letterSpacing:'0.24em', color:c.textSubtle, textTransform:'uppercase' }}>
          {lang==='ru'?'БИОГРАФИЯ':'BIOGRAPHY'}
        </span>
        <span className="mono" style={{
          fontSize:8.5, letterSpacing:'0.18em', padding:'2px 7px', borderRadius:8,
          color:c.accentLight, border:`1px solid ${isDark?'rgba(160,130,255,0.4)':'rgba(110,80,220,0.35)'}`,
          background: isDark?'rgba(124,91,255,0.12)':'rgba(124,91,255,0.08)',
        }}>AI</span>
      </div>
      <div ref={boxRef} className="serif" style={{
        maxWidth:720, fontSize:16, lineHeight:1.7, color:c.text, letterSpacing:'-0.005em',
        maxHeight: clamped ? COLLAPSED_PX : 'none', overflow:'hidden',
        WebkitMaskImage: clamped ? 'linear-gradient(180deg, #000 58%, transparent 100%)' : 'none',
        maskImage: clamped ? 'linear-gradient(180deg, #000 58%, transparent 100%)' : 'none',
      }}>
        <MarkdownText text={bio} />
      </div>
      {overflowing && (
        <button onClick={() => setExpanded(e => !e)} style={{
          marginTop:10, background:'transparent', border:'none', cursor:'pointer',
          color:c.accentLight, fontSize:12.5, fontFamily:'inherit', padding:0, letterSpacing:'0.02em',
        }}>
          {expanded ? (lang==='ru'?'свернуть ↑':'collapse ↑') : (lang==='ru'?'читать дальше ↓':'read more ↓')}
        </button>
      )}
    </div>
  );
}

// Stamp tilts cycle through both directions so neighbouring cards never match —
// "hand-stamped" without true randomness (stable across re-renders).
const ATLAS_STAMP_TILT = [-4, 3, -2, 5, -5, 2];

function AtlasFactsShelf({ facts, isDark, lang }) {
  const c = useColors(isDark);
  const isMobile = useIsMobile();
  const shelfRef = useRef(null);
  const [pos, setPos] = useState({ idx:0, atStart:true, atEnd:false });
  const CARD_W = isMobile ? 168 : 300, GAP = 14;
  const recompute = () => {
    const el = shelfRef.current; if (!el) return;
    const idx = Math.round(el.scrollLeft / (CARD_W + GAP));
    setPos({
      idx: Math.max(0, Math.min(idx, facts.length - 1)),
      atStart: el.scrollLeft <= 4,
      atEnd: el.scrollLeft + el.clientWidth >= el.scrollWidth - 6,
    });
  };
  const nudge = (dir) => {
    const el = shelfRef.current; if (!el) return;
    el.scrollBy({ left: dir * (CARD_W + GAP) * 2, behavior:'smooth' });
  };
  const showNav = facts.length > 3;
  const arrowStyle = (off) => ({
    width: isMobile ? 40 : 30, height: isMobile ? 40 : 30, borderRadius:'50%', display:'grid', placeItems:'center',
    cursor: off ? 'default' : 'pointer', fontSize:14, color:c.textMuted,
    opacity: off ? 0.35 : 1, ...atlasGlass(isDark),
  });
  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14 }}>
        <span className="mono" style={{ fontSize:10, letterSpacing:'0.24em', color:c.textSubtle, textTransform:'uppercase' }}>
          {lang==='ru'?'ФАКТЫ':'FACTS'} · {facts.length}
        </span>
        {showNav && (
          <div style={{ display:'flex', gap:6 }}>
            <button onClick={() => nudge(-1)} disabled={pos.atStart} style={arrowStyle(pos.atStart)}>‹</button>
            <button onClick={() => nudge(1)} disabled={pos.atEnd} style={arrowStyle(pos.atEnd)}>›</button>
          </div>
        )}
      </div>
      <div style={{ position:'relative' }}>
        <div ref={shelfRef} className="atlas-shelf" onScroll={recompute} style={{
          display:'flex', gap:GAP, overflowX:'auto', scrollSnapType:'x mandatory',
          padding:'4px 2px 6px',
        }}>
          {facts.map((f, i) => (
            // Archive index card: perforated spine + a hand-stamped number that
            // lies at its own pseudo-random tilt and straightens on hover. The
            // card itself never rotates (rotated text rasterizes blurry); the
            // hover lift is a pure translate — see .atlas-fact-card.
            <div key={i} className="atlas-fact-card" style={{
              flex:`0 0 ${CARD_W}px`, scrollSnapAlign:'start',
              '--stamp-tilt': `${ATLAS_STAMP_TILT[i % ATLAS_STAMP_TILT.length]}deg`,
            }}>
              <span className="atlas-fact-grain" aria-hidden="true" />
              <span className="atlas-fact-stamp mono">
                {lang==='ru'?'ФАКТ':'FACT'} · {String(i+1).padStart(2,'0')}
              </span>
              <div className="serif" style={{ marginTop:13, fontSize:16, lineHeight:1.68, color:c.text }}>{f}</div>
            </div>
          ))}
        </div>
        {showNav && !pos.atEnd && (
          <div aria-hidden="true" style={{
            position:'absolute', right:0, top:0, bottom:6, width:64, pointerEvents:'none',
            background:`linear-gradient(90deg, transparent, ${c.bg})`,
          }} />
        )}
      </div>
      {showNav && (
        <div style={{ display:'flex', gap:4, justifyContent:'center', marginTop:12 }}>
          {facts.map((_, i) => (
            <span key={i} style={{
              width: i === pos.idx ? 16 : 7, height:2.5, borderRadius:2,
              background: i === pos.idx ? c.accentLight : (isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'),
              transition:'all .25s ease',
            }} />
          ))}
        </div>
      )}
    </div>
  );
}

function AtlasAlbumCard({ album, artistName, isDark, lang, onOpen, width }) {
  const c = useColors(isDark);
  const [hover, setHover] = useState(false);
  const W = width || 172;
  const hue = atlasHue(album.title, artistName);
  const coverRef = useRef(null);
  const liked = album.liked_track_count || 0;
  const deckEase = 'cubic-bezier(.22,.9,.3,1)';
  const open = () => {
    const r = coverRef.current && coverRef.current.getBoundingClientRect();
    onOpen(album, r ? { top:r.top, left:r.left, width:r.width, height:r.height } : null);
  };
  return (
    <div
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      onClick={open}
      style={{ width:W, cursor:'pointer', position:'relative', zIndex: hover ? 12 : 1 }}
    >
      <div ref={coverRef} style={{ position:'relative', width:W, height:W }}>
        {/* Vinyl sneaks out from behind the sleeve on hover — same disc as the home deck */}
        <div aria-hidden="true" style={{
          position:'absolute', top:'4%', left:'6%', width:Math.round(W*0.92), height:Math.round(W*0.92), zIndex:1,
          transform: hover ? `translateX(${Math.round(W*0.52)}px) rotate(10deg)` : 'translateX(6px)',
          transition:`transform .5s ${deckEase}`,
        }}>
          <div className="vinyl-disc">
            <span className="vinyl-label" style={{
              background:`radial-gradient(circle at 38% 32%, oklch(62% 0.16 ${hue}), oklch(45% 0.16 ${(hue+45)%360}) 70%)`,
            }} />
          </div>
        </div>
        <div style={{
          position:'relative', zIndex:2, width:W, height:W, borderRadius:12, overflow:'hidden',
          transform: hover ? 'translateX(-7px) rotate(-2deg)' : 'none',
          transition:`transform .5s ${deckEase}, box-shadow .5s ease`,
          boxShadow: hover ? '0 18px 42px rgba(124,91,255,0.32)' : '0 6px 18px rgba(0,0,0,0.3)',
        }}>
          <AlbumCover title={album.title} artist={artistName} size={W} isDark={isDark} coverPath={album.cover_art_path} radius={12} fluid />
          {/* sleeve sheen */}
          <div aria-hidden="true" style={{
            position:'absolute', inset:0, borderRadius:12, pointerEvents:'none',
            background:'linear-gradient(115deg, rgba(255,255,255,0.14) 0%, transparent 34%, transparent 68%, rgba(0,0,0,0.22) 100%)',
          }} />
        </div>
      </div>
      <div style={{ marginTop:10, display:'flex', alignItems:'baseline', gap:6, minWidth:0 }}>
        <span className="serif" style={{ fontSize:15, color:c.text, letterSpacing:'-0.01em', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{album.title}</span>
        {liked >= 3 && (
          <span
            title={lang==='ru'
              ? `${liked} ${atlasRuPlural(liked,'любимый трек','любимых трека','любимых треков')}`
              : `${liked} liked tracks`}
            style={{ color:'oklch(78% 0.14 85)', fontSize:13, flexShrink:0, textShadow:'0 0 12px rgba(212,165,90,0.5)' }}
          >★</span>
        )}
      </div>
      <div className="mono" style={{ marginTop:3, fontSize:10, letterSpacing:'0.16em', color:c.textSubtle, textTransform:'uppercase' }}>
        {album.year || '—'} · {album.tracks.length} {lang==='ru' ? atlasRuPlural(album.tracks.length,'ТРЕК','ТРЕКА','ТРЕКОВ') : (album.tracks.length===1?'TRACK':'TRACKS')}
      </div>
    </div>
  );
}

function AtlasAlbumStrip({ albums, artistName, isDark, lang, onOpen }) {
  const c = useColors(isDark);
  const wrapRef = useRef(null);
  const scrollRef = useRef(null);
  const [wrapW, setWrapW] = useState(0);
  const [nav, setNav] = useState({ atStart: true, atEnd: false });
  const recompute = () => {
    const el = scrollRef.current; if (!el) return;
    setNav({
      atStart: el.scrollLeft <= 4,
      atEnd: el.scrollLeft + el.clientWidth >= el.scrollWidth - 6,
    });
  };
  useEffect(() => {
    const measure = () => { if (wrapRef.current) setWrapW(wrapRef.current.clientWidth); };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);
  // Re-read scroll edges whenever the rail width or the album set changes.
  useEffect(() => { recompute(); }, [wrapW, albums.length]);

  const isMobile = useIsMobile();
  const CARD = isMobile ? 132 : 172, GAP = isMobile ? 16 : 28, CAP_H = 64, AXIS_H = 44;
  // Breathing room around the absolutely-placed cards so the hover lift/rotate
  // (and the vinyl that slides out) isn't clipped at the rail's left edge.
  const PAD_X = 24, PAD_TOP = 16;
  const dated = albums.filter(a => a.year).slice().sort((a, b) => a.year - b.year);
  const undated = albums.filter(a => !a.year);
  const years = dated.map(a => a.year);
  const minY = years.length ? Math.min(...years) : 0;
  const maxY = years.length ? Math.max(...years) : 0;
  // Timeline needs ≥3 dated albums and an actual year spread; otherwise a plain
  // airy wrap reads better than a one-note axis.
  const timeline = dated.length >= 3 && maxY > minY;

  const kicker = (
    <span className="mono" style={{ fontSize:10, letterSpacing:'0.24em', color:c.textSubtle, textTransform:'uppercase' }}>
      {lang==='ru'?'АЛЬБОМЫ':'ALBUMS'} · {albums.length}{timeline ? ` · ${minY}–${maxY}` : ''}
    </span>
  );

  if (!timeline) {
    return (
      <div ref={wrapRef}>
        <div style={{ marginBottom:16 }}>{kicker}</div>
        <div style={{ display:'flex', flexWrap:'wrap', gap:GAP, rowGap:34 }}>
          {albums.map(a => (
            <AtlasAlbumCard key={a.title} album={a} artistName={artistName} isDark={isDark} lang={lang} onOpen={onOpen} width={CARD} />
          ))}
        </div>
      </div>
    );
  }

  // Even pitch: albums sit chronologically left→right at a fixed step. Release
  // dates set the ORDER, never the gap — so reissues and dry spells read the same.
  const STEP = CARD + GAP;
  const ordered = [...dated, ...undated];
  const xs = ordered.map((_, i) => PAD_X + i * STEP);
  const lastX = xs.length ? xs[xs.length - 1] : PAD_X;
  const totalW = lastX + CARD + PAD_X;
  const axisY = PAD_TOP + CARD + CAP_H;
  const tickColor = isDark ? 'rgba(187,168,255,0.5)' : 'rgba(110,80,220,0.45)';
  const scrolls = totalW > wrapW + 2;

  const nudge = (dir) => {
    const el = scrollRef.current; if (!el) return;
    el.scrollBy({ left: dir * STEP * 2, behavior: 'smooth' });
  };
  // No visible scrollbar and a plain wheel mouse can't scroll sideways — so map
  // vertical wheel intent onto the horizontal rail.
  const onWheel = (e) => {
    const el = scrollRef.current; if (!el) return;
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
    if (el.scrollWidth <= el.clientWidth) return;
    el.scrollLeft += e.deltaY;
    e.preventDefault();
  };
  const arrowStyle = (off) => ({
    width: isMobile ? 40 : 30, height: isMobile ? 40 : 30, borderRadius:'50%', display:'grid', placeItems:'center',
    cursor: off ? 'default' : 'pointer', fontSize:14, color:c.textMuted,
    opacity: off ? 0.35 : 1, ...atlasGlass(isDark),
  });

  return (
    <div ref={wrapRef}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
        {kicker}
        {scrolls && (
          <div style={{ display:'flex', gap:6 }}>
            <button onClick={() => nudge(-1)} disabled={nav.atStart} style={arrowStyle(nav.atStart)}>‹</button>
            <button onClick={() => nudge(1)} disabled={nav.atEnd} style={arrowStyle(nav.atEnd)}>›</button>
          </div>
        )}
      </div>
      <div style={{ position:'relative' }}>
        <div ref={scrollRef} className="atlas-shelf" onScroll={recompute} onWheel={onWheel}
             style={{ overflowX:'auto' }}>
          <div style={{ position:'relative', width:totalW, height:axisY + AXIS_H }}>
            {ordered.map((a, i) => (
              <div key={a.title} style={{ position:'absolute', left:xs[i], top:PAD_TOP }}>
                <AtlasAlbumCard album={a} artistName={artistName} isDark={isDark} lang={lang} onOpen={onOpen} width={CARD} />
              </div>
            ))}
            {/* time axis + glowing year ticks */}
            <div aria-hidden="true" style={{
              position:'absolute', left:PAD_X, width:Math.max(0, totalW - PAD_X * 2), top:axisY + 12, height:1,
              background: isDark
                ? 'linear-gradient(90deg, rgba(187,168,255,0.45), rgba(187,168,255,0.08))'
                : 'linear-gradient(90deg, rgba(110,80,220,0.4), rgba(110,80,220,0.06))',
            }} />
            {ordered.map((a, i) => (
              <Fragment key={`tick-${a.title}`}>
                <span aria-hidden="true" style={{ position:'absolute', left:xs[i] + CARD/2, top:axisY + 6, width:1, height:7, background:tickColor }} />
                <span className="mono" style={{
                  position:'absolute', left:xs[i] + CARD/2, top:axisY + 16, transform:'translateX(-50%)',
                  fontSize:13, fontWeight:600, letterSpacing:'0.06em', whiteSpace:'nowrap',
                  padding:'2px 9px', borderRadius:9,
                  color: isDark ? '#ece4ff' : '#4326a8',
                  background: isDark ? 'rgba(124,91,255,0.18)' : 'rgba(124,91,255,0.10)',
                  boxShadow: isDark
                    ? '0 0 16px rgba(124,91,255,0.55), inset 0 0 0 1px rgba(187,168,255,0.45)'
                    : '0 0 12px rgba(124,91,255,0.28), inset 0 0 0 1px rgba(124,91,255,0.30)',
                  textShadow: isDark ? '0 0 10px rgba(187,168,255,0.55)' : 'none',
                }}>{a.year || '—'}</span>
              </Fragment>
            ))}
          </div>
        </div>
        {scrolls && !nav.atEnd && (
          <div aria-hidden="true" style={{
            position:'absolute', right:0, top:0, bottom:0, width:64, pointerEvents:'none',
            background:`linear-gradient(90deg, transparent, ${c.bg})`,
          }} />
        )}
      </div>
    </div>
  );
}

function ArtistAtlasSection({
  isDark, lang, artistSlug, visible, onNav,
  onPlayTrack, navigateToArtist, playerTrack, audioPlaying,
}) {
  const c = useColors(isDark);
  const [data, setData] = useState(null);
  const [status, setStatus] = useState('idle');  // 'idle'|'loading'|'loaded'|'error'
  const [albumModal, setAlbumModal] = useState(null);  // { album, originRect }
  const [enterEpoch, setEnterEpoch] = useState(0);
  const [condensed, setCondensed] = useState(false);
  const [activeId, setActiveId] = useState(null);
  const scrollRef = useRef(null);
  const heroRef = useRef(null);
  const bioRef = useRef(null);
  const factsRef = useRef(null);
  const albumsRef = useRef(null);

  // Replay the .lib-rise cascade on each visit (section stays mounted, see App).
  useEffect(() => { if (visible !== false) setEnterEpoch(e => e + 1); }, [visible]);

  useEffect(() => {
    if (!visible || !artistSlug) return;
    let cancelled = false;
    setStatus('loading');
    setAlbumModal(null);
    setCondensed(false);
    setActiveId(null);
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
    apiFetch(`/artists/${encodeURIComponent(artistSlug)}?lang=${encodeURIComponent(lang)}`)
      .then(res => { if (!cancelled) { setData(res); setStatus('loaded'); } })
      .catch(() => { if (!cancelled) setStatus('error'); });
    return () => { cancelled = true; };
  }, [visible, artistSlug, lang]);

  // Scroll plumbing: condensed dock past the hero + active anchor highlight.
  const handleScroll = () => {
    const sc = scrollRef.current;
    if (!sc) return;
    const heroH = heroRef.current ? heroRef.current.offsetHeight : 340;
    setCondensed(sc.scrollTop > heroH - 70);
    const probe = sc.scrollTop + 170;
    let act = null;
    [['bio', bioRef], ['facts', factsRef], ['albums', albumsRef]].forEach(([id, r]) => {
      if (r.current && r.current.offsetTop <= probe) act = id;
    });
    if (act) setActiveId(act);
  };

  const shell = (msg) => (
    <div style={{ flex:1, display:'grid', placeItems:'center', background:c.bg }}>
      <span className="mono" style={{ fontSize:11, letterSpacing:'0.24em', color:c.textSubtle, textTransform:'uppercase' }}>{msg}</span>
    </div>
  );
  if (!artistSlug) return shell(lang==='ru'?'Артист не выбран':'No artist selected');
  if (status === 'loading' || status === 'idle') return shell(lang==='ru'?'Загрузка…':'Loading…');
  if (status === 'error' || !data) return shell(lang==='ru'?'Не удалось загрузить артиста':'Could not load artist');

  const sectionsCfg = [
    data.bio ? { id:'bio', label: lang==='ru'?'БИО':'BIO' } : null,
    (data.facts || []).length ? { id:'facts', label: lang==='ru'?'ФАКТЫ':'FACTS' } : null,
    (data.albums || []).length ? { id:'albums', label: lang==='ru'?'АЛЬБОМЫ':'ALBUMS' } : null,
  ].filter(Boolean);
  const sectionRefMap = { bio: bioRef, facts: factsRef, albums: albumsRef };
  const prefersReduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const scrollToSection = (id) => {
    const el = sectionRefMap[id] && sectionRefMap[id].current;
    if (el) el.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth', block:'start' });
  };
  const effectiveActive = activeId || (sectionsCfg[0] && sectionsCfg[0].id) || null;

  // Breadcrumb mini-EQ: is the track playing right now by THIS artist?
  const playingHere = !!(audioPlaying && playerTrack && playerTrack.artist && data.name &&
    playerTrack.artist.toLowerCase().includes(data.name.toLowerCase()));

  // ▶ Play artist — autoplay queue seeded by first track of first album
  const onSpin = () => {
    const seed = data.albums[0] && data.albums[0].tracks[0];
    if (!seed || !onPlayTrack) return;
    onPlayTrack(
      { track: seed, score: 1, matched_on: 'lyrics' },
      data.albums.flatMap(a => a.tracks.map(t => ({ track: t, score: 1, matched_on: 'lyrics' }))),
    );
    onNav?.('player');
  };

  // ArtistAlbum → AlbumSummary-ish shape the shared AlbumModal expects
  // (note: modal tracks use `duration`, aggregate tracks use `duration_sec`).
  const toModalAlbum = (album) => ({
    album_title: album.title,
    primary_artist: data.name,
    primary_artist_slug: data.slug,
    cover_art_path: album.cover_art_path,
    year: album.year,
    track_count: album.tracks.length,
    duration_seconds: album.tracks.reduce((s, t) => s + (t.duration_sec || 0), 0),
    tracks: album.tracks.map(t => ({
      track_id: t.track_id, title: t.title, artist: t.artist,
      duration: t.duration_sec, year: t.year, cover_art_path: t.cover_art_path,
    })),
  });
  const openAlbum = (album, rect) => setAlbumModal({ album: toModalAlbum(album), originRect: rect });

  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', position:'relative', background:c.bg }}>
      {/* Condensed glass dock — fades in once the hero scrolls away */}
      <div className="atlas-glare" onMouseMove={atlasGlareMove} style={{
        position:'absolute', top:12, left:26, right:26, zIndex:40,
        display:'flex', alignItems:'center', gap:16, padding:'8px 12px 8px 18px', borderRadius:14,
        ...atlasGlass(isDark),
        opacity: condensed ? 1 : 0,
        transform: condensed ? 'translateY(0)' : 'translateY(-14px)',
        pointerEvents: condensed ? 'auto' : 'none',
        transition:'opacity .28s ease, transform .28s ease',
      }}>
        <span className="serif" style={{
          fontSize:17, color:c.text, letterSpacing:'-0.01em',
          whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', minWidth:0,
        }}>{data.name}</span>
        <div style={{ flex:1 }} />
        <AtlasAnchorPills sections={sectionsCfg} activeId={effectiveActive} onAnchor={scrollToSection} isDark={isDark} compact />
        <AtlasSpinButton onSpin={onSpin} lang={lang} compact />
      </div>

      <div ref={scrollRef} onScroll={handleScroll} style={{ flex:1, overflowY:'auto', position:'relative' }}>
        <div key={`atlas-${enterEpoch}`}>
          <AtlasHero data={data} isDark={isDark} lang={lang} onNav={onNav} heroRef={heroRef} playingHere={playingHere} />

          {/* Floating glass dock overlapping the hero's bottom edge */}
          <div className="lib-rise" style={{ maxWidth:1120, margin:'0 auto', padding:'0 32px', position:'relative', zIndex:6 }}>
            <div className="atlas-glare" onMouseMove={atlasGlareMove} style={{
              marginTop:-34, borderRadius:16, padding:'10px 14px',
              display:'flex', alignItems:'center', justifyContent:'space-between', gap:14,
              ...atlasGlass(isDark),
            }}>
              <AtlasAnchorPills sections={sectionsCfg} activeId={effectiveActive} onAnchor={scrollToSection} isDark={isDark} />
              <AtlasSpinButton onSpin={onSpin} lang={lang} />
            </div>
          </div>

          {/* position+z lift the column's text above the light field's tail
              (a positioned z-auto hero paints over later static siblings) */}
          <div style={{ maxWidth:1120, margin:'0 auto', padding:'34px 32px 110px', display:'flex', flexDirection:'column', gap:46, position:'relative', zIndex:2 }}>
            {data.bio && (
              <section ref={bioRef} className="lib-rise" style={{ '--lib-d':'0.08s', scrollMarginTop:84 }}>
                <AtlasBio bio={data.bio} isDark={isDark} lang={lang} />
              </section>
            )}
            {(data.facts || []).length > 0 && (
              <section ref={factsRef} className="lib-rise" style={{ '--lib-d':'0.16s', scrollMarginTop:84 }}>
                <AtlasFactsShelf facts={data.facts} isDark={isDark} lang={lang} />
              </section>
            )}
            {(data.albums || []).length > 0 && (
              <section ref={albumsRef} className="lib-rise" style={{ '--lib-d':'0.24s', scrollMarginTop:84 }}>
                <AtlasAlbumStrip albums={data.albums} artistName={data.name} isDark={isDark} lang={lang} onOpen={openAlbum} />
              </section>
            )}
            {sectionsCfg.length === 0 && (
              <div style={{ color:c.textMuted, fontSize:13 }}>
                {lang==='ru' ? 'Пока пусто — обогатите библиотеку, чтобы здесь появились факты и альбомы.' : 'Nothing here yet — enrich your library to fill this page.'}
              </div>
            )}
          </div>
        </div>
      </div>

      {albumModal && (
        <AlbumModal
          album={albumModal.album}
          originRect={albumModal.originRect}
          onClose={() => setAlbumModal(null)}
          onPlayTrack={onPlayTrack}
          navigateToArtist={navigateToArtist}
          isDark={isDark}
          lang={lang}
        />
      )}
    </div>
  );
}

// ─── Phase A: Login + Register screens ────────────────────────────────────
function LoginScreen({ instanceMode, onAuthSuccess, lang }) {
  const [tab, setTab]           = useState('login'); // 'login' | 'register'
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [invite, setInvite]     = useState(() => {
    // Pre-fill from URL hash like #/register?invite=ABCDEFGH1234
    const m = window.location.hash.match(/invite=([\w-]{12})/);
    return m ? m[1] : '';
  });
  const [error, setError]   = useState('');
  const [busy, setBusy]     = useState(false);

  useEffect(() => {
    if (window.location.hash.startsWith('#/register') && instanceMode === 'server') {
      setTab('register');
    }
  }, [instanceMode]);

  const canRegister = instanceMode === 'server';

  const onSubmit = async (e) => {
    e.preventDefault();
    setError(''); setBusy(true);
    try {
      const path = tab === 'login' ? '/auth/login' : '/auth/register';
      const body = tab === 'login'
        ? { email: email.trim(), password }
        : { email: email.trim(), password, invite_code: invite.trim() };
      const res = await fetch(API + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setStoredAuth(data);
      onAuthSuccess(data);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'grid', placeItems: 'center',
      background: 'radial-gradient(ellipse at top, #1a1a24 0%, #0a0a10 100%)',
      color: '#eee',
    }}>
      <form onSubmit={onSubmit} style={{
        width: 360, padding: 32, borderRadius: 16,
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
      }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 4 }}>MusiX</h1>
        <div style={{ fontSize: 12, color: '#888', marginBottom: 20 }}>
          {instanceMode === 'sharing'
            ? (lang === 'ru' ? 'личный self-host' : 'personal self-host')
            : (lang === 'ru' ? 'multi-account сервер' : 'multi-account server')}
        </div>

        {canRegister && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <button type="button" onClick={() => { setTab('login'); setError(''); }}
              style={{
                flex: 1, padding: '8px 12px', borderRadius: 8,
                background: tab === 'login' ? 'rgba(124,91,255,0.2)' : 'transparent',
                color: tab === 'login' ? '#bba8ff' : '#888',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>{lang === 'ru' ? 'Войти' : 'Log in'}</button>
            <button type="button" onClick={() => { setTab('register'); setError(''); }}
              style={{
                flex: 1, padding: '8px 12px', borderRadius: 8,
                background: tab === 'register' ? 'rgba(124,91,255,0.2)' : 'transparent',
                color: tab === 'register' ? '#bba8ff' : '#888',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>{lang === 'ru' ? 'Регистрация' : 'I have an invite'}</button>
          </div>
        )}

        <input
          type="email" required autoFocus
          placeholder={lang === 'ru' ? 'email' : 'email'}
          value={email}
          onChange={e => setEmail(e.target.value)}
          style={{
            width: '100%', padding: '10px 12px', marginBottom: 10,
            borderRadius: 8, background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.08)', color: '#eee',
          }}
        />
        <input
          type="password" required
          placeholder={lang === 'ru' ? 'пароль' : 'password'}
          value={password}
          onChange={e => setPassword(e.target.value)}
          style={{
            width: '100%', padding: '10px 12px', marginBottom: 10,
            borderRadius: 8, background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.08)', color: '#eee',
          }}
        />
        {tab === 'register' && (
          <input
            type="text" required
            placeholder={lang === 'ru' ? 'инвайт-код (12 символов)' : 'invite code (12 chars)'}
            value={invite}
            onChange={e => setInvite(e.target.value)}
            maxLength={12}
            style={{
              width: '100%', padding: '10px 12px', marginBottom: 10,
              borderRadius: 8, background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.08)', color: '#eee',
              fontFamily: 'monospace', letterSpacing: '0.1em',
            }}
          />
        )}

        {error && (
          <div style={{
            padding: '8px 12px', marginBottom: 10, borderRadius: 8,
            background: 'rgba(255,80,80,0.12)', color: '#ff9090',
            fontSize: 13,
          }}>{error}</div>
        )}

        <button type="submit" disabled={busy}
          style={{
            width: '100%', padding: '12px', borderRadius: 8,
            background: busy ? 'rgba(124,91,255,0.4)' : 'rgba(124,91,255,0.7)',
            color: '#fff', fontWeight: 600, fontSize: 14,
            border: 'none', cursor: busy ? 'wait' : 'pointer',
            marginTop: 6,
          }}>
          {busy
            ? (lang === 'ru' ? 'Подождите…' : 'Please wait…')
            : (tab === 'login'
                ? (lang === 'ru' ? 'Войти' : 'Log in')
                : (lang === 'ru' ? 'Создать аккаунт' : 'Create account'))}
        </button>

        {!canRegister && (
          <div style={{ fontSize: 11, color: '#666', marginTop: 16, lineHeight: 1.5 }}>
            {lang === 'ru'
              ? 'Регистрация недоступна — sharing mode. Запустите scripts/create_owner для первого пользователя.'
              : 'Registration disabled in sharing mode. Run scripts/create_owner to seed the owner.'}
          </div>
        )}
      </form>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
// ─── useIsMobile — runtime viewport detection (Phase 1 mobile shell) ──────────
// Client detection happens in the browser, not via server-side User-Agent: the
// shell swaps to a bottom tab bar + mini-player + full-screen player below 768px
// and reacts live to rotate / resize / split-screen.
// Spec: docs/superpowers/specs/2026-06-24-mobile-responsive-design.md
function useIsMobile() {
  const MQ = '(max-width: 768px)';
  const [mobile, setMobile] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia(MQ).matches);
  useEffect(() => {
    const mql = window.matchMedia(MQ);
    const onChange = e => setMobile(e.matches);
    mql.addEventListener('change', onChange);
    setMobile(mql.matches);  // sync in case the breakpoint flipped before mount
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return mobile;
}

// ─── BottomTabBar — mobile bottom navigation ──────────────────────────────────
// Replaces FloatingIconNav on phones. Four destinations; the player is reached
// through MiniPlayerBar, not a tab. Rendered as a flex child in the App column so
// it reserves its own height (content never hides behind it); safe-area inset
// keeps it clear of the notch/home-indicator.
function BottomTabBar({ section, onNav, isDark, lang }) {
  const c = useColors(isDark);
  const accent = 'oklch(60% 0.18 270)';
  const inactive = isDark ? 'rgba(238,238,243,0.52)' : 'rgba(22,22,32,0.5)';
  const items = [
    { id:'home', label: lang==='ru'?'Главная':'Home',
      icon:(a)=><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={a?2.1:1.8} strokeLinecap="round" strokeLinejoin="round"><path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v10h14V10"/></svg> },
    { id:'search', label: lang==='ru'?'Поиск':'Search',
      icon:(a)=><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={a?2.1:1.8} strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> },
    { id:'library', label: lang==='ru'?'Библиотека':'Library',
      icon:(a)=><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={a?2.1:1.8} strokeLinecap="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg> },
    { id:'recommend', label: lang==='ru'?'Рекомендации':'For You',
      icon:(a)=><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={a?2.1:1.8} strokeLinecap="round" strokeLinejoin="round"><path d="M12 2 15 8l6 1-4.5 4.5L18 20l-6-3-6 3 1.5-6.5L3 9l6-1z"/></svg> },
  ];
  const barBg = isDark
    ? 'linear-gradient(180deg, rgba(22,22,28,0.97), rgba(14,14,19,0.98))'
    : 'linear-gradient(180deg, rgba(255,255,255,0.97), rgba(244,243,249,0.98))';
  const topBorder = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(22,22,32,0.10)';
  return (
    <nav className="mobile-tabbar" style={{
      flexShrink:0, display:'flex', alignItems:'stretch', background: barBg,
      borderTop:`1px solid ${topBorder}`,
      backdropFilter:'blur(18px) saturate(1.1)', WebkitBackdropFilter:'blur(18px) saturate(1.1)',
      paddingBottom:'env(safe-area-inset-bottom, 0px)', zIndex:30,
    }}>
      {items.map(item => {
        const active = section === item.id;
        return (
          <button key={item.id} onClick={()=>onNav(item.id)} title={item.label}
            style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center',
              justifyContent:'center', gap:3, padding:'8px 2px 7px', background:'transparent',
              border:0, cursor:'pointer', color: active ? accent : inactive,
              transition:'color .18s ease', minWidth:0 }}>
            <span style={{ display:'flex' }}>{item.icon(active)}</span>
            <span style={{ fontFamily:"'JetBrains Mono', ui-monospace, monospace", fontWeight:600,
              fontSize:9, letterSpacing:'0.03em', whiteSpace:'nowrap', overflow:'hidden',
              textOverflow:'ellipsis', maxWidth:'100%' }}>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}

// ─── MiniPlayerBar — mobile persistent now-playing strip above the tab bar ────
// Reuses the cover-URL convention from MiniPlaybackPopout. Tapping the strip opens
// the full-screen player; prev/play/next + add-to-playlist act in place.
function MiniPlayerBar({ track, audio, isDark, lang, onOpen, onPrev, onNext, onAddToPlaylist }) {
  const c = useColors(isDark);
  const rawCover = (track && (track.cover_art_path || track.coverArt)) || null;
  const cover = rawCover ? thumbCoverUrl(rawCover.startsWith('http') ? rawCover : `${API}${rawCover}`) : null;
  const title = track?.title || '—';
  const artist = track?.artist || '';
  const isPlaying = !!(audio && audio.isPlaying);
  const currentTime = useCurrentTime();
  const duration = audio?.duration || 0;
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  const barBg = isDark
    ? 'linear-gradient(180deg, rgba(30,30,38,0.97), rgba(24,24,30,0.98))'
    : 'linear-gradient(180deg, rgba(255,255,255,0.97), rgba(247,246,251,0.98))';
  const topBorder = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(22,22,32,0.08)';
  const coverBg = isDark ? '#1a1a22' : '#f0eff5';
  const accent = 'oklch(60% 0.18 270)';

  return (
    <div className="mobile-miniplayer" onClick={onOpen} style={{
      flexShrink:0, display:'flex', alignItems:'center', gap:10, padding:'7px 12px',
      background: barBg, borderTop:`1px solid ${topBorder}`,
      backdropFilter:'blur(18px) saturate(1.1)', WebkitBackdropFilter:'blur(18px) saturate(1.1)',
      cursor:'pointer', position:'relative', zIndex:31,
    }}>
      <div style={{ position:'absolute', top:0, left:0, height:2, width:`${progress*100}%`,
        background:accent, borderRadius:'0 2px 2px 0', transition:'width .2s linear' }} />
      <div style={{ width:46, height:46, borderRadius:9, overflow:'hidden', background:coverBg,
        flexShrink:0, display:'grid', placeItems:'center' }}>
        {cover
          ? <img src={cover} alt="" style={{ width:'100%', height:'100%', objectFit:'cover' }} />
          : <span className="serif-display" style={{ color:'#d4a55a', fontSize:20, fontStyle:'normal' }}>{title[0]?.toUpperCase() || '?'}</span>}
      </div>
      <div style={{ flex:1, minWidth:0, display:'flex', flexDirection:'column', gap:2 }}>
        <div style={{ fontSize:13, fontWeight:600, color:c.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{title}</div>
        <div style={{ fontSize:11, color:c.textMuted, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{artist}</div>
      </div>
      <button title={lang==='ru'?'В плейлист':'Add to playlist'}
        onClick={(e)=>{ e.stopPropagation(); onAddToPlaylist && onAddToPlaylist(track?.track_id, e.currentTarget); }}
        style={{ flexShrink:0, width:34, height:40, display:'grid',
          placeItems:'center', background:'transparent', border:0, color:c.textMuted, cursor:'pointer', fontSize:19, lineHeight:1 }}>＋</button>
      <button title={lang==='ru'?'Предыдущий':'Previous'}
        onClick={(e)=>{ e.stopPropagation(); onPrev && onPrev(); }}
        style={{ flexShrink:0, width:34, height:40, display:'grid',
          placeItems:'center', background:'transparent', border:0, color:c.text, cursor:'pointer' }}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="19 20 9 12 19 4"/><rect x="5" y="4" width="2" height="16"/></svg>
      </button>
      <button title={isPlaying ? (lang==='ru'?'Пауза':'Pause') : (lang==='ru'?'Играть':'Play')}
        onClick={(e)=>{ e.stopPropagation(); audio?.togglePlay?.(); }}
        style={{ flexShrink:0, width:40, height:40, borderRadius:20, display:'grid',
          placeItems:'center', background:'transparent', border:0, color:c.text, cursor:'pointer' }}>
        {isPlaying
          ? <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          : <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="7 4 19 12 7 20 7 4"/></svg>}
      </button>
      <button title={lang==='ru'?'Следующий':'Next'}
        onClick={(e)=>{ e.stopPropagation(); onNext && onNext(); }}
        style={{ flexShrink:0, width:34, height:40, display:'grid',
          placeItems:'center', background:'transparent', border:0, color:c.text, cursor:'pointer' }}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 4 15 12 5 20"/><rect x="17" y="4" width="2" height="16"/></svg>
      </button>
    </div>
  );
}

// ─── SPA routing (spec 2026-07-10-spa-routing-design) ────────────────────────
// The URL is a projection of App's `section` state (+ artist slug). No router
// library on purpose: sections must stay mounted (visibility-toggled) so audio
// and per-section state survive navigation; a route-per-element router would
// unmount them. FastAPI's SPA catch-all and the PWA navigateFallback already
// serve index.html for these paths, so F5 and deep links work.
const SECTION_PATHS = {
  home: '/', search: '/search', recommend: '/recommend',
  library: '/library', player: '/player',
};

function pathForSection(section, artistSlug) {
  if (section === 'artist' && artistSlug) return `/artist/${encodeURIComponent(artistSlug)}`;
  return SECTION_PATHS[section] || '/';
}

function parseAppPath(pathname) {
  const clean = (pathname || '/').replace(/\/+$/, '') || '/';
  const artist = clean.match(/^\/artist\/([^/]+)$/);
  if (artist) {
    try { return { section: 'artist', slug: decodeURIComponent(artist[1]) }; }
    catch { return null; }
  }
  for (const [section, path] of Object.entries(SECTION_PATHS)) {
    if (clean === path) return { section };
  }
  return null;
}

// Push a history entry while an overlay is open so the browser/OS back
// gesture closes the overlay instead of leaving the page (critical in the
// installed PWA, where "back" otherwise exits the app). Closing through the
// overlay's own UI pops the entry it pushed, keeping history clean. Only for
// overlays whose close path can't race a section navigation in the same tick
// (e.g. the settings panel); the library album modal closes INTO a play-track
// navigation, so it must not use this.
function useHistoryOverlay(isOpen, onClose, key) {
  const openRef = useRef(false);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (isOpen && !openRef.current) {
      openRef.current = true;
      window.history.pushState({ app: true, overlay: key }, '', window.location.href);
    } else if (!isOpen && openRef.current) {
      openRef.current = false;
      if (window.history.state && window.history.state.overlay === key) window.history.back();
    }
  }, [isOpen, key]);
  useEffect(() => {
    const onPop = () => {
      if (openRef.current && !(window.history.state && window.history.state.overlay === key)) {
        openRef.current = false;
        onCloseRef.current();
      }
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [key]);
}

function App({ instanceMode = 'sharing', onLogout = () => {} }) {
  const [isDark, setDark] = useState(() => (localStorage.getItem('musix_theme') || 'dark') === 'dark');
  const [lang, setLang]   = useState(() => localStorage.getItem('musix_lang') || 'ru');

  // Reflect theme on <body data-theme> so CSS class overrides (.panel-v3, .pill-v3,
  // .mono-label, .ask-ai-btn, .atmospheric-bg) can target light/dark independently.
  useEffect(() => {
    document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  // SPA routing: the initial section comes from the URL (deep links / F5).
  // '/player' can't be restored across a reload — the queue lives in memory —
  // so it lands on home; the first URL-sync effect below replaceState()s the
  // path back to '/' without creating a history entry (same for unknown paths).
  const initialRouteRef = useRef(parseAppPath(window.location.pathname));
  const [section, setSection] = useState(() => { // home | search | recommend | library | player | artist
    const s = initialRouteRef.current?.section;
    return (s && s !== 'player') ? s : 'home';
  });
  // Artist Atlas: slug is stored alongside the section so the section type
  // 'artist' can render the ArtistAtlasSection with the right artist. Setting
  // a non-null slug + switching section in one call lets the section preserve
  // its own internal state (fetched aggregate, active tab) across navigation.
  const [activeArtistSlug, setActiveArtistSlug] = useState(initialRouteRef.current?.slug || null);
  const navigateToArtist = useCallback((slug) => {
    if (!slug) return;
    setActiveArtistSlug(slug);
    setSection('artist');
  }, []);

  // Mobile shell (Phase 1): below 768px the layout swaps to a bottom tab bar +
  // mini-player + full-screen player overlay. The player reuses the existing
  // 'player' section; mobilePrevSectionRef remembers which tab to return to when
  // the full-screen player is dismissed.
  const isMobile = useIsMobile();
  const mobilePrevSectionRef = useRef('library');
  const openMobilePlayer = useCallback(() => {
    setSection(s => { if (s !== 'player') mobilePrevSectionRef.current = s; return 'player'; });
  }, []);
  const closeMobilePlayer = useCallback(() => {
    // The /player history entry was pushed by the URL-sync effect when the
    // overlay opened — prefer history.back() so the ✕ button and the OS back
    // gesture walk the same path and history doesn't accumulate a forward
    // loop. The state marker is only absent when /player wasn't our push.
    if (window.history.state && window.history.state.app) {
      window.history.back();
    } else {
      setSection(mobilePrevSectionRef.current || 'library');
    }
  }, []);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [themeAnim, setThemeAnim] = useState(false);
  const [appState, setAppState] = useState('checking');

  // Phase D: migrate collection-suffixed localStorage keys → user-id-scoped.
  // useMemo (not useEffect) so it runs DURING App's first render, before any
  // child hook (useChatHistory / useRecentSearches / useTrackChat) reads a key.
  React.useMemo(() => runCollectionLocalStorageMigration_d(localStorage.getItem('musix_user_id')), []);

  const [collections, setCollections] = useState([]);
  const [userPoints, setUserPoints] = useState(0);
  const aiStatus = useAIStatus();

  // Plan 19 follow-up: lift usePlaylists + AddToPlaylistPopover to App-level
  // so PlayerSection, SearchSection, LibrarySection share one source of truth
  // and the popover can be triggered from any track-row in any surface.
  const appPlaylists = usePlaylists();
  const [addToPopoverInfo, setAddToPopoverInfo] = useState(null);  // { trackId, anchor } | null
  const openAddToPlaylist = useCallback((trackId, anchor) => {
    if (!trackId) return;
    setAddToPopoverInfo({ trackId, anchor });
  }, []);
  const closeAddToPlaylist = useCallback(() => setAddToPopoverInfo(null), []);

  // ── Spotlight (find-and-play) + search handoff from the landing ──────────
  // spotlightOpen: the global ⌘K overlay. searchHandoff: a one-shot query the
  // landing (lyrics field) or the spotlight («ещё») throws into SearchSection;
  // ts makes repeated identical queries re-fire. mode 'auto' → AI chat when
  // the assistant is up, classic grid otherwise; 'grid' forces the classic tab.
  const [spotlightOpen, setSpotlightOpen] = useState(false);
  const [searchHandoff, setSearchHandoff] = useState(null);  // { query, mode, ts } | null
  const handoffToSearch = useCallback((query, mode = 'auto') => {
    setSearchHandoff({ query, mode, ts: Date.now() });
    setSection('search');
  }, []);
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === 'k' || e.key === 'K' || e.key === 'л' || e.key === 'Л')) {
        e.preventDefault();
        setSpotlightOpen(o => !o);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const [stats, setStats] = useState(null);

  // Player state
  const [playerPlaylist, setPlayerPlaylist] = useState([]);
  const [playerTrack, setPlayerTrack] = useState(null);
  const [lyricsMode, setLyricsMode] = useState(false);

  // Toast state
  const [toast, setToast] = useState(null); // { message: string, id: number } | null

  const showToast = useCallback((message) => {
    const id = Date.now() + Math.random();
    setToast({ message, id });
    setTimeout(() => {
      setToast(prev => (prev && prev.id === id ? null : prev));
    }, 2400);
  }, []);

  // Reset cover face when the track changes — new song should start on cover.
  useEffect(() => { setLyricsMode(false); }, [playerTrack?.track_id]);
  // Mirror for the popstate handler above (declared before playerTrack).
  useEffect(() => { playerTrackRef2.current = playerTrack; }, [playerTrack]);
  // Played track ids live in a ref (not React state) because they only get
  // read from the autoplay-queue exclude_ids param — never rendered. A ref
  // sidesteps the stale-closure trap that would force every consumer to
  // re-subscribe via useCallback deps to see the latest set.
  const playedTrackIdsRef = useRef(new Set());

  // Shared audio controller — lives at App level so audio survives navigation
  const audio = useAudioPlayer();

  // ── SPA routing effects ────────────────────────────────────────────────────
  // playerTrack is declared further down (transpiled const→var, so reading it
  // during render up here would see undefined) — the popstate handler reads it
  // at event time through this ref instead.
  const playerTrackRef2 = useRef(null);

  // State → URL. After a popstate the URL already matches, so the pathname
  // check breaks the loop and back/forward never double-push. The first sync
  // uses replaceState: it only fires for deep links that need normalizing
  // (unknown path, unrestorable /player) and must not mint a history entry.
  const firstUrlSyncRef = useRef(true);
  useEffect(() => {
    if (appState !== 'ready') return; // boot/onboarding screens own the viewport
    const path = pathForSection(section, activeArtistSlug);
    if (window.location.pathname === path) { firstUrlSyncRef.current = false; return; }
    const method = firstUrlSyncRef.current ? 'replaceState' : 'pushState';
    window.history[method]({ app: true }, '', path);
    firstUrlSyncRef.current = false;
  }, [section, activeArtistSlug, appState]);

  // URL → state (browser back/forward).
  useEffect(() => {
    // A leftover overlay entry can be current after a reload (history.state
    // survives F5) — strip the marker so useHistoryOverlay can't misread it.
    if (window.history.state && window.history.state.overlay) {
      window.history.replaceState({ app: true }, '', window.location.href);
    }
    const onPopState = () => {
      const route = parseAppPath(window.location.pathname) || { section: 'home' };
      if (route.section === 'player' && !playerTrackRef2.current) {
        // A /player entry from a previous page load can't be restored (the
        // queue lives in memory) — land on home instead of an empty player.
        window.history.replaceState({ app: true }, '', '/');
        setSection('home');
        return;
      }
      if (route.section === 'artist' && route.slug) setActiveArtistSlug(route.slug);
      setSection(route.section);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Back gesture closes the settings overlay instead of leaving the section
  // (spec phase 3). SettingsPanel's own ✕ pops the entry it pushed.
  useHistoryOverlay(settingsOpen, () => setSettingsOpen(false), 'settings');

  // Global keyboard shortcuts (Task 11 + Task 12)
  useGlobalKeyboardShortcuts({
    audio,
    onNavToSection: setSection,
    onToggleLyrics: () => { if (section === 'player') setLyricsMode(m => !m); },
    onCloseLyrics:  () => { if (section === 'player') setLyricsMode(false); },
  });

  // Best-effort playback event on tab close / navigation away.
  useEffect(() => {
    // audio.audioRef is a stable useRef from the hook — closure reads .current
    // at fire time, so the empty deps are safe. [audio] would tear down + re-add
    // on every App re-render (audio is a new object literal each useAudioPlayer
    // render) for no benefit, and could in principle leave a gap if a re-render
    // races with unload.
    const onUnload = () => {
      // Flush the accumulated listen (real played seconds), deduped by the
      // playEmitted guard so it can't double up with the pagehide listener.
      flushAccumulatedListen(audio.audioRef.current);
    };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const c = useColors(isDark);

  // Stream token for <audio> URLs: mint on mount, refresh every 20 min
  // (server TTL is 60 min) so a long listening session never goes stale.
  useEffect(() => {
    refreshStreamToken();
    const t = setInterval(refreshStreamToken, 20 * 60 * 1000);
    return () => clearInterval(t);
  }, []);

  // Boot
  useEffect(() => {
    apiFetch('/library/collections')
      .then(res => {
        if (!res.qdrant_available) { setAppState('no-qdrant'); return; }
        const cols = res.collections || [];
        setCollections(cols);
        const up = res.user_points || 0;
        setUserPoints(up);
        setAppState(up > 0 ? 'ready' : 'onboarding');
      })
      .catch(() => setAppState('no-qdrant'));
  }, []);

  const loadCollections = useCallback(() =>
    apiFetch('/library/collections').then(data => {
      setCollections(data.collections || []);
      setUserPoints(data.user_points || 0);
    }).catch(() => {}), []);

  // Indexing job tracking lives at App level (spec phase 2): the SSE
  // subscription survives closing the settings panel and section navigation.
  const indexingJob = useIndexingJob({ onCompleted: loadCollections });

  // Resume after a reload / navigation away: the server keeps the per-account
  // job slot, and the SSE stream replays a full progress snapshot to late
  // subscribers — so ask it instead of trusting any client-side memory.
  useEffect(() => {
    apiFetch('/library/status')
      .then(st => {
        if (st && st.job_id && (st.overall_status === 'running' || st.overall_status === 'pending')) {
          indexingJob.attach(st.job_id, { resumed: true });
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePlayTrack = (track, results) => {
    // Any manual play exits stream mode — the user took the wheel.
    streamActiveRef.current = false;
    setStreamActive(false);
    setStreamAdapt(null);
    // Replacing the queue voids the shuffle: the fresh queue starts unshuffled.
    resetShuffle();
    // Search/Recommend pass HIT shape ({ track:{track_id,title,...}, score, matched_on }).
    // playerTrack must be FLAT so LandingPlayer / NowPlayingPebble / MiniPlaybackPopout /
    // PlayerSection.findIndex (which compares h.track.track_id === initialTrack.track_id)
    // all work. playerPlaylist stays HIT[] because PlayerSection's right rail renders score/matched_on.
    const flatTrack = track && track.track ? track.track : track;
    // Single-track entry points (stats widgets: song map, loved/skipped columns,
    // top-track card) have no surrounding queue and pass results=[]. PlayerSection's
    // start-playback effect bails when the playlist is empty (initialPlaylist.length
    // === 0), so the track would never load or play — only the App-level cover blur
    // (which reads playerTrack) would update. Synthesize a 1-item playlist from the
    // track itself so the now-playing track is always present in the queue.
    const queue = (results && results.length)
      ? results
      : (flatTrack?.track_id ? [(track && track.track) ? track : { track: flatTrack, score: 1 }] : []);
    setPlayerPlaylist(queue);
    setPlayerTrack(flatTrack);
    setSection('player');

    // Library/Recently/Playlists ship slim track shapes (LikedSongTrack et al.)
    // that lack `file_path`, `lyrics`, `producer`, `samples`, `genre` — fields the
    // player UI (lyrics back-face, info pills, chat context, queue genre-pill)
    // reads directly off currentTrack and queue items. Enrich the ENTIRE queue
    // in one batch /metadata/tracks?ids= call so switching to any queue item
    // doesn't show partial metadata. Slim items are detected by undefined file_path
    // (Search/Recommend hits arrive with file_path populated and are skipped).
    const slimIds = new Set();
    for (const h of queue) {
      const t = (h && h.track) ? h.track : h;
      if (t && t.track_id && t.file_path === undefined) slimIds.add(t.track_id);
    }
    if (flatTrack?.track_id && flatTrack.file_path === undefined) slimIds.add(flatTrack.track_id);
    if (slimIds.size === 0) return;
    const ids = Array.from(slimIds);
    apiFetch(
      `/metadata/tracks?ids=${encodeURIComponent(ids.join(','))}` +
      ``
    )
      .then(full => {
        if (!Array.isArray(full) || full.length === 0) return;
        const byId = new Map();
        for (const t of full) { if (t && t.track_id) byId.set(t.track_id, t); }
        // Preserve any slim-shape extras (e.g. liked_at, added_at) by spreading slim first.
        // Then overlay ONLY the non-empty fresh fields — a null/empty value from the
        // enrichment endpoint must never clobber a populated slim one (e.g. album, year,
        // or artist_refs the list-source already carried for a collaboration).
        const mergeOne = (slim) => {
          const fresh = byId.get(slim.track_id);
          if (!fresh) return slim;
          const merged = { ...slim };
          for (const [k, v] of Object.entries(fresh)) {
            const empty = v === null || v === undefined || v === ''
              || (Array.isArray(v) && v.length === 0);
            if (!empty) merged[k] = v;
          }
          merged.duration = fresh.duration_sec ?? slim.duration;
          return merged;
        };
        setPlayerTrack(prev => (prev && byId.has(prev.track_id) ? mergeOne(prev) : prev));
        setPlayerPlaylist(prev => prev.map(h => {
          const slim = (h && h.track) ? h.track : null;
          if (!slim || !byId.has(slim.track_id)) return h;
          return { ...h, track: mergeOne(slim) };
        }));
      })
      .catch(() => {/* keep slim shape — player still works, just no lyrics */});
  };

  // Called by PlayerSection when the user skips / picks a different playlist
  // item — keeps App.playerTrack in lockstep with what's actually playing,
  // so Landing/Pebble/Popout don't go stale.
  const handleTrackChange = (track) => {
    if (!track) return;
    setPlayerTrack(track);
    // Mutate the ref directly — no re-render needed since nothing reads
    // playedTrackIds during render. The autoplay-queue fetcher reads the
    // current value at call time, so it always sees the freshest set.
    if (!playedTrackIdsRef.current.has(track.track_id)) {
      playedTrackIdsRef.current.add(track.track_id);
    }
    // Stream prefetch: keep 1-2 tracks of runway. The buffer is deliberately
    // SMALL — it is the only adaptation lag the stateless backend has.
    if (streamActiveRef.current) {
      const idx = playerPlaylist.findIndex(h => ((h && h.track) ? h.track : h).track_id === track.track_id);
      if (idx >= 0 && playerPlaylist.length - idx <= 2) {
        fetchStreamChunk().then(fresh => {
          if (fresh.length && streamActiveRef.current) setPlayerPlaylist(prev => [...prev, ...fresh]);
        });
      }
    }
  };

  // Queue a track (from the player's similar/contrast rail) to play NEXT.
  // Inserts into the single source-of-truth queue (App.playerPlaylist) so it
  // survives PlayerSection's prop re-sync; marked _noInfluence so its eventual
  // playback does NOT feed the "For You" taste profile (the user hand-picked it).
  const handleQueueNext = (track) => {
    if (!track || !track.track_id) return;
    const hit = { track, score: 0, matched_on: 'audio', _queuedNext: true, _noInfluence: true };
    setPlayerPlaylist(prev => {
      const cur = playerTrack;
      const idx = cur ? prev.findIndex(h => ((h && h.track) ? h.track : h).track_id === cur.track_id) : -1;
      const at = idx >= 0 ? idx + 1 : prev.length;
      return [...prev.slice(0, at), hit, ...prev.slice(at)];
    });
    if (track.file_path === undefined) {
      apiFetch(`/metadata/tracks?ids=${encodeURIComponent(track.track_id)}`)
        .then(full => {
          if (!Array.isArray(full) || !full.length) return;
          const fresh = full[0];
          if (!fresh || !fresh.track_id) return;
          setPlayerPlaylist(prev => prev.map(h => (
            (h && h.track && h.track.track_id === fresh.track_id)
              ? { ...h, track: { ...h.track, ...fresh, duration: fresh.duration_sec ?? h.track.duration } }
              : h
          )));
        })
        .catch(() => {});
    }
    showToast(lang === 'ru' ? 'Поставлено следующим' : 'Queued next');
  };

  // Home mini-player prev/next. On the home screen PlayerSection is unmounted,
  // so its [initialTrack] effect (which normally drives setSrc + play) never
  // fires — changing playerTrack alone would update the strip but not the
  // audio. Drive the element directly here (same pattern as playTrackAt), then
  // hand off to handleTrackChange for the shared bookkeeping (playerTrack sync
  // + stream prefetch).
  const handleHomeTrackChange = (track) => {
    if (!track || !track.track_id) return;
    if (audio) {
      markPlaybackInteracted(audio.audioRef?.current);
      audio.setSrc(buildStreamUrl(track.track_id), { trackId: track.track_id }, { autoplay: true });
    }
    handleTrackChange(track);
  };

  // ── Queue shuffle ──────────────────────────────────────────────────────
  // Toggle-with-restore over the SOURCE-OF-TRUTH queue (playerPlaylist), so
  // the visible queue stays truthful: ON physically Fisher–Yates-shuffles the
  // unplayed tail (played + now-playing stay put, same lock as drag-reorder);
  // OFF restores the tail to the pre-shuffle snapshot. Tracks that appeared
  // after ON (queue-next, autoplay top-ups) aren't in the snapshot — they sink
  // to the end of the tail keeping their relative order. Hidden entirely in
  // stream mode (the wave owns its order), and reset on every queue
  // replacement (handlePlayTrack / startStream) — a new queue starts honest.
  // Session-only: nothing persisted. Design: 2026-07-10-queue-shuffle-design.
  const [shuffleOn, setShuffleOn] = useState(false);
  const preShuffleOrderRef = useRef(null);   // track_id[] of the tail at ON time
  const resetShuffle = () => { setShuffleOn(false); preShuffleOrderRef.current = null; };

  const toggleShuffle = () => {
    const list = playerPlaylist || [];
    const idOf = (h) => ((h && h.track) ? h.track : h)?.track_id;
    const curId = playerTrack?.track_id;
    const idx = curId ? list.findIndex(h => idOf(h) === curId) : -1;
    const splitAt = idx + 1;                 // idx === -1 → shuffle the whole list
    const head = list.slice(0, splitAt);
    const tail = list.slice(splitAt);
    if (!shuffleOn) {
      preShuffleOrderRef.current = tail.map(idOf);
      const shuffled = tail.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      setPlayerPlaylist([...head, ...shuffled]);
      setShuffleOn(true);
    } else {
      const order = preShuffleOrderRef.current || [];
      const pos = new Map(order.map((id, i) => [id, i]));
      const known = tail.filter(h => pos.has(idOf(h)));
      const unknown = tail.filter(h => !pos.has(idOf(h)));
      known.sort((a, b) => pos.get(idOf(a)) - pos.get(idOf(b)));
      setPlayerPlaylist([...head, ...known, ...unknown]);
      resetShuffle();
    }
  };

  // ── Stream mode («Поток») ──────────────────────────────────────────────
  // Stateless personalized radio: GET /recommend/stream/next rebuilds the
  // taste profile from playback_events + reactions on every call. The
  // frontend keeps a 1-2 track prefetch and DROPS the unplayed tail after a
  // strong signal (like/dislike/skip) so adaptation is felt immediately.
  const streamActiveRef = useRef(false);
  const streamFetchingRef = useRef(false);
  // Reactive mirror of streamActiveRef for the UI (the ref alone is invisible
  // to render). Drives the For-You orb's play/pause icon + animation: the orb
  // must know whether the live queue IS the wave. Kept in lockstep with the ref
  // at every write-site below.
  const [streamActive, setStreamActive] = useState(false);
  // «Подстроились под твой вайб»: the latest chunk's session_adaptation
  // payload (null when session contributions are indistinguishable — the
  // server only sends it after a fire/replay, never for uniform listening).
  const [streamAdapt, setStreamAdapt] = useState(null);

  const fetchStreamChunk = async ({ excludeQueue = true } = {}) => {
    if (streamFetchingRef.current) return [];
    streamFetchingRef.current = true;
    try {
      // exclude_ids carries ONLY the prefetch buffer (tracks already issued but
      // not yet played — the stateless-gap cover). Anti-repeat of already-PLAYED
      // tracks is now owned by the server's round floor; we deliberately no longer
      // send the last-N played here — that client-side hard list used to re-create
      // the «Поток пуст» exhaustion on a long session (design 2026-06-14).
      const exclude = new Set();
      if (excludeQueue) {
        playerPlaylist.forEach(h => {
          const t = (h && h.track) ? h.track : h;
          if (t && t.track_id) exclude.add(t.track_id);
        });
      }
      const ex = Array.from(exclude).slice(0, 50).join(',');
      const data = await apiFetch(
        `/recommend/stream/next?session_id=${encodeURIComponent(getSessionId())}&n=3` +
        (ex ? `&exclude_ids=${encodeURIComponent(ex)}` : '')
      );
      setStreamAdapt(data.session_adaptation || null);
      return (data.tracks || []).map(t => ({
        track: t, score: t.score || 0, matched_on: 'stream', _stream: true, _pool: t.pool,
      }));
    } catch (e) {
      console.warn('[stream] chunk fetch failed', e);
      return [];
    } finally {
      streamFetchingRef.current = false;
    }
  };

  const startStream = async () => {
    streamActiveRef.current = true;
    setStreamActive(true);
    resetShuffle();   // the wave owns its order — shuffle can't apply to it
    const hits = await fetchStreamChunk({ excludeQueue: false });
    if (!hits.length) {
      streamActiveRef.current = false;
      setStreamActive(false);
      setStreamAdapt(null);
      showToast(lang === 'ru' ? 'Библиотека пуста — добавьте музыку' : 'Library is empty — add your music');
      return;
    }
    setPlayerPlaylist(hits);
    setPlayerTrack(hits[0].track);
    setSection('player');
  };

  // Strong signal during stream playback: the prefetched tail was chosen by
  // the PRE-signal profile — drop it (keeping one runway track on skip so the
  // jump is instant) and refetch with the updated profile.
  const handleStreamSignal = async (kind, track) => {
    if (!streamActiveRef.current || !track) return;
    setPlayerPlaylist(prev => {
      const idx = prev.findIndex(h => ((h && h.track) ? h.track : h).track_id === track.track_id);
      if (idx < 0) return prev;
      return prev.slice(0, kind === 'skip' ? idx + 2 : idx + 1);
    });
    const fresh = await fetchStreamChunk();
    if (fresh.length && streamActiveRef.current) {
      setPlayerPlaylist(prev => [...prev, ...fresh]);
    }
  };

  // When the current playlist runs out, this fetches similar tracks from the
  // backend and appends them. Called by PlayerSection's 'ended' handler when
  // there is no next track left in playerPlaylist.
  const ensureAutoplayQueue = async (seedTrack) => {
    // Stream mode owns the queue: exhaustion means the prefetch missed (e.g.
    // a failed fetch) — top up from the stream, not from CLAP-autoplay.
    if (streamActiveRef.current) {
      const fresh = await fetchStreamChunk();
      if (fresh.length && streamActiveRef.current) {
        setPlayerPlaylist(prev => [...prev, ...fresh]);
        return fresh;
      }
      return [];
    }
    if (!seedTrack?.track_id) return [];
    const excluded = Array.from(playedTrackIdsRef.current).slice(-200).join(',');
    try {
      const data = await apiFetch(
        `/recommend/autoplay-queue` +
        `?seed_track_id=${encodeURIComponent(seedTrack.track_id)}` +
        (excluded ? `&exclude_ids=${encodeURIComponent(excluded)}` : '') +
        `&limit=20`
      );
      const newTracks = data.tracks || [];
      if (newTracks.length === 0) return [];
      // PlayerSection's playlist is HIT[]-shaped ({track, score, matched_on}),
      // so wrap each fetched TrackMetadata accordingly.
      const newHits = newTracks.map(t => ({
        track: t,
        score: 0,
        matched_on: 'audio',
        _autoplay: true,
      }));
      setPlayerPlaylist(prev => [...prev, ...newHits]);
      return newHits;
    } catch (e) {
      console.warn('[autoplay] failed to fetch queue', e);
      return [];
    }
  };

  // Auto-advance when a track ends while the user is NOT on the player screen
  // (PlayerSection's own 'ended' listener only attaches when section==='player').
  const advancePlaybackOffPlayer = async () => {
    const list = playerPlaylist;
    const cur = playerTrack;
    if (!cur) return;
    const idx = list.findIndex(h => ((h && h.track) ? h.track : h).track_id === cur.track_id);
    const nextHit = idx >= 0 ? list[idx + 1] : null;
    if (nextHit) {
      const t = (nextHit.track) ? nextHit.track : nextHit;
      // Synchronous src+play inside the 'ended' event — see setSrc for why
      // (background-tab timers are throttled/frozen once audio stops).
      audio.setSrc(buildStreamUrl(t.track_id), { trackId: t.track_id, noInfluence: !!nextHit._noInfluence }, { autoplay: true });
      handleTrackChange(t);
      return;
    }
    // Exhausted: top up, then advance to the first fresh track.
    let fresh = [];
    if (streamActiveRef.current) {
      fresh = await fetchStreamChunk();
      if (fresh.length) setPlayerPlaylist(prev => [...prev, ...fresh]);
    } else {
      fresh = (await ensureAutoplayQueue(cur)) || [];
    }
    const firstFresh = fresh[0];
    if (firstFresh) {
      const t = firstFresh.track ? firstFresh.track : firstFresh;
      audio.setSrc(buildStreamUrl(t.track_id), { trackId: t.track_id, noInfluence: !!firstFresh._noInfluence }, { autoplay: true });
      handleTrackChange(t);
    }
  };

  // Battery: pause every CSS animation while the tab is hidden / screen is
  // off (audio keeps playing). The CSS rule lives on html.app-hidden.
  useEffect(() => {
    const onVis = () => document.documentElement.classList.toggle('app-hidden', document.hidden);
    document.addEventListener('visibilitychange', onVis);
    onVis();
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  // Off-player prev (mini-player button): a plain index step back — no queue
  // top-up semantics needed at the head of the list.
  const stepBackOffPlayer = () => {
    const list = playerPlaylist;
    const cur = playerTrack;
    if (!cur) return;
    const idx = list.findIndex(h => ((h && h.track) ? h.track : h).track_id === cur.track_id);
    const prevHit = idx > 0 ? list[idx - 1] : null;
    if (!prevHit) return;
    const t = prevHit.track ? prevHit.track : prevHit;
    audio.setSrc(buildStreamUrl(t.track_id), { trackId: t.track_id, noInfluence: !!prevHit._noInfluence }, { autoplay: true });
    handleTrackChange(t);
  };

  // Off-player auto-advance. On the player screen, PlayerSection owns the 'ended'
  // listener (it receives `audio`); everywhere else attach ours so a finished
  // track on Home / "For You" still rolls to the next one.
  useEffect(() => {
    const el = audio?.audioRef?.current;
    if (!el || section === 'player') return;
    const onEnded = () => { advancePlaybackOffPlayer(); };
    el.addEventListener('ended', onEnded);
    return () => el.removeEventListener('ended', onEnded);
  }, [section, audio?.audioRef, playerTrack?.track_id, playerPlaylist]);

  // Prefetch the NEXT queue track once the CURRENT one is fully buffered.
  // Gating on canplaythrough/HAVE_ENOUGH_DATA means a starving stream on a
  // slow connection keeps the whole pipe — prefetch only spends the idle
  // bandwidth after the current track is safe. Runs at App level so it works
  // on every screen (PlayerSection may be unmounted on Home).
  useEffect(() => {
    const el = audio?.audioRef?.current;
    if (!el || !playerTrack?.track_id) return;
    const list = playerPlaylist || [];
    const idx = list.findIndex(h => ((h && h.track) ? h.track : h).track_id === playerTrack.track_id);
    const nextHit = idx >= 0 ? list[idx + 1] : null;
    const nextId = nextHit ? ((nextHit.track) ? nextHit.track : nextHit).track_id : null;
    if (!nextId) return;
    let fired = false;
    const kick = () => { if (!fired) { fired = true; prefetchNextTrack(nextId); } };
    if (el.readyState >= 4) { kick(); return; }   // HAVE_ENOUGH_DATA already
    el.addEventListener('canplaythrough', kick);
    return () => el.removeEventListener('canplaythrough', kick);
  }, [playerTrack?.track_id, playerPlaylist, audio?.audioRef]);

  // Warm up the CURRENT track when it streams over the network (not from a
  // prefetched blob:). Kick on canplaythrough (healthy stream, idle bandwidth)
  // OR after 5s of wall time — a starving stream may never reach
  // canplaythrough, and that's exactly the case that needs the full download.
  // 'waiting' mid-play + completed warmup Blob → hot-swap src in place.
  useEffect(() => {
    const el = audio?.audioRef?.current;
    const tid = playerTrack?.track_id;
    if (!el || !tid) return;
    if ((el.currentSrc || el.src || '').startsWith('blob:')) return;
    let fired = false;
    const kick = () => { if (!fired) { fired = true; warmupCurrentTrack(tid); } };
    const timer = setTimeout(kick, 5000);
    if (el.readyState >= 4) kick();
    else el.addEventListener('canplaythrough', kick);
    const onWaiting = () => {
      // Ignore the initial load ('waiting' before playback ever started).
      if (!el.currentTime) return;
      const url = takeWarmupBlob(tid);
      if (url) audio.hotSwapSrc(url);
    };
    el.addEventListener('waiting', onWaiting);
    return () => {
      clearTimeout(timer);
      el.removeEventListener('canplaythrough', kick);
      el.removeEventListener('waiting', onWaiting);
    };
  }, [playerTrack?.track_id, audio?.audioRef]);

  // Stats for landing
  useEffect(() => {
    if (appState !== 'ready') return;
    const col = ``;
    apiFetch(`/library/stats${col}`).then(setStats).catch(()=>{});
  }, [appState]);

  const handleTheme = () => {
    setThemeAnim(true);
    setTimeout(() => setThemeAnim(false), 380);
    setDark(d => { const next = !d; localStorage.setItem('musix_theme', next?'dark':'light'); return next; });
  };
  const handleLang = (l) => { setLang(l); localStorage.setItem('musix_lang', l); };

  // Boot state screens
  if (appState === 'checking') return (
    <div style={{ width:'100vw', height:'100vh', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
      background: isDark ? '#0a0a0e' : '#f2f1f6', gap:'18px' }}>
      <BrandMark size={50} isDark={isDark} />
      <Spinner size={20} />
    </div>
  );
  if (appState === 'no-qdrant') return <NoQdrantScreen isDark={isDark} lang={lang} />;
  if (appState === 'onboarding') return (
    <OnboardingScreen isDark={isDark} lang={lang} onLang={handleLang} onTheme={handleTheme}
      indexingJob={indexingJob}
      onDone={() => { loadCollections(); setAppState('ready'); }} />
  );

  const sectionMap = {
    search:    SearchSection,
    recommend: RecommendSection,
    library:   LibrarySection,
    player:    PlayerSection,
    artist:    ArtistAtlasSection,
  };
  const ActiveSection = sectionMap[section];

  return (
    <div className={`app-shell${themeAnim ? ' theme-transition theme-fade' : ''}`} style={{
      display:'flex', flexDirection: isMobile ? 'column' : 'row',
      width:'100vw', overflow:'hidden', background: c.bg,
      position: 'relative',
    }}>
      {/* Full-bleed ambient wash for the player view — sits behind the
          transparent floating nav AND the section content so the blurred-cover
          glow no longer cuts off at the rail's edge. */}
      {section === 'player' && <PlayerAmbient track={playerTrack} isDark={isDark} />}
      {section === 'search' && <SearchAmbient isDark={isDark} />}
      <div className="app-main-area" style={{
        flex:1, minWidth:0, minHeight:0, display:'flex', position:'relative',
      }}>
      {/* Home stays permanently mounted too (same visibility-toggle pattern as
          sectionMap below) — it used to fully unmount on nav-away, which threw
          out the wave/vibes state and re-fired their fetches (LLM call
          included) on every return to the home page. */}
      <div style={{
        display: 'flex',
        flex: section === 'home' ? 1 : 0,
        width: section === 'home' ? 'auto' : 0,
        minWidth: 0,
        flexDirection: 'column',
        overflow: 'hidden',
        position: section === 'home' ? 'relative' : 'absolute',
        visibility: section === 'home' ? 'visible' : 'hidden',
      }}>
        <LandingScreen
          isDark={isDark} lang={lang}
          onLang={handleLang} onTheme={handleTheme} onSettings={() => setSettingsOpen(true)}
          onNav={(id) => setSection(id)}

          hasLibrary={userPoints > 0}
          stats={stats}
          playerTrack={playerTrack}
          playerPlaylist={playerPlaylist}
          onTrackChange={handleHomeTrackChange}
          onPlayTrack={handlePlayTrack}
          onStartStream={startStream}
          streamActive={streamActive}
          audio={audio}
          navigateToArtist={navigateToArtist}
          onOpenSpotlight={() => setSpotlightOpen(true)}
          onSearchLyrics={(q) => handoffToSearch(q, 'auto')}
          aiActive={!!(aiStatus && aiStatus.aiActive)}
        />
      </div>
      <Fragment>
          {!isMobile && section !== 'home' && (
          <FloatingIconNav
            section={section}
            onNav={setSection}
            isDark={isDark}
            lang={lang}
            onSettings={() => setSettingsOpen(true)}
            currentTrack={playerTrack}
            audio={audio}
            playlist={playerPlaylist}
            onTrackChange={handleTrackChange}
          />
          )}
          {/* Render all sections; visibility toggled (NOT display:none) to preserve audio across navigation */}
          {Object.entries(sectionMap).map(([id, Comp]) => (
            <div key={id} style={{
              display: 'flex',
              flex: section === id ? 1 : 0,
              width: section === id ? 'auto' : 0,
              minWidth: 0,
              flexDirection: 'column',
              overflow: 'hidden',
              position: section === id ? 'relative' : 'absolute',
              visibility: section === id ? 'visible' : 'hidden',
              animation: section === id ? 'fadeIn 0.3s ease' : 'none',
            }}>
              <Comp
                isDark={isDark} lang={lang}
                onPlayTrack={handlePlayTrack}
                playerTrack={playerTrack}
                onTrackChange={id === 'player' ? handleTrackChange : undefined}
                onRequestAutoplay={id === 'player' ? ensureAutoplayQueue : undefined}
                onStreamSignal={id === 'player' ? handleStreamSignal : undefined}
                onStartStream={id === 'recommend' ? startStream : undefined}
                audio={id === 'player' ? audio : undefined}
                initialPlaylist={id === 'player' ? playerPlaylist : undefined}
                initialTrack={id === 'player' ? playerTrack : undefined}
                visible={section === id}
                lyricsMode={id === 'player' ? lyricsMode : false}
                onToggleLyrics={id === 'player' ? () => setLyricsMode(m => !m) : undefined}
                onCloseLyrics={id === 'player' ? () => setLyricsMode(false) : undefined}
                showToast={id === 'player' ? showToast : undefined}
                artistSlug={id === 'artist' ? activeArtistSlug : undefined}
                audioPlaying={id === 'artist' ? !!(audio && audio.isPlaying) : undefined}
                navigateToArtist={navigateToArtist}
                aiStatus={aiStatus}
                onNav={setSection}
                onAddToPlaylist={openAddToPlaylist}
                searchHandoff={id === 'search' ? searchHandoff : undefined}
                playlistsListing={appPlaylists}
                onQueueNext={id === 'player' ? handleQueueNext : undefined}
                onReorderQueue={id === 'player' ? setPlayerPlaylist : undefined}
                shuffleOn={id === 'player' ? shuffleOn : undefined}
                onToggleShuffle={id === 'player' ? toggleShuffle : undefined}
                streamActive={id === 'player' ? streamActive : undefined}
                streamAdapt={id === 'player' ? streamAdapt : undefined}
              />
            </div>
          ))}
        </Fragment>
        {isMobile && section === 'player' && (
          <button onClick={closeMobilePlayer}
            aria-label={lang==='ru'?'Свернуть плеер':'Minimize player'}
            title={lang==='ru'?'Свернуть':'Close'} style={{
              position:'absolute', top:'calc(env(safe-area-inset-top, 0px) + 10px)', left:12, zIndex:50,
              width:40, height:40, borderRadius:20, display:'grid', placeItems:'center',
              background: isDark ? 'rgba(20,20,26,0.55)' : 'rgba(255,255,255,0.62)',
              backdropFilter:'blur(10px)', WebkitBackdropFilter:'blur(10px)',
              border:`1px solid ${isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}`,
              color: c.text, cursor:'pointer',
            }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6"/></svg>
          </button>
        )}
      </div>
      {isMobile && section !== 'player' && playerTrack && (
        <MiniPlayerBar track={playerTrack} audio={audio} isDark={isDark} lang={lang} onOpen={openMobilePlayer}
          onPrev={stepBackOffPlayer} onNext={advancePlaybackOffPlayer} onAddToPlaylist={openAddToPlaylist} />
      )}
      {isMobile && section !== 'player' && (
        <BottomTabBar section={section} onNav={setSection} isDark={isDark} lang={lang} />
      )}

      {settingsOpen && (
        <SettingsPanel isDark={isDark} lang={lang}
          onClose={() => setSettingsOpen(false)}
          onCollectionsUpdate={loadCollections}
          indexingJob={indexingJob}

          aiStatus={aiStatus}
          onTheme={handleTheme}
          onLang={handleLang}
          collections={collections}
          userPoints={userPoints}
          onLogout={onLogout}
          instanceMode={instanceMode}
          showToast={showToast} />
      )}

      {/* Background indexing indicator: the job started in Settings (or was
          resumed after a reload) keeps reporting while the panel is closed. */}
      {indexingJob.status === 'running' && !settingsOpen && (
        <IndexingStatusPill isDark={isDark} lang={lang}
          stepStatus={indexingJob.stepStatus} stageProgress={indexingJob.stageProgress}
          onClick={() => setSettingsOpen(true)} />
      )}

      {toast && (
        <div
          role="status"
          aria-live="polite"
          style={{
            position: 'fixed',
            top: 20,
            right: 20,
            zIndex: 9999,
            padding: '10px 16px',
            borderRadius: 10,
            background: isDark
              ? 'linear-gradient(180deg, rgba(30,30,38,0.95), rgba(22,22,28,0.95))'
              : 'linear-gradient(180deg, rgba(255,255,255,0.95), rgba(245,244,250,0.95))',
            color: isDark ? '#fff' : '#161620',
            border: `1px solid ${isDark ? 'rgba(255,255,255,0.10)' : 'rgba(22,22,32,0.10)'}`,
            boxShadow: isDark
              ? '0 12px 30px rgba(0,0,0,0.5)'
              : '0 12px 30px rgba(40,30,60,0.18)',
            fontSize: 13,
            maxWidth: 320,
            animation: 'toastIn 220ms cubic-bezier(0.22, 0.9, 0.3, 1)',
            backdropFilter: 'blur(12px) saturate(1.1)',
            WebkitBackdropFilter: 'blur(12px) saturate(1.1)',
          }}
        >
          {toast.message}
        </div>
      )}

      {/* Root-level audio — survives navigation since it's never unmounted or
          hidden. Deliberately UNCONTROLLED: useAudioPlayer.setSrc assigns
          el.src directly so track switches happen synchronously inside the
          'ended' handler (a controlled src waited for a React commit — with
          the screen off mobile Chrome froze the page in that silent gap and
          background playback died at every track change). */}
      <audio
        ref={audio.initAudio}
        // "auto" (was "metadata"): on a slow connection "metadata" delayed the
        // real download until play() — the user stared at the buffering veil.
        // Let the browser buffer ahead; the next-track blob prefetch above is
        // gated on canplaythrough so the two never compete.
        preload="auto"
        crossOrigin="anonymous"
        style={{ position:'absolute', width:0, height:0, pointerEvents:'none' }}
      />

      {/* Plan 19 follow-up: App-level AddToPlaylistPopover shared across all sections */}
      {addToPopoverInfo && (
        <AddToPlaylistPopover
          trackId={addToPopoverInfo.trackId}
          anchor={addToPopoverInfo.anchor}
          onClose={closeAddToPlaylist}
          listing={appPlaylists}
          lang={lang}
        />
      )}

      {/* Spotlight find-and-play — global overlay (🔍 on the landing, ⌘K anywhere).
          z 1500 keeps it under the AddToPlaylist popover (z 2000) so «+» works
          without leaving the spotlight. */}
      <SpotlightSearch open={spotlightOpen} onClose={() => setSpotlightOpen(false)}
        isDark={isDark} lang={lang}
        onPlayTrack={handlePlayTrack}
        onAddToPlaylist={openAddToPlaylist}
        onMore={(q) => handoffToSearch(q, 'grid')} />
    </div>
  );
}

// ─── First-Run Setup Wizard ──────────────────────────────────────────────────
// Spec: docs/superpowers/specs/2026-06-10-first-run-setup-wizard-design.md
// Rendered by Root when GET /instance/config → 404 (no owner yet). The mode
// step is the single point that calls POST /instance/setup; success stores the
// returned JWT and the wizard continues authenticated.

// Slider tiers — must mirror TEXT_MODELS in app/resources/model_registry.py.
// model:null = omit text_model (backend default jina-small).
const WIZ_TIERS = [
  { model: null,                            dim: 512,
    ru: 'Скорость', en: 'Speed',
    noteRu: 'только английский',         noteEn: 'English only' },
  { model: 'intfloat/multilingual-e5-base', dim: 768,
    ru: 'Баланс',   en: 'Balanced',
    noteRu: 'мультиязычный',             noteEn: 'multilingual' },
  { model: 'Qwen/Qwen3-Embedding-0.6B',     dim: 1024,
    ru: 'Качество', en: 'Quality',
    noteRu: 'мультиязычный · медленнее', noteEn: 'multilingual · slower' },
];

// Step order branches on the chosen mode. Server: the owner sets instance-wide
// AI + embedding policy (no music — members bring their own libraries later).
// Sharing: single user adds music, optionally AI, then indexes inline.
// The first three steps are identical in both, so stepIndex stays correct while
// mode is still null (welcome/account/mode).
const stepsSharing = (ru) => [
  // "accountMode" creates the owner AND picks the app mode — this is the only
  // place the mode is chosen (owner, first-time setup); members never see it.
  { key:'accountMode', label: ru ? 'Режим приложения' : 'App mode' },
  { key:'music',       label: ru ? 'Откуда музыка?' : 'Music source' },
  { key:'ai',          label: ru ? 'Гуру (AI)' : 'Guru (AI)' },
  { key:'indexing',    label: ru ? 'Подготовка библиотеки' : 'Preparing library' },
  // Terminal step — lights up on the wizard's "All set" screen before entering.
  { key:'done',        label: ru ? 'Готово!' : 'Done!' },
];
const stepsServer = (ru) => [
  { key:'accountMode', label: ru ? 'Режим приложения' : 'App mode' },
  { key:'policy',      label: ru ? 'Настройки системы' : 'System settings' },
];

// Stage keys come from the shared JobTracker (same SSE stream the existing
// IndexingModal/UploadIndexingWizard consume). KEY ORDER = display order, kept
// identical to IndexingProgress.stages so every indexing flow agrees: lyrics +
// audio start together, dense after lyrics, facts overlap, analysis last. The
// backend reports granular progress for every key here (including `dense` via
// the "lyrics" → DENSE callback), so all show a real X/Y bar. `metadata`
// (MusicBrainz) is skipped server-side and intentionally omitted.
const WIZ_STAGE_LABELS = {
  lyrics:   { ru: 'Тексты песен',       en: 'Lyrics' },
  audio:    { ru: 'Анализ звучания',   en: 'Sound analysis' },
  dense:    { ru: 'Подготовка поиска',  en: 'Search setup' },
  facts:    { ru: 'Факты о треках',    en: 'Track facts' },
  analysis: { ru: 'Похожие треки',   en: 'Similar tracks' },
};

// ─── Onboarding redesign: shared liquid-glass primitives ─────────────────
function DriftBackdrop({ variant }) {
  return (
    <div className={'ob-drift' + (variant === 'success' ? ' ob-drift-success' : '')} aria-hidden="true">
      <div className="ob-blob ob-blob-a"></div><div className="ob-blob ob-blob-b"></div><div className="ob-blob ob-blob-c"></div>
    </div>
  );
}
function GlassCard({ children, style, className }) {
  return <div className={'ob-glass' + (className ? ' ' + className : '')} style={style}>{children}</div>;
}
function FeatureCard({ icon, title, body, premium }) {
  return (
    <div className={'ob-feat ' + (premium ? 'ob-feat-premium' : 'ob-feat-hover')} style={{ padding:'clamp(15px, 1.3vw, 26px)' }}>
      <div style={{ fontSize:'clamp(20px, 1.7vw, 32px)', lineHeight:1 }}>{icon}</div>
      <div style={{ fontWeight:600, fontSize:'clamp(14px, 1.05vw, 20px)', margin:'10px 0 5px' }}>{title}</div>
      <div style={{ fontSize:'clamp(12.5px, 0.92vw, 16px)', lineHeight:1.45, opacity:.72 }}>{body}</div>
    </div>
  );
}
// The «Я» tile used everywhere Yandex Music appears — kept as an element so the
// welcome premium card matches the source-pick card visually.
function YandexTile({ size = '1.05em' }) {
  return (
    <span aria-hidden style={{ display:'inline-flex', width:size, height:size, borderRadius:'0.28em',
      background:'linear-gradient(135deg, #ffcc00, #ff5c5c)', color:'#1a1a1a',
      alignItems:'center', justifyContent:'center', fontWeight:800, fontSize:'0.72em', lineHeight:1 }}>Я</span>
  );
}
function WelcomeSlide({ ru, c, onStart }) {
  const feats = [
    { icon:'🎯', title: ru?'Прозрачные рекомендации':'Transparent recommendations', body: ru?'Настраивай: больше любимого или старого. Оценивай свои острова вкуса.':'Tune it: more loved or more old. Rate your taste islands.' },
    { icon:'🔎', title: ru?'Всё про песню':'Everything about a song', body: ru?'Как создавалась, о чём, что семплирует. Тыкни в строчку — гуру объяснит.':'How it was made, what it means, what it samples. Tap a line — the guru explains.' },
    { icon:'💡', title: ru?'Другая сторона артистов':'The other side of artists', body: ru?'Маколей Калкин — крёстный отец детей Майкла Джексона. И ещё сотни фактов.':"Macaulay Culkin is godfather to Michael Jackson's kids. And hundreds more facts." },
    { icon:'🌐', title: ru?'Свой сервер':'Your own server', body: ru?'Зови друзей. Библиотека без ИИ-музыки и ремиксов — песня всегда доступна.':'Invite friends. A library with no AI music or remixes — a song never disappears.' },
  ];
  const premiumFeats = [
    { icon:<YandexTile />, title: ru?'Импорт из Яндекс Музыки':'Import from Yandex Music',
      body: ru?'Лайкнутые песни и плейлисты переезжают в один клик.':'Liked songs and playlists move over in one click.' },
    { icon:'✨', title: ru?'Метаданные лучшего качества':'Better metadata quality',
      body: ru?'Тексты, обложки и информация о песнях подтягиваются в улучшенном качестве — даже для ваших собственных файлов.':'Lyrics, covers and song info arrive in higher quality — even for your own files.' },
  ];
  const gold = PREMIUM_GOLD;
  return (
    <div style={{ display:'flex', gap:'clamp(28px, 3vw, 60px)', alignItems:'center', flexWrap:'wrap' }}>
      <div style={{ flex:'1 1 340px', minWidth:290 }}>
        <div className="mono" style={{ fontSize:'clamp(11px, 0.85vw, 14px)', letterSpacing:'.26em', textTransform:'uppercase', opacity:.5 }}>{ru?'— Добро пожаловать':'— Welcome'}</div>
        <h1 className="serif" style={{ fontSize:'clamp(40px, 4vw, 78px)', lineHeight:1.04, letterSpacing:'-.02em', margin:'clamp(12px,1vw,22px) 0 clamp(10px,0.8vw,16px)' }}>{ru?'Добро пожаловать в ':'Welcome to '}<span style={{ color:'oklch(62% 0.2 275)' }}>MusiX</span>!</h1>
        <div style={{ fontSize:'clamp(17px, 1.5vw, 27px)', opacity:.65 }}>{ru?'Ваш личный музыкальный гуру':'Your personal music guru'}</div>
        <button className="ske-accent" style={{ marginTop:'clamp(22px, 1.9vw, 34px)', padding:'clamp(13px,1vw,18px) clamp(28px,2.4vw,46px)', borderRadius:12, fontSize:'clamp(14.5px, 1.1vw, 19px)', fontWeight:600, border:'none', cursor:'pointer' }} onClick={onStart}>{ru?'В мир музыки →':'Into the music →'}</button>
      </div>
      <div style={{ flex:'1 1 380px', minWidth:300 }}>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'clamp(12px, 1vw, 20px)' }}>{feats.map((f,i)=><FeatureCard key={i} icon={f.icon} title={f.title} body={f.body} />)}</div>
        <PremiumGate>
          <div style={{ display:'flex', alignItems:'center', gap:'12px', margin:'clamp(16px, 1.4vw, 26px) 0 clamp(10px, 0.9vw, 16px)' }}>
            <span style={{ flex:1, height:1, background:`linear-gradient(90deg, transparent, ${gold.replace(')', ' / 0.45)')})` }} />
            <span className="mono" style={{ fontSize:'clamp(10px, 0.78vw, 12.5px)', letterSpacing:'.22em', color:gold, whiteSpace:'nowrap' }}>
              ★ {ru?'С PREMIUM ВАМ ДОСТУПНО':'AVAILABLE WITH PREMIUM'}
            </span>
            <span style={{ flex:1, height:1, background:`linear-gradient(90deg, ${gold.replace(')', ' / 0.45)')}, transparent)` }} />
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'clamp(12px, 1vw, 20px)' }}>{premiumFeats.map((f,i)=><FeatureCard key={i} icon={f.icon} title={f.title} body={f.body} premium />)}</div>
        </PremiumGate>
      </div>
    </div>
  );
}
function SetupRail({ ru, steps, currentKey }) {
  const idx = steps.findIndex(s => s.key === currentKey);
  return (
    <div className="ob-rail" style={{ flex:'0 0 188px', display:'flex', flexDirection:'column', gap:2, position:'relative', zIndex:1 }}>
      <div className="mono" style={{ fontSize:11, letterSpacing:'.24em', textTransform:'uppercase', opacity:.5, marginBottom:12 }}>{ru?'◆ Настройка':'◆ Setup'}</div>
      {steps.filter(s => s.key !== 'welcome').map((s, i) => {
        const my = steps.findIndex(x => x.key === s.key);
        const state = my < idx ? 'done' : (my === idx ? 'now' : 'todo');
        return (
          <div key={s.key} style={{ display:'flex', alignItems:'center', gap:12, padding:11, borderRadius:10, background: state==='now'?'rgba(154,133,255,.12)':'transparent' }}>
            <span style={{ width:28, height:28, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontFamily:'Georgia,serif', fontWeight:600, fontSize:12,
              background: state==='done'?'linear-gradient(180deg,oklch(67% .18 270),oklch(54% .22 280))':'transparent',
              color: state==='done'?'#fff':(state==='now'?'oklch(70% .16 270)':'rgba(128,128,128,.6)'),
              boxShadow: state==='now'?'0 0 0 2px oklch(62% .2 275),0 0 12px oklch(62% .2 275 / .5)':(state==='todo'?'inset 0 0 0 1px rgba(128,128,128,.3)':'none') }}>{state==='done'?'✓':(i+1)}</span>
            <span style={{ fontSize:13.5, fontWeight: state==='now'?600:400, opacity: state==='todo'?.5:1 }}>{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}
function ModeCard({ sel, onClick, title, body, note }) {
  return (
    <div onClick={onClick} className="ob-hoverlift" style={{ cursor:'pointer', background:'var(--ob-card-bg)', borderRadius:12, padding:13,
        boxShadow: sel?'inset 0 0 0 2px oklch(62% .2 275),0 6px 18px rgba(124,92,255,.3)':'inset 0 0 0 1px var(--ob-card-edge),0 3px 10px rgba(0,0,0,.18)' }}>
      <div style={{ fontWeight:600, fontSize:13 }}>{title}</div>
      <div style={{ fontSize:11.5, opacity:.7, lineHeight:1.4, marginTop:3 }}>{body}</div>
      {note && <div style={{ fontSize:10.5, color:'#cdbfff', background:'rgba(139,116,255,.12)', border:'1px solid rgba(139,116,255,.3)', borderRadius:8, padding:'6px 9px', marginTop:8 }}>{note}</div>}
    </div>
  );
}

function SetupWizard({ onComplete }) {
  const [isDark, setDark] = useState(() => (localStorage.getItem('musix_theme') || 'dark') === 'dark');
  const [lang, setLang]   = useState(() => localStorage.getItem('musix_lang') || 'ru');
  const c = useColors(isDark);
  const ru = lang === 'ru';

  const [step, setStep] = useState('welcome');

  // account
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [accError, setAccError] = useState('');

  // mode + the /instance/setup call
  const [mode, setMode]               = useState(null);   // 'sharing' | 'server'
  const [setupBusy, setSetupBusy]     = useState(false);
  const [setupError, setSetupError]   = useState('');
  const [alreadyInit, setAlreadyInit] = useState(false);

  // music — folder (sharing) / uploads (server) + quality slider.
  // jobId: async indexing job to watch; syncDoneCount: sharing-mode small
  // libraries can finish inline (no job_id) — remember the count instead.
  const [tier, setTier]                     = useState(1);   // default: Balanced
  const [folderPath, setFolderPath]         = useState('');
  const [picking, setPicking]               = useState(false);
  const [files, setFiles]                   = useState([]);
  const [uploadProgress, setUploadProgress] = useState([]);
  const [musicBusy, setMusicBusy]           = useState(false);
  const [musicError, setMusicError]         = useState('');
  const [jobId, setJobId]                   = useState(null);
  const [syncDoneCount, setSyncDoneCount]   = useState(null);

  // ai — OpenAI-compatible endpoint, probed via existing POST /system/llm-status.
  const [llmUrl, setLlmUrl]           = useState(() => localStorage.getItem('llm_base_url') || '');
  const [llmModel, setLlmModel]       = useState(() => localStorage.getItem('llm_model') || '');
  const [llmKey, setLlmKey]           = useState('');     // API key — server-side secret, never localStorage
  const [probe, setProbe]             = useState(null);   // null | {available, error}
  const [probing, setProbing]         = useState(false);
  const [aiBusy, setAiBusy]           = useState(false);
  const [aiConnected, setAiConnected] = useState(false);

  // policy — server-mode instance settings step (PATCH /instance/settings)
  const [policyBusy, setPolicyBusy]   = useState(false);
  const [policyError, setPolicyError] = useState('');

  // indexing — mirrors UploadIndexingWizard's SSE consumption.
  const [stepStatus, setStepStatus]       = useState({});
  const [stageProgress, setStageProgress] = useState({});
  const [etas, setEtas]                   = useState({});
  const etaRef                            = useRef({});
  const [indexError, setIndexError]       = useState(null);
  const [indexDone, setIndexDone]         = useState(false);
  // AI enrichment as indexing stages (sharing: opt-out; runs the 3 AI tasks
  // and polls /library/ai-index/status to fill their bars before finishing).
  const [enrichWithAi, setEnrichWithAi]   = useState(true);
  const [aiStatus, setAiStatus]           = useState(null);
  const [aiRunning, setAiRunning]         = useState(false);
  const [aiDone, setAiDone]               = useState(false);
  // Server-mode upload jobs run the guru phase server-side and stream its
  // progress over the same SSE (`ai_stages`). When present, it both renders
  // live bars and tells the client NOT to re-run the tasks after completion.
  const [sseAiStages, setSseAiStages]     = useState(null);

  const handleTheme = () => setDark(d => {
    const next = !d; localStorage.setItem('musix_theme', next ? 'dark' : 'light'); return next;
  });
  const handleLang = (l) => { setLang(l); localStorage.setItem('musix_lang', l); };

  const railSteps = mode === 'server' ? stepsServer(ru) : stepsSharing(ru);
  // Rail highlight: the indexing step splits visually into "Подготовка библиотеки" (running)
  // and "Готово" (the wizard's "All set" screen, shown once everything finished).
  const indexingFinished = indexDone && (!(aiConnected && enrichWithAi) || aiDone);
  const railCurrentKey = step === 'indexing' && indexingFinished ? 'done' : step;

  // Server mode: owner configures instance policy instead of bringing music.
  const advanceFromSetup = () => setStep(mode === 'server' ? 'policy' : 'music');
  const advanceFromMusic = () => setStep('ai');
  const advanceFromAi = () => {
    // Something to watch? → progress screen. Music skipped? → done.
    if (jobId || syncDoneCount !== null) setStep('indexing');
    else onComplete({ mode });
  };

  const probeLlm = async () => {
    setProbing(true); setProbe(null);
    try {
      // apiFetch (not raw fetch): /system/llm-status is behind the JWT gate.
      // The wizard reaches this step AFTER /instance/setup stored the token,
      // so the header is available — a raw fetch here returned the 401 body
      // {"detail": "missing or invalid Authorization header"} as the "probe".
      const res = await apiFetch('/system/llm-status', {
        method:'POST',
        body: JSON.stringify({ base_url: llmUrl.trim() || null, model: llmModel.trim() || null }),
      });
      setProbe(res);
    } catch (e) {
      setProbe({ available:false, error:String(e.message || e) });
    } finally { setProbing(false); }
  };

  const connectAi = async () => {
    if (aiBusy) return;
    setAiBusy(true);
    try {
      localStorage.setItem('llm_base_url', llmUrl.trim());
      localStorage.setItem('llm_model', llmModel.trim());
      await apiFetch('/library/ai-enabled', { method:'PATCH', body: JSON.stringify({ enabled:true }) });
      // Unify with server mode: persist AI config to instance settings so the
      // backend resolver (not per-browser localStorage) is the source of truth.
      // Sharing has a single user — the owner — so this PATCH is authorized.
      await apiFetch('/instance/settings', { method:'PATCH', body: JSON.stringify({
        llm_base_url: llmUrl.trim(),
        llm_model: llmModel.trim(),
        ...(llmKey.trim() ? { llm_api_key: llmKey.trim() } : {}),
        embed_model: WIZ_TIERS[tier].model,
        ai_enabled: true,
      }) });
      setAiConnected(true);
      advanceFromAi();
    } catch (e) {
      setProbe({ available:false, error:String(e.message || e) });
    } finally { setAiBusy(false); }
  };

  // Server mode: save instance-wide policy (embedding + optional AI) and finish.
  // No music here — members upload under their own accounts; the owner can index
  // their own library afterwards via the normal upload flow.
  const submitPolicy = async (enableAi) => {
    if (policyBusy) return;
    setPolicyBusy(true); setPolicyError('');
    try {
      const body = { embed_model: WIZ_TIERS[tier].model, clap_enabled: true };
      if (enableAi) {
        body.llm_base_url = llmUrl.trim();
        body.llm_model = llmModel.trim();
        if (llmKey.trim()) body.llm_api_key = llmKey.trim();
        body.ai_enabled = true;
      } else {
        body.ai_enabled = false;
      }
      await apiFetch('/instance/settings', { method:'PATCH', body: JSON.stringify(body) });
      applyTierToLocalStorage();   // owner's own search session uses the same model
      onComplete({ mode });
    } catch (e) {
      setPolicyError(String(e.message || e));
    } finally { setPolicyBusy(false); }
  };

  const retryMusic = () => {
    setIndexError(null); setIndexDone(false);
    setStepStatus({}); setStageProgress({});
    setJobId(null); setSyncDoneCount(null);
    setStep('music');
  };

  useEffect(() => {
    if (step !== 'indexing') return;
    if (!jobId) {
      // Sharing-mode small library finished inline — nothing to stream.
      if (syncDoneCount !== null) setIndexDone(true);
      return;
    }
    const evt = new EventSource(`${API}/index/progress/${jobId}`);
    evt.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.stages) {
          const statusMap = { completed:'done', failed:'failed', running:'running', pending:'pending' };
          setStepStatus(prev => {
            const next = { ...prev };
            for (const [k, v] of Object.entries(data.stages)) next[k] = statusMap[v.status] || v.status || 'pending';
            return next;
          });
          setStageProgress(prev => {
            const next = { ...prev };
            for (const [k, v] of Object.entries(data.stages)) {
              next[k] = {
                current: v.current ?? prev[k]?.current ?? 0,
                total:   v.total   ?? prev[k]?.total   ?? 0,
              };
            }
            return next;
          });
          setEtas(prev => ({ ...prev, ...computeStageEtas(data.stages, etaRef.current, statusMap, ru, Date.now()) }));
        }
        if (data.ai_stages) setSseAiStages(data.ai_stages);
        if (data.overall_status === 'completed') { evt.close(); setIndexDone(true); }
        else if (data.overall_status === 'failed') { evt.close(); setIndexError(data.error || data.message || 'failed'); }
      } catch {}
    };
    // No evt.close() in onerror: EventSource auto-reconnects (spec §5).
    return () => evt.close();
  }, [step, jobId, syncDoneCount]);

  // After core indexing completes: if the owner connected AI and kept enrichment
  // on, run the three AI tasks as visible stages (poll their status) before the
  // finish. shouldEnrich=false → finish immediately. Member mode (T3) reuses the
  // same idea driven by ai_available.
  const shouldEnrich = aiConnected && enrichWithAi;
  useEffect(() => {
    if (!indexDone) return;
    if (!shouldEnrich) { setAiDone(true); return; }
    // The backend already ran (and streamed) the guru phase inside the job —
    // re-running the tasks client-side would just re-poll a finished state.
    if (sseAiStages) { setAiDone(true); return; }
    if (aiRunning || aiDone) return;
    let cancelled = false;
    const tasks = ['artist_bio', 'refined_facts', 'sonic_vibe'];
    const terminal = (s) => s === 'done' || s === 'failed' || s === 'cancelled';
    (async () => {
      setAiRunning(true);
      for (const task of tasks) {
        try {
          await apiFetch(`/library/ai-index/${task}`, { method:'POST', body: JSON.stringify({
            lang,
            llm_base_url: localStorage.getItem('llm_base_url') || undefined,
            llm_model: localStorage.getItem('llm_model') || undefined,
          }) });
        } catch (e) { /* 409 already-running etc. — non-fatal */ }
      }
      for (let i = 0; i < 800 && !cancelled; i++) {
        let st = null;
        try { st = await apiFetch('/library/ai-index/status'); } catch (e) {}
        if (st && !cancelled) setAiStatus(st);
        if (st && tasks.every(t => st[t] && terminal(st[t].status))) break;
        await new Promise(r => setTimeout(r, 1500));
      }
      if (!cancelled) { setAiRunning(false); setAiDone(true); }
    })();
    return () => { cancelled = true; };
  }, [indexDone, shouldEnrich, sseAiStages]);

  const applyTierToLocalStorage = () => {
    // Mirror the Settings panel behavior: search must use the same model the
    // collection is indexed with; server also pins it in collection_settings
    // at the end of the job.
    const m = WIZ_TIERS[tier].model;
    if (m) localStorage.setItem('text_model', m);
    else localStorage.removeItem('text_model');
  };

  const handlePickFolder = async () => {
    setPicking(true); setMusicError('');
    try {
      const res = await apiFetch('/library/pick-folder');
      if (res.path) setFolderPath(res.path);
    } catch (e) { setMusicError(e.message); }
    finally { setPicking(false); }
  };

  const startIndexingSharing = async () => {
    if (!folderPath.trim() || musicBusy) return;
    setMusicBusy(true); setMusicError('');
    applyTierToLocalStorage();
    try {
      const body = { folder_path: folderPath.trim() };
      if (WIZ_TIERS[tier].model) body.text_model = WIZ_TIERS[tier].model;
      const res = await apiFetch('/library/index', { method:'POST', body: JSON.stringify(body) });
      if (res.status === 'failed') throw new Error(res.message || 'indexing failed');
      if (res.job_id) setJobId(res.job_id);
      else setSyncDoneCount(res.count || 0);
      advanceFromMusic();
    } catch (e) {
      setMusicError(/HTTP 503/.test(String(e.message))
        ? (ru ? 'Поиск временно недоступен — запустите docker-compose up -d и повторите.'
              : 'Search is temporarily unavailable — run docker-compose up -d and retry.')
        : e.message);
    } finally { setMusicBusy(false); }
  };

  const onWizPick = (e) => {
    const picked = Array.from(e.target.files || []);
    setFiles(picked);
    setUploadProgress(picked.map(f => ({ name: f.name, status: 'queued' })));
  };
  const onWizDrop = (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files || []).filter(
      f => /\.(flac|mp3|m4a|aac|ogg|wav|opus)$/i.test(f.name),
    );
    setFiles(dropped);
    setUploadProgress(dropped.map(f => ({ name: f.name, status: 'queued' })));
  };

  const startIndexingServer = async () => {
    if (!files.length || musicBusy) return;
    setMusicBusy(true); setMusicError('');
    applyTierToLocalStorage();
    try {
      // Sequential like ServerOnboardingScreen — keeps server memory bounded.
      const ids = [];
      for (let i = 0; i < files.length; i++) {
        setUploadProgress(p => p.map((row, j) => j === i ? { ...row, status:'uploading' } : row));
        try {
          const fd = new FormData();
          fd.append('file', files[i], files[i].name);
          const res = await apiFetch('/library/upload', { method:'POST', body: fd });
          ids.push(res.upload_id);
          setUploadProgress(p => p.map((row, j) => j === i ? { ...row, status:'done' } : row));
        } catch (e) {
          setUploadProgress(p => p.map((row, j) => j === i ? { ...row, status:'failed', error: e.message } : row));
        }
      }
      if (!ids.length) throw new Error(ru ? 'Ни один файл не загрузился' : 'No files were uploaded');
      const body = { upload_ids: ids, lang: ru ? 'ru' : 'en' };
      if (WIZ_TIERS[tier].model) body.text_model = WIZ_TIERS[tier].model;
      const res = await apiFetch('/library/upload/batch-commit', { method:'POST', body: JSON.stringify(body) });
      setJobId(res.job_id);
      advanceFromMusic();
    } catch (e) { setMusicError(e.message); }
    finally { setMusicBusy(false); }
  };

  const skipMusic = () => { setJobId(null); setSyncDoneCount(null); advanceFromMusic(); };

  const emailOk = /\S+@\S+\.\S+/.test(email.trim());
  const passOk  = password.length >= 6;

  const continueAccountMode = () => {
    if (!emailOk) { setAccError(ru ? 'Похоже, это не email' : 'That does not look like an email'); return; }
    if (!passOk)  { setAccError(ru ? 'Пароль — минимум 6 символов' : 'Password must be at least 6 characters'); return; }
    if (!mode)    { setAccError(ru ? 'Выберите, где живёт музыка' : 'Choose where your music lives'); return; }
    setAccError('');
    submitSetup();
  };

  const submitSetup = async () => {
    if (!mode || setupBusy) return;
    setSetupBusy(true); setSetupError('');
    try {
      const res = await fetch(API + '/instance/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password, mode }),
      });
      if (res.status === 409) { setAlreadyInit(true); return; }
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();   // { token, user }
      setStoredAuth(data);
      advanceFromSetup();
    } catch (err) {
      setSetupError(String(err.message || err));
    } finally {
      setSetupBusy(false);
    }
  };

  const renderWelcome = () => (
    <div style={{ position:'relative' }}>
      <DriftBackdrop />
      <GlassCard style={{ padding:'34px 32px' }}>
        <WelcomeSlide ru={ru} c={c} onStart={() => setStep('accountMode')} />
      </GlassCard>
    </div>
  );

  const renderAccountMode = () => (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      <div className="mono" style={{ fontSize:'11px', letterSpacing:'0.24em', textTransform:'uppercase', color:c.textSubtle }}>
        {ru ? 'Шаг 1' : 'Step 1'}
      </div>
      <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', margin:'7px 0 4px' }}>
        {ru ? 'Создадим ваш аккаунт' : "Let's create your account"}
      </h2>
      <p style={{ fontSize:'13px', color:c.textMuted, lineHeight:'1.5', marginBottom:'16px' }}>
        {ru ? 'Этот аккаунт станет владельцем инстанса.' : 'This account becomes the instance owner.'}
      </p>
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>EMAIL</div>
      <input type="email" autoFocus value={email} onChange={e => setEmail(e.target.value)}
        className={ske('inset', isDark)}
        style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none', color:c.text, fontSize:'15px', outline:'none', marginBottom:'14px' }} />
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>
        {ru ? 'ПАРОЛЬ · МИН. 6 СИМВОЛОВ' : 'PASSWORD · MIN 6 CHARS'}
      </div>
      <input type="password" value={password} onChange={e => setPassword(e.target.value)}
        className={ske('inset', isDark)}
        style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none', color:c.text, fontSize:'15px', outline:'none', marginBottom:'18px' }} />

      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'10px' }}>
        {ru ? 'ГДЕ ЖИВЁТ ВАША МУЗЫКА?' : 'WHERE DOES YOUR MUSIC LIVE?'}
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'11px' }}>
        <ModeCard sel={mode==='sharing'} onClick={() => setMode('sharing')}
          title={ru ? '📁 Локальная папка' : '📁 Local folder'}
          body={ru ? 'Музыка уже на этом компьютере. Индексируется на месте, один пользователь.'
                   : 'Music is already on this machine. Indexed in place, single user.'} />
        <ModeCard sel={mode==='server'} onClick={() => setMode('server')}
          title={ru ? '🌐 Сервер' : '🌐 Server'}
          body={ru ? 'Загрузка через браузер, можно звать друзей.' : 'Upload via the browser, invite friends.'}
          note={mode==='server' ? (ru ? '⚑ Этот аккаунт станет админским. Чтобы слушать музыку — позже войдёте обычным участником.'
                                       : '⚑ This account becomes the admin. To listen, sign in later as a regular member.') : null} />
      </div>
      <div style={{ fontSize:'12px', color:'#b08950', marginTop:'12px' }}>
        ⚠ {ru ? 'Выбор режима фиксируется. Поменять потом можно только скриптом миграции.'
              : 'The mode is locked in. Changing it later requires a migration script.'}
      </div>

      {(accError || setupError) && (
        <div style={{ padding:'9px 13px', marginTop:'14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{accError || setupError}</div>
      )}
      <div style={{ display:'flex', justifyContent:'flex-end', marginTop:'18px' }}>
        <button onClick={continueAccountMode} disabled={setupBusy} className="ske-accent"
          style={{ padding:'12px 24px', borderRadius:'12px', fontSize:'15px', fontWeight:'600', letterSpacing:'0.06em',
            opacity: setupBusy ? 0.5 : 1, cursor: setupBusy ? 'wait' : 'pointer',
            display:'flex', alignItems:'center', gap:'8px' }}>
          {setupBusy ? <><Spinner size={14} color="white" /> {ru ? 'СОЗДАЁМ…' : 'CREATING…'}</> : (ru ? 'Продолжить →' : 'Continue →')}
        </button>
      </div>
    </div>
  );

  // account + mode merged into renderAccountMode (above).

  const renderSlider = () => (
    <div style={{ marginTop:'20px', marginBottom:'4px' }}>
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'2px' }}>
        {ru ? 'КАЧЕСТВО ОБРАБОТКИ ТЕКСТОВ ПЕСЕН' : 'LYRICS-SEARCH QUALITY'}
      </div>
      <div style={{ fontSize:'12px', color:c.textMuted, marginBottom:'12px' }}>
        {ru ? 'Точность поиска по словам внутри песен.' : 'How precisely you can search words inside songs.'}
      </div>
      <SkeRange min={0} max={2} step={1} value={tier} animated
        onChange={setTier} disabled={musicBusy}
        accent="oklch(62% 0.2 275)" style={{ width:'100%' }}
        ariaLabel={ru ? 'Качество обработки текстов песен' : 'Lyrics-search quality'} />
      <div style={{ display:'flex', marginTop:'6px' }}>
        {WIZ_TIERS.map((t, i) => (
          <div key={i} onClick={() => !musicBusy && setTier(i)}
            style={{ flex:1, textAlign: i === 0 ? 'left' : i === 2 ? 'right' : 'center', cursor:'pointer' }}>
            <div style={{ fontSize:'13.5px', fontWeight: tier === i ? '600' : '400', color: tier === i ? 'oklch(62% 0.2 275)' : c.textMuted }}>
              {ru ? t.ru : t.en}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderMusic = () => {
    // Server mode offers BOTH paths: a host folder (owner runs the host, the
    // backend allows owner folder-indexing in server mode) or browser uploads.
    // Picked files take precedence — picking them is the more explicit action.
    const serverUsesUpload = mode === 'server' && files.length > 0;
    const canStartMusic = serverUsesUpload || !!folderPath.trim();
    return (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'6px' }}>
        {ru ? 'Добавьте музыку' : 'Add your music'}
      </h2>
      <p style={{ fontSize:'13px', color:c.textMuted, lineHeight:'1.5', marginBottom:'20px' }}>
        {mode === 'server'
          ? (ru ? 'Укажите папку с музыкой на сервере — или загрузите файлы, они будут храниться под вашим аккаунтом. FLAC, MP3, M4A.'
                : 'Point at a music folder on the server — or upload files to store them under your account. FLAC, MP3, M4A.')
          : (ru ? 'Укажите папку с музыкой на этом компьютере — добавим её в библиотеку.'
                : 'Point at a music folder on this machine — we add it to your library.')}
      </p>

      <div style={{ display:'flex', gap:'8px' }}>
        <button onClick={handlePickFolder} disabled={picking || musicBusy}
          className={ske('btn', isDark)} style={{ padding:'11px 16px', borderRadius:'10px', fontSize:'14px',
            color:c.textMuted, display:'flex', alignItems:'center', gap:'6px', flexShrink:0,
            cursor: picking || musicBusy ? 'not-allowed' : 'pointer' }}>
          {picking ? <Spinner size={12} /> : '📁'}
          {picking ? '…' : (ru ? 'Выбрать' : 'Pick')}
        </button>
        <input value={folderPath} onChange={e => setFolderPath(e.target.value)} disabled={musicBusy}
          placeholder="C:\Music"
          className={ske('inset', isDark)} style={{ flex:1, padding:'11px 13px', borderRadius:'10px', border:'none',
            color:c.text, fontSize:'15px', outline:'none', fontFamily:"'JetBrains Mono', monospace" }} />
      </div>

      {mode === 'server' && (
        <div>
          <div className="mono" style={{ textAlign:'center', fontSize:'11px', color:c.textSubtle,
            letterSpacing:'0.22em', margin:'14px 0 10px' }}>
            {ru ? '— ИЛИ ЗАГРУЗИТЕ ФАЙЛЫ —' : '— OR UPLOAD FILES —'}
          </div>
          <div onDragOver={(e) => e.preventDefault()} onDrop={onWizDrop}
            style={{ padding:'28px 20px', borderRadius:'14px', border:`2px dashed ${c.border}`, textAlign:'center' }}>
            <div style={{ fontSize:'34px', marginBottom:'8px' }}>📁</div>
            <p style={{ fontSize:'13px', color:c.textMuted, marginBottom:'10px' }}>
              {ru ? 'Перетащите аудиофайлы сюда, или' : 'Drag audio files here, or'}
            </p>
            <input type="file" multiple
              accept=".flac,.mp3,.m4a,.aac,.ogg,.wav,.opus"
              onChange={onWizPick} disabled={musicBusy}
              style={{ display:'none' }} id="wiz-picker" />
            <label htmlFor="wiz-picker" className="ske-accent"
              style={{ display:'inline-block', padding:'9px 18px', borderRadius:'10px', fontSize:'13px',
                cursor:'pointer', opacity: musicBusy ? 0.5 : 1, pointerEvents: musicBusy ? 'none' : 'auto' }}>
              {ru ? 'ВЫБРАТЬ ФАЙЛЫ' : 'PICK FILES'}
            </label>
          </div>
          {uploadProgress.length > 0 && (
            <div style={{ maxHeight:'160px', overflow:'auto', marginTop:'10px' }}>
              {uploadProgress.map((p, i) => (
                <div key={i} style={{ display:'flex', justifyContent:'space-between',
                  padding:'5px 0', borderBottom:`1px solid ${c.border}`, fontSize:'12px' }}>
                  <span style={{ overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{p.name}</span>
                  <span style={{ flexShrink:0, marginLeft:'10px',
                    color: p.status === 'done' ? 'oklch(70% 0.18 145)' : p.status === 'failed' ? c.red : c.textSubtle }}>
                    {p.status === 'done' ? '✓' : p.status === 'failed' ? `✗ ${p.error || ''}` : p.status === 'uploading' ? '…' : '·'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {renderSlider()}

      {musicError && (
        <div style={{ padding:'9px 13px', marginTop:'14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{musicError}</div>
      )}

      <button onClick={serverUsesUpload ? startIndexingServer : startIndexingSharing}
        disabled={musicBusy || !canStartMusic}
        className="ske-accent" style={{
          width:'100%', marginTop:'18px', padding:'13px 20px', borderRadius:'12px',
          fontSize:'15px', fontWeight:'600', letterSpacing:'0.08em',
          opacity: musicBusy || !canStartMusic ? 0.5 : 1,
          cursor: musicBusy ? 'wait' : 'pointer',
          display:'flex', alignItems:'center', justifyContent:'center', gap:'8px' }}>
        {musicBusy
          ? <><Spinner size={14} color="white" /> {ru ? 'ЗАПУСКАЕМ…' : 'STARTING…'}</>
          : (ru ? '▶ ДОБАВИТЬ МУЗЫКУ' : '▶ ADD MUSIC')}
      </button>
      <button onClick={skipMusic} disabled={musicBusy}
        style={{ width:'100%', marginTop:'10px', padding:'10px', borderRadius:'10px', background:'transparent',
          border:`1px solid ${c.border}`, color:c.textMuted, fontSize:'13px', cursor:'pointer' }}>
        {ru ? 'Пропустить — добавлю музыку позже' : 'Skip — I will add music later'}
      </button>
    </div>
    );
  };

  const renderAi = () => (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'6px' }}>
        {ru ? 'Подключить AI?' : 'Connect AI?'}
      </h2>
      <p style={{ fontSize:'13px', color:c.textMuted, lineHeight:'1.5', marginBottom:'20px' }}>
        {ru ? 'Чат о треках, «звуковые вайбы», биографии артистов. Нужен OpenAI-совместимый сервер (LM Studio, Ollama…). Можно настроить позже в настройках.'
            : 'Track chat, sonic vibes, artist bios. Needs an OpenAI-compatible server (LM Studio, Ollama…). You can set this up later in Settings.'}
      </p>
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>BASE URL</div>
      <input value={llmUrl} onChange={e => { setLlmUrl(e.target.value); setProbe(null); }}
        placeholder="http://localhost:1234/v1" disabled={aiBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>
        {ru ? 'МОДЕЛЬ' : 'MODEL'}
      </div>
      <input value={llmModel} onChange={e => { setLlmModel(e.target.value); setProbe(null); }}
        placeholder="qwen2.5-14b-instruct" disabled={aiBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>
        {ru ? 'API-КЛЮЧ · НЕОБЯЗАТЕЛЬНО' : 'API KEY · OPTIONAL'}
      </div>
      <input type="password" value={llmKey} onChange={e => setLlmKey(e.target.value)}
        placeholder={ru ? 'для локального сервера обычно не нужен' : 'usually not needed for a local server'} disabled={aiBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div style={{ display:'flex', alignItems:'center', gap:'10px', marginBottom:'18px' }}>
        <button onClick={probeLlm} disabled={probing || !llmUrl.trim()}
          className={ske('btn', isDark)} style={{ padding:'9px 16px', borderRadius:'10px', fontSize:'13px',
            color:c.textMuted, cursor: probing || !llmUrl.trim() ? 'not-allowed' : 'pointer',
            display:'flex', alignItems:'center', gap:'6px' }}>
          {probing ? <Spinner size={12} /> : '⚡'}
          {ru ? 'Проверить соединение' : 'Test connection'}
        </button>
        {probe && (
          <span style={{ fontSize:'12px', color: probe.available ? 'oklch(70% 0.18 145)' : c.red }}>
            {probe.available
              ? (ru ? '✓ модель найдена' : '✓ model found')
              : `✗ ${probe.error || (ru ? 'недоступно' : 'unavailable')}`}
          </span>
        )}
      </div>
      {probe && probe.available && (
        <div className="ob-glass" style={{ padding:'14px 16px', marginBottom:'16px',
             background:'linear-gradient(180deg,rgba(154,133,255,.16),rgba(154,133,255,.07))',
             boxShadow:'inset 0 0 0 1px rgba(154,133,255,.35),0 0 26px rgba(124,92,255,.18)' }}>
          <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
            <span style={{ fontWeight:600, fontSize:'13.5px' }}>✨ {ru ? 'Сразу обогатить библиотеку' : 'Enrich the library right away'}</span>
            <span className="mono" style={{ fontSize:'9px', letterSpacing:'.12em', color:'#cfc4ff', background:'rgba(154,133,255,.2)', border:'1px solid rgba(154,133,255,.4)', borderRadius:'6px', padding:'3px 6px' }}>
              {ru ? 'РЕКОМЕНДУЕМ' : 'RECOMMENDED'}
            </span>
            <label style={{ marginLeft:'auto', cursor:'pointer', display:'flex', alignItems:'center' }}>
              <input type="checkbox" checked={enrichWithAi} onChange={e => setEnrichWithAi(e.target.checked)}
                style={{ width:'16px', height:'16px', accentColor:'oklch(62% 0.2 275)', cursor:'pointer' }} />
            </label>
          </div>
          <div style={{ fontSize:'12.5px', color:c.textMuted, lineHeight:'1.5', marginTop:'8px' }}>
            {ru ? 'Гуру напишет биографии артистов, углубит факты и опишет звучание песен — прямо при добавлении музыки. Чуть дольше, зато всё готово с первого входа.'
                : 'The guru writes artist bios, deepens facts and describes how songs sound — inside this same indexing. A bit longer, but ready from the first launch.'}
          </div>
        </div>
      )}
      <div style={{ display:'flex', gap:'10px' }}>
        <button onClick={() => { setAiConnected(false); advanceFromAi(); }} disabled={aiBusy}
          style={{ flex:1, padding:'12px', borderRadius:'10px', background:'transparent',
            border:`1px solid ${c.border}`, color:c.textMuted, fontSize:'14px', cursor:'pointer' }}>
          {ru ? 'Пропустить' : 'Skip'}
        </button>
        <button onClick={connectAi} disabled={aiBusy || !llmUrl.trim() || !llmModel.trim()}
          className="ske-accent" style={{ flex:1, padding:'12px', borderRadius:'10px', fontSize:'14px', fontWeight:'600',
            opacity: aiBusy || !llmUrl.trim() || !llmModel.trim() ? 0.5 : 1,
            cursor: aiBusy ? 'wait' : 'pointer',
            display:'flex', alignItems:'center', justifyContent:'center', gap:'8px' }}>
          {aiBusy ? <Spinner size={14} color="white" /> : null}
          {ru ? 'ПОДКЛЮЧИТЬ →' : 'CONNECT →'}
        </button>
      </div>
    </div>
  );

  const renderPolicy = () => (
    <div className="ob-glass" style={{ padding:'26px 28px' }}>
      <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'6px' }}>
        {ru ? 'Настройки системы' : 'System settings'}
      </h2>
      <p style={{ fontSize:'13px', color:c.textMuted, lineHeight:'1.5', marginBottom:'18px' }}>
        {ru ? 'Эти параметры — общие для всего сервера. Участники по инвайтам только загружают свою музыку; модель и AI вы задаёте один раз.'
            : 'These apply to the whole server. Invited members only upload their music; you choose the model and AI once.'}
      </p>

      {renderSlider()}

      <div style={{ height:'1px', background:c.border, margin:'22px 0 18px' }} />

      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'10px' }}>
        {ru ? 'AI (НЕОБЯЗАТЕЛЬНО)' : 'AI (OPTIONAL)'}
      </div>
      <p style={{ fontSize:'12px', color:c.textMuted, lineHeight:'1.5', marginBottom:'14px' }}>
        {ru ? 'OpenAI-совместимый сервер (LM Studio, Ollama…). Ключ хранится на сервере и не показывается участникам.'
            : 'OpenAI-compatible server (LM Studio, Ollama…). The key is stored server-side and never shown to members.'}
      </p>
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>BASE URL</div>
      <input value={llmUrl} onChange={e => { setLlmUrl(e.target.value); setProbe(null); }}
        placeholder="http://localhost:1234/v1" disabled={policyBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>{ru ? 'МОДЕЛЬ' : 'MODEL'}</div>
      <input value={llmModel} onChange={e => { setLlmModel(e.target.value); setProbe(null); }}
        placeholder="qwen2.5-14b-instruct" disabled={policyBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div className="mono" style={{ fontSize:'12px', color:c.textSubtle, letterSpacing:'0.22em', marginBottom:'8px' }}>{ru ? 'API-КЛЮЧ · НЕОБЯЗАТЕЛЬНО' : 'API KEY · OPTIONAL'}</div>
      <input type="password" value={llmKey} onChange={e => setLlmKey(e.target.value)}
        placeholder={ru ? 'для локального сервера обычно не нужен' : 'usually not needed for a local server'} disabled={policyBusy}
        className={ske('inset', isDark)} style={{ width:'100%', padding:'11px 13px', borderRadius:'10px', border:'none',
          color:c.text, fontSize:'14px', outline:'none', fontFamily:"'JetBrains Mono', monospace", marginBottom:'14px' }} />
      <div style={{ display:'flex', alignItems:'center', gap:'10px', marginBottom:'18px' }}>
        <button onClick={probeLlm} disabled={probing || !llmUrl.trim()}
          className={ske('btn', isDark)} style={{ padding:'9px 16px', borderRadius:'10px', fontSize:'13px',
            color:c.textMuted, cursor: probing || !llmUrl.trim() ? 'not-allowed' : 'pointer',
            display:'flex', alignItems:'center', gap:'6px' }}>
          {probing ? <Spinner size={12} /> : '⚡'}
          {ru ? 'Проверить соединение' : 'Test connection'}
        </button>
        {probe && (
          <span style={{ fontSize:'12px', color: probe.available ? 'oklch(70% 0.18 145)' : c.red }}>
            {probe.available ? (ru ? '✓ модель найдена' : '✓ model found') : `✗ ${probe.error || (ru ? 'недоступно' : 'unavailable')}`}
          </span>
        )}
      </div>

      {policyError && (
        <div style={{ padding:'9px 13px', marginBottom:'14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px' }}>{policyError}</div>
      )}

      <div style={{ display:'flex', gap:'10px' }}>
        <button onClick={() => submitPolicy(false)} disabled={policyBusy}
          style={{ flex:1, padding:'13px', borderRadius:'10px', background:'transparent',
            border:`1px solid ${c.border}`, color:c.textMuted, fontSize:'14px', cursor: policyBusy ? 'wait' : 'pointer' }}>
          {ru ? 'Без AI' : 'Without AI'}
        </button>
        <button onClick={() => submitPolicy(true)} disabled={policyBusy || !llmUrl.trim() || !llmModel.trim()}
          className="ske-accent" style={{ flex:1, padding:'13px', borderRadius:'10px', fontSize:'14px', fontWeight:'600',
            opacity: policyBusy || !llmUrl.trim() || !llmModel.trim() ? 0.5 : 1,
            cursor: policyBusy ? 'wait' : 'pointer', display:'flex', alignItems:'center', justifyContent:'center', gap:'8px' }}>
          {policyBusy ? <Spinner size={14} color="white" /> : null}
          {ru ? 'СОХРАНИТЬ И ЗАВЕРШИТЬ →' : 'SAVE & FINISH →'}
        </button>
      </div>
    </div>
  );

  const aiStageDefs = [
    { key:'artist_bio',    label: ru ? 'Биографии артистов' : 'Artist bios' },
    { key:'refined_facts', label: ru ? 'Углубление фактов' : 'Deeper facts' },
    { key:'sonic_vibe',    label: ru ? 'Звучание песен' : 'Song vibes' },
  ];
  const renderStageBar = (label, st, pct, indeterminate, eta, trailing) => (
    <div key={label} style={{ marginBottom:'12px' }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:'4px' }}>
        <span style={{ display:'flex', alignItems:'center', gap:'8px', minWidth:0 }}>
          <span className="mono" style={{ fontSize:'12px', letterSpacing:'0.12em', color: st === 'running' ? c.text : c.textSubtle }}>{label}</span>
          {trailing}
        </span>
        <span style={{ display:'flex', alignItems:'center', gap:'8px', fontSize:'11px', color: st === 'failed' ? c.red : c.textSubtle }}>
          {eta && st === 'running' && <span style={{ fontVariantNumeric:'tabular-nums', opacity:.85 }}>{eta}</span>}
          <span>{st === 'done' ? '✓' : st === 'failed' ? '✗' : st === 'running' ? '…' : '·'}</span>
        </span>
      </div>
      <div style={{ height:'5px', borderRadius:'3px', background:c.border, overflow:'hidden', position:'relative' }}>
        {indeterminate && st === 'running'
          ? <div className="ob-indet" />
          : <div style={{ height:'100%', width:`${pct}%`, transition:'width 0.4s',
              background: st === 'done' ? 'oklch(63% 0.17 142)' : 'linear-gradient(90deg, oklch(65% 0.18 270), oklch(75% 0.17 280))' }} />}
      </div>
    </div>
  );

  const renderIndexing = () => {
    const finished = indexDone && (!shouldEnrich || aiDone);
    return (
    <div className="ob-glass" style={finished
      ? { padding:'30px 28px', textAlign:'center', boxShadow:'inset 0 1px 0 var(--ob-glass-sheen),0 12px 50px rgba(0,0,0,.28),0 0 70px rgba(95,208,138,.22)' }
      : { padding:'26px 28px' }}>

      {!finished && (
        <h2 className="serif" style={{ fontSize:'26px', letterSpacing:'-0.02em', marginBottom:'14px' }}>
          {indexError ? (ru ? 'Что-то пошло не так' : 'Something went wrong')
                      : (ru ? 'Собираем вашу библиотеку…' : 'Building your library…')}
        </h2>
      )}

      {!finished && <ProcessingModeBadge isDark={isDark} lang={lang} style={{ marginBottom:'18px' }} />}

      {!finished && !indexError && Object.keys(WIZ_STAGE_LABELS).map(k => {
        const st = stepStatus[k] || (indexDone ? 'done' : 'pending');
        const pr = stageProgress[k] || { current: 0, total: 0 };
        const pct = st === 'done' ? 100 : pr.total > 0 ? Math.round(100 * pr.current / pr.total) : 0;
        return (
          <Fragment key={k}>
            {renderStageBar(ru ? WIZ_STAGE_LABELS[k].ru : WIZ_STAGE_LABELS[k].en, st, pct, k === 'dense', etas[k],
              k === 'facts' ? <PremiumMetaHint isDark={isDark} lang={lang} /> : null)}
          </Fragment>
        );
      })}

      {/* Server-side guru phase — live bars straight from the job's SSE stream. */}
      {!finished && !indexError && sseAiStages && (
        <>
          <div className="mono" style={{ display:'flex', alignItems:'center', gap:'9px', margin:'16px 0 10px',
            fontSize:'11px', letterSpacing:'0.2em', textTransform:'uppercase', color:'#c3b8ff' }}>
            ✨ {ru ? 'С помощью гуру' : 'With the guru'}
          </div>
          <GuruStagesFromSse ru={ru} c={c} aiStages={sseAiStages} />
          <button onClick={() => onComplete({ mode })}
            style={{ marginTop:'6px', background:'transparent', border:'none', color:c.textSubtle, fontSize:'12px', textDecoration:'underline', cursor:'pointer' }}>
            {ru ? 'Войти, не дожидаясь' : 'Enter without waiting'}
          </button>
        </>
      )}

      {!finished && !indexError && !sseAiStages && shouldEnrich && (indexDone || aiRunning) && (
        <>
          <div className="mono" style={{ display:'flex', alignItems:'center', gap:'9px', margin:'16px 0 10px',
            fontSize:'11px', letterSpacing:'0.2em', textTransform:'uppercase', color:'#c3b8ff' }}>
            ✨ {ru ? 'С помощью гуру' : 'With the guru'}
          </div>
          {aiStageDefs.map(s => {
            const j = aiStatus && aiStatus[s.key];
            const status = j ? j.status : 'pending';
            const st = status === 'done' ? 'done'
                     : status === 'failed' ? 'failed'
                     : (status === 'running' || status === 'queued') ? 'running' : 'pending';
            const pct = j && j.n_total > 0 ? Math.round(100 * j.n_done / j.n_total) : (st === 'done' ? 100 : 0);
            return renderStageBar(s.label, st, pct);
          })}
          <button onClick={() => onComplete({ mode })}
            style={{ marginTop:'6px', background:'transparent', border:'none', color:c.textSubtle, fontSize:'12px', textDecoration:'underline', cursor:'pointer' }}>
            {ru ? 'Войти, не дожидаясь' : 'Enter without waiting'}
          </button>
        </>
      )}

      {indexError && (
        <div>
          <div style={{ padding:'10px 14px', borderRadius:'10px', background:c.redBg, color:c.red, fontSize:'13px', marginBottom:'16px' }}>
            {indexError}
          </div>
          <div style={{ display:'flex', gap:'10px' }}>
            <button onClick={retryMusic}
              className="ske-accent" style={{ flex:1, padding:'12px', borderRadius:'10px', fontSize:'14px', fontWeight:'600', cursor:'pointer' }}>
              {ru ? '↻ ПОВТОРИТЬ' : '↻ RETRY'}
            </button>
            <button onClick={() => onComplete({ mode })}
              style={{ flex:1, padding:'12px', borderRadius:'10px', background:'transparent',
                border:`1px solid ${c.border}`, color:c.textMuted, fontSize:'14px', cursor:'pointer' }}>
              {ru ? 'В приложение' : 'Into the app'}
            </button>
          </div>
        </div>
      )}

      {finished && (
        <div style={{ display:'flex', flexDirection:'column', alignItems:'center' }}>
          <div style={{ fontSize:'60px', lineHeight:1, marginBottom:'12px',
            filter:'drop-shadow(0 0 26px rgba(95,208,138,.9)) drop-shadow(0 0 8px rgba(95,208,138,.75))' }}>✨</div>
          <div className="mono" style={{ fontSize:'11px', letterSpacing:'0.24em', textTransform:'uppercase', color:'#a9ecc4' }}>{ru ? 'Всё готово' : 'All set'}</div>
          <h2 className="serif" style={{ fontSize:'28px', margin:'8px 0' }}>{ru ? 'Гуру познакомился с вашей музыкой' : 'The guru has met your music'}</h2>
          <p style={{ fontSize:'13.5px', color:c.textMuted, lineHeight:'1.6', maxWidth:'380px' }}>
            {ru ? 'Библиотека собрана и понимает ваш вкус. Поток, факты и чат — уже ждут.' : 'Your library is built and gets your taste. Stream, facts and chat are ready.'}
          </p>
          <button onClick={() => onComplete({ mode })} className="ske-accent"
            style={{ marginTop:'18px', padding:'13px 28px', borderRadius:'12px', fontSize:'15px', fontWeight:'600', letterSpacing:'0.06em', cursor:'pointer' }}>
            {ru ? 'Открыть MusiX →' : 'Open MusiX →'}
          </button>
        </div>
      )}
    </div>
    );
  };

  if (alreadyInit) {
    return (
      <div className="grain" style={{
        width:'100vw', height:'100vh', display:'flex', alignItems:'center', justifyContent:'center',
        background: isDark
          ? 'radial-gradient(ellipse at top, #15151b 0%, #07070a 100%)'
          : 'radial-gradient(ellipse at top, #fafaff 0%, #e3e2e8 100%)',
        color:c.text, padding:'24px' }}>
        <div className={ske('panel', isDark)} style={{ maxWidth:'440px', padding:'36px 32px', borderRadius:'20px', textAlign:'center' }}>
          <h2 className="serif" style={{ fontSize:'28px', marginBottom:'12px' }}>
            {ru ? 'Инстанс уже настроен' : 'Instance already set up'}
          </h2>
          <p style={{ fontSize:'14px', color:c.textMuted, lineHeight:'1.6', marginBottom:'20px' }}>
            {ru ? 'Кто-то уже создал владельца на этом сервере. Перейдите ко входу.'
                : 'Someone already created the owner on this server. Proceed to login.'}
          </p>
          <button onClick={() => window.location.reload()} className="ske-accent"
            style={{ width:'100%', padding:'12px 20px', borderRadius:'12px', fontSize:'15px', fontWeight:'600', cursor:'pointer' }}>
            {ru ? 'ПЕРЕЙТИ КО ВХОДУ' : 'GO TO LOGIN'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="grain ob-root" style={{
      '--ob-glass-bg': isDark ? 'rgba(255,255,255,.055)' : 'rgba(255,255,255,.62)',
      '--ob-glass-sheen': isDark ? 'rgba(255,255,255,.12)' : 'rgba(255,255,255,.9)',
      '--ob-glass-edge': isDark ? 'rgba(255,255,255,.18)' : 'rgba(0,0,0,.10)',
      '--ob-card-bg': isDark ? 'linear-gradient(180deg,#1d1d23,#131318)' : 'linear-gradient(180deg,#ffffff,#eef0f5)',
      '--ob-card-edge': isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.06)',
      '--ob-blob1':'#7d5cff', '--ob-blob2':'#3aa0ff', '--ob-blob3':'#c061ff',
      width:'100vw', height:'100vh', overflow:'auto', position:'relative',
      background: isDark
        ? 'radial-gradient(ellipse at top, #15151b 0%, #0a0a0e 60%, #07070a 100%)'
        : 'radial-gradient(ellipse at top, #fafaff 0%, #ececf3 60%, #e3e2e8 100%)',
      color: c.text, display:'flex', flexDirection:'column',
    }}>
      <div style={{ display:'flex', justifyContent:'space-between', padding:'24px 32px' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
          <BrandMark size={36} isDark={isDark} />
          <span className="serif" style={{ fontSize:'28px', letterSpacing:'-0.02em' }}>Musi<i style={{ color:'oklch(62% 0.2 275)' }}>X</i></span>
        </div>
        <TopRightControls isDark={isDark} lang={lang} onLang={handleLang} onTheme={handleTheme} onSettings={() => {}} />
      </div>
      <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center', padding:'24px 32px 48px' }}>
        <div style={{ width:'100%', maxWidth: step === 'welcome' ? 'min(1100px, 94vw)' : '880px' }}>
          {step === 'welcome' ? renderWelcome() : (
            <div style={{ position:'relative', display:'flex', gap:24 }}>
              <DriftBackdrop />
              <SetupRail ru={ru} steps={railSteps} currentKey={railCurrentKey} />
              <div style={{ flex:1, position:'relative', zIndex:1, minWidth:0 }}>
                {step === 'accountMode' && renderAccountMode()}
                {step === 'music'    && renderMusic()}
                {step === 'ai'       && renderAi()}
                {step === 'policy'   && renderPolicy()}
                {step === 'indexing' && renderIndexing()}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Phase A: Root — handles auth gating before mounting App ──────────────
function Root() {
  const [token, setToken] = useState(() => getStoredToken());
  const [instanceMode, setInstanceMode] = useState(null);     // 'sharing' | 'server' | null
  const [instanceLoaded, setInstanceLoaded] = useState(false);
  const [bootError, setBootError] = useState('');
  const [needsSetup, setNeedsSetup] = useState(false);

  // Boot: fetch /instance/config (no auth required) to learn the mode.
  // BEFORE rendering anything else — this drives both the LoginScreen UX
  // (show 'I have an invite' tab only in server mode) and the App layout.
  // Declared via useEffect deps=[]: runs exactly once on mount.
  useEffect(() => {
    fetch(API + '/instance/config')
      .then(r => {
        if (r.status === 404) {
          // Instance not initialized — first visitor runs the setup wizard.
          setNeedsSetup(true);
          setInstanceLoaded(true);
          return null;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        if (data && data.mode) setInstanceMode(data.mode);
        setInstanceLoaded(true);
      })
      .catch(err => {
        setBootError(`Cannot reach server: ${err.message || err}`);
        setInstanceLoaded(true);
      });
  }, []);

  // After login, store auth and flip the token state so we re-render into App.
  const onAuthSuccess = useCallback(({ token: t }) => {
    setToken(t);
  }, []);

  if (!instanceLoaded) {
    return (
      <div style={{
        minHeight: '100vh', display: 'grid', placeItems: 'center',
        background: '#0a0a10', color: '#666', fontFamily: 'sans-serif',
      }}>
        Loading…
      </div>
    );
  }

  if (needsSetup) {
    return (
      <SetupWizard onComplete={({ mode }) => {
        // Owner created + JWT already in localStorage (setStoredAuth ran
        // inside the wizard). Flip state so App mounts without a reload.
        setInstanceMode(mode);
        setNeedsSetup(false);
        setToken(getStoredToken());
      }} />
    );
  }

  if (bootError) {
    return (
      <div style={{
        minHeight: '100vh', display: 'grid', placeItems: 'center',
        background: '#0a0a10', color: '#aaa', padding: 24,
        fontFamily: 'monospace', fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap',
      }}>
        {bootError}
      </div>
    );
  }

  if (!token) {
    return <LoginScreen instanceMode={instanceMode} onAuthSuccess={onAuthSuccess} lang={localStorage.getItem('musix_lang') || 'ru'} />;
  }

  // Token present. Server-mode owner runs the instance, not the music app:
  // mount the admin dashboard (invites/members/AI policy + logout) INSTEAD of
  // <App>. Role comes from localStorage — populated by setStoredAuth on both
  // login and /instance/setup. To listen, the owner logs out and signs in as a
  // regular member. Members + sharing-mode users mount <App> unchanged.
  const role = localStorage.getItem('musix_user_role') || '';
  if (instanceMode === 'server' && role === 'owner') {
    return <OwnerAdminDashboard onLogout={() => { clearStoredAuth(); setToken(''); }} />;
  }

  // We pass instanceMode down so App can branch on it (e.g. hide the Invites
  // panel in sharing mode).
  return <App instanceMode={instanceMode} onLogout={() => { clearStoredAuth(); setToken(''); }} />;
}

ReactDOM.createRoot(document.getElementById('root')).render(<Root />);
