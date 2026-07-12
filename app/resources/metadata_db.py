"""SQLite-backed metadata store for artist/song facts.

Architecture: Qdrant (vectors + search) + SQLite (structured facts).

The DB file lives at ``cache/metadata.db`` alongside the existing cache
directories. All queries are scoped to a ``collection_name`` so that the
frontend only sees facts for the currently selected collection.

Language support is baked into the schema (``lang`` column) but all
methods currently default to ``'en'``. A future pass will add LLM-driven
translations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["MetadataDB", "canonical_track_id"]

_HEX32 = frozenset("0123456789abcdef")


def canonical_track_id(track_id: str) -> str:
    """Normalize a track id to the dashed-UUID form Qdrant emits.

    Indexing used to mint point ids as ``uuid4().hex`` (32 hex chars, no
    dashes) while Qdrant canonicalizes any UUID input to the dashed form in
    every read — so the same track had two spellings depending on the data
    source. Non-UUID ids pass through unchanged.
    """
    if (
        isinstance(track_id, str)
        and len(track_id) == 32
        and all(c in _HEX32 for c in track_id.lower())
    ):
        return (
            f"{track_id[0:8]}-{track_id[8:12]}-{track_id[12:16]}"
            f"-{track_id[16:20]}-{track_id[20:32]}"
        ).lower()
    return track_id

import os as _os
DB_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
# MUSIX_METADATA_DB env override lets tests + CLI scripts point at a tmp DB
# without monkey-patching. Falls back to the default cache location.
DB_PATH = Path(_os.environ.get("MUSIX_METADATA_DB", str(DB_DIR / "metadata.db")))

_SCHEMA_SQL: Tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS artists (
        slug TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        collection_name TEXT,
        mbid TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS artist_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_slug TEXT NOT NULL REFERENCES artists(slug) ON DELETE CASCADE,
        lang TEXT NOT NULL DEFAULT 'en',
        fact TEXT NOT NULL,
        category TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS songs (
        slug TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist_slug TEXT NOT NULL REFERENCES artists(slug) ON DELETE CASCADE,
        collection_name TEXT,
        recording_mbid TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS song_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        song_slug TEXT NOT NULL REFERENCES songs(slug) ON DELETE CASCADE,
        lang TEXT NOT NULL DEFAULT 'en',
        fact TEXT NOT NULL,
        category TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS track_reactions (
        collection_name TEXT NOT NULL,
        track_id TEXT NOT NULL,
        reaction TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (collection_name, track_id)
    )""",
    """CREATE TABLE IF NOT EXISTS collection_settings (
        collection_name TEXT PRIMARY KEY,
        text_model TEXT,
        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_af_artist_lang ON artist_facts(artist_slug, lang)",
    "CREATE INDEX IF NOT EXISTS idx_sf_song_lang ON song_facts(song_slug, lang)",
    "CREATE INDEX IF NOT EXISTS idx_artists_collection ON artists(collection_name)",
    "CREATE INDEX IF NOT EXISTS idx_songs_collection ON songs(collection_name)",
    # Sonic Descriptor columns (added in Plan 1)
    "ALTER TABLE songs ADD COLUMN sonic_tags_json TEXT",
    "ALTER TABLE songs ADD COLUMN sonic_class TEXT",
    "ALTER TABLE songs ADD COLUMN sonic_class_confidence REAL",
    "ALTER TABLE songs ADD COLUMN audio_signature TEXT",
    # Playback history (added in Plan 3 Task 2)
    """CREATE TABLE IF NOT EXISTS playback_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT    NOT NULL,
        collection_name TEXT    NOT NULL,
        track_id        TEXT    NOT NULL,
        played_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        played_sec      REAL    NOT NULL,
        total_dur       REAL,
        skipped_early   INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE INDEX IF NOT EXISTS idx_playback_track
          ON playback_events(collection_name, track_id)""",
    """CREATE INDEX IF NOT EXISTS idx_playback_session
          ON playback_events(session_id)""",
    """CREATE INDEX IF NOT EXISTS idx_playback_at
          ON playback_events(collection_name, played_at)""",
    # Огонёк/Вода journal (2026-06-29 fire-wave recsys). Append-only: every fire
    # or water gesture is one row. Aggregations derive ephemeral session anchors
    # (the «волна»), the long-term island deposit, and the computed «favorites»
    # rank that replaced heart-likes.
    """CREATE TABLE IF NOT EXISTS taste_signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT    NOT NULL,
        session_id      TEXT    NOT NULL,
        track_id        TEXT    NOT NULL,
        kind            TEXT    NOT NULL,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE INDEX IF NOT EXISTS idx_taste_signals_coll_created
          ON taste_signals(collection_name, created_at)""",
    """CREATE INDEX IF NOT EXISTS idx_taste_signals_session
          ON taste_signals(session_id)""",
    """CREATE INDEX IF NOT EXISTS idx_taste_signals_coll_track
          ON taste_signals(collection_name, track_id)""",
    # AI Indexing (added in Plan 3 Task 11)
    """CREATE TABLE IF NOT EXISTS ai_indexing_jobs (
        job_id          TEXT PRIMARY KEY,
        task_type       TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        lang            TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'queued',
        n_total         INTEGER NOT NULL DEFAULT 0,
        n_done          INTEGER NOT NULL DEFAULT 0,
        n_failed        INTEGER NOT NULL DEFAULT 0,
        started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        finished_at     TIMESTAMP,
        error           TEXT
    )""",
    """CREATE INDEX IF NOT EXISTS idx_ai_jobs_lookup
          ON ai_indexing_jobs(collection_name, task_type, started_at DESC)""",
    # Sonic Vibe cache (added in Plan 3 Task 14)
    """CREATE TABLE IF NOT EXISTS sonic_vibes (
        track_id        TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        lang            TEXT NOT NULL,
        phrase          TEXT NOT NULL,
        generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (track_id, collection_name, lang)
    )""",
    # Refined Facts cache (added in Plan 3 Task 15)
    # Key is (scope, scope_key, lang) — collection-INDEPENDENT. scope_key is a
    # stable slug (song_slug from artist+title, or artist_slug), so the same
    # song/artist refined once is reused across every collection. collection_name
    # is kept only as provenance (last writer) for the per-collection cache reset.
    # Pre-existing DBs are migrated from the old (…, collection_name, lang) PK by
    # _migrate_refined_facts_key().
    """CREATE TABLE IF NOT EXISTS refined_facts (
        scope           TEXT NOT NULL,         -- 'song' or 'artist'
        scope_key       TEXT NOT NULL,         -- song_slug (song) or artist_slug (artist)
        lang            TEXT NOT NULL,
        collection_name TEXT,                  -- provenance only; NOT part of key
        refined_json    TEXT NOT NULL,         -- JSON array of {"text": str}
        generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (scope, scope_key, lang)
    )""",
    # Artist Bio cache (Plan 5)
    """CREATE TABLE IF NOT EXISTS artist_bios (
        artist_slug     TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        lang            TEXT NOT NULL,
        bio_text        TEXT NOT NULL,
        generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (artist_slug, collection_name, lang)
    )""",
    # Custom Playlists (Plan 19)
    """CREATE TABLE IF NOT EXISTS playlists (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT    NOT NULL,
        name            TEXT    NOT NULL,
        description     TEXT,
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE(collection_name, name)
    )""",
    """CREATE TABLE IF NOT EXISTS playlist_tracks (
        playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
        track_id    TEXT    NOT NULL,
        position    INTEGER NOT NULL,
        added_at    TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (playlist_id, track_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_playlist_tracks_position ON playlist_tracks(playlist_id, position)",
    "CREATE INDEX IF NOT EXISTS idx_playlists_collection ON playlists(collection_name)",
    # Stream RecSys: cached LLM texts (listener portrait + island names).
    # source_hash fingerprints the profile inputs (island members) — when the
    # taste drifts the hash changes and readers treat the row as stale.
    """CREATE TABLE IF NOT EXISTS recsys_llm_texts (
        collection_name TEXT NOT NULL,
        kind            TEXT NOT NULL,
        lang            TEXT NOT NULL,
        source_hash     TEXT NOT NULL,
        content_json    TEXT NOT NULL,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (collection_name, kind, lang)
    )""",
    # Phase A: Auth Foundation — users + invites + instance_config
    """CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'member'
                        CHECK (role IN ('owner', 'member')),
        created_at    REAL NOT NULL,
        last_login_at REAL,
        premium       INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    # Premium tier flag (display-only; no server-side gating). Idempotent —
    # the duplicate-column catch in init() skips it on already-migrated DBs.
    "ALTER TABLE users ADD COLUMN premium INTEGER NOT NULL DEFAULT 0",
    """CREATE TABLE IF NOT EXISTS invites (
        code         TEXT PRIMARY KEY,
        created_by   TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at   REAL NOT NULL,
        expires_at   REAL NOT NULL,
        consumed_by  TEXT REFERENCES users(id) ON DELETE SET NULL,
        consumed_at  REAL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_invites_unused ON invites (consumed_at) WHERE consumed_at IS NULL",
    """CREATE TABLE IF NOT EXISTS instance_config (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        mode        TEXT NOT NULL CHECK (mode IN ('sharing', 'server')),
        created_at  REAL NOT NULL
    )""",
    # Server-mode onboarding: instance-level settings (LLM endpoint/key/model,
    # embedding model, AI/CLAP flags) written via the wizard/Settings and read
    # by SettingsService with precedence DB > env > default. Key-value so keys
    # mirror env var names (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, EMBED_MODEL,
    # CLAP_ENABLED, AI_ENABLED) and new keys need no migration.
    """CREATE TABLE IF NOT EXISTS instance_settings (
        key         TEXT PRIMARY KEY,
        value       TEXT,
        updated_at  REAL NOT NULL
    )""",
    # Phase C: server-mode upload pipeline. One row per file upload — survives
    # restart so an interrupted batch-commit can be resumed via /library/upload/{id}.
    """CREATE TABLE IF NOT EXISTS pending_uploads (
        upload_id          TEXT PRIMARY KEY,
        account_id         TEXT NOT NULL REFERENCES users(id),
        sha256             TEXT NOT NULL,
        original_filename  TEXT NOT NULL,
        size_bytes         INTEGER NOT NULL,
        storage_path       TEXT NOT NULL,
        status             TEXT NOT NULL,
        track_id           TEXT,
        error              TEXT,
        created_at         REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_pending_uploads_account_status ON pending_uploads (account_id, status)",
    # ── Track metadata mirror (SQLite copy of Qdrant payload card fields) ──
    # Populated during indexing / upload so the artist-aggregate endpoint can
    # read from a single indexed SQL query instead of paginating through Qdrant.
    # JSON-like fields (artists, artist_slugs, sonic_tags, sonic_axes) are stored
    # as TEXT (json.dumps).  artist_slugs also gets a join table for fast lookup.
    """CREATE TABLE IF NOT EXISTS track_metadata (
        collection_name       TEXT NOT NULL,
        track_id              TEXT NOT NULL,
        title                 TEXT NOT NULL,
        artist                TEXT NOT NULL,
        artists               TEXT,           -- JSON array
        artist_slugs          TEXT,           -- JSON array (also in join table)
        primary_artist_slug   TEXT,
        album                 TEXT,
        year                  INTEGER,
        genre                 TEXT,
        duration              REAL,
        file_path             TEXT,
        cover_art_path        TEXT,
        producer              TEXT,
        label                 TEXT,
        track_number          INTEGER,
        disc_number           INTEGER,
        bitrate_kbps          INTEGER,
        sonic_tags            TEXT,           -- JSON array
        sonic_axes            TEXT,           -- JSON object
        created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (collection_name, track_id)
    )""",
    """CREATE TABLE IF NOT EXISTS track_artist_slugs (
        collection_name TEXT NOT NULL,
        track_id        TEXT NOT NULL,
        artist_slug     TEXT NOT NULL,
        PRIMARY KEY (collection_name, track_id, artist_slug),
        FOREIGN KEY (collection_name, track_id)
            REFERENCES track_metadata(collection_name, track_id) ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_tas_slug_collection ON track_artist_slugs(artist_slug, collection_name)",
    "CREATE INDEX IF NOT EXISTS idx_tm_album ON track_metadata(album, collection_name)",
    # Yandex Music import: one row per account holding the (encrypted) OAuth
    # token so re-imports / enrichment don't require re-login. The token blob is
    # Fernet-encrypted by app.services.yandex.token_store — never store plaintext.
    """CREATE TABLE IF NOT EXISTS yandex_accounts (
        account_id  TEXT PRIMARY KEY REFERENCES users(id),
        enc_token   TEXT NOT NULL,
        yandex_uid  TEXT,
        expires_at  REAL,
        linked_at   REAL NOT NULL
    )""",
    # Yandex Music import: dedup + per-track report. (account_id, yandex_track_id)
    # is unique so a re-run skips already-imported tracks; status/reason drive the
    # end-of-job report (skipped tracks with their reason).
    """CREATE TABLE IF NOT EXISTS yandex_imports (
        account_id       TEXT NOT NULL REFERENCES users(id),
        yandex_track_id  TEXT NOT NULL,
        upload_id        TEXT,
        track_id         TEXT,
        status           TEXT NOT NULL,
        reason           TEXT,
        imported_at      REAL NOT NULL,
        PRIMARY KEY (account_id, yandex_track_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_yandex_imports_account ON yandex_imports (account_id, status)",
)


def _slugify(text: str) -> str:
    return "-".join(text.lower().split())


class MetadataDB:
    """Lazy singleton wrapper around a local SQLite database.

    The connection is created on first access. Call :meth:`init` to ensure
    the schema exists.
    """

    # Legacy/test hook: integration conftest patches _connect to store a single
    # shared connection here; close() still honors it. Production code paths
    # use the per-thread map below and leave _instance as None.
    _instance: Optional[sqlite3.Connection] = None

    # One connection PER THREAD, keyed by thread ident. CPython's sqlite3
    # connections are NOT safe for concurrent use from multiple threads even
    # with check_same_thread=False: parallel requests through FastAPI's
    # threadpool raced on the shared connection and produced
    # sqlite3.InterfaceError ("bad parameter or other API misuse") plus
    # silently wrong/None rows — get_current_user then 401'd valid tokens and
    # the frontend logged users out in a loop. Each entry stores the DB path
    # it was opened against so tests that repoint DB_PATH get a fresh
    # connection instead of a stale one aimed at the previous file.
    _connections: Dict[int, Tuple[str, sqlite3.Connection]] = {}
    _connections_lock = threading.Lock()

    # AudioDB enrichment columns (Plan: audiodb-enrichment Task 2).
    # Idempotent ALTER TABLE migration adds these to ``artists`` if missing.
    _AUDIODB_COLUMNS = (
        ("audiodb_bio", "TEXT"),
        ("mood", "TEXT"),
        ("country_code", "TEXT"),
        ("country", "TEXT"),
        ("label", "TEXT"),
        ("cutout_path", "TEXT"),
        ("thumb_path", "TEXT"),
        ("audiodb_mbid", "TEXT"),
        ("audiodb_fetched_at", "TIMESTAMP"),
    )

    @classmethod
    def _migrate_audiodb_columns(cls, conn: sqlite3.Connection) -> None:
        """Add AudioDB enrichment columns to artists table if missing.

        Idempotent: checks PRAGMA table_info first so re-runs don't raise.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(artists)").fetchall()}
        for col_name, col_type in cls._AUDIODB_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE artists ADD COLUMN {col_name} {col_type}")
        conn.commit()

    @classmethod
    def _connect(cls) -> sqlite3.Connection:
        """Return the CALLING THREAD's connection, opening it lazily.

        WAL journal mode makes concurrent per-thread connections safe:
        readers don't block the writer, and busy_timeout serializes
        competing writers instead of failing fast.
        """
        tid = threading.get_ident()
        path = str(DB_PATH)
        with cls._connections_lock:
            entry = cls._connections.get(tid)
            if entry is not None and entry[0] == path:
                return entry[1]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False not for sharing (each thread has its own
        # connection) but so close() can close worker-thread connections at
        # shutdown/test teardown from a different thread.
        conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Without a busy timeout, two concurrent writes (e.g. parallel
        # /library/upload from two accounts) can hit "database is locked"
        # immediately; this makes the loser wait for the lock instead
        # (Phase C concurrency).
        conn.execute("PRAGMA busy_timeout=5000")

        with cls._connections_lock:
            stale = cls._connections.get(tid)
            cls._connections[tid] = (path, conn)
        if stale is not None:
            # Same thread, different DB_PATH (test repointed it) — drop the old one.
            try:
                stale[1].close()
            except Exception:
                pass
        logger.info("[MetadataDB] Connected to %s (thread %s)", path, tid)
        return conn

    @classmethod
    def get(cls) -> sqlite3.Connection:
        """Return this thread's connection (creates it lazily)."""
        return cls._connect()

    @classmethod
    def init(cls) -> None:
        """Create tables, indexes, and any new ALTER TABLE statements idempotently."""
        conn = cls._connect()
        for sql in _SCHEMA_SQL:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                # ALTER TABLE ... ADD COLUMN re-runs raise "duplicate column name"; ignore.
                if "duplicate column" not in str(e).lower():
                    raise

        # ── Idempotent column migrations (additive, never destructive) ───────
        # Plan 3: MusicBrainz scaffold (data-only; no harvesting yet)
        cls._ensure_columns(conn, "songs", {
            "producers":    "TEXT",
            "label":        "TEXT",
            "samples_json": "TEXT",
            "mbid":         "TEXT",
        })
        # The artists table may or may not exist in older databases. Skip
        # silently if missing.
        existing_tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "artists" in existing_tables:
            cls._ensure_columns(conn, "artists", {"mbid": "TEXT"})
            # AudioDB enrichment columns (idempotent — see _AUDIODB_COLUMNS).
            cls._migrate_audiodb_columns(conn)

        # AI Mode infrastructure (Plan 6) — per-collection opt-in for live-LLM
        # features. DEFAULT 1 means existing rows immediately report on,
        # matching the pre-flag world where AI features were always available.
        # ai_enabled — AI Mode (Plan 6). axis_norm_stats + stream_liked_share —
        # Stream RecSys: per-collection sonic-axis mean/std (JSON, recomputed on
        # each indexing run) and the liked/new slider default.
        cls._ensure_columns(conn, "collection_settings", {
            "ai_enabled": "INTEGER NOT NULL DEFAULT 1",
            "axis_norm_stats": "TEXT",
            "stream_liked_share": "REAL",
        })

        # AI indexing — distinguish "processed" from "silently skipped" so the
        # UI can tell when a job completed with literally zero LLM work done
        # (e.g. tracks lacked sonic_tags + facts). Without this column the
        # status payload reports n_done = n_total and looks like a real run.
        cls._ensure_columns(conn, "ai_indexing_jobs", {
            "n_skipped": "INTEGER NOT NULL DEFAULT 0",
        })

        # Stream RecSys idle rule: did the user touch ANY control during this
        # track (like/skip/pause/seek)? NULL = legacy events, treated as
        # interacted so old history keeps moving the taste profile.
        cls._ensure_columns(conn, "playback_events", {
            "interacted": "INTEGER",
        })

        # Per-event influence flag: hand-queued tracks (Task 5 _noInfluence) must
        # not shape the taste profile. Default 1 = legacy rows treated as influencing.
        cls._ensure_columns(conn, "playback_events", {
            "influence": "INTEGER NOT NULL DEFAULT 1",
        })

        # Phase B: per-user settings live as columns on the (Phase A) users
        # table — model selection + CLAP toggle, keyed by users.id. Additive so
        # it coexists with the auth columns regardless of migration order.
        if "users" in existing_tables:
            cls._ensure_columns(conn, "users", {
                # v2-small-en matches what collections are actually indexed with
                # and is loadable without trust_remote_code (jina-v3 needs custom_st).
                "text_model_name": "TEXT NOT NULL DEFAULT 'jinaai/jina-embeddings-v2-small-en'",
                "clap_enabled":    "INTEGER NOT NULL DEFAULT 1",
            })

        # Facts are unique per (slug, lang, fact) and per-slug refined facts are
        # collection-independent. Both migrations are idempotent no-ops once applied.
        if "artist_facts" in existing_tables:
            cls._migrate_dedup_facts(conn)
        if "refined_facts" in existing_tables:
            cls._migrate_refined_facts_key(conn)

        cls._migrate_dashed_track_ids(conn, existing_tables)

        conn.commit()
        logger.info("[MetadataDB] Schema initialised")

    # Tables whose track ids may still carry the legacy undashed uuid4().hex
    # spelling (written by indexing / upload flows before point ids were minted
    # in the dashed form Qdrant emits).
    _TRACK_ID_TABLES: Tuple[Tuple[str, str], ...] = (
        ("track_metadata", "track_id"),
        ("track_artist_slugs", "track_id"),
        ("playlist_tracks", "track_id"),
        ("track_reactions", "track_id"),
        ("playback_events", "track_id"),
        ("taste_signals", "track_id"),
        ("sonic_vibes", "track_id"),
        ("pending_uploads", "track_id"),
        ("yandex_imports", "track_id"),
    )

    @classmethod
    def _migrate_dashed_track_ids(
        cls, conn: sqlite3.Connection, existing_tables: set,
    ) -> None:
        """Rewrite legacy undashed uuid4().hex track ids to the dashed form.

        Qdrant canonicalizes UUID point ids to the dashed spelling in every
        read, so any SQLite row keyed by the undashed spelling never matches a
        live Qdrant id (playlists resolved to "lost" tracks, the player's
        metadata batch couldn't merge, top-pairs lookups missed). Idempotent:
        the WHERE clause matches nothing once all ids are dashed. UPDATE OR
        IGNORE skips rows whose dashed twin already exists; the follow-up
        DELETE drops those now-redundant undashed duplicates.
        """
        dashed_expr = (
            "lower(substr({c},1,8) || '-' || substr({c},9,4) || '-' || "
            "substr({c},13,4) || '-' || substr({c},17,4) || '-' || substr({c},21))"
        )
        # 32 hex chars, no dashes. NB: SQLite GLOB negates classes with '^'
        # (not the Unix-shell '!'), so a positive 32-class pattern is the
        # least error-prone spelling.
        hex32_glob = "[0-9a-f]" * 32
        legacy_where = f"lower({{c}}) GLOB '{hex32_glob}'"
        # track_artist_slugs references track_metadata(track_id) with no ON
        # UPDATE action, so rewriting the parent key under foreign_keys=ON
        # would raise. The pragma is a no-op inside a transaction — commit
        # whatever init() has done so far, toggle, migrate, toggle back.
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            for table, col in cls._TRACK_ID_TABLES:
                if table not in existing_tables:
                    continue
                where = legacy_where.format(c=col)
                try:
                    n_legacy = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {where}"
                    ).fetchone()[0]
                    if not n_legacy:
                        continue
                    conn.execute(
                        f"UPDATE OR IGNORE {table} SET {col} = {dashed_expr.format(c=col)}"
                        f" WHERE {where}"
                    )
                    conn.execute(f"DELETE FROM {table} WHERE {where}")
                    logger.info(
                        "[MetadataDB] migrated %d undashed track ids in %s",
                        n_legacy, table,
                    )
                except sqlite3.OperationalError:
                    logger.warning(
                        "[MetadataDB] dashed-id migration failed for %s", table,
                        exc_info=True,
                    )
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    @classmethod
    def _migrate_dedup_facts(cls, conn: sqlite3.Connection) -> None:
        """Collapse duplicate fact rows and enforce uniqueness going forward.

        Re-indexing the same folder used to append identical fact rows (no dedup
        on insert, no UNIQUE constraint), so a single artist could accumulate the
        same fact N times. Keep the lowest-id row per (slug, lang, fact), then add
        UNIQUE indexes so future INSERT OR IGNORE silently drops repeats.
        Idempotent: the DELETEs are no-ops once unique, indexes use IF NOT EXISTS.

        Best-effort: a legacy/malformed schema (e.g. a pre-slug ``songs`` table
        that breaks the song_facts foreign key) makes SQLite raise on the first
        DML. A migration must never abort app startup, so such tables are logged
        and skipped — the live schema this ships with is always well-formed.
        """
        for table, key_cols in (
            ("artist_facts", "artist_slug, lang, fact"),
            ("song_facts", "song_slug, lang, fact"),
        ):
            try:
                conn.execute(
                    f"DELETE FROM {table} WHERE id NOT IN "
                    f"(SELECT MIN(id) FROM {table} GROUP BY {key_cols})"
                )
                conn.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{table} ON {table}({key_cols})"
                )
            except sqlite3.OperationalError as e:
                logger.warning("[MetadataDB] dedup migration skipped for %s: %s", table, e)

    @classmethod
    def _migrate_refined_facts_key(cls, conn: sqlite3.Connection) -> None:
        """Drop collection_name from the refined_facts primary key.

        scope_key is already a collection-independent slug, so keying refined
        facts by collection forced the LLM refinement to re-run (and re-store)
        per collection. Rebuild the table with PK (scope, scope_key, lang),
        keeping the most-recent row per group. collection_name survives as a
        plain provenance column. Idempotent: only rebuilds while the live PK
        still contains collection_name.
        """
        cols = conn.execute("PRAGMA table_info(refined_facts)").fetchall()
        # row layout: (cid, name, type, notnull, dflt, pk). pk>0 ⇒ in primary key.
        collection_in_pk = any(c[1] == "collection_name" and c[5] > 0 for c in cols)
        if not collection_in_pk:
            return  # already migrated
        conn.executescript(
            """
            CREATE TABLE refined_facts_new (
                scope           TEXT NOT NULL,
                scope_key       TEXT NOT NULL,
                lang            TEXT NOT NULL,
                collection_name TEXT,
                refined_json    TEXT NOT NULL,
                generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (scope, scope_key, lang)
            );
            INSERT INTO refined_facts_new
                (scope, scope_key, lang, collection_name, refined_json, generated_at)
            SELECT scope, scope_key, lang, collection_name, refined_json, generated_at
            FROM refined_facts rf
            WHERE rf.rowid = (
                SELECT rf2.rowid FROM refined_facts rf2
                WHERE rf2.scope = rf.scope AND rf2.scope_key = rf.scope_key
                      AND rf2.lang = rf.lang
                ORDER BY rf2.generated_at DESC, rf2.rowid DESC
                LIMIT 1
            );
            DROP TABLE refined_facts;
            ALTER TABLE refined_facts_new RENAME TO refined_facts;
            """
        )
        logger.info("[MetadataDB] refined_facts re-keyed to (scope, scope_key, lang)")

    @classmethod
    def _ensure_columns(cls, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        """Add columns that don't already exist on the given table.

        ``columns`` maps column-name -> SQL type. No-op for any column that
        already exists. Safe to call on every startup.
        """
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sqltype in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Drop any cached connections and clear the init flag — test only."""
        cls.close()

    # ── Artists ──

    @classmethod
    def upsert_artist(cls, slug: str, name: str, collection_name: str, mbid: Optional[str] = None) -> None:
        conn = cls._connect()
        conn.execute(
            """INSERT INTO artists (slug, name, collection_name, mbid)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   name=excluded.name,
                   collection_name=excluded.collection_name""",
            (slug, name, collection_name, mbid),
        )
        conn.commit()

    @classmethod
    def get_artist_audiodb(cls, slug: str, collection_name: str) -> dict | None:
        """Return dict of audiodb fields for the artist (or None if no row exists).

        A row that exists but has audiodb_fetched_at IS NULL returns the dict with
        all-None values, so the caller can distinguish 'never fetched' from
        'no such artist'."""
        conn = cls._connect()
        row = conn.execute(
            """SELECT audiodb_bio, mood, country_code, country, label,
                      cutout_path, thumb_path, audiodb_mbid, audiodb_fetched_at
               FROM artists WHERE slug = ? AND collection_name = ?""",
            (slug, collection_name),
        ).fetchone()
        if not row:
            return None
        fetched_at = row[8]
        return {
            "audiodb_bio": row[0],
            "mood": row[1],
            "country_code": row[2],
            "country": row[3],
            "label": row[4],
            "cutout_path": row[5],
            "thumb_path": row[6],
            "audiodb_mbid": row[7],
            "audiodb_fetched_at": (
                fetched_at.isoformat() if hasattr(fetched_at, "isoformat")
                else (str(fetched_at) if fetched_at else None)
            ),
        }

    @classmethod
    def upsert_artist_audiodb(
        cls, *, slug: str, collection_name: str,
        audiodb_bio: str | None, mood: str | None,
        country_code: str | None, country: str | None, label: str | None,
        cutout_path: str | None, thumb_path: str | None,
        audiodb_mbid: str | None,
    ) -> None:
        """INSERT new row if (slug, collection_name) missing, else UPDATE in place.

        Always sets audiodb_fetched_at = CURRENT_TIMESTAMP. If the row doesn't
        exist yet, inserts with name=slug as fallback (real name will be set later
        by upsert_artist when artist_facts processes the same artist)."""
        conn = cls._connect()
        existing = conn.execute(
            "SELECT 1 FROM artists WHERE slug = ? AND collection_name = ?",
            (slug, collection_name),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE artists SET
                     audiodb_bio = ?, mood = ?, country_code = ?, country = ?,
                     label = ?, cutout_path = ?, thumb_path = ?,
                     audiodb_mbid = ?, audiodb_fetched_at = CURRENT_TIMESTAMP
                   WHERE slug = ? AND collection_name = ?""",
                (audiodb_bio, mood, country_code, country, label,
                 cutout_path, thumb_path, audiodb_mbid, slug, collection_name),
            )
        else:
            conn.execute(
                """INSERT INTO artists
                   (slug, name, collection_name, audiodb_bio, mood, country_code,
                    country, label, cutout_path, thumb_path, audiodb_mbid,
                    audiodb_fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(slug) DO UPDATE SET
                       audiodb_bio = excluded.audiodb_bio,
                       mood = excluded.mood,
                       country_code = excluded.country_code,
                       country = excluded.country,
                       label = excluded.label,
                       cutout_path = excluded.cutout_path,
                       thumb_path = excluded.thumb_path,
                       audiodb_mbid = excluded.audiodb_mbid,
                       audiodb_fetched_at = CURRENT_TIMESTAMP""",
                (slug, slug, collection_name, audiodb_bio, mood, country_code,
                 country, label, cutout_path, thumb_path, audiodb_mbid),
            )
        conn.commit()

    @classmethod
    def get_artist_slug(cls, name: str, collection_name: str) -> Optional[str]:
        """Return the stored slug for an artist in a collection, or None."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT slug FROM artists WHERE name = ? AND collection_name = ? LIMIT 1",
            (name, collection_name),
        ).fetchone()
        return row[0] if row else None

    # ── Artist facts ──

    @classmethod
    def add_artist_fact(
        cls,
        slug: str,
        collection_name: str,
        fact_text: str,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """Ensure the artist row exists, then insert a fact."""
        conn = cls._connect()
        # Make sure artist exists (slug is derived from name)
        artist_name = slug.replace("-", " ")  # best-effort reverse slugify
        conn.execute(
            """INSERT OR IGNORE INTO artists (slug, name, collection_name)
               VALUES (?, ?, ?)""",
            (slug, artist_name, collection_name),
        )
        conn.execute(
            """INSERT OR IGNORE INTO artist_facts (artist_slug, lang, fact, category, source)
               VALUES (?, 'en', ?, ?, ?)""",
            (slug, fact_text, category, source),
        )
        conn.commit()

    @classmethod
    def add_artist_facts_batch(
        cls,
        slug: str,
        collection_name: str,
        facts: List[str],
        source: Optional[str] = None,
    ) -> None:
        """Insert multiple facts for an artist at once."""
        conn = cls._connect()
        artist_name = slug.replace("-", " ")
        conn.execute(
            """INSERT INTO artists (slug, name, collection_name)
               VALUES (?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   name=excluded.name,
                   collection_name=excluded.collection_name""",
            (slug, artist_name, collection_name),
        )
        conn.executemany(
            """INSERT OR IGNORE INTO artist_facts (artist_slug, lang, fact, source)
               VALUES (?, 'en', ?, ?)""",
            [(slug, f, source) for f in facts],
        )
        conn.commit()

    @classmethod
    def get_artist_facts(cls, slug: str, collection_name: str) -> List[str]:
        """Return all English facts for an artist in a collection."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT af.fact FROM artist_facts af
               JOIN artists a ON a.slug = af.artist_slug
               WHERE af.artist_slug = ? AND a.collection_name = ? AND af.lang = 'en'
               ORDER BY af.id""",
            (slug, collection_name),
        ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _join_facts_by_slug(rows) -> Dict[str, str]:
        """Group ``(slug, fact)`` rows into ``{slug: facts joined by \\n\\n}``,
        preserving row order (matches the previous flat-file format)."""
        grouped: Dict[str, List[str]] = {}
        for slug, fact in rows:
            grouped.setdefault(slug, []).append(fact)
        return {slug: "\n\n".join(facts) for slug, facts in grouped.items()}

    @classmethod
    def get_all_artist_facts_by_collection(cls, collection_name: str) -> Dict[str, str]:
        """Return a dict of ``{artist_slug: joined_facts_text}`` for a collection.

        Facts within each artist are joined with ``\\n\\n`` to match the
        previous flat-file format.
        """
        conn = cls._connect()
        rows = conn.execute(
            """SELECT af.artist_slug, af.fact FROM artist_facts af
               JOIN artists a ON a.slug = af.artist_slug
               WHERE a.collection_name = ? AND af.lang = 'en'
               ORDER BY af.artist_slug, af.id""",
            (collection_name,),
        ).fetchall()

        return cls._join_facts_by_slug(rows)

    # ── Songs ──

    @classmethod
    def upsert_song(
        cls,
        slug: str,
        title: str,
        artist_slug: str,
        collection_name: str,
        recording_mbid: Optional[str] = None,
    ) -> None:
        conn = cls._connect()
        conn.execute(
            """INSERT INTO songs (slug, title, artist_slug, collection_name, recording_mbid)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   title=excluded.title,
                   artist_slug=excluded.artist_slug,
                   collection_name=excluded.collection_name,
                   recording_mbid=excluded.recording_mbid""",
            (slug, title, artist_slug, collection_name, recording_mbid),
        )
        conn.commit()

    # ── Song facts ──

    @classmethod
    def add_song_fact(
        cls,
        slug: str,
        collection_name: str,
        fact_text: str,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        conn = cls._connect()
        # Derive artist_slug and title from song slug (format: artist-song)
        parts = slug.split("-", 1)
        artist_slug = parts[0] if len(parts) > 1 else slug
        artist_name = artist_slug.replace("-", " ")
        # Ensure artist exists before inserting song (foreign key constraint)
        conn.execute(
            """INSERT INTO artists (slug, name, collection_name)
               VALUES (?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   name=excluded.name,
                   collection_name=excluded.collection_name""",
            (artist_slug, artist_name, collection_name),
        )
        conn.execute(
            """INSERT INTO songs (slug, title, artist_slug, collection_name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   title=excluded.title,
                   artist_slug=excluded.artist_slug,
                   collection_name=excluded.collection_name""",
            (slug, slug.replace("-", " "), artist_slug, collection_name),
        )
        conn.execute(
            """INSERT OR IGNORE INTO song_facts (song_slug, lang, fact, category, source)
               VALUES (?, 'en', ?, ?, ?)""",
            (slug, fact_text, category, source),
        )
        conn.commit()

    @classmethod
    def add_song_facts_batch(
        cls,
        slug: str,
        collection_name: str,
        facts: List[str],
        source: Optional[str] = None,
    ) -> None:
        conn = cls._connect()
        parts = slug.split("-", 1)
        artist_slug = parts[0] if len(parts) > 1 else slug
        artist_name = artist_slug.replace("-", " ")
        # Ensure artist exists before inserting song (foreign key constraint)
        conn.execute(
            """INSERT INTO artists (slug, name, collection_name)
               VALUES (?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   name=excluded.name,
                   collection_name=excluded.collection_name""",
            (artist_slug, artist_name, collection_name),
        )
        conn.execute(
            """INSERT INTO songs (slug, title, artist_slug, collection_name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   title=excluded.title,
                   artist_slug=excluded.artist_slug,
                   collection_name=excluded.collection_name""",
            (slug, slug.replace("-", " "), artist_slug, collection_name),
        )
        conn.executemany(
            """INSERT OR IGNORE INTO song_facts (song_slug, lang, fact, source)
               VALUES (?, 'en', ?, ?)""",
            [(slug, f, source) for f in facts],
        )
        conn.commit()

    @classmethod
    def get_song_facts(cls, slug: str, collection_name: str) -> List[str]:
        """Return all English facts for a song in a collection."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT sf.fact FROM song_facts sf
               JOIN songs s ON s.slug = sf.song_slug
               WHERE sf.song_slug = ? AND s.collection_name = ? AND sf.lang = 'en'
               ORDER BY sf.id""",
            (slug, collection_name),
        ).fetchall()
        return [r[0] for r in rows]

    @classmethod
    def get_all_song_facts_by_collection(cls, collection_name: str) -> Dict[str, str]:
        """Return ``{song_slug: joined_facts_text}`` for a collection."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT sf.song_slug, sf.fact FROM song_facts sf
               JOIN songs s ON s.slug = sf.song_slug
               WHERE s.collection_name = ? AND sf.lang = 'en'
               ORDER BY sf.song_slug, sf.id""",
            (collection_name,),
        ).fetchall()

        return cls._join_facts_by_slug(rows)

    @classmethod
    def count_facts_for_collection(cls, collection_name: str) -> int:
        """Return the total number of raw facts (artist + song) for a collection.

        Used by the ``refined_facts`` AI task to set ``n_total`` to the actual
        number of facts that will be fed into the LLM, rather than a
        misleading track count.
        """
        conn = cls._connect()
        # Artist facts scoped to collection via the artists table
        artist_row = conn.execute(
            "SELECT COUNT(*) FROM artist_facts af "
            "JOIN artists a ON a.slug = af.artist_slug "
            "WHERE a.collection_name = ? AND af.lang = 'en'",
            (collection_name,),
        ).fetchone()
        # Song facts scoped to collection via the songs table
        song_row = conn.execute(
            "SELECT COUNT(*) FROM song_facts sf "
            "JOIN songs s ON s.slug = sf.song_slug "
            "WHERE s.collection_name = ? AND sf.lang = 'en'",
            (collection_name,),
        ).fetchone()
        return (artist_row[0] or 0) + (song_row[0] or 0)

    # ── Random facts ──

    @classmethod
    def get_random_facts(
        cls,
        collection_name: str,
        limit: int = 5,
        lang: str = "en",
    ) -> List[dict]:
        """Return ``limit`` random facts from the collection's fact pool.

        Prefers REFINED facts (``refined_facts`` — the AI-filtered short pools,
        keyed per entity) in the requested language, then refined English, and
        only then tops up from the raw ``artist_facts``/``song_facts`` pool
        (which is English-only). One fact per entity, so the picks never all
        come from the same artist/song. Scoped to the collection via the
        ``artists``/``songs`` join — refined rows themselves are
        collection-independent (see :meth:`get_refined_facts`).

        Returns list of dicts:
        ``{"fact", "context", "type", "artist", "artist_slug", "title"}``
        (``title`` is None for artist facts) — the extra attribution fields
        let the route resolve a cover / artist image for the home strip.
        """
        import json as _json
        import random as _random

        conn = cls._connect()
        out: List[dict] = []
        seen: set = set()   # (type, context) — one fact per entity

        langs = [lang] + (["en"] if lang != "en" else [])
        for lng in langs:
            if len(out) >= limit:
                break
            rows = conn.execute(
                """
                SELECT refined_json, context, type, artist, artist_slug, title FROM (
                    SELECT
                        rf.refined_json,
                        a.name AS context,
                        'artist' AS type,
                        a.name AS artist,
                        a.slug AS artist_slug,
                        NULL AS title
                    FROM refined_facts rf
                    JOIN artists a ON a.slug = rf.scope_key
                    WHERE rf.scope = 'artist' AND rf.lang = ?
                      AND a.collection_name = ?

                    UNION ALL

                    SELECT
                        rf.refined_json,
                        a.name || ' — ' || s.title AS context,
                        'song' AS type,
                        a.name AS artist,
                        a.slug AS artist_slug,
                        s.title AS title
                    FROM refined_facts rf
                    JOIN songs s ON s.slug = rf.scope_key
                    JOIN artists a ON a.slug = s.artist_slug
                    WHERE rf.scope = 'song' AND rf.lang = ?
                      AND s.collection_name = ?
                )
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (lng, collection_name, lng, collection_name, limit * 4),
            ).fetchall()
            for refined_json, context, ftype, artist, artist_slug, title in rows:
                if len(out) >= limit:
                    break
                key = (ftype, context)
                if key in seen:
                    continue
                try:
                    texts = [
                        (item.get("text") or "").strip()
                        for item in _json.loads(refined_json)
                        if isinstance(item, dict)
                    ]
                except Exception:
                    texts = []
                texts = [t for t in texts if t]
                if not texts:
                    # Explicit [] — AI ran and kept nothing for this entity.
                    # Mark it consumed so the raw top-up below can't resurface
                    # the very facts the AI rejected (same semantics as
                    # get_track_facts honouring an explicit empty refined list).
                    seen.add(key)
                    continue
                out.append({
                    "fact": _random.choice(texts),
                    "context": context,
                    "type": ftype,
                    "artist": artist,
                    "artist_slug": artist_slug,
                    "title": title,
                })
                seen.add(key)

        if len(out) < limit:
            rows = conn.execute(
                """
                SELECT fact, context, type, artist, artist_slug, title FROM (
                    SELECT
                        af.fact,
                        a.name AS context,
                        'artist' AS type,
                        a.name AS artist,
                        a.slug AS artist_slug,
                        NULL AS title
                    FROM artist_facts af
                    JOIN artists a ON a.slug = af.artist_slug
                    WHERE a.collection_name = ? AND af.lang = 'en'

                    UNION ALL

                    SELECT
                        sf.fact,
                        a.name || ' — ' || s.title AS context,
                        'song' AS type,
                        a.name AS artist,
                        a.slug AS artist_slug,
                        s.title AS title
                    FROM song_facts sf
                    JOIN songs s ON s.slug = sf.song_slug
                    JOIN artists a ON a.slug = s.artist_slug
                    WHERE s.collection_name = ? AND sf.lang = 'en'
                )
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (collection_name, collection_name, limit * 4),
            ).fetchall()
            for fact, context, ftype, artist, artist_slug, title in rows:
                if len(out) >= limit:
                    break
                key = (ftype, context)
                if key in seen:
                    continue
                out.append({
                    "fact": fact, "context": context, "type": ftype,
                    "artist": artist, "artist_slug": artist_slug, "title": title,
                })
                seen.add(key)
        return out

    # ── Convenience helpers ──

    @classmethod
    def ensure_artist(cls, name: str, collection_name: str) -> str:
        """Ensure an artist row exists and return its slug."""
        slug = _slugify(name)
        cls.upsert_artist(slug, name, collection_name)
        return slug

    @classmethod
    def ensure_song(
        cls,
        artist: str,
        title: str,
        collection_name: str,
    ) -> Tuple[str, str]:
        """Ensure song + artist rows exist. Returns (artist_slug, song_slug)."""
        artist_slug = cls.ensure_artist(artist, collection_name)
        song_slug = _slugify(artist) + "-" + _slugify(title)
        cls.upsert_song(song_slug, title, artist_slug, collection_name)
        return artist_slug, song_slug

    @classmethod
    def close(cls) -> None:
        """Close every cached connection (shutdown + test teardown).

        Closes both the per-thread map and the legacy _instance slot (the
        integration conftest's patched _connect still stores there).
        Worker-thread connections are closed from the calling thread —
        allowed because they are opened with check_same_thread=False."""
        with cls._connections_lock:
            conns = [conn for (_path, conn) in cls._connections.values()]
            cls._connections.clear()
        if cls._instance is not None:
            conns.append(cls._instance)
            cls._instance = None
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass

    # ── Track reactions ──

    @classmethod
    def set_reaction(
        cls,
        track_id: str,
        collection_name: str,
        reaction: Literal["like", "dislike"] | None,
    ) -> None:
        """Upsert or delete a track reaction scoped by collection."""
        conn = cls._connect()
        if reaction is None:
            conn.execute(
                "DELETE FROM track_reactions WHERE track_id = ? AND collection_name = ?",
                (track_id, collection_name),
            )
        else:
            conn.execute(
                """INSERT INTO track_reactions (track_id, collection_name, reaction)
                   VALUES (?, ?, ?)
                   ON CONFLICT(track_id, collection_name) DO UPDATE SET
                       reaction=excluded.reaction,
                       updated_at=CURRENT_TIMESTAMP""",
                (track_id, collection_name, reaction),
            )
        conn.commit()

    @classmethod
    def get_reaction(
        cls,
        track_id: str,
        collection_name: str,
    ) -> Literal["like", "dislike"] | None:
        """Return the stored reaction for a track in a collection, or None."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT reaction FROM track_reactions WHERE track_id = ? AND collection_name = ?",
            (track_id, collection_name),
        ).fetchone()
        return row[0] if row else None

    @classmethod
    def get_liked_track_ids_with_updated_at(
        cls, collection_name: str
    ) -> list[tuple[str, str]]:
        """Return list of (track_id, updated_at ISO string) for tracks with
        reaction='like' in the given collection, ordered newest-first."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT track_id, updated_at FROM track_reactions "
            "WHERE collection_name = ? AND reaction = 'like' "
            "ORDER BY updated_at DESC",
            (collection_name,),
        ).fetchall()
        # updated_at is declared TIMESTAMP — with PARSE_DECLTYPES the sqlite3
        # adapter returns it as a datetime.datetime. Coerce to ISO string so
        # the API response model (Pydantic str field) accepts it directly.
        # Use .isoformat() (T-separator) so `new Date(liked_at)` in the
        # browser parses reliably across engines; str(datetime) yields a
        # space-separated form that Safari/older browsers reject.
        return [
            (r[0], r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]))
            for r in rows
        ]

    # Tables holding per-collection Qdrant point ids that re-indexing orphans.
    _TRACK_REF_TABLES = ("track_reactions", "playback_events")
    _PRUNE_DELETE_CHUNK = 500

    @classmethod
    def prune_orphaned_tracks(
        cls, collection_name: str, live_ids: set[str]
    ) -> dict[str, int]:
        """Delete rows whose track_id is no longer a live Qdrant point id.

        Re-indexing drops + recreates the collection with fresh ``uuid4`` point
        ids, leaving the old ids dangling in ``track_reactions`` /
        ``playback_events``. Those orphans surface in «Поток» as empty-payload
        «—» tracks that 404 on /stream and /lyrics. Called after a fit completes
        with the collection's current id set. Returns ``{table: rows_deleted}``.
        """
        conn = cls._connect()
        removed: dict[str, int] = {}
        for table in cls._TRACK_REF_TABLES:
            rows = conn.execute(
                f"SELECT DISTINCT track_id FROM {table} WHERE collection_name = ?",
                (collection_name,),
            ).fetchall()
            orphans = [r[0] for r in rows if r[0] not in live_ids]
            deleted = 0
            for i in range(0, len(orphans), cls._PRUNE_DELETE_CHUNK):
                batch = orphans[i: i + cls._PRUNE_DELETE_CHUNK]
                placeholders = ",".join("?" * len(batch))
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE collection_name = ? "
                    f"AND track_id IN ({placeholders})",
                    (collection_name, *batch),
                )
                deleted += cur.rowcount
            removed[table] = deleted
        conn.commit()
        return removed

    @classmethod
    def get_reactions_for_tracks(
        cls, collection_name: str, track_ids: list[str]
    ) -> dict[str, str]:
        """Return a mapping of track_id -> reaction for known tracks only.

        Tracks without a reaction row are omitted from the result.
        Caller should treat missing keys as 'no reaction'.

        Uses a single SELECT with ``IN (?, ?, ...)`` placeholders to avoid
        N+1 queries when the autoplay pipeline filters dozens of candidates.
        ``track_ids`` is interpolated as positional parameters — no SQL
        injection surface despite the dynamic placeholder count.
        """
        if not track_ids:
            return {}
        placeholders = ",".join("?" * len(track_ids))
        conn = cls._connect()
        rows = conn.execute(
            f"SELECT track_id, reaction FROM track_reactions "
            f"WHERE collection_name = ? AND track_id IN ({placeholders})",
            (collection_name, *track_ids),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Collection settings (per-collection text_model, etc.) ──

    @classmethod
    def set_collection_text_model(cls, collection_name: str, text_model: Optional[str]) -> None:
        """Record which text model a collection was indexed with.

        Idempotent upsert. Pass ``text_model=None`` to clear the binding (rare —
        usually a collection always has exactly one model).
        """
        conn = cls._connect()
        conn.execute(
            """INSERT INTO collection_settings (collection_name, text_model, indexed_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(collection_name) DO UPDATE SET
                 text_model = excluded.text_model,
                 indexed_at = CURRENT_TIMESTAMP
            """,
            (collection_name, text_model),
        )
        conn.commit()

    @classmethod
    def get_collection_text_model(cls, collection_name: str) -> Optional[str]:
        """Return the text model used to index this collection, or None if unset."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT text_model FROM collection_settings WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return row[0] if row else None

    @classmethod
    def get_collection_settings(cls, collection_name: str) -> Optional[Dict]:
        """Return full settings dict for a collection, or None if unset."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT text_model, indexed_at FROM collection_settings WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        if row is None:
            return None
        return {"text_model": row[0], "indexed_at": row[1]}

    # ── AI mode (Plan 6) ──

    @classmethod
    def get_collection_ai_enabled(cls, collection_name: str) -> bool:
        """Returns True when no row exists — pre-migration collections that
        never had `collection_settings` written should remain AI-on. The
        IndexingModal explicitly persists 0/1 for newly-created collections."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT ai_enabled FROM collection_settings WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return bool(row[0]) if row else True

    @classmethod
    def set_collection_ai_enabled(cls, collection_name: str, enabled: bool) -> None:
        conn = cls._connect()
        conn.execute(
            """INSERT INTO collection_settings (collection_name, ai_enabled)
               VALUES (?, ?)
               ON CONFLICT(collection_name) DO UPDATE SET
                 ai_enabled = excluded.ai_enabled""",
            (collection_name, 1 if enabled else 0),
        )
        conn.commit()

    # ── Stream RecSys: sonic-axis normalisation stats ──

    @classmethod
    def set_axis_norm_stats(cls, collection_name: str, stats: Dict) -> None:
        """Persist per-collection axis mean/std as JSON.

        ``stats`` shape: ``{"version": str, "n": int,
        "mean": {axis: float}, "std": {axis: float}}`` — recomputed after every
        indexing run so z-scores stay honest as the collection grows.
        """
        conn = cls._connect()
        conn.execute(
            """INSERT INTO collection_settings (collection_name, axis_norm_stats)
               VALUES (?, ?)
               ON CONFLICT(collection_name) DO UPDATE SET
                 axis_norm_stats = excluded.axis_norm_stats""",
            (collection_name, json.dumps(stats)),
        )
        conn.commit()

    @classmethod
    def set_recsys_llm_text(
        cls, collection_name: str, kind: str, lang: str,
        source_hash: str, content: Dict,
    ) -> None:
        """Upsert a cached LLM text (one row per collection+kind+lang)."""
        conn = cls._connect()
        conn.execute(
            """INSERT INTO recsys_llm_texts
                 (collection_name, kind, lang, source_hash, content_json, created_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(collection_name, kind, lang) DO UPDATE SET
                 source_hash = excluded.source_hash,
                 content_json = excluded.content_json,
                 created_at = CURRENT_TIMESTAMP""",
            (collection_name, kind, lang, source_hash, json.dumps(content)),
        )
        conn.commit()

    @classmethod
    def get_recsys_llm_text(
        cls, collection_name: str, kind: str, lang: str,
        source_hash: Optional[str] = None,
    ) -> Optional[Dict]:
        """Return cached content, or None when absent / corrupt / stale.

        When ``source_hash`` is provided, a stored row with a different hash is
        treated as stale (the taste profile moved on) and None is returned.
        """
        conn = cls._connect()
        row = conn.execute(
            "SELECT source_hash, content_json FROM recsys_llm_texts "
            "WHERE collection_name = ? AND kind = ? AND lang = ?",
            (collection_name, kind, lang),
        ).fetchone()
        if row is None:
            return None
        if source_hash is not None and row[0] != source_hash:
            return None
        try:
            return json.loads(row[1])
        except (TypeError, ValueError):
            logger.warning("[MetadataDB] corrupt recsys_llm_texts row for %s/%s", collection_name, kind)
            return None

    @classmethod
    def set_stream_liked_share(cls, collection_name: str, share: float) -> None:
        """Persist the liked/new slider position (0.0–1.0) for the stream."""
        conn = cls._connect()
        conn.execute(
            """INSERT INTO collection_settings (collection_name, stream_liked_share)
               VALUES (?, ?)
               ON CONFLICT(collection_name) DO UPDATE SET
                 stream_liked_share = excluded.stream_liked_share""",
            (collection_name, float(share)),
        )
        conn.commit()

    @classmethod
    def get_stream_liked_share(cls, collection_name: str) -> Optional[float]:
        conn = cls._connect()
        row = conn.execute(
            "SELECT stream_liked_share FROM collection_settings WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    @classmethod
    def get_axis_norm_stats(cls, collection_name: str) -> Optional[Dict]:
        """Return the stored axis stats dict, or None when absent/corrupt."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT axis_norm_stats FROM collection_settings WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            logger.warning("[MetadataDB] corrupt axis_norm_stats for %s — ignoring", collection_name)
            return None

    # ── Playback history ──

    @classmethod
    def record_playback_event(
        cls,
        *,
        session_id: str,
        collection_name: str,
        track_id: str,
        played_sec: float,
        total_dur: float | None,
        interacted: bool | None = None,
        influence: bool = True,
    ) -> int:
        """Insert a playback event. Returns the new row id.

        ``skipped_early`` is derived server-side: when ``total_dur`` is known,
        ``True`` requires BOTH ``played_sec < 30`` AND
        ``played_sec / total_dur < 0.30`` (so a short track played to completion
        is not falsely flagged). When ``total_dur`` is missing, falls back to
        the absolute 30-second threshold alone.

        ``influence=False`` marks events that should NOT shape the taste profile
        (e.g. hand-queued tracks). They are still stored for anti-repeat purposes.
        """
        if total_dur and total_dur > 0.0:
            # Both signals required to count as a skip — short play AND short ratio.
            skipped_early = played_sec < 30.0 and played_sec / total_dur < 0.30
        else:
            # No duration available: fall back to absolute threshold.
            skipped_early = played_sec < 30.0
        conn = cls._connect()
        cur = conn.execute(
            "INSERT INTO playback_events "
            "(session_id, collection_name, track_id, played_sec, total_dur, skipped_early, interacted, influence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, collection_name, track_id, played_sec, total_dur,
             1 if skipped_early else 0,
             None if interacted is None else (1 if interacted else 0),
             1 if influence else 0),
        )
        conn.commit()
        return int(cur.lastrowid)

    @classmethod
    def get_playback_signals(
        cls, collection_name: str, limit: int = 2000,
    ) -> list[Dict]:
        """Last ``limit`` playback events in CHRONOLOGICAL order (Stream RecSys).

        Chronology matters: replay detection and the idle rule scan a session's
        events in play order. ``played_at`` is normalised to ``datetime``;
        ``interacted`` to ``bool | None`` (NULL = legacy = treated as action).
        """
        from datetime import datetime as _dt
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id, played_sec, total_dur, played_at, session_id, interacted, influence
               FROM playback_events
               WHERE collection_name = ?
               ORDER BY id DESC LIMIT ?""",
            (collection_name, limit),
        ).fetchall()
        out: List[Dict] = []
        for r in reversed(rows):
            played_at = r[3]
            if not hasattr(played_at, "isoformat"):
                try:
                    played_at = _dt.fromisoformat(str(played_at))
                except ValueError:
                    continue  # unparseable timestamp — drop the row, not the request
            out.append({
                "track_id": r[0],
                "played_sec": float(r[1] or 0.0),
                "total_dur": float(r[2]) if r[2] is not None else None,
                "played_at": played_at,
                "session_id": r[4],
                "interacted": None if r[5] is None else bool(r[5]),
                "influence": True if r[6] is None else bool(r[6]),
            })
        return out

    @classmethod
    def get_reactions_with_updated_at(
        cls, collection_name: str,
    ) -> list[Tuple[str, str, str]]:
        """All reactions as ``(track_id, reaction, updated_at_iso)`` (Stream RecSys)."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT track_id, reaction, updated_at FROM track_reactions "
            "WHERE collection_name = ?",
            (collection_name,),
        ).fetchall()
        return [
            (r[0], r[1],
             r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]))
            for r in rows
        ]

    # ── Taste signals (огонёк/вода journal) ──

    @classmethod
    def record_taste_signal(
        cls, *, session_id: str, collection_name: str, track_id: str, kind: str,
    ) -> int:
        """Append one огонёк/вода gesture (kind 'fire'|'water'). Returns row id."""
        conn = cls._connect()
        cur = conn.execute(
            "INSERT INTO taste_signals (collection_name, session_id, track_id, kind) "
            "VALUES (?, ?, ?, ?)",
            (collection_name, session_id, track_id, kind),
        )
        conn.commit()
        return int(cur.lastrowid)

    @classmethod
    def get_taste_signals(
        cls, collection_name: str, *, session_id: str | None = None,
        limit: int = 5000,
    ) -> list[Tuple[str, str, str]]:
        """Fire/water signals as ``(track_id, kind, created_at_iso)``, newest first.

        ``session_id`` restricts to one session (the ephemeral wave); omit it for
        the full history (island deposit + «favorites» rank). Decay is
        order-independent, so newest-first ordering is just for the cap.
        """
        conn = cls._connect()
        if session_id is None:
            rows = conn.execute(
                "SELECT track_id, kind, created_at FROM taste_signals "
                "WHERE collection_name = ? ORDER BY id DESC LIMIT ?",
                (collection_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT track_id, kind, created_at FROM taste_signals "
                "WHERE collection_name = ? AND session_id = ? ORDER BY id DESC LIMIT ?",
                (collection_name, session_id, limit),
            ).fetchall()
        return [
            (r[0], r[1],
             r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]))
            for r in rows
        ]

    @classmethod
    def get_recent_tracks(
        cls, collection_name: str, limit: int = 50,
    ) -> list[tuple[str, str, int]]:
        """Returns list of (track_id, last_played_iso, play_count_non_skipped),
        deduped by track_id, ordered by last_played DESC."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id,
                      MAX(played_at) AS last_played,
                      SUM(CASE WHEN skipped_early=0 THEN 1 ELSE 0 END) AS plays
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY track_id
               ORDER BY last_played DESC
               LIMIT ?""",
            (collection_name, limit),
        ).fetchall()
        # `played_at` is TIMESTAMP-typed; under PARSE_DECLTYPES it comes back as datetime.
        # Coerce to ISO format string (T-separated) for JSON clients (mirror Task 4 fix).
        return [
            (r[0],
             r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
             int(r[2] or 0))
            for r in rows
        ]

    @classmethod
    def get_play_recency_map(cls, collection_name: str) -> dict[str, str]:
        """Return {track_id: last_played_iso} for all played tracks in a
        collection. Used by Rediscover (oldest gap) and For-You seed (recency)."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id, MAX(played_at) AS last_played
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY track_id""",
            (collection_name,),
        ).fetchall()
        result = {}
        for r in rows:
            track_id = r[0]
            played_at = r[1]
            # PARSE_DECLTYPES may return datetime or string depending on SQLite mode.
            # Convert to ISO format with T separator for consistent client parsing.
            if hasattr(played_at, "isoformat"):
                result[track_id] = played_at.isoformat()
            else:
                # If it's a string, ensure T-separated format (replace space with T)
                s = str(played_at)
                result[track_id] = s.replace(" ", "T", 1) if " " in s else s
        return result

    @classmethod
    def get_listening_total(cls, collection_name: str) -> tuple[float, str | None]:
        """Return (total_played_sec, first_played_iso_or_None)."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(played_sec), 0), MIN(played_at) "
            "FROM playback_events WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        first_played = row[1]
        first_played_iso = (
            first_played.isoformat() if hasattr(first_played, "isoformat") else
            (str(first_played) if first_played is not None else None)
        )
        return float(row[0] or 0), first_played_iso

    @classmethod
    def get_top_played_track(cls, collection_name: str) -> tuple[str, int] | None:
        """Return (track_id, play_count_non_skipped) for the most-played track, or None."""
        conn = cls._connect()
        row = conn.execute(
            """SELECT track_id, COUNT(*) AS plays
               FROM playback_events
               WHERE collection_name = ? AND skipped_early = 0
               GROUP BY track_id
               ORDER BY plays DESC
               LIMIT 1""",
            (collection_name,),
        ).fetchone()
        return (row[0], int(row[1])) if row else None

    @classmethod
    def get_peak_hour(
        cls, collection_name: str, tz_offset_minutes: int = 0,
    ) -> int | None:
        """Return the most-frequent hour-of-day (0-23) across all non-skipped events, or None.

        ``played_at`` is stored in UTC (SQLite ``CURRENT_TIMESTAMP``). To report
        the hour in the *user's* local time, the caller passes their UTC offset
        in minutes (e.g. UTC+3 → +180); the offset is applied as a SQLite
        datetime modifier before the hour is extracted. ``tz_offset_minutes`` is
        coerced to int, so it cannot inject into the modifier string.
        """
        modifier = f"{int(tz_offset_minutes):+d} minutes"
        conn = cls._connect()
        row = conn.execute(
            """SELECT CAST(strftime('%H', played_at, ?) AS INT) AS h, COUNT(*) AS n
               FROM playback_events
               WHERE collection_name = ? AND skipped_early = 0
               GROUP BY h
               ORDER BY n DESC
               LIMIT 1""",
            (modifier, collection_name),
        ).fetchone()
        return int(row[0]) if row else None

    @classmethod
    def get_play_counts_by_track(
        cls, collection_name: str
    ) -> dict[str, int]:
        """Non-skipped play counts grouped by track_id."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT track_id, COUNT(*) FROM playback_events "
            "WHERE collection_name = ? AND skipped_early = 0 "
            "GROUP BY track_id",
            (collection_name,),
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    @classmethod
    def get_plays_by_local_day(
        cls, collection_name: str, tz_offset_minutes: int = 0,
    ) -> list[tuple[str, int]]:
        """Return [(local_date 'YYYY-MM-DD', play_count), ...] ordered by date.

        Counts every play event (skipped or not). ``played_at`` is UTC; the
        caller's offset is applied as a SQLite datetime modifier (same pattern
        as ``get_peak_hour``; coerced to int so it can't inject)."""
        modifier = f"{int(tz_offset_minutes):+d} minutes"
        conn = cls._connect()
        rows = conn.execute(
            """SELECT strftime('%Y-%m-%d', played_at, ?) AS d, COUNT(*) AS n
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY d ORDER BY d""",
            (modifier, collection_name),
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows if r[0]]

    @classmethod
    def get_weekly_listening_summary(
        cls, collection_name: str, tz_offset_minutes: int = 0,
    ) -> tuple[float, str | None, int]:
        """Return (seconds_listened, top_genre, discoveries) for the CURRENT
        calendar week — Monday through today, in the caller's local time
        (same modifier pattern as ``get_plays_by_local_day``).

        ``discoveries`` = tracks whose first-ever playback event falls inside
        this week, i.e. music the user heard for the first time. The counter
        grows through the week and resets every Monday.

        SQLite Monday idiom: ``weekday 1`` moves FORWARD to Monday (or stays
        if already Monday), so stepping back 6 days first always lands on the
        Monday of the current local week."""
        modifier = f"{int(tz_offset_minutes):+d} minutes"
        week_start = "date('now', ?, '-6 days', 'weekday 1')"
        conn = cls._connect()
        total_row = conn.execute(
            f"""SELECT COALESCE(SUM(played_sec), 0)
               FROM playback_events
               WHERE collection_name = ?
                 AND date(played_at, ?) >= {week_start}""",
            (collection_name, modifier, modifier),
        ).fetchone()
        genre_row = conn.execute(
            f"""SELECT tm.genre, COUNT(*) AS plays
               FROM playback_events pe
               JOIN track_metadata tm
                 ON tm.collection_name = pe.collection_name AND tm.track_id = pe.track_id
               WHERE pe.collection_name = ?
                 AND pe.skipped_early = 0
                 AND date(pe.played_at, ?) >= {week_start}
                 AND tm.genre IS NOT NULL AND tm.genre != ''
               GROUP BY tm.genre
               ORDER BY plays DESC
               LIMIT 1""",
            (collection_name, modifier, modifier),
        ).fetchone()
        # MIN over the TEXT timestamps is safe: CURRENT_TIMESTAMP's
        # "YYYY-MM-DD HH:MM:SS" sorts lexicographically = chronologically.
        disc_row = conn.execute(
            f"""SELECT COUNT(*) FROM (
                   SELECT track_id, MIN(played_at) AS first_played
                   FROM playback_events
                   WHERE collection_name = ?
                   GROUP BY track_id
               )
               WHERE date(first_played, ?) >= {week_start}""",
            (collection_name, modifier, modifier),
        ).fetchone()
        return (
            float(total_row[0] or 0),
            (genre_row[0] if genre_row else None),
            int(disc_row[0] or 0),
        )

    @classmethod
    def get_plays_by_local_hour(
        cls, collection_name: str, tz_offset_minutes: int = 0,
    ) -> list[int]:
        """Return a 24-element list of play counts per local hour (index 0-23)."""
        modifier = f"{int(tz_offset_minutes):+d} minutes"
        conn = cls._connect()
        rows = conn.execute(
            """SELECT CAST(strftime('%H', played_at, ?) AS INT) AS h, COUNT(*) AS n
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY h""",
            (modifier, collection_name),
        ).fetchall()
        out = [0] * 24
        for h, n in rows:
            if h is not None and 0 <= int(h) <= 23:
                out[int(h)] = int(n)
        return out

    @classmethod
    def get_top_track_on_local_day(
        cls, collection_name: str, day: str, tz_offset_minutes: int = 0,
    ) -> tuple[str, int] | None:
        """(track_id, plays) of the most-played non-skipped track on a local day."""
        modifier = f"{int(tz_offset_minutes):+d} minutes"
        conn = cls._connect()
        row = conn.execute(
            """SELECT track_id, COUNT(*) AS n
               FROM playback_events
               WHERE collection_name = ?
                 AND strftime('%Y-%m-%d', played_at, ?) = ?
                 AND skipped_early = 0
               GROUP BY track_id ORDER BY n DESC LIMIT 1""",
            (collection_name, modifier, day),
        ).fetchone()
        return (row[0], int(row[1])) if row else None

    @classmethod
    def get_track_completion_stats(
        cls, collection_name: str,
    ) -> list[tuple[str, int, float | None, float]]:
        """Per-track engagement: [(track_id, plays, avg_completion|None, skip_rate)].

        ``completion`` = played_sec / total_dur (clamped to 1.0), averaged only
        over plays that have a known ``total_dur`` (NULLs ignored by AVG, so the
        per-track value is None when no play was timed). ``skip_rate`` is the
        fraction of plays flagged ``skipped_early``."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id,
                      COUNT(*) AS plays,
                      AVG(CASE WHEN total_dur IS NOT NULL AND total_dur > 0
                               THEN min(played_sec * 1.0 / total_dur, 1.0) END) AS comp,
                      AVG(skipped_early) AS skip_rate
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY track_id""",
            (collection_name,),
        ).fetchall()
        return [
            (r[0], int(r[1]), (None if r[2] is None else float(r[2])), float(r[3] or 0))
            for r in rows
        ]

    @classmethod
    def get_engagement_detail_stats(
        cls, collection_name: str,
    ) -> list[tuple[str, int, float | None, int, int, float | None]]:
        """Richer per-track engagement for the "what you finish / abandon" panel.

        Returns ``[(track_id, plays, avg_completion|None, finish_count,
        skip_count, avg_skip_sec|None)]`` where:

          - ``finish_count`` = number of plays heard through to the end (≥90% of
            a known duration). Plays without a timed duration can't be judged and
            don't count.
          - ``skip_count`` = number of plays flagged ``skipped_early``.
          - ``avg_skip_sec`` = mean seconds heard before those early skips (None
            when the track was never skipped early)."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id,
                      COUNT(*) AS plays,
                      AVG(CASE WHEN total_dur IS NOT NULL AND total_dur > 0
                               THEN min(played_sec * 1.0 / total_dur, 1.0) END) AS comp,
                      SUM(CASE WHEN total_dur IS NOT NULL AND total_dur > 0
                                AND played_sec >= 0.9 * total_dur
                               THEN 1 ELSE 0 END) AS finishes,
                      SUM(CASE WHEN skipped_early = 1 THEN 1 ELSE 0 END) AS skips,
                      AVG(CASE WHEN skipped_early = 1 THEN played_sec END) AS avg_skip_sec
               FROM playback_events
               WHERE collection_name = ?
               GROUP BY track_id""",
            (collection_name,),
        ).fetchall()
        return [
            (
                r[0], int(r[1]),
                (None if r[2] is None else float(r[2])),
                int(r[3] or 0), int(r[4] or 0),
                (None if r[5] is None else float(r[5])),
            )
            for r in rows
        ]

    @classmethod
    def get_overall_completion(cls, collection_name: str) -> float | None:
        """Library-wide mean play completion (0..1), or None if no timed plays."""
        conn = cls._connect()
        row = conn.execute(
            """SELECT AVG(min(played_sec * 1.0 / total_dur, 1.0))
               FROM playback_events
               WHERE collection_name = ? AND total_dur IS NOT NULL AND total_dur > 0""",
            (collection_name,),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    # ── AI Indexing jobs (Plan 3 Task 11) ──

    @classmethod
    def record_ai_job(
        cls, job_id: str, task_type: str, collection_name: str,
        lang: str, n_total: int,
    ) -> None:
        """Insert a new job row in 'queued' status."""
        conn = cls._connect()
        conn.execute(
            "INSERT INTO ai_indexing_jobs "
            "(job_id, task_type, collection_name, lang, status, n_total) "
            "VALUES (?, ?, ?, ?, 'queued', ?)",
            (job_id, task_type, collection_name, lang, n_total),
        )
        conn.commit()

    @classmethod
    def update_ai_job(
        cls,
        job_id: str,
        *,
        status: Optional[str] = None,
        n_done: Optional[int] = None,
        n_failed: Optional[int] = None,
        n_skipped: Optional[int] = None,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        """Patch a job row. Only non-None fields are written.

        ``finished=True`` sets ``finished_at = CURRENT_TIMESTAMP``.
        """
        sets: list[str] = []
        params: list = []
        if status is not None:
            sets.append("status = ?"); params.append(status)
        if n_done is not None:
            sets.append("n_done = ?"); params.append(n_done)
        if n_failed is not None:
            sets.append("n_failed = ?"); params.append(n_failed)
        if n_skipped is not None:
            sets.append("n_skipped = ?"); params.append(n_skipped)
        if error is not None:
            sets.append("error = ?"); params.append(error)
        if finished:
            sets.append("finished_at = CURRENT_TIMESTAMP")
        if not sets:
            return
        params.append(job_id)
        conn = cls._connect()
        conn.execute(
            f"UPDATE ai_indexing_jobs SET {', '.join(sets)} WHERE job_id = ?",
            params,
        )
        conn.commit()

    @classmethod
    def get_latest_ai_job(
        cls, collection_name: str, task_type: str,
    ) -> Optional[dict]:
        """Return the most-recent job row for the given (collection, task_type)
        as a plain dict, or None if no job exists."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT job_id, task_type, collection_name, lang, status, "
            "       n_total, n_done, n_failed, n_skipped, "
            "       started_at, finished_at, error "
            "FROM ai_indexing_jobs "
            "WHERE collection_name = ? AND task_type = ? "
            "ORDER BY started_at DESC, rowid DESC LIMIT 1",
            (collection_name, task_type),
        ).fetchone()
        if not row:
            return None
        keys = ["job_id", "task_type", "collection_name", "lang", "status",
                "n_total", "n_done", "n_failed", "n_skipped",
                "started_at", "finished_at", "error"]
        return dict(zip(keys, row))

    # ── Sonic Descriptor ──

    @classmethod
    def upsert_sonic_descriptor(
        cls,
        song_slug: str,
        tags: Optional[List[Dict]] = None,
        sonic_class: Optional[str] = None,
        confidence: Optional[float] = None,
        audio_signature: Optional[str] = None,
    ) -> None:
        """Persist Sonic Descriptor fields for a track. Pass None to leave a field unchanged."""
        import json as _json
        conn = cls._connect()
        sets = []
        params: list = []
        if tags is not None:
            sets.append("sonic_tags_json = ?")
            params.append(_json.dumps(tags))
        if sonic_class is not None:
            sets.append("sonic_class = ?")
            params.append(sonic_class)
        if confidence is not None:
            sets.append("sonic_class_confidence = ?")
            params.append(confidence)
        if audio_signature is not None:
            sets.append("audio_signature = ?")
            params.append(audio_signature)
        if not sets:
            return
        params.append(song_slug)
        conn.execute(f"UPDATE songs SET {', '.join(sets)} WHERE slug = ?", params)
        conn.commit()

    @classmethod
    def get_sonic_descriptor(cls, song_slug: str) -> Optional[Dict]:
        """Return dict with tags / sonic_class / sonic_class_confidence / audio_signature, or None if song unknown."""
        import json as _json
        conn = cls._connect()
        row = conn.execute(
            "SELECT sonic_tags_json, sonic_class, sonic_class_confidence, audio_signature FROM songs WHERE slug = ?",
            (song_slug,),
        ).fetchone()
        if row is None:
            return None
        tags_json, sclass, conf, sig = row
        return {
            "tags": _json.loads(tags_json) if tags_json else [],
            "sonic_class": sclass,
            "sonic_class_confidence": conf,
            "audio_signature": sig,
        }

    @classmethod
    def get_sonic_facets(cls, top_k: int = 50) -> dict:
        """Return aggregate counts of sonic_tags across the songs table.

        Returns
        -------
        {
          "tags": [{"value": str, "count": int}, ...]  # sorted desc by count, capped at top_k
        }
        """
        import json as _json
        from collections import Counter

        conn = cls._connect()
        tag_rows = conn.execute(
            "SELECT sonic_tags_json FROM songs "
            "WHERE sonic_tags_json IS NOT NULL AND sonic_tags_json != ''"
        ).fetchall()

        tag_counter: Counter[str] = Counter()
        for (raw,) in tag_rows:
            try:
                tags = _json.loads(raw)
            except (TypeError, ValueError, _json.JSONDecodeError):
                continue
            if not isinstance(tags, list):
                continue
            for t in tags:
                if isinstance(t, dict) and "tag" in t:
                    tag_counter[t["tag"]] += 1
                elif isinstance(t, str):
                    tag_counter[t] += 1

        return {
            "tags": [
                {"value": v, "count": n}
                for v, n in tag_counter.most_common(top_k)
            ],
        }

    # ── Sonic Vibe cache (Plan 3 Task 14) ──

    @classmethod
    def get_sonic_vibe(
        cls, track_id: str, collection_name: str, lang: str,
    ) -> Optional[dict]:
        """Return cached vibe ({phrase, generated_at}) or None."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT phrase, generated_at FROM sonic_vibes "
            "WHERE track_id = ? AND collection_name = ? AND lang = ?",
            (track_id, collection_name, lang),
        ).fetchone()
        if not row:
            return None
        return {"phrase": row[0], "generated_at": str(row[1])}

    @classmethod
    def set_sonic_vibe(
        cls, track_id: str, collection_name: str, lang: str, phrase: str,
    ) -> None:
        """Upsert a vibe. Overwrites existing row for same (track, collection, lang)."""
        conn = cls._connect()
        conn.execute(
            "INSERT INTO sonic_vibes (track_id, collection_name, lang, phrase) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(track_id, collection_name, lang) DO UPDATE SET "
            "phrase = excluded.phrase, generated_at = CURRENT_TIMESTAMP",
            (track_id, collection_name, lang, phrase),
        )
        conn.commit()

    @classmethod
    def delete_sonic_vibes(cls, collection_name: str) -> int:
        """Drop all cached vibes for the given collection. Returns rows deleted."""
        conn = cls._connect()
        cur = conn.execute(
            "DELETE FROM sonic_vibes WHERE collection_name = ?",
            (collection_name,),
        )
        conn.commit()
        return int(cur.rowcount)

    # ── Refined Facts cache (Plan 3 Task 15) ──

    @classmethod
    def get_refined_facts(
        cls, *, scope: str, scope_key: str, collection_name: str, lang: str,
    ) -> Optional[list[str]]:
        """Return refined facts as a plain text list, or None if no refined row exists.

        Note: an EXPLICIT empty list (set_refined_facts(refined=[])) is a valid
        signal — "AI indexed, judged nothing interesting". The caller should
        respect that by returning [] instead of falling back to originals.
        """
        import json as _json
        conn = cls._connect()
        # collection_name is accepted for signature stability but intentionally
        # ignored: refined facts are keyed by (scope, scope_key, lang) so a
        # song/artist refined under any collection is reused everywhere.
        row = conn.execute(
            "SELECT refined_json FROM refined_facts "
            "WHERE scope = ? AND scope_key = ? AND lang = ?",
            (scope, scope_key, lang),
        ).fetchone()
        if not row:
            return None
        try:
            arr = _json.loads(row[0])
            return [item.get("text", "") for item in arr if isinstance(item, dict)]
        except Exception:
            return []

    @classmethod
    def set_refined_facts(
        cls, *, scope: str, scope_key: str, collection_name: str,
        lang: str, refined: list[str],
    ) -> None:
        """Upsert a refined-facts row. Empty `refined=[]` is explicit and
        signals 'AI judged nothing interesting' — not 'no AI run yet'."""
        import json as _json
        payload = _json.dumps([{"text": t} for t in refined], ensure_ascii=False)
        conn = cls._connect()
        # Key is (scope, scope_key, lang); collection_name is stored as provenance
        # (last writer) and updated on conflict so the cache-reset-by-collection
        # admin action still has something meaningful to match on.
        conn.execute(
            "INSERT INTO refined_facts (scope, scope_key, lang, collection_name, refined_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(scope, scope_key, lang) DO UPDATE SET "
            "refined_json = excluded.refined_json, "
            "collection_name = excluded.collection_name, "
            "generated_at = CURRENT_TIMESTAMP",
            (scope, scope_key, lang, collection_name, payload),
        )
        conn.commit()

    @classmethod
    def delete_refined_facts(cls, collection_name: str) -> int:
        """Drop all refined-fact rows for the given collection. Returns rows deleted."""
        conn = cls._connect()
        cur = conn.execute(
            "DELETE FROM refined_facts WHERE collection_name = ?",
            (collection_name,),
        )
        conn.commit()
        return int(cur.rowcount)

    # ── Artist Bio cache (Plan 5) ──

    @classmethod
    def get_artist_bio(
        cls, artist_slug: str, collection_name: str, lang: str,
    ) -> Optional[str]:
        conn = cls._connect()
        row = conn.execute(
            "SELECT bio_text FROM artist_bios "
            "WHERE artist_slug = ? AND collection_name = ? AND lang = ?",
            (artist_slug, collection_name, lang),
        ).fetchone()
        return row[0] if row else None

    @classmethod
    def set_artist_bio(
        cls, artist_slug: str, collection_name: str, lang: str, bio_text: str,
    ) -> None:
        conn = cls._connect()
        conn.execute(
            "INSERT INTO artist_bios (artist_slug, collection_name, lang, bio_text) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(artist_slug, collection_name, lang) DO UPDATE SET "
            "bio_text = excluded.bio_text, generated_at = CURRENT_TIMESTAMP",
            (artist_slug, collection_name, lang, bio_text),
        )
        conn.commit()

    @classmethod
    def delete_artist_bios(cls, collection_name: str) -> int:
        conn = cls._connect()
        cur = conn.execute(
            "DELETE FROM artist_bios WHERE collection_name = ?",
            (collection_name,),
        )
        conn.commit()
        return int(cur.rowcount)

    # ── Bulk refined facts loaders (for search service caching) ──

    @classmethod
    def _get_all_refined_facts(cls, scope: str) -> Dict[str, str]:
        """Shared loader for refined artist/song facts.

        Returns ``{scope_key: joined_refined_text}`` from the ``refined_facts``
        table for the given ``scope`` ('artist' | 'song'). Each row's
        ``refined_json`` is a JSON array of ``{"text": str}`` objects; entries
        without text are dropped. Refined facts are collection-independent
        (keyed by slug), so no collection filter is applied — callers merge by
        slug and keep only the slugs present in their collection.
        """
        import json as _json

        conn = cls._connect()
        rows = conn.execute(
            "SELECT scope_key, refined_json FROM refined_facts WHERE scope = ?",
            (scope,),
        ).fetchall()

        result: Dict[str, str] = {}
        for slug, json_str in rows:
            try:
                arr = _json.loads(json_str)
                texts = [
                    item.get("text", "") for item in arr
                    if isinstance(item, dict) and item.get("text")
                ]
                if texts:
                    result[slug] = "\n\n".join(texts)
            except Exception:
                pass
        return result

    @classmethod
    def get_all_refined_artist_facts(cls, collection_name: str) -> Dict[str, str]:
        """Return ``{artist_slug: joined_refined_text}`` across all collections.

        Reads from the ``refined_facts`` table (scope='artist'). collection_name
        is accepted for signature stability but not used to filter — refined
        facts are collection-independent (see :meth:`_get_all_refined_facts`).
        """
        return cls._get_all_refined_facts("artist")

    @classmethod
    def get_all_refined_song_facts(cls, collection_name: str) -> Dict[str, str]:
        """Return ``{song_slug: joined_refined_text}`` across all collections.

        Reads from the ``refined_facts`` table (scope='song'); ``scope_key`` is
        the song_slug (same format as song_facts table slugs). collection_name is
        accepted for signature stability but not used to filter — refined facts
        are collection-independent (see :meth:`_get_all_refined_facts`). Keyed by
        song_slug so the search service can merge into TrackHit.song_facts using
        the same key as load_all_song_facts_for_collection().
        """
        return cls._get_all_refined_facts("song")

    # ─── Playlists CRUD (Plan 19) ────────────────────────────────────────
    @classmethod
    def _row_to_dict(cls, row) -> dict | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    @classmethod
    def create_playlist(cls, collection_name: str, name: str, description: str | None) -> int:
        """Insert a new playlist. Raises sqlite3.IntegrityError on (collection_name, name) collision."""
        conn = cls._connect()
        cur = conn.execute(
            "INSERT INTO playlists (collection_name, name, description) VALUES (?, ?, ?)",
            (collection_name, name, description),
        )
        conn.commit()
        return int(cur.lastrowid)

    @classmethod
    def list_playlists(cls, collection_name: str) -> list[dict]:
        """Return playlist rows for a collection, ordered by updated_at DESC."""
        conn = cls._connect()
        conn.row_factory = __import__("sqlite3").Row
        try:
            rows = conn.execute(
                "SELECT id, collection_name, name, description, created_at, updated_at "
                "FROM playlists WHERE collection_name = ? ORDER BY updated_at DESC, id DESC",
                (collection_name,),
            ).fetchall()
            return [cls._row_to_dict(r) for r in rows]
        finally:
            conn.row_factory = None

    @classmethod
    def get_playlist_row(cls, playlist_id: int) -> dict | None:
        conn = cls._connect()
        conn.row_factory = __import__("sqlite3").Row
        try:
            row = conn.execute(
                "SELECT id, collection_name, name, description, created_at, updated_at "
                "FROM playlists WHERE id = ?",
                (playlist_id,),
            ).fetchone()
            return cls._row_to_dict(row)
        finally:
            conn.row_factory = None

    @classmethod
    def touch_playlist(cls, playlist_id: int) -> None:
        """Update `updated_at` to now. Used after any mutation."""
        conn = cls._connect()
        conn.execute(
            "UPDATE playlists SET updated_at = datetime('now') WHERE id = ?",
            (playlist_id,),
        )
        conn.commit()

    @classmethod
    def update_playlist(
        cls,
        playlist_id: int,
        *,
        name: str | None,
        description: str | None,
        clear_description: bool = False,
    ) -> None:
        """Update name and/or description. Pass clear_description=True to set description to NULL."""
        conn = cls._connect()
        fields = []
        params: list = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if description is not None:
            fields.append("description = ?")
            params.append(description)
        elif clear_description:
            fields.append("description = NULL")
        if not fields:
            return  # no-op
        fields.append("updated_at = datetime('now')")
        params.append(playlist_id)
        conn.execute(f"UPDATE playlists SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()

    @classmethod
    def delete_playlist(cls, playlist_id: int) -> None:
        """Delete a playlist. CASCADE handles playlist_tracks."""
        conn = cls._connect()
        conn.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
        conn.commit()

    @classmethod
    def add_track_to_playlist(cls, playlist_id: int, track_id: str) -> int:
        """Append track with position = max(position) + 1. Returns the new position.
        Raises sqlite3.IntegrityError if already present (UNIQUE)."""
        track_id = canonical_track_id(track_id)
        conn = cls._connect()
        cur = conn.execute(
            "SELECT COALESCE(MAX(position), 0) FROM playlist_tracks WHERE playlist_id = ?",
            (playlist_id,),
        )
        next_pos = int(cur.fetchone()[0]) + 1
        conn.execute(
            "INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, ?, ?)",
            (playlist_id, track_id, next_pos),
        )
        conn.execute(
            "UPDATE playlists SET updated_at = datetime('now') WHERE id = ?",
            (playlist_id,),
        )
        conn.commit()
        return next_pos

    @classmethod
    def remove_track_from_playlist(cls, playlist_id: int, track_id: str) -> bool:
        """Returns True if a row was removed, False if no match."""
        conn = cls._connect()
        cur = conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id = ? AND track_id = ?",
            (playlist_id, track_id),
        )
        if cur.rowcount == 0:
            conn.commit()
            return False
        conn.execute(
            "UPDATE playlists SET updated_at = datetime('now') WHERE id = ?",
            (playlist_id,),
        )
        conn.commit()
        return True

    @classmethod
    def list_playlist_tracks(cls, playlist_id: int) -> list[dict]:
        """Return playlist_tracks rows in position-ascending order."""
        conn = cls._connect()
        conn.row_factory = __import__("sqlite3").Row
        try:
            rows = conn.execute(
                "SELECT playlist_id, track_id, position, added_at "
                "FROM playlist_tracks WHERE playlist_id = ? ORDER BY position ASC",
                (playlist_id,),
            ).fetchall()
            return [cls._row_to_dict(r) for r in rows]
        finally:
            conn.row_factory = None

    @classmethod
    def reorder_playlist(cls, playlist_id: int, track_ids: list[str]) -> None:
        """Replace positions for the given playlist according to `track_ids` order.
        Validates that `track_ids` is exactly the current member set. Raises ValueError on mismatch.
        Renumbers densely 1..N inside a single transaction."""
        conn = cls._connect()
        current = {r["track_id"] for r in cls.list_playlist_tracks(playlist_id)}
        requested = set(track_ids)
        if len(track_ids) != len(requested):
            raise ValueError("duplicate track_ids in reorder payload")
        missing = current - requested
        unexpected = requested - current
        if missing or unexpected:
            raise ValueError(
                f"track_ids set mismatch (missing={sorted(missing)}, unexpected={sorted(unexpected)})"
            )
        try:
            conn.execute("BEGIN")
            for i, tid in enumerate(track_ids, start=1):
                conn.execute(
                    "UPDATE playlist_tracks SET position = ? WHERE playlist_id = ? AND track_id = ?",
                    (i, playlist_id, tid),
                )
            conn.execute(
                "UPDATE playlists SET updated_at = datetime('now') WHERE id = ?",
                (playlist_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @classmethod
    def track_in_playlist(cls, playlist_id: int, track_id: str) -> bool:
        conn = cls._connect()
        row = conn.execute(
            "SELECT 1 FROM playlist_tracks WHERE playlist_id = ? AND track_id = ?",
            (playlist_id, track_id),
        ).fetchone()
        return row is not None

    @classmethod
    def playlists_containing_track(cls, collection_name: str, track_id: str) -> list[int]:
        """Return playlist ids in this collection that contain the given track."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT p.id FROM playlists p "
            "JOIN playlist_tracks pt ON pt.playlist_id = p.id "
            "WHERE p.collection_name = ? AND pt.track_id = ?",
            (collection_name, track_id),
        ).fetchall()
        return [int(r[0]) for r in rows]

    # ─── Phase A: Users CRUD ───────────────────────────────────────────────
    @classmethod
    def create_user(
        cls, *, user_id: str, email: str, password_hash: str,
        role: str, created_at: float,
    ) -> None:
        """Insert a new user. Raises sqlite3.IntegrityError on duplicate email
        or invalid role. Caller is responsible for lower-casing email."""
        conn = cls._connect()
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, email, password_hash, role, created_at),
        )
        conn.commit()

    @classmethod
    def get_user_by_email(cls, email: str) -> dict | None:
        """Return user row dict or None. Email lookup is case-SENSITIVE — caller
        must lower-case the input string (and we lower-case at write time, so
        stored rows are already canonical)."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT id, email, password_hash, role, created_at, last_login_at, premium "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "email": row[1], "password_hash": row[2],
            "role": row[3], "created_at": row[4], "last_login_at": row[5],
            "premium": row[6],
        }

    @classmethod
    def get_user_by_id(cls, user_id: str) -> dict | None:
        """Return user row dict or None."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT id, email, password_hash, role, created_at, last_login_at, premium "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "email": row[1], "password_hash": row[2],
            "role": row[3], "created_at": row[4], "last_login_at": row[5],
            "premium": row[6],
        }

    @classmethod
    def update_last_login(cls, user_id: str, ts: float) -> bool:
        """Update last_login_at; return True if the user row was updated."""
        conn = cls._connect()
        cur = conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (ts, user_id),
        )
        conn.commit()
        return cur.rowcount > 0

    @classmethod
    def has_owner(cls) -> bool:
        """True if any user with role='owner' exists. Used by create_owner
        idempotency guard."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT 1 FROM users WHERE role = 'owner' LIMIT 1"
        ).fetchone()
        return row is not None

    @classmethod
    def delete_user(cls, user_id: str) -> bool:
        """Delete a user row; return True if a row was removed. Used to roll back
        a registration that lost the invite-claim race."""
        conn = cls._connect()
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0

    # ── Owner admin: per-account stats + full account deletion ─────────────
    @classmethod
    def account_stats(cls, collection_name: str) -> Dict[str, float]:
        """Library summary for one account's collection: track count, total
        seconds listened, and liked-track count. Cheap COUNT/SUM scans — fine for
        the small member rosters a personal server has. Reads SQLite only, so it
        works even when Qdrant is down."""
        conn = cls._connect()
        songs = conn.execute(
            "SELECT COUNT(*) FROM track_metadata WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()[0]
        listened = conn.execute(
            "SELECT COALESCE(SUM(played_sec), 0) FROM playback_events WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()[0]
        likes = conn.execute(
            "SELECT COUNT(*) FROM track_reactions "
            "WHERE collection_name = ? AND reaction = 'like'",
            (collection_name,),
        ).fetchone()[0]
        return {
            "songs": int(songs or 0),
            "listened_sec": float(listened or 0.0),
            "likes": int(likes or 0),
        }

    @classmethod
    def list_account_cover_paths(cls, collection_name: str) -> list[str]:
        """Distinct non-empty ``cover_art_path`` values for an account's tracks.
        Captured BEFORE deletion so the reference-counted cover cleanup knows
        which files to re-check afterwards."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT DISTINCT cover_art_path FROM track_metadata "
            "WHERE collection_name = ? AND cover_art_path IS NOT NULL "
            "AND cover_art_path != ''",
            (collection_name,),
        ).fetchall()
        return [r[0] for r in rows]

    @classmethod
    def cover_reference_count(cls, cover_art_path: str) -> int:
        """How many track rows (across ALL accounts) still reference this cover.
        Called after an account's rows are deleted: 0 ⇒ the cover file is now an
        orphan and safe to unlink; >0 ⇒ another account shares it — keep the
        file. This is what makes cover cleanup safe despite content-addressed
        (hash-named) covers being shared between accounts."""
        conn = cls._connect()
        return conn.execute(
            "SELECT COUNT(*) FROM track_metadata WHERE cover_art_path = ?",
            (cover_art_path,),
        ).fetchone()[0]

    @classmethod
    def delete_account_data(cls, user_id: str, collection_name: str) -> None:
        """Remove ALL data belonging to one account, in a single transaction.

        Two scoping keys: most tables key on ``collection_name`` (= acct_{id});
        the upload / Yandex tables key on ``account_id`` (= user_id). FK cascades
        (foreign_keys=ON, set per connection) handle child rows —
        track_metadata→track_artist_slugs, playlists→playlist_tracks, and
        users→invites(created_by). ``invites.consumed_by`` is ON DELETE SET NULL,
        so the invite this member used survives as an already-consumed row.

        Deliberately does NOT touch the global, slug-keyed knowledge base
        (artists / songs / artist_facts / song_facts / refined_facts): those are
        shared across accounts; their ``collection_name`` is provenance only.

        Atomic: any failure rolls the whole thing back — never a half-deleted
        account.
        """
        conn = cls._connect()
        by_collection = (
            "track_reactions", "collection_settings", "playback_events",
            "ai_indexing_jobs", "sonic_vibes", "artist_bios",
            "recsys_llm_texts", "playlists", "track_metadata",
        )
        by_account = ("pending_uploads", "yandex_accounts", "yandex_imports")
        try:
            for tbl in by_collection:
                conn.execute(
                    f"DELETE FROM {tbl} WHERE collection_name = ?",
                    (collection_name,),
                )
            for tbl in by_account:
                conn.execute(
                    f"DELETE FROM {tbl} WHERE account_id = ?",
                    (user_id,),
                )
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @classmethod
    def list_users_with_invite(cls) -> list[dict]:
        """All users with the invite code each registered through (LEFT JOIN on
        invites.consumed_by). The owner / pre-invite accounts get invite_code
        None. Ordered oldest-first. Powers the owner 'Members' admin view."""
        conn = cls._connect()
        rows = conn.execute(
            """SELECT u.id, u.email, u.role, u.created_at, u.last_login_at, i.code
               FROM users u
               LEFT JOIN invites i ON i.consumed_by = u.id
               ORDER BY u.created_at ASC"""
        ).fetchall()
        return [
            {
                "id": r[0], "email": r[1], "role": r[2], "created_at": r[3],
                "last_login_at": r[4], "invite_code": r[5],
            }
            for r in rows
        ]

    # ── Phase B: per-user settings (columns on the users table) ────────────
    @classmethod
    def get_user_settings(cls, user_id: str) -> Optional[Dict]:
        """Return {'text_model_name': str, 'clap_enabled': bool} for an existing
        user, or None if the user doesn't exist. Settings are columns on the
        users table with defaults, so any user created via create_user/
        create_owner already carries sane values."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT text_model_name, clap_enabled FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {"text_model_name": row[0], "clap_enabled": bool(row[1])}

    @classmethod
    def update_user_settings(
        cls,
        user_id: str,
        text_model_name: Optional[str] = None,
        clap_enabled: Optional[bool] = None,
    ) -> None:
        """Partial update — only non-None args are written. No-op when both are
        None. Targets an existing user row (creating users is Phase A's job)."""
        if text_model_name is None and clap_enabled is None:
            return
        conn = cls._connect()
        fields, values = [], []
        if text_model_name is not None:
            fields.append("text_model_name = ?")
            values.append(text_model_name)
        if clap_enabled is not None:
            fields.append("clap_enabled = ?")
            values.append(1 if clap_enabled else 0)
        values.append(user_id)
        conn.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()

    # ── Pending uploads (Phase C: server mode) ──

    @classmethod
    def create_pending_upload(
        cls, *,
        account_id: str,
        sha256: str,
        original_filename: str,
        size_bytes: int,
        storage_path: str,
    ) -> str:
        """Insert a row with status='uploaded'. Returns the generated upload_id."""
        import time
        import uuid as _uuid
        conn = cls._connect()
        upload_id = _uuid.uuid4().hex
        conn.execute(
            """INSERT INTO pending_uploads
               (upload_id, account_id, sha256, original_filename, size_bytes,
                storage_path, status, track_id, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'uploaded', NULL, NULL, ?)""",
            (upload_id, account_id, sha256, original_filename, size_bytes,
             storage_path, time.time()),
        )
        conn.commit()
        return upload_id

    @classmethod
    def get_pending_upload(cls, upload_id: str) -> dict | None:
        conn = cls._connect()
        row = conn.execute(
            """SELECT upload_id, account_id, sha256, original_filename,
                      size_bytes, storage_path, status, track_id, error, created_at
               FROM pending_uploads WHERE upload_id = ?""",
            (upload_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "upload_id": row[0], "account_id": row[1], "sha256": row[2],
            "original_filename": row[3], "size_bytes": row[4],
            "storage_path": row[5], "status": row[6], "track_id": row[7],
            "error": row[8], "created_at": row[9],
        }

    @classmethod
    def update_pending_upload_status(
        cls, upload_id: str, *,
        status: str,
        track_id: str | None = None,
        error: str | None = None,
    ) -> None:
        conn = cls._connect()
        conn.execute(
            """UPDATE pending_uploads
               SET status = ?, track_id = COALESCE(?, track_id),
                   error = COALESCE(?, error)
               WHERE upload_id = ?""",
            (status, track_id, error, upload_id),
        )
        conn.commit()

    @classmethod
    def list_pending_uploads_by_account(
        cls, account_id: str, *, status: str | None = None,
    ) -> list[dict]:
        conn = cls._connect()
        if status:
            cursor = conn.execute(
                """SELECT upload_id, account_id, sha256, original_filename,
                          size_bytes, storage_path, status, track_id, error, created_at
                   FROM pending_uploads WHERE account_id = ? AND status = ?
                   ORDER BY created_at""",
                (account_id, status),
            )
        else:
            cursor = conn.execute(
                """SELECT upload_id, account_id, sha256, original_filename,
                          size_bytes, storage_path, status, track_id, error, created_at
                   FROM pending_uploads WHERE account_id = ?
                   ORDER BY created_at""",
                (account_id,),
            )
        return [
            {
                "upload_id": r[0], "account_id": r[1], "sha256": r[2],
                "original_filename": r[3], "size_bytes": r[4],
                "storage_path": r[5], "status": r[6], "track_id": r[7],
                "error": r[8], "created_at": r[9],
            }
            for r in cursor.fetchall()
        ]

    @classmethod
    def find_done_upload_by_sha(cls, *, account_id: str, sha256: str) -> dict | None:
        """Idempotency lookup: has this account already uploaded + indexed this SHA?"""
        conn = cls._connect()
        row = conn.execute(
            """SELECT upload_id, account_id, sha256, original_filename,
                      size_bytes, storage_path, status, track_id, error, created_at
               FROM pending_uploads
               WHERE account_id = ? AND sha256 = ? AND status = 'done'
               ORDER BY created_at DESC LIMIT 1""",
            (account_id, sha256),
        ).fetchone()
        if not row:
            return None
        return {
            "upload_id": row[0], "account_id": row[1], "sha256": row[2],
            "original_filename": row[3], "size_bytes": row[4],
            "storage_path": row[5], "status": row[6], "track_id": row[7],
            "error": row[8], "created_at": row[9],
        }

    @classmethod
    def purge_old_pending_uploads(cls, older_than_seconds: int = 7 * 86400) -> int:
        """Delete ``status='done'`` rows older than the cutoff. Returns count deleted."""
        import time
        conn = cls._connect()
        cutoff = time.time() - older_than_seconds
        cursor = conn.execute(
            "DELETE FROM pending_uploads WHERE status = 'done' AND created_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount or 0

    # ─── Yandex Music import CRUD ──────────────────────────────────────────
    @classmethod
    def save_yandex_account(
        cls, *, account_id: str, enc_token: str,
        yandex_uid: str | None = None, expires_at: float | None = None,
    ) -> None:
        """Upsert the encrypted Yandex token for an account (re-link overwrites)."""
        import time
        conn = cls._connect()
        conn.execute(
            """INSERT INTO yandex_accounts
                 (account_id, enc_token, yandex_uid, expires_at, linked_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(account_id) DO UPDATE SET
                 enc_token = excluded.enc_token,
                 yandex_uid = excluded.yandex_uid,
                 expires_at = excluded.expires_at,
                 linked_at = excluded.linked_at""",
            (account_id, enc_token, yandex_uid, expires_at, time.time()),
        )
        conn.commit()

    @classmethod
    def get_yandex_account(cls, account_id: str) -> dict | None:
        conn = cls._connect()
        row = conn.execute(
            """SELECT account_id, enc_token, yandex_uid, expires_at, linked_at
               FROM yandex_accounts WHERE account_id = ?""",
            (account_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "account_id": row[0], "enc_token": row[1], "yandex_uid": row[2],
            "expires_at": row[3], "linked_at": row[4],
        }

    @classmethod
    def delete_yandex_account(cls, account_id: str) -> bool:
        """Unlink: drop the stored token. Returns True if a row was removed."""
        conn = cls._connect()
        cur = conn.execute(
            "DELETE FROM yandex_accounts WHERE account_id = ?", (account_id,),
        )
        conn.commit()
        return (cur.rowcount or 0) > 0

    @classmethod
    def get_imported_yandex_track_ids(cls, account_id: str) -> set[str]:
        """Return yandex_track_ids already imported (any non-skipped/failed status).

        Used to dedup a re-import: a track that previously downloaded or indexed
        is skipped; a prior 'skipped'/'failed' row is retried.
        """
        conn = cls._connect()
        rows = conn.execute(
            """SELECT yandex_track_id FROM yandex_imports
               WHERE account_id = ? AND status IN ('downloaded', 'indexed')""",
            (account_id,),
        ).fetchall()
        return {r[0] for r in rows}

    @classmethod
    def upsert_yandex_import(
        cls, *, account_id: str, yandex_track_id: str, status: str,
        upload_id: str | None = None, track_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record/update the import outcome for one Yandex track."""
        import time
        conn = cls._connect()
        conn.execute(
            """INSERT INTO yandex_imports
                 (account_id, yandex_track_id, upload_id, track_id, status, reason, imported_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id, yandex_track_id) DO UPDATE SET
                 upload_id = COALESCE(excluded.upload_id, yandex_imports.upload_id),
                 track_id = COALESCE(excluded.track_id, yandex_imports.track_id),
                 status = excluded.status,
                 reason = excluded.reason,
                 imported_at = excluded.imported_at""",
            (account_id, yandex_track_id, upload_id, track_id, status, reason, time.time()),
        )
        conn.commit()

    # ─── Phase A: Invites CRUD ─────────────────────────────────────────────
    @classmethod
    def create_invite(
        cls, code: str, created_by: str, created_at: float, expires_at: float,
    ) -> None:
        """Insert a new invite row. Raises sqlite3.IntegrityError on duplicate code
        or unknown created_by user."""
        conn = cls._connect()
        conn.execute(
            "INSERT INTO invites (code, created_by, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (code, created_by, created_at, expires_at),
        )
        conn.commit()

    @classmethod
    def get_invite(cls, code: str) -> dict | None:
        conn = cls._connect()
        row = conn.execute(
            "SELECT code, created_by, created_at, expires_at, consumed_by, consumed_at "
            "FROM invites WHERE code = ?",
            (code,),
        ).fetchone()
        if row is None:
            return None
        return {
            "code": row[0], "created_by": row[1],
            "created_at": row[2], "expires_at": row[3],
            "consumed_by": row[4], "consumed_at": row[5],
        }

    @classmethod
    def consume_invite(cls, code: str, *, consumed_by: str, consumed_at: float) -> bool:
        """Atomically claim an open invite. The `consumed_at IS NULL` predicate in
        the UPDATE makes this a compare-and-swap: only the first concurrent caller
        wins. Returns True if this call claimed it, False if the code was unknown
        or already consumed (lets AuthService roll back a racing registration)."""
        conn = cls._connect()
        cur = conn.execute(
            "UPDATE invites SET consumed_by = ?, consumed_at = ? "
            "WHERE code = ? AND consumed_at IS NULL",
            (consumed_by, consumed_at, code),
        )
        conn.commit()
        return cur.rowcount > 0

    @classmethod
    def list_invites(cls, *, include_consumed: bool = False) -> list[dict]:
        """Return all invites (newest first). When include_consumed=False, only
        rows with consumed_at IS NULL are returned."""
        conn = cls._connect()
        if include_consumed:
            sql = (
                "SELECT code, created_by, created_at, expires_at, consumed_by, consumed_at "
                "FROM invites ORDER BY created_at DESC"
            )
            rows = conn.execute(sql).fetchall()
        else:
            sql = (
                "SELECT code, created_by, created_at, expires_at, consumed_by, consumed_at "
                "FROM invites WHERE consumed_at IS NULL ORDER BY created_at DESC"
            )
            rows = conn.execute(sql).fetchall()
        return [
            {
                "code": r[0], "created_by": r[1],
                "created_at": r[2], "expires_at": r[3],
                "consumed_by": r[4], "consumed_at": r[5],
            }
            for r in rows
        ]

    @classmethod
    def delete_invite(cls, code: str) -> bool:
        """Delete an invite by code; return True if a row was removed."""
        conn = cls._connect()
        cur = conn.execute("DELETE FROM invites WHERE code = ?", (code,))
        conn.commit()
        return cur.rowcount > 0

    # ─── Phase A: Instance Config ──────────────────────────────────────────
    @classmethod
    def get_instance_config(cls) -> dict | None:
        """Return {"mode": ..., "created_at": ...} or None if instance never
        initialized."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT mode, created_at FROM instance_config WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return {"mode": row[0], "created_at": row[1]}

    @classmethod
    def set_instance_config(cls, *, mode: str, created_at: float) -> None:
        """Write the single instance_config row. Raises sqlite3.IntegrityError
        if a row already exists (per spec §4.2 — mode is locked at first write).
        Use scripts/change_instance_mode.py to migrate (not in Phase A scope)."""
        conn = cls._connect()
        conn.execute(
            "INSERT INTO instance_config (id, mode, created_at) VALUES (1, ?, ?)",
            (mode, created_at),
        )
        conn.commit()

    # ─── Instance settings (key-value; read via SettingsService) ───────────
    @classmethod
    def get_instance_setting(cls, key: str) -> str | None:
        """Return the stored value for ``key`` or None if unset. None means
        'fall through to env/default' — it is NOT the same as an empty string."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT value FROM instance_settings WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row is not None else None

    @classmethod
    def get_all_instance_settings(cls) -> dict[str, str | None]:
        """Return {key: value} for every stored setting (no env/default merge)."""
        conn = cls._connect()
        rows = conn.execute(
            "SELECT key, value FROM instance_settings"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    @classmethod
    def set_instance_settings(
        cls, mapping: dict[str, str | None], updated_at: float,
    ) -> None:
        """Upsert each key in ``mapping``. A None/empty value DELETES the key so
        the resolver falls back to env/default (rather than storing an empty
        string that would mask the env var). No-op on an empty mapping."""
        if not mapping:
            return
        conn = cls._connect()
        for key, value in mapping.items():
            if value is None or value == "":
                conn.execute(
                    "DELETE FROM instance_settings WHERE key = ?", (key,)
                )
            else:
                conn.execute(
                    """INSERT INTO instance_settings (key, value, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                         value = excluded.value,
                         updated_at = excluded.updated_at""",
                    (key, value, updated_at),
                )
        conn.commit()

    # ── Library aggregation from SQLite ──

    @classmethod
    def count_distinct_artist_slugs(cls, collection_name: str) -> int:
        """Count distinct artists in a collection from track_artist_slugs.

        Returns 0 if the mirror is empty (pre-backfill) — callers should treat
        0 as "unknown" and fall back to a Qdrant scan.
        """
        conn = cls._connect()
        row = conn.execute(
            "SELECT COUNT(DISTINCT artist_slug) FROM track_artist_slugs"
            " WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return int(row[0] or 0)

    @classmethod
    def get_distinct_artist_slugs_from_sqlite(cls, collection_name: str) -> list[dict]:
        """Return list of {slug, name, track_count, album_count} for all artists
        in a collection, read from track_artist_slugs + track_metadata.

        Returns [] if the tables are empty (pre-backfill).
        """
        conn = cls._connect()
        rows = conn.execute(
            """
            SELECT
                tas.artist_slug,
                MAX(tm.artist) AS artist_name,
                COUNT(DISTINCT tas.track_id) AS track_count,
                COUNT(DISTINCT tm.album) AS album_count
            FROM track_artist_slugs tas
            JOIN track_metadata tm
                ON tas.collection_name = tm.collection_name
               AND tas.track_id = tm.track_id
            WHERE tas.collection_name = ?
            GROUP BY tas.artist_slug
            ORDER BY track_count DESC
            """,
            (collection_name,),
        ).fetchall()
        return [
            {
                "slug": r[0],
                "name": r[1] or r[0],
                "track_count": r[2],
                "album_count": r[3],
            }
            for r in rows
        ]

    @classmethod
    def get_library_albums_from_sqlite(cls, collection_name: str) -> list[dict]:
        """Return per-album aggregates for a collection, read from track_metadata.

        Each album dict carries: album, track_count, primary_artist,
        feat_artists (raw names), top_genres, year, year_range, cover_art_path,
        and ``tracks`` (list of {track_id, title, artist, duration, year,
        cover_art_path}).

        ``primary_artist`` / ``feat_artists`` / ``year`` / ``year_range`` /
        ``top_genres`` mirror the Qdrant ``get_albums`` aggregation exactly
        (majority vote on the raw ``artist`` tag, min/max year), so the SQLite
        fast-path and the Qdrant fallback produce identical cards. The slug for
        each artist name is derived in the service layer.

        Returns [] if the table is empty (pre-backfill).
        """
        from collections import Counter

        conn = cls._connect()

        # Single query: all tracks grouped by album, with album-level aggregates.
        # We fetch every row and group in Python to avoid complex window functions.
        rows = conn.execute(
            """
            SELECT album, title, artist, duration, year, cover_art_path,
                   track_id, genre
            FROM track_metadata
            WHERE collection_name = ? AND album IS NOT NULL AND album != ''
            ORDER BY album, track_id
            """,
            (collection_name,),
        ).fetchall()

        if not rows:
            return []

        # Group by album (case-insensitive key, preserves first-seen casing)
        groups: dict[str, dict] = {}
        for album, title, artist, duration, year, cover_art_path, track_id, genre in rows:
            key = album.lower()
            if key not in groups:
                groups[key] = {
                    "album": album,
                    "track_count": 0,
                    "cover_art_path": cover_art_path,
                    "tracks": [],
                    "_artist_counter": Counter(),
                    "_year_counter": Counter(),
                    "_genre_counter": Counter(),
                }
            g = groups[key]
            g["track_count"] += 1
            # Use first non-NULL cover_art_path
            if not g["cover_art_path"] and cover_art_path:
                g["cover_art_path"] = cover_art_path
            artist_name = (artist or "").strip()
            g["_artist_counter"][artist_name] += 1
            if year:
                g["_year_counter"][year] += 1
            gn = (genre or "").strip()
            if gn:
                g["_genre_counter"][gn] += 1
            g["tracks"].append({
                "track_id": track_id,
                "title": title or "—",
                "artist": artist or "—",
                "duration": duration,
                "year": year,
                "cover_art_path": cover_art_path,
            })

        # Finalise album-level aggregates (majority-vote primary, year/range).
        result = []
        for g in groups.values():
            artists = g.pop("_artist_counter")
            primary = sorted(
                artists.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0] if artists else "—"
            feat = [
                a for a, _ in sorted(
                    [(a, c) for a, c in artists.items() if a != primary],
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ]
            year_counter = g.pop("_year_counter")
            year = None
            year_range = None
            if year_counter:
                ys = list(year_counter.elements())
                ymin, ymax = min(ys), max(ys)
                if ymin == ymax:
                    year = ymin
                else:
                    year_range = f"{ymin}—{ymax}"
            genre_counter = g.pop("_genre_counter")
            g["primary_artist"] = primary or "—"
            g["feat_artists"] = feat
            g["top_genres"] = [gn for gn, _ in genre_counter.most_common(3)]
            g["year"] = year
            g["year_range"] = year_range
            result.append(g)

        # Sort: year DESC (year_range albums sort by their min year), then album.
        def _year_key(a):
            if a["year"]:
                return a["year"]
            if a["year_range"]:
                try:
                    return int(a["year_range"].split("—")[0])
                except (ValueError, IndexError):
                    return 0
            return 0

        result.sort(key=lambda a: (-_year_key(a), a["album"]))
        return result

    # ── Track metadata mirror (SQLite copy of Qdrant payload) ──

    @classmethod
    def upsert_track_metadata(cls, collection_name: str, track_id: str, payload: dict) -> None:
        """INSERT OR REPLACE a track's card metadata + artist slug join rows.

        Called during indexing / upload for every track so the artist-aggregate
        endpoint can read from SQLite instead of paginating Qdrant.
        """
        track_id = canonical_track_id(track_id)
        conn = cls._connect()
        conn.execute(
            """INSERT INTO track_metadata
                (collection_name, track_id, title, artist, artists, artist_slugs,
                 primary_artist_slug, album, year, genre, duration, file_path,
                 cover_art_path, producer, label, track_number, disc_number,
                 bitrate_kbps, sonic_tags, sonic_axes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(collection_name, track_id) DO UPDATE SET
                 title=excluded.title, artist=excluded.artist,
                 artists=excluded.artists, artist_slugs=excluded.artist_slugs,
                 primary_artist_slug=excluded.primary_artist_slug,
                 album=excluded.album, year=excluded.year, genre=excluded.genre,
                 duration=excluded.duration, file_path=excluded.file_path,
                 cover_art_path=excluded.cover_art_path,
                 producer=excluded.producer, label=excluded.label,
                 track_number=excluded.track_number, disc_number=excluded.disc_number,
                 bitrate_kbps=excluded.bitrate_kbps,
                 sonic_tags=excluded.sonic_tags, sonic_axes=excluded.sonic_axes""",
            (
                collection_name, track_id,
                payload.get("title") or "",
                payload.get("artist") or "",
                json.dumps(payload.get("artists") or []),
                json.dumps(payload.get("artist_slugs") or []),
                payload.get("primary_artist_slug"),
                payload.get("album"),
                payload.get("year"),
                payload.get("genre"),
                payload.get("duration"),
                payload.get("file_path"),
                payload.get("cover_art_path"),
                payload.get("producer"),
                payload.get("label"),
                payload.get("track_number"),
                payload.get("disc_number"),
                payload.get("bitrate_kbps"),
                json.dumps(payload.get("sonic_tags") or []),
                json.dumps(payload.get("sonic_axes") or {}),
            ),
        )
        # Update join table: delete old slugs, insert new ones
        conn.execute(
            "DELETE FROM track_artist_slugs WHERE collection_name = ? AND track_id = ?",
            (collection_name, track_id),
        )
        for slug in (payload.get("artist_slugs") or []):
            conn.execute(
                "INSERT OR IGNORE INTO track_artist_slugs (collection_name, track_id, artist_slug) VALUES (?, ?, ?)",
                (collection_name, track_id, slug),
            )
        conn.commit()

    @classmethod
    def upsert_track_metadata_bulk(cls, collection_name: str, rows: list) -> None:
        """Bulk upsert multiple tracks in one transaction.

        ``rows`` is a list of ``(track_id, payload_dict)`` tuples.
        """
        conn = cls._connect()
        try:
            for track_id, payload in rows:
                track_id = canonical_track_id(track_id)
                conn.execute(
                    """INSERT INTO track_metadata
                        (collection_name, track_id, title, artist, artists, artist_slugs,
                         primary_artist_slug, album, year, genre, duration, file_path,
                         cover_art_path, producer, label, track_number, disc_number,
                         bitrate_kbps, sonic_tags, sonic_axes)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                      ON CONFLICT(collection_name, track_id) DO UPDATE SET
                        title=excluded.title, artist=excluded.artist,
                        artists=excluded.artists, artist_slugs=excluded.artist_slugs,
                        primary_artist_slug=excluded.primary_artist_slug,
                        album=excluded.album, year=excluded.year, genre=excluded.genre,
                        duration=excluded.duration, file_path=excluded.file_path,
                        cover_art_path=excluded.cover_art_path,
                        producer=excluded.producer, label=excluded.label,
                        track_number=excluded.track_number, disc_number=excluded.disc_number,
                        bitrate_kbps=excluded.bitrate_kbps,
                        sonic_tags=excluded.sonic_tags, sonic_axes=excluded.sonic_axes""",
                    (
                        collection_name, track_id,
                        payload.get("title") or "",
                        payload.get("artist") or "",
                        json.dumps(payload.get("artists") or []),
                        json.dumps(payload.get("artist_slugs") or []),
                        payload.get("primary_artist_slug"),
                        payload.get("album"),
                        payload.get("year"),
                        payload.get("genre"),
                        payload.get("duration"),
                        payload.get("file_path"),
                        payload.get("cover_art_path"),
                        payload.get("producer"),
                        payload.get("label"),
                        payload.get("track_number"),
                        payload.get("disc_number"),
                        payload.get("bitrate_kbps"),
                        json.dumps(payload.get("sonic_tags") or []),
                        json.dumps(payload.get("sonic_axes") or {}),
                    ),
                )
                conn.execute(
                    "DELETE FROM track_artist_slugs WHERE collection_name = ? AND track_id = ?",
                    (collection_name, track_id),
                )
                for slug in (payload.get("artist_slugs") or []):
                    conn.execute(
                        "INSERT OR IGNORE INTO track_artist_slugs (collection_name, track_id, artist_slug) VALUES (?, ?, ?)",
                        (collection_name, track_id, slug),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @classmethod
    def get_tracks_for_artist(cls, collection_name: str, artist_slug: str) -> list[dict]:
        """Return all tracks where artist_slug matches — single indexed query.

        Returns rows as dicts with column names as keys. Empty list when no tracks.
        """
        conn = cls._connect()
        rows = conn.execute(
            """SELECT tm.collection_name, tm.track_id, tm.title, tm.artist, tm.artists,
                      tm.artist_slugs, tm.primary_artist_slug, tm.album, tm.year,
                      tm.genre, tm.duration, tm.file_path, tm.cover_art_path,
                      tm.producer, tm.label, tm.track_number, tm.disc_number,
                      tm.bitrate_kbps, tm.sonic_tags, tm.sonic_axes
               FROM track_metadata tm
               INNER JOIN track_artist_slugs tas
                   ON tas.collection_name = tm.collection_name
                   AND tas.track_id = tm.track_id
               WHERE tas.artist_slug = ? AND tm.collection_name = ?""",
            (artist_slug, collection_name),
        ).fetchall()
        cols = ["collection_name", "track_id", "title", "artist", "artists",
                "artist_slugs", "primary_artist_slug", "album", "year",
                "genre", "duration", "file_path", "cover_art_path",
                "producer", "label", "track_number", "disc_number",
                "bitrate_kbps", "sonic_tags", "sonic_axes"]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            # Deserialize JSON fields
            for field in ("artists", "artist_slugs", "sonic_tags"):
                val = d.get(field)
                if isinstance(val, str):
                    try:
                        d[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        d[field] = []
                else:
                    d[field] = d.get(field) or []
            val = d.get("sonic_axes")
            if isinstance(val, str):
                try:
                    d["sonic_axes"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d["sonic_axes"] = {}
            result.append(d)
        return result

    @classmethod
    def clear_track_metadata(cls, collection_name: str) -> None:
        """Remove all track metadata for a collection (called before reindex)."""
        conn = cls._connect()
        conn.execute("DELETE FROM track_metadata WHERE collection_name = ?", (collection_name,))
        conn.execute("DELETE FROM track_artist_slugs WHERE collection_name = ?", (collection_name,))
        conn.commit()

    @classmethod
    def delete_track_metadata(cls, collection_name: str, track_id: str) -> None:
        """Remove a single track from the mirror (track_metadata + slug join).

        Called when a track is deleted from Qdrant so the SQLite mirror — now the
        source of truth for the albums grid, artist pages, recommendations,
        stats and catalog search — doesn't keep serving a ghost row.
        """
        conn = cls._connect()
        conn.execute(
            "DELETE FROM track_metadata WHERE collection_name = ? AND track_id = ?",
            (collection_name, track_id),
        )
        conn.execute(
            "DELETE FROM track_artist_slugs WHERE collection_name = ? AND track_id = ?",
            (collection_name, track_id),
        )
        conn.commit()

    @classmethod
    def get_track_count_for_collection(cls, collection_name: str) -> int:
        """Return the number of tracks stored in track_metadata for a collection."""
        conn = cls._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM track_metadata WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _duration_range_buckets(durations: list) -> list[str]:
        """IQR-based duration bucket label per input duration.

        Mirrors ``app.resources.qdrant_payload.prepare_metadata`` so the SQLite
        light-points path reproduces the same six-bucket histogram the indexer
        baked into the Qdrant payload. Non-numeric durations map to ``""``.
        """
        import numpy as np

        nums = [d for d in durations if isinstance(d, (int, float))]
        if not nums:
            return ["" for _ in durations]
        arr = np.array(nums, dtype=float)
        p25 = np.percentile(arr, 25)
        p50 = np.percentile(arr, 50)
        p75 = np.percentile(arr, 75)
        iqr = p50 - p25
        lower = p25 - 1.5 * iqr
        upper = p75 + 1.5 * iqr
        max_dur = float(arr.max())
        labels = [
            f"0-{int(round(lower))}",
            f"{int(round(lower))}-{int(round(p25))}",
            f"{int(round(p25))}-{int(round(p50))}",
            f"{int(round(p50))}-{int(round(p75))}",
            f"{int(round(p75))}-{int(round(upper))}",
            f"{int(round(upper))}-{int(max_dur)}",
        ]

        def _label(d):
            if not isinstance(d, (int, float)):
                return ""
            if d < lower:
                return labels[0]
            if d < p25:
                return labels[1]
            if d < p50:
                return labels[2]
            if d <= p75:
                return labels[3]
            if d <= upper:
                return labels[4]
            return labels[5]

        return [_label(d) for d in durations]

    @classmethod
    def get_light_points(cls, collection_name: str) -> list:
        """SQLite-backed equivalent of ``qdrant_utils.light_points``.

        Returns ``[(track_id, payload_dict)]`` where each payload carries the
        same keys as ``LIGHT_PAYLOAD_FIELDS`` (card fields + ``sonic_axes``),
        read from the ``track_metadata`` mirror instead of a full Qdrant scroll.
        ``duration_range`` is recomputed over this collection's durations so the
        library stats histogram keeps working without a Qdrant round-trip.

        Returns ``[]`` when the mirror is empty (pre-backfill) so callers can
        fall back to a Qdrant scroll.
        """
        conn = cls._connect()
        rows = conn.execute(
            """SELECT track_id, title, artist, artists, artist_slugs,
                      primary_artist_slug, album, year, genre, duration,
                      file_path, cover_art_path, sonic_axes
               FROM track_metadata WHERE collection_name = ?""",
            (collection_name,),
        ).fetchall()
        if not rows:
            return []

        def _json_or(val, default):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return default
            return val if val is not None else default

        buckets = cls._duration_range_buckets([r[9] for r in rows])
        out = []
        for r, bucket in zip(rows, buckets):
            (track_id, title, artist, artists, artist_slugs, primary_artist_slug,
             album, year, genre, duration, file_path, cover_art_path,
             sonic_axes) = r
            payload = {
                "title": title or "",
                "artist": artist or "",
                "artists": _json_or(artists, []),
                "artist_slugs": _json_or(artist_slugs, []),
                "primary_artist_slug": primary_artist_slug,
                "album": album,
                "year": year,
                "genre": genre,
                "duration": duration,
                "duration_range": bucket,
                "file_path": file_path,
                "cover_art_path": cover_art_path,
                "sonic_axes": _json_or(sonic_axes, {}),
            }
            out.append((str(track_id), payload))
        return out

    @classmethod
    def get_track_by_id(cls, collection_name: str, track_id: str) -> dict | None:
        """Return a single track by track_id, or None."""
        conn = cls._connect()
        row = conn.execute(
            """SELECT title, artist, artists, artist_slugs, primary_artist_slug,
                      album, year, genre, duration, file_path, cover_art_path,
                      producer, label, track_number, disc_number, bitrate_kbps,
                      sonic_tags, sonic_axes
               FROM track_metadata
               WHERE collection_name = ? AND track_id = ?""",
            (collection_name, track_id),
        ).fetchone()
        if row is None:
            return None
        cols = ["title", "artist", "artists", "artist_slugs", "primary_artist_slug",
                "album", "year", "genre", "duration", "file_path", "cover_art_path",
                "producer", "label", "track_number", "disc_number", "bitrate_kbps",
                "sonic_tags", "sonic_axes"]
        d = dict(zip(cols, row))
        d["track_id"] = track_id
        for field in ("artists", "artist_slugs", "sonic_tags"):
            val = d.get(field)
            if isinstance(val, str):
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        val = d.get("sonic_axes")
        if isinstance(val, str):
            try:
                d["sonic_axes"] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d["sonic_axes"] = {}
        return d

    @classmethod
    def get_browse_rows(cls, collection_name: str, limit: int | None = None) -> list[dict]:
        """Lightweight rows for the /library/browse search: track_id + title +
        artist + album + cover_art_path (+ year/genre/duration for the
        spotlight quick-search rows), read from the mirror.

        ``limit`` caps the no-query "first N" mode; omit it (None) for the
        scored-search mode that needs every row. Returns [] when the mirror is
        empty so the route can fall back to a Qdrant scroll.
        """
        conn = cls._connect()
        sql = ("SELECT track_id, title, artist, album, cover_art_path, "
               "year, genre, duration "
               "FROM track_metadata WHERE collection_name = ?")
        params: tuple = (collection_name,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (collection_name, limit)
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "track_id": str(r[0]),
                "title": r[1] or "",
                "artist": r[2] or "",
                "album": r[3] or "",
                "cover_art_path": r[4],
                "year": r[5],
                "genre": r[6],
                "duration": r[7],
            }
            for r in rows
        ]

    @classmethod
    def get_track_ids_for_collection(cls, collection_name: str) -> list[str]:
        """All track_ids stored in the mirror for a collection (no payload).

        Mirrors the id-only Qdrant scroll the rediscover picker did. Returns []
        when the mirror is empty so the caller can fall back to Qdrant.
        """
        conn = cls._connect()
        rows = conn.execute(
            "SELECT track_id FROM track_metadata WHERE collection_name = ?",
            (collection_name,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    @classmethod
    def get_year_facets_from_sqlite(cls, collection_name: str | None = None) -> dict:
        """Decade-bucket counts from the mirror's ``year`` column, keyed exactly
        like the Qdrant ``year_range`` payload (``f"{decade}-{decade+9}"``).

        Aggregates a single collection, or all collections when
        ``collection_name`` is None. Returns {} when the mirror is empty so the
        route can fall back to a Qdrant scroll.
        """
        from collections import Counter

        conn = cls._connect()
        if collection_name is not None:
            rows = conn.execute(
                "SELECT year FROM track_metadata "
                "WHERE collection_name = ? AND year IS NOT NULL",
                (collection_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT year FROM track_metadata WHERE year IS NOT NULL",
            ).fetchall()
        counter: Counter[str] = Counter()
        for (year,) in rows:
            try:
                y = int(year)
            except (TypeError, ValueError):
                continue
            decade = (y // 10) * 10
            counter[f"{decade}-{decade + 9}"] += 1
        return dict(counter)
