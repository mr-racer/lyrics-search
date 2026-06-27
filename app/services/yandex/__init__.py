"""Yandex Music import — bring a user's playlists / liked tracks into MusiX.

Sub-modules:
  token_store    — Fernet-encrypted per-account OAuth token persistence
  client_factory — build a (throttled) yandex_music.Client, with or without a token
  auth           — device-flow login (show code → poll → store token)
  playlists      — list a user's playlists + the synthetic "likes" source
  downloader     — download one track (flac→mp3→aac) + tag/cover/lyrics embed
  enrichment     — feature #2: fill empty metadata fields from the Yandex catalog
  importer       — orchestrate an import job → pending_uploads → existing index flow

Server-mode only (writes into ``media/<account_id>/audio/``), like /library/upload.
"""
