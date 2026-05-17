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

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["MetadataDB"]

DB_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
DB_PATH = DB_DIR / "metadata.db"

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
    """CREATE TABLE IF NOT EXISTS refined_facts (
        scope           TEXT NOT NULL,         -- 'song' or 'artist'
        scope_key       TEXT NOT NULL,         -- track_id (song) or artist_slug (artist)
        collection_name TEXT NOT NULL,
        lang            TEXT NOT NULL,
        refined_json    TEXT NOT NULL,         -- JSON array of {"text": str}
        generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (scope, scope_key, collection_name, lang)
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
)


def _slugify(text: str) -> str:
    return "-".join(text.lower().split())


class MetadataDB:
    """Lazy singleton wrapper around a local SQLite database.

    The connection is created on first access. Call :meth:`init` to ensure
    the schema exists.
    """

    _instance: Optional[sqlite3.Connection] = None

    @classmethod
    def _connect(cls) -> sqlite3.Connection:
        if cls._instance is None:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            cls._instance = conn
            logger.info("[MetadataDB] Connected to %s", DB_PATH)
        return cls._instance

    @classmethod
    def get(cls) -> sqlite3.Connection:
        """Return the shared connection (creates it lazily)."""
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

        # AI Mode infrastructure (Plan 6) — per-collection opt-in for live-LLM
        # features. DEFAULT 1 means existing rows immediately report on,
        # matching the pre-flag world where AI features were always available.
        cls._ensure_columns(conn, "collection_settings", {
            "ai_enabled": "INTEGER NOT NULL DEFAULT 1",
        })

        # AI indexing — distinguish "processed" from "silently skipped" so the
        # UI can tell when a job completed with literally zero LLM work done
        # (e.g. tracks lacked sonic_tags + facts). Without this column the
        # status payload reports n_done = n_total and looks like a real run.
        cls._ensure_columns(conn, "ai_indexing_jobs", {
            "n_skipped": "INTEGER NOT NULL DEFAULT 0",
        })

        conn.commit()
        logger.info("[MetadataDB] Schema initialised")

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
        """Drop any cached connection and clear the init flag — test only."""
        if cls._instance is not None:
            try:
                cls._instance.close()
            except Exception:
                pass
        cls._instance = None

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
            """INSERT INTO artist_facts (artist_slug, lang, fact, category, source)
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
            """INSERT INTO artist_facts (artist_slug, lang, fact, source)
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

        result: Dict[str, List[str]] = {}
        for slug, fact in rows:
            result.setdefault(slug, []).append(fact)
        return {slug: "\n\n".join(facts) for slug, facts in result.items()}

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
            """INSERT INTO song_facts (song_slug, lang, fact, category, source)
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
            """INSERT INTO song_facts (song_slug, lang, fact, source)
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

        result: Dict[str, List[str]] = {}
        for slug, fact in rows:
            result.setdefault(slug, []).append(fact)
        return {slug: "\n\n".join(facts) for slug, facts in result.items()}

    # ── Random facts ──

    @classmethod
    def get_random_facts(
        cls,
        collection_name: str,
        limit: int = 5,
    ) -> List[dict]:
        """Return ``limit`` random facts from the collection's fact pool.

        Pool includes both ``artist_facts`` (with artist name as context) and
        ``song_facts`` (with ``"Artist — Song"`` as context).  Filtered by
        collection and ``lang='en'``.

        Returns list of dicts: ``{"fact": str, "context": str, "type": str}``.
        """
        conn = cls._connect()
        rows = conn.execute(
            """
            SELECT fact, context, type FROM (
                SELECT
                    af.fact,
                    a.name AS context,
                    'artist' AS type
                FROM artist_facts af
                JOIN artists a ON a.slug = af.artist_slug
                WHERE a.collection_name = ? AND af.lang = 'en'

                UNION ALL

                SELECT
                    sf.fact,
                    a.name || ' — ' || s.title AS context,
                    'song' AS type
                FROM song_facts sf
                JOIN songs s ON s.slug = sf.song_slug
                JOIN artists a ON a.slug = s.artist_slug
                WHERE s.collection_name = ? AND sf.lang = 'en'
            )
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (collection_name, collection_name, limit),
        ).fetchall()
        return [{"fact": r[0], "context": r[1], "type": r[2]} for r in rows]

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
        """Close the shared connection (mainly for tests)."""
        conn = cls._instance
        if conn:
            conn.close()
            cls._instance = None

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
    ) -> int:
        """Insert a playback event. Returns the new row id.

        ``skipped_early`` is derived server-side: when ``total_dur`` is known,
        ``True`` requires BOTH ``played_sec < 30`` AND
        ``played_sec / total_dur < 0.30`` (so a short track played to completion
        is not falsely flagged). When ``total_dur`` is missing, falls back to
        the absolute 30-second threshold alone.
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
            "(session_id, collection_name, track_id, played_sec, total_dur, skipped_early) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, collection_name, track_id, played_sec, total_dur,
             1 if skipped_early else 0),
        )
        conn.commit()
        return int(cur.lastrowid)

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
        row = conn.execute(
            "SELECT refined_json FROM refined_facts "
            "WHERE scope = ? AND scope_key = ? AND collection_name = ? AND lang = ?",
            (scope, scope_key, collection_name, lang),
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
        conn.execute(
            "INSERT INTO refined_facts (scope, scope_key, collection_name, lang, refined_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(scope, scope_key, collection_name, lang) DO UPDATE SET "
            "refined_json = excluded.refined_json, generated_at = CURRENT_TIMESTAMP",
            (scope, scope_key, collection_name, lang, payload),
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
    def get_all_refined_artist_facts(cls, collection_name: str) -> Dict[str, str]:
        """Return ``{artist_slug: joined_refined_text}`` for a collection.

        Reads from the ``refined_facts`` table (scope='artist'). Each row's
        ``refined_json`` is a JSON array of ``{"text": str}`` objects.
        """
        import json as _json

        conn = cls._connect()
        rows = conn.execute(
            "SELECT scope_key, refined_json FROM refined_facts "
            "WHERE scope = ? AND collection_name = ?",
            ("artist", collection_name),
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
    def get_all_refined_song_facts(cls, collection_name: str) -> Dict[str, str]:
        """Return ``{song_slug: joined_refined_text}`` for a collection.

        Reads from the ``refined_facts`` table (scope='song'). The ``scope_key``
        is the song_slug (same format as song_facts table slugs).

        Returns a dict keyed by song_slug so the search service can merge
        refined facts into TrackHit.song_facts using the same key as
        load_all_song_facts_for_collection().
        """
        import json as _json

        conn = cls._connect()
        rows = conn.execute(
            "SELECT scope_key, refined_json FROM refined_facts "
            "WHERE scope = ? AND collection_name = ?",
            ("song", collection_name),
        ).fetchall()

        result: Dict[str, str] = {}
        for track_id, json_str in rows:
            try:
                arr = _json.loads(json_str)
                texts = [
                    item.get("text", "") for item in arr
                    if isinstance(item, dict) and item.get("text")
                ]
                if texts:
                    result[track_id] = "\n\n".join(texts)
            except Exception:
                pass
        return result
