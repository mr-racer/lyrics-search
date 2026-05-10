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
    "CREATE INDEX IF NOT EXISTS idx_af_artist_lang ON artist_facts(artist_slug, lang)",
    "CREATE INDEX IF NOT EXISTS idx_sf_song_lang ON song_facts(song_slug, lang)",
    "CREATE INDEX IF NOT EXISTS idx_artists_collection ON artists(collection_name)",
    "CREATE INDEX IF NOT EXISTS idx_songs_collection ON songs(collection_name)",
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
        """Create tables & indexes if they don't exist yet."""
        conn = cls._connect()
        for sql in _SCHEMA_SQL:
            conn.execute(sql)
        conn.commit()
        logger.info("[MetadataDB] Schema initialised")

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
