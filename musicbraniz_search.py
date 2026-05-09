import time
import musicbrainzngs
from typing import Optional


class MusicBrainzLookup:
    """Wrapper around musicbrainzngs for enriched metadata lookups."""

    def __init__(self, app_name: str, app_version: str, app_contact: str):
        musicbrainzngs.set_useragent(app_name, app_version, app_contact)
        self._cache: dict = {}

    # ── Resolution ──

    def resolve_recording_id(self, title: str, artist: str) -> Optional[str]:
        """Find recording MBID by title + artist name."""
        cache_key = f"rec_id:{title}:{artist}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            artist_search = musicbrainzngs.search_artists(artist=artist, limit=1)
            if not artist_search["artist-count"]:
                return None
            artist_id = artist_search["artist-list"][0]["id"]

            rec_search = musicbrainzngs.search_recordings(
                query=f'recording:"{title}" artist:{artist_id}',
                limit=1,
            )
            if not rec_search["recording-count"]:
                return None
            rec_id = rec_search["recording-list"][0]["id"]
            self._cache[cache_key] = rec_id
            return rec_id
        except Exception:
            return None

    def resolve_release_id(
        self, recording_id: str, prefer_best: bool = True
    ) -> Optional[str]:
        """Get release MBID for a recording.

        If prefer_best=True: pick release where status=Official + quality=high.
        Falls back to any official, then to absolute first.
        """
        rec_data = self._get_recording(recording_id, ["releases"])
        if not rec_data:
            return None

        releases = rec_data.get("release-list", [])
        if not releases:
            return None

        if prefer_best:
            # Priority 1: official + high quality
            for r in releases:
                if r.get("status") == "Official" and r.get("quality") == "high":
                    return r["id"]
            # Priority 2: any official
            for r in releases:
                if r.get("status") == "Official":
                    return r["id"]

        # Fallback: absolute first
        return releases[0]["id"]

    # ── Recording-level (songs) ──

    def get_recording_producers(self, recording_id: str) -> list[str]:
        """Return producer names for a recording."""
        rec_data = self._get_recording(recording_id, ["artist-rels"])
        if not rec_data:
            return []

        producers = []
        for rel in rec_data.get("artist-relation-list", []):
            if rel.get("type") == "producer":
                artist = rel.get("artist", {})
                name = artist.get("name")
                if name:
                    producers.append(name)
        return producers

    def get_recording_samples(self, recording_id: str) -> list[dict]:
        """Return recordings this track samples (forward relations)."""
        return self._get_sample_relations(recording_id, direction="forward")

    def get_recording_sampled_by(self, recording_id: str) -> list[dict]:
        """Return recordings that sample this track (backward relations)."""
        return self._get_sample_relations(recording_id, direction="backward")

    # ── Release-level (albums) ──

    def get_release_date(self, release_id: str) -> Optional[str]:
        """Return release date string (YYYY, YYYY-MM, or YYYY-MM-DD)."""
        rel_data = self._get_release(release_id)
        return rel_data.get("date") if rel_data else None

    def get_release_year(self, release_id: str) -> Optional[int]:
        """Parse date string and return year as int."""
        date_str = self.get_release_date(release_id)
        return self._parse_year(date_str)

    def get_release_labels(self, release_id: str) -> list[dict]:
        """Return label info: name, catalog_number, country."""
        rel_data = self._get_release(release_id, ["labels"])
        if not rel_data:
            return []

        labels = []
        for info in rel_data.get("label-info-list", []):
            label = info.get("label", {})
            labels.append({
                "name": label.get("name", ""),
                "id": label.get("id"),
                "country": label.get("country"),
                "catalog_number": info.get("catalog-number", ""),
            })
        return labels

    # ── Convenience (recording → release bridge) ──

    def get_recording_year(self, recording_id: str) -> Optional[int]:
        """Bridge: resolve best release, return its year as int."""
        release_id = self.resolve_release_id(recording_id)
        if not release_id:
            return None
        return self.get_release_year(release_id)

    def get_recording_labels(self, recording_id: str) -> list[dict]:
        """Bridge: resolve best release, then get its labels."""
        release_id = self.resolve_release_id(recording_id)
        if not release_id:
            return []
        return self.get_release_labels(release_id)

    # ── Internal helpers ──

    def _get_recording(self, recording_id: str, includes: list[str] | None = None) -> Optional[dict]:
        """Fetch recording data, cached by MBID."""
        cache_key = f"rec:{recording_id}:{','.join(sorted(includes or []))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = musicbrainzngs.get_recording_by_id(
                recording_id, includes=includes or []
            )
            rec_data = result.get("recording")
            self._cache[cache_key] = rec_data
            time.sleep(1)  # MusicBrainz rate limit: 1 req/sec
            return rec_data
        except Exception:
            return None

    def _get_release(
        self, release_id: str, includes: list[str] | None = None
    ) -> Optional[dict]:
        """Fetch release data, cached by MBID."""
        cache_key = f"rel:{release_id}:{','.join(sorted(includes or []))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = musicbrainzngs.get_release_by_id(
                release_id, includes=includes or []
            )
            rel_data = result.get("release")
            self._cache[cache_key] = rel_data
            time.sleep(1)  # MusicBrainz rate limit: 1 req/sec
            return rel_data
        except Exception:
            return None

    def _get_sample_relations(
        self, recording_id: str, direction: str
    ) -> list[dict]:
        """Extract sample relations (forward=contains samples, backward=sampled by)."""
        rec_data = self._get_recording(recording_id, ["recording-rels"])
        if not rec_data:
            return []

        results = []
        for rel in rec_data.get("recording-relation-list", []):
            rel_type = rel.get("type", "")
            if rel_type not in ("samples material", "samples", "sampled in"):
                continue
            if rel.get("direction") != direction:
                continue

            rec = rel.get("recording", {})
            title = rec.get("title", "")
            artist_cred = rec.get("artist-credit", [])
            artist = ""
            if artist_cred:
                artist = artist_cred[0].get("artist", {}).get("name", "")

            if title or artist:
                results.append({"title": title, "artist": artist, "id": rec.get("id")})

        return results

    @staticmethod
    def _parse_year(date_str: Optional[str]) -> Optional[int]:
        """Convert MB date string (YYYY, YYYY-MM, YYYY-MM-DD) to int year."""
        if not date_str:
            return None
        try:
            return int(date_str.split("-")[0])
        except (ValueError, IndexError):
            return None
