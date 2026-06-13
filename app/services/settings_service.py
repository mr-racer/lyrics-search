"""Instance-level settings: LLM endpoint/key/model, embedding model, AI/CLAP
flags — admin-owned, shared by every account in both modes.

Resolution precedence for each key is **DB > env var > built-in default**:

  1. ``instance_settings`` table row (written via the wizard / Settings panel)
  2. the mirrored environment variable (so ops can preset via docker-compose/.env)
  3. a hard-coded sane default

Keys mirror env var names so the chain is uniform. The one exception is the LLM
API key, whose env fallback is the conventional ``OPENAI_API_KEY``.

The API key is a SECRET: never return it raw across the API boundary — use
``public_view()`` which masks it, and never log ``llm_api_key()``.
"""
from __future__ import annotations

import os
from typing import Optional

from app.resources.metadata_db import MetadataDB


# Setting key  ->  (env var name, built-in default)
# The key is what's stored in instance_settings and sent over the API.
_SPEC: dict[str, tuple[str, Optional[str]]] = {
    "LLM_BASE_URL": ("LLM_BASE_URL", None),
    "LLM_MODEL":    ("LLM_MODEL", None),
    "LLM_API_KEY":  ("OPENAI_API_KEY", "lm-studio"),
    "EMBED_MODEL":  ("EMBED_MODEL", None),
    "CLAP_ENABLED": ("CLAP_ENABLED", "1"),
    "AI_ENABLED":   ("AI_ENABLED", "0"),
}

SECRET_KEYS = frozenset({"LLM_API_KEY"})
BOOL_KEYS = frozenset({"CLAP_ENABLED", "AI_ENABLED"})

# Sentinel a PATCH sends for a secret it does NOT want to change. The frontend
# never receives the real key, so it echoes this back when the field is left
# untouched; the service then skips writing that key.
UNCHANGED_SENTINEL = "__unchanged__"


def _parse_bool(raw: Optional[str]) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class SettingsService:
    """Stateless resolver over MetadataDB + os.environ. Cheap to construct;
    a module-level singleton is exposed via app.api.dependencies."""

    def __init__(self, db=MetadataDB):
        self.db = db

    # ── core resolver ─────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[str]:
        """Effective raw string value for ``key`` (DB > env > default)."""
        if key not in _SPEC:
            raise KeyError(f"unknown instance setting {key!r}")
        env_name, default = _SPEC[key]
        db_val = self.db.get_instance_setting(key)
        if db_val is not None and db_val != "":
            return db_val
        env_val = os.getenv(env_name)
        if env_val is not None and env_val != "":
            return env_val
        return default

    def db_value(self, key: str) -> Optional[str]:
        """The DB-stored override for ``key`` only (no env/default), or None if
        unset. Used by llm_client to give admin instance-settings precedence
        over per-request (localStorage) values while keeping env/default as a
        lower fallback. Never raises — a broken/uninitialized DB resolves to
        None so AI paths fall back gracefully."""
        if key not in _SPEC:
            raise KeyError(f"unknown instance setting {key!r}")
        try:
            v = self.db.get_instance_setting(key)
        except Exception:
            return None
        return v or None

    def source(self, key: str) -> str:
        """Where the effective value comes from: 'db' | 'env' | 'default'."""
        env_name, _ = _SPEC[key]
        db_val = self.db.get_instance_setting(key)
        if db_val is not None and db_val != "":
            return "db"
        env_val = os.getenv(env_name)
        if env_val is not None and env_val != "":
            return "env"
        return "default"

    # ── typed accessors ───────────────────────────────────────────────────
    def llm_base_url(self) -> Optional[str]:
        return self.get("LLM_BASE_URL")

    def llm_model(self) -> Optional[str]:
        return self.get("LLM_MODEL")

    def llm_api_key(self) -> Optional[str]:
        return self.get("LLM_API_KEY")

    def embed_model(self) -> Optional[str]:
        return self.get("EMBED_MODEL")

    def clap_enabled(self) -> bool:
        return _parse_bool(self.get("CLAP_ENABLED"))

    def ai_enabled(self) -> bool:
        return _parse_bool(self.get("AI_ENABLED"))

    def ai_available(self) -> bool:
        """Public-config flag: AI features show only when the admin enabled AI
        AND a working LLM endpoint is configured."""
        return self.ai_enabled() and bool(self.llm_base_url())

    # ── write (owner-only is enforced at the route) ───────────────────────
    def set_many(self, patch: dict[str, Optional[str]], *, updated_at: float) -> None:
        """Persist a partial update. Rules:
          - keys not in _SPEC are ignored (graceful, no error);
          - a secret key whose value is the UNCHANGED_SENTINEL is skipped;
          - bool keys are normalized to '1'/'0';
          - None / '' DELETES the key (resolver falls back to env/default).
        """
        clean: dict[str, Optional[str]] = {}
        for key, value in patch.items():
            if key not in _SPEC:
                continue
            if key in SECRET_KEYS and value == UNCHANGED_SENTINEL:
                continue
            if key in BOOL_KEYS and value is not None and value != "":
                value = "1" if _parse_bool(value) else "0"
            clean[key] = value
        self.db.set_instance_settings(clean, updated_at)

    # ── read for the owner Settings UI (secret masked) ────────────────────
    def public_view(self) -> dict[str, dict]:
        """{key: {value, source}} for the owner UI. The API key is masked: its
        value is null and an extra 'has_value' bool tells the UI whether a key
        is configured. Bool keys are returned as 'true'/'false' strings via
        their effective value (UI parses)."""
        out: dict[str, dict] = {}
        for key in _SPEC:
            src = self.source(key)
            if key in SECRET_KEYS:
                eff = self.get(key)
                out[key] = {"value": None, "source": src, "has_value": bool(eff)}
            else:
                out[key] = {"value": self.get(key), "source": src}
        return out


# Module-level singleton — settings are global and the service holds no state.
settings_service = SettingsService()
