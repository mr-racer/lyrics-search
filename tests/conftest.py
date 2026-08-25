"""Shared fixtures for the MusiX test suite."""

import contextlib
import os
import pytest
import sys
import types
from pathlib import Path
from _pytest.python import Package

# musicbrainzngs is always stubbed — no test needs the real client.
sys.modules.setdefault("musicbrainzngs", types.ModuleType("musicbrainzngs"))

# Stub heavy ML deps before any app.* module loads them — UNLESS this run is the
# live-stack suite (tests/docker/), which needs the REAL torch / CLAP /
# sentence-transformers and a reachable Qdrant. The docker runner exports
# MUSIX_LIVE_STACK=1 (see scripts/run_docker_tests.sh).
if not os.environ.get("MUSIX_LIVE_STACK"):
    # Point Qdrant at a name that cannot resolve, BEFORE app.* imports read it.
    #
    # Every TestClient(create_app()) runs the lifespan, which opens a Qdrant
    # connection. The suite expects that to fail — the tests assert limited-mode
    # behaviour — but "fail" and "fail quickly" are not the same thing: a
    # loopback port that is filtered rather than closed (a Windows firewall
    # default) makes the connect sit until it times out. That was 2.2 s per
    # client construction, paid by a few hundred function-scoped fixtures, and it
    # was the whole reason this suite took ten minutes instead of seconds.
    #
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737): guaranteed unroutable, and an IP
    # literal so nothing waits on DNS either. _block_outbound_network below
    # refuses it at connect() the moment it is dialled. Set unconditionally, not
    # setdefault: a developer with QDRANT_URL exported for the real app should
    # still get fast, hermetic tests. tests/docker/ opts out via MUSIX_LIVE_STACK.
    os.environ["QDRANT_URL"] = "http://192.0.2.1:6333"

    sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

    _torch_stub = types.ModuleType("torch")

    class _OutOfMemoryError(RuntimeError):
        """Stand-in for ``torch.OutOfMemoryError`` (a RuntimeError there too).

        Named here because the registry's sparse leg CATCHES it: shrinking the
        batch after an allocator failure is behaviour, not plumbing, and a stub
        without this class would leave that path untestable off a GPU.
        """

    _torch_stub.OutOfMemoryError = _OutOfMemoryError
    _torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda i=0: "stub",
        empty_cache=lambda: None,
        OutOfMemoryError=_OutOfMemoryError,
    )
    _torch_stub.device = lambda x: "cpu"
    _torch_stub.Tensor = object  # dummy for scipy is_torch_array check
    # ModelRegistry names these when it asks for an fp16 GPU load or an fp32 CPU
    # one. Sentinels are enough — nothing under the stubs ever runs a kernel.
    _torch_stub.float16 = "float16"
    _torch_stub.float32 = "float32"

    @contextlib.contextmanager
    def _no_grad():
        yield

    _torch_stub.no_grad = _no_grad
    _torch_stub.inference_mode = _no_grad
    _torch_stub.long = "long"

    class _CatResult(list):
        """What the stub's ``torch.cat`` returns: ROWS, not batches.

        Flattened on purpose. The sparse leg encodes its texts out of order
        (longest first, so the biggest batch either fits or fails immediately)
        and permutes the rows back afterwards. Row-level fakes are what let a
        test assert that the permutation actually restores the caller's order —
        the one place where that reordering could silently corrupt a ranking.
        """

        # MILCO hands its rows back on the CPU (``milco.py`` calls ``.cpu()``
        # per batch), and the leg reads ``.device`` off the concatenation to
        # build the permutation index on the same one.
        device = "cpu"

        def coalesce(self):
            return self

    _torch_stub.cat = lambda tensors, dim=0: _CatResult(
        [row for t in tensors for row in t])
    _torch_stub.tensor = lambda data, **kw: list(data)
    _torch_stub.index_select = lambda t, dim, idx: _CatResult([t[i] for i in idx])
    sys.modules.setdefault("torch", _torch_stub)

    _st_stub = types.ModuleType("sentence_transformers")
    _st_stub.SentenceTransformer = object  # dummy; never instantiated in unit tests
    sys.modules.setdefault("sentence_transformers", _st_stub)


def pytest_configure(config):
    """Patch Package.setup to skip importing the project root __init__.py.

    The project root has an __init__.py with relative imports that fail when
    pytest tries to treat the root as a test package. We skip the import
    for the project root only.
    """
    root_dir = Path(__file__).resolve().parent.parent
    original_setup = Package.setup

    def _patched_setup(self):
        if self.path.parent == root_dir or self.path == root_dir:
            return  # skip root __init__.py import
        original_setup(self)

    Package.setup = _patched_setup


# Hosts a test may legitimately reach: the loopback services a developer might
# have running (Qdrant, SearXNG, a local LLM). Everything else is the internet.
_LOCAL_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "", None,
})


class OutboundNetworkBlocked(RuntimeError):
    """Raised when a test tries to reach a host outside loopback."""


def _is_local(host) -> bool:
    if host in _LOCAL_HOSTS:
        return True
    h = str(host).strip("[]").lower()
    return h in _LOCAL_HOSTS or h.startswith("127.") or h.endswith(".localhost")


@pytest.fixture(autouse=True)
def _block_outbound_network(monkeypatch):
    """Fail fast instead of dialling the internet.

    Two entire suites used to spend most of their wall clock on real outbound
    calls: ``llm_client`` resolves to ``api.openai.com`` whenever LLM_BASE_URL is
    unset (with the built-in ``lm-studio`` key, so every call was a TLS handshake
    ending in 401), and the facts services hit TheAudioDB / Deezer / Genius. Each
    of those is a DNS lookup plus a round trip per call, which is why the
    integration suite ran for minutes and why its results depended on whether the
    machine had a working uplink.

    Blocking at ``getaddrinfo`` means no DNS wait and no connect timeout — the
    call raises immediately, and the production code paths under test already
    treat an unreachable endpoint as a state to degrade from. A test that WANTS a
    remote response has to stub it, which is what these tests believed they were
    doing all along.

    The live-stack suite (tests/docker/) is exempt: it exists to talk to real
    services.
    """
    if os.environ.get("MUSIX_LIVE_STACK"):
        yield
        return

    import socket

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if not _is_local(host):
            raise OutboundNetworkBlocked(
                f"outbound network blocked in tests: {host}:{port} — "
                f"stub this call (see tests/conftest.py::_block_outbound_network)"
            )
        return real_getaddrinfo(host, port, *args, **kwargs)

    def guarded_connect(self, address, *args, **kwargs):
        # Backstop for a connection made to a literal IP, which never asks DNS.
        if isinstance(address, tuple) and address and not _is_local(address[0]):
            raise OutboundNetworkBlocked(
                f"outbound network blocked in tests: {address[0]} — "
                f"stub this call (see tests/conftest.py::_block_outbound_network)"
            )
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield


@pytest.fixture(autouse=True)
def _clear_light_payload_cache():
    """Reset the per-collection light-payload cache before each test.

    ``app.resources.qdrant_utils.light_points`` memoises a full-collection
    scroll keyed by collection name with a 90s TTL. Tests reuse collection
    names (e.g. "c") with different fake clients, so without this the second
    case would read the first case's cached points. Cheap and isolating.
    """
    from app.resources.qdrant_utils import invalidate_light_cache
    from app.services.library_catalog import invalidate as invalidate_catalog
    from app.services.similarity_service import clear_load_memo
    invalidate_light_cache()
    clear_load_memo()
    # The assistant's library catalog is memoised the same way and for the same
    # reason (a full scan plus a token index). Without this the second test to
    # use a collection name reads the first one's library.
    invalidate_catalog()
    yield
    invalidate_light_cache()
    clear_load_memo()
    invalidate_catalog()


@pytest.fixture
def sample_track():
    """A standard track metadata dict for unit tests."""
    return {
        "title": "Blinding Lights",
        "artist": "The Weeknd",
        "album": "After Hours",
        "year": 2020,
        "genre": "Pop",
        "duration": 200,
        "lyrics": "I've been on my own for long enough\nNow I've got somebody...",
        "file_path": "/music/the-weeknd/blinding-lights.flac",
    }


@pytest.fixture
def sample_tracks_data():
    """Dict keyed by 'Artist — Title' as prepare_metadata expects."""
    return {
        "The Weeknd — Blinding Lights": {
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "album": "After Hours",
            "year": 2020,
            "genre": "Pop",
            "duration": 200,
            "lyrics": "lyrics here with enough length to pass the filter threshold comfortably",
            "file_path": "/music/1.flac",
        },
        "Dua Lipa — Levitating": {
            "title": "Levitating",
            "artist": "Dua Lipa",
            "album": "Future Nostalgia",
            "year": 2020,
            "genre": "Disco",
            "duration": 180,
            "lyrics": "another set of lyrics that is long enough to pass the minimum length filter check",
            "file_path": "/music/2.flac",
        },
        "Kendrick — HUMBLE": {
            "title": "HUMBLE.",
            "artist": "Kendrick",
            "album": "DAMN.",
            "year": 2017,
            "genre": "Hip-Hop",
            "duration": 177,
            "lyrics": "sir this is a fake but long enough lyrics text to ensure it passes the filter",
            "file_path": "/music/3.flac",
        },
    }


@pytest.fixture
def sample_vectors():
    """Small set of normalized vectors for similarity tests."""
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0],
    ]


@pytest.fixture
def mock_qdrant_point():
    """Factory for creating mock ScoredPoint objects."""
    def _make_point(track_id="abc123", score=0.85, payload=None):
        from qdrant_client.models import ScoredPoint

        return ScoredPoint(
            id=track_id,
            version=1,
            score=score,
            payload=payload
            or {
                "title": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "year": 2020,
                "genre": "Pop",
                "duration": 200,
                "lyrics": "test lyrics for the song that are long enough",
                "file_path": "/test/file.flac",
            },
            vector={},
        )

    return _make_point


# ── Phase C: tiny audio fixtures (server-mode upload tests) ──
# Defined here in the ROOT tests/conftest.py so they resolve for tests under
# tests/unit/ and tests/integration/ (a conftest in tests/fixtures/ would only
# be visible to tests under that directory). Fixture FILES live in
# tests/fixtures/audio/ (generated by generate_fixtures.py, committed).

_AUDIO_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture
def audio_bytes():
    """Return raw bytes for a named fixture: audio_bytes('tiny.flac')."""

    def _load(name: str) -> bytes:
        p = _AUDIO_FIXTURE_DIR / name
        if not p.exists():
            pytest.skip(f"fixture missing: {p} (run tests/fixtures/audio/generate_fixtures.py)")
        return p.read_bytes()

    return _load


@pytest.fixture
def audio_path():
    """Return absolute path to a named fixture: audio_path('tiny.flac')."""

    def _load(name: str) -> Path:
        p = _AUDIO_FIXTURE_DIR / name
        if not p.exists():
            pytest.skip(f"fixture missing: {p}")
        return p

    return _load
