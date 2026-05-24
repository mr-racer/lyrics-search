from app.domain.models import HomeTrack, RediscoverResponse, ForYouSeedResponse


def test_home_track_minimal():
    t = HomeTrack(track_id="1", title="T", artist="A")
    assert t.album is None and t.cover_art_path is None


def test_rediscover_response_defaults():
    r = RediscoverResponse(collection_name="c")
    assert r.track is None and r.never_played is False and r.last_played is None


def test_for_you_seed_response_defaults():
    s = ForYouSeedResponse(collection_name="c")
    assert s.seed_track_id is None and s.track is None and s.source == "random"
