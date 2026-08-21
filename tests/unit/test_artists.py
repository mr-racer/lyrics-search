"""Consolidated artist tests: splitter, payload build, payload->track surfacing."""

from app.services.artist_split import (
    split_artists, artist_slugs, primary_artist, name_for_slug, artist_refs,
    parse_title_feat, artist_refs_for_track, display_title_for_track,
    display_name_for_slug,
)
from app.services.indexing_service import _build_payload_for_upsert
from app.api.routes.artists import _track_from_payload


# --- from test_artist_split.py (already class-based, kept as-is) ---

class TestSplitArtists:
    def test_solo_artist_unchanged(self):
        assert split_artists("Kanye West") == ["Kanye West"]

    def test_comma_separator(self):
        assert split_artists("Kanye West, Sia") == ["Kanye West", "Sia"]

    def test_feat_dot(self):
        assert split_artists("Kanye West feat. Rihanna") == ["Kanye West", "Rihanna"]

    def test_ft_and_featuring(self):
        assert split_artists("Drake ft Future") == ["Drake", "Future"]
        assert split_artists("Drake featuring Future") == ["Drake", "Future"]

    def test_ampersand_and_plus_and_x(self):
        assert split_artists("Calvin Harris & Dua Lipa") == ["Calvin Harris", "Dua Lipa"]
        assert split_artists("MGK x Travis") == ["MGK", "Travis"]

    def test_primary_is_first(self):
        assert primary_artist("Kanye West, Sia") == "Kanye West"

    def test_dedupe_preserves_order(self):
        assert split_artists("Sia, Sia") == ["Sia"]

    def test_empty_and_none(self):
        assert split_artists("") == []
        assert split_artists("   ") == []

    def test_vs_separator(self):
        assert split_artists("Jay-Z vs Nas") == ["Jay-Z", "Nas"]

    def test_with_separator(self):
        assert split_artists("Santana with Rob Thomas") == ["Santana", "Rob Thomas"]


class TestKnownGroups:
    def test_earth_wind_and_fire_not_split(self):
        assert split_artists("Earth, Wind & Fire") == ["Earth, Wind & Fire"]

    def test_florence_plus_machine_not_split(self):
        assert split_artists("Florence + the Machine") == ["Florence + the Machine"]

    def test_acdc_not_split(self):
        assert split_artists("AC/DC") == ["AC/DC"]

    def test_charli_xcx_not_split_by_x(self):
        # No space-bounded 'x' token, so the trailing x in XCX must not split.
        assert split_artists("Charli XCX") == ["Charli XCX"]


class TestSlugsAndFalseInclusion:
    def test_artist_slugs(self):
        assert artist_slugs("Kanye West, Sia") == ["kanye-west", "sia"]

    def test_alias_resolves_to_canonical_slug(self):
        assert artist_slugs("Ye") == ["kanye-west"]

    def test_ye_not_found_inside_kanye(self):
        # The whole point: 'ye' is a distinct slug, never a substring match.
        assert "ye" not in artist_slugs("Kanye West")
        assert "ye" not in artist_slugs("Yeah Yeah Yeahs")


class TestArtistRefs:
    """artist_refs pairs each canonical slug with its display name, aligned —
    used to render per-participant clickable artist credits on track rows."""

    def test_aligned_name_slug_pairs(self):
        assert artist_refs("Kanye West, Sia") == [
            {"name": "Kanye West", "slug": "kanye-west"},
            {"name": "Sia", "slug": "sia"},
        ]

    def test_feat_collaborator_named(self):
        assert artist_refs("Dua Lipa x Angele") == [
            {"name": "Dua Lipa", "slug": "dua-lipa"},
            {"name": "Angele", "slug": "angele"},
        ]

    def test_alias_collapse_stays_aligned(self):
        # "Ye" and "Kanye West" both resolve to kanye-west; dedup to ONE ref
        # whose name is the first tagged participant. Zipping split_artists with
        # artist_slugs would misalign here (2 names vs 1 slug).
        assert artist_refs("Ye, Kanye West") == [
            {"name": "Ye", "slug": "kanye-west"},
        ]

    def test_cyrillic(self):
        assert artist_refs("Король и Шут") == [
            {"name": "Король и Шут", "slug": "король-и-шут"},
        ]

    def test_empty_and_none(self):
        assert artist_refs("") == []
        assert artist_refs(None) == []


class TestNameForSlug:
    def test_picks_primary_participant(self):
        assert name_for_slug("Dua Lipa x Angele", "dua-lipa") == "Dua Lipa"

    def test_picks_featured_participant(self):
        # The whole point of the fix: the feat's page gets the feat's name.
        assert name_for_slug("Dua Lipa x Angele", "angele") == "Angele"

    def test_solo(self):
        assert name_for_slug("Dua Lipa", "dua-lipa") == "Dua Lipa"

    def test_no_match_returns_none(self):
        assert name_for_slug("Dua Lipa x Angele", "taylor-swift") is None

    def test_alias_resolves_to_canonical_slug(self):
        # "Ye" maps to kanye-west by alias; the tagged name "Ye" is returned.
        assert name_for_slug("Ye, Sia", "kanye-west") == "Ye"

    def test_none_input(self):
        assert name_for_slug(None, "x") is None


# --- from test_build_payload_artists.py ---

class TestBuildPayloadArtists:
    def test_payload_has_split_artist_fields(self):
        info = {
            "lyrics": "some words here long enough",
            "title": "Heartless",
            "artist": "Kanye West, Sia",
            "album": "808s",
        }
        p = _build_payload_for_upsert(info, slug=None)
        assert p["artist"] == "Kanye West, Sia"            # raw preserved for display
        assert p["artists"] == ["Kanye West", "Sia"]
        assert p["artist_slugs"] == ["kanye-west", "sia"]
        assert p["primary_artist_slug"] == "kanye-west"

    def test_payload_solo_artist(self):
        info = {"lyrics": "x", "title": "T", "artist": "Radiohead", "album": "A"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["artists"] == ["Radiohead"]
        assert p["artist_slugs"] == ["radiohead"]
        assert p["primary_artist_slug"] == "radiohead"

    def test_payload_missing_artist(self):
        info = {"lyrics": "x", "title": "T", "artist": "", "album": "A"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["artists"] == []
        assert p["artist_slugs"] == []
        assert p["primary_artist_slug"] is None

    def test_payload_carries_track_and_disc_number(self):
        info = {
            "lyrics": "x", "title": "T", "artist": "A", "album": "Alb",
            "track_number": 5, "disc_number": 2,
        }
        p = _build_payload_for_upsert(info, slug=None)
        assert p["track_number"] == 5
        assert p["disc_number"] == 2

    def test_payload_track_number_defaults_to_none(self):
        info = {"lyrics": "x", "title": "T", "artist": "A", "album": "Alb"}
        p = _build_payload_for_upsert(info, slug=None)
        assert p["track_number"] is None
        assert p["disc_number"] is None


# --- from test_track_from_payload_artists.py ---

class TestTrackFromPayloadArtists:
    def test_uses_payload_fields_when_present(self):
        p = {
            "title": "Heartless", "artist": "Kanye West, Sia",
            "artists": ["Kanye West", "Sia"],
            "artist_slugs": ["kanye-west", "sia"],
            "primary_artist_slug": "kanye-west",
            "duration": 200,
        }
        t = _track_from_payload("id1", p)
        assert t.artists == ["Kanye West", "Sia"]
        assert t.primary_artist_slug == "kanye-west"

    def test_falls_back_to_split_when_fields_absent(self):
        p = {"title": "T", "artist": "Kanye West feat. Sia", "duration": 200}
        t = _track_from_payload("id1", p)
        assert t.artists == ["Kanye West", "Sia"]
        assert t.primary_artist_slug == "kanye-west"

    def test_empty_artist_yields_none(self):
        p = {"title": "T", "artist": "", "artists": [], "artist_slugs": [], "duration": 0}
        t = _track_from_payload("id1", p)
        assert t.artists is None
        assert t.primary_artist_slug is None

    def test_primary_derived_from_slugs_when_primary_absent(self):
        p = {"title": "T", "artist": "Kanye West, Sia",
             "artists": ["Kanye West", "Sia"], "artist_slugs": ["kanye-west", "sia"],
             "duration": 200}
        t = _track_from_payload("id1", p)
        assert t.primary_artist_slug == "kanye-west"


# --- feat-in-title parsing (2026-07-19: 388/5630 library tracks carry feat in the title) ---

class TestParseTitleFeat:
    def test_plain_title_untouched(self):
        r = parse_title_feat("Heartless")
        assert r.clean_title == "Heartless"
        assert r.feat_names == [] and r.with_names == []

    def test_paren_feat(self):
        r = parse_title_feat("Bangarang (ft. Sirah)")
        assert r.clean_title == "Bangarang"
        assert r.feat_names == ["Sirah"]

    def test_bracket_feat(self):
        r = parse_title_feat("Beautiful [feat. Enrique Iglesias]")
        assert r.clean_title == "Beautiful"
        assert r.feat_names == ["Enrique Iglesias"]

    def test_capital_feat_spelling(self):
        r = parse_title_feat("Best Friend (Feat. Yelawolf)")
        assert r.clean_title == "Best Friend"
        assert r.feat_names == ["Yelawolf"]

    def test_multi_comma_and_and(self):
        r = parse_title_feat("The Sweet Prince (feat. Ajay Prasanna, Johnny Marr and Anoushka Shankar)")
        assert r.clean_title == "The Sweet Prince"
        assert r.feat_names == ["Ajay Prasanna", "Johnny Marr", "Anoushka Shankar"]

    def test_ampersand_inside_feat_chunk(self):
        r = parse_title_feat("On the Road (ft. Meek Mill & Lil Baby)")
        assert r.feat_names == ["Meek Mill", "Lil Baby"]

    def test_and_capitalized(self):
        r = parse_title_feat("4 Minutes (feat. Justin Timberlake And Timbaland)")
        assert r.feat_names == ["Justin Timberlake", "Timbaland"]

    def test_trailing_text_after_paren_survives(self):
        r = parse_title_feat("Starboy (ft. Daft Punk) [Kygo Remix]")
        assert r.clean_title == "Starboy [Kygo Remix]"
        assert r.feat_names == ["Daft Punk"]

    def test_prefix_inside_paren_stays_in_title(self):
        r = parse_title_feat("Lighters (Bad Meets Evil Feat. Bruno Mars)")
        assert r.clean_title == "Lighters (Bad Meets Evil)"
        assert r.feat_names == ["Bruno Mars"]

    def test_bare_ft_to_end_of_string(self):
        r = parse_title_feat("Ride ft. MUTEMATH")
        assert r.clean_title == "Ride"
        assert r.feat_names == ["MUTEMATH"]

    def test_with_paren_is_co_artist_not_feat(self):
        r = parse_title_feat("Kids [with Robbie Williams]")
        assert r.clean_title == "Kids"
        assert r.with_names == ["Robbie Williams"]
        assert r.feat_names == []

    def test_with_trailing_remaster_survives(self):
        r = parse_title_feat("The Motown Song (with The Temptations) (2008 Remaster)")
        assert r.clean_title == "The Motown Song (2008 Remaster)"
        assert r.with_names == ["The Temptations"]

    def test_with_oxford_comma_ampersand(self):
        r = parse_title_feat("Roc The Mic-Remix [With Freeway, Beanie Sigel, & Murphy Lee]")
        assert r.with_names == ["Freeway", "Beanie Sigel", "Murphy Lee"]

    def test_bare_with_not_honored(self):
        # Way too many legit titles contain the word "with".
        r = parse_title_feat("Dancing with a Stranger")
        assert r.clean_title == "Dancing with a Stranger"
        assert r.with_names == []

    def test_ft_without_name_is_not_a_feat(self):
        # Foals — "10,000 Ft." (the only library false-positive candidate).
        r = parse_title_feat("10,000 Ft.")
        assert r.clean_title == "10,000 Ft."
        assert r.feat_names == []

    def test_word_containing_ft_untouched(self):
        r = parse_title_feat("Shoplifters of the World Unite")
        assert r.clean_title == "Shoplifters of the World Unite"
        assert r.feat_names == []

    def test_idempotent_on_clean_output(self):
        once = parse_title_feat("Bangarang (ft. Sirah)")
        twice = parse_title_feat(once.clean_title)
        assert twice.clean_title == once.clean_title
        assert twice.feat_names == []

    def test_empty_and_none(self):
        assert parse_title_feat("").clean_title == ""
        assert parse_title_feat(None).clean_title == ""


class TestArtistRefsForTrack:
    def test_payload_arrays_with_feat_slugs_are_authoritative(self):
        p = {
            "title": "Bangarang", "artist": "Skrillex",
            "artists": ["Skrillex", "Sirah"],
            "artist_slugs": ["skrillex", "sirah"],
            "feat_artist_slugs": ["sirah"],
        }
        refs = artist_refs_for_track(p)
        assert refs == [
            {"name": "Skrillex", "slug": "skrillex", "role": "main"},
            {"name": "Sirah", "slug": "sirah", "role": "feat"},
        ]

    def test_legacy_payload_falls_back_to_title_parse(self):
        # Pre-backfill payload: arrays exist but carry no title-feats.
        p = {
            "title": "Bangarang (ft. Sirah)", "artist": "Skrillex",
            "artists": ["Skrillex"], "artist_slugs": ["skrillex"],
        }
        refs = artist_refs_for_track(p)
        assert {"name": "Sirah", "slug": "sirah", "role": "feat"} in refs
        assert refs[0] == {"name": "Skrillex", "slug": "skrillex", "role": "main"}

    def test_with_artist_billed_as_main(self):
        p = {"title": "Kids [with Robbie Williams]", "artist": "Kylie Minogue"}
        refs = artist_refs_for_track(p)
        assert refs == [
            {"name": "Kylie Minogue", "slug": "kylie-minogue", "role": "main"},
            {"name": "Robbie Williams", "slug": "robbie-williams", "role": "main"},
        ]

    def test_tag_feat_gets_feat_role(self):
        # feat in the ARTIST tag: same split as before, but the participant
        # after the keyword now carries an explicit feat role (previously the
        # frontend faked this by position, mislabeling «A & B» duos).
        p = {"title": "Heartless", "artist": "Kanye West feat. Sia"}
        refs = artist_refs_for_track(p)
        assert [(r["slug"], r["role"]) for r in refs] == [
            ("kanye-west", "main"), ("sia", "feat"),
        ]

    def test_tag_duo_stays_co_primary(self):
        p = {"title": "One Kiss", "artist": "Calvin Harris & Dua Lipa"}
        refs = artist_refs_for_track(p)
        assert [(r["slug"], r["role"]) for r in refs] == [
            ("calvin-harris", "main"), ("dua-lipa", "main"),
        ]

    def test_dedupe_feat_already_in_tag_upgrades_role(self):
        # No duplicate ref; the title's feat signal upgrades the existing
        # (role-less) credit to feat instead of being dropped.
        p = {"title": "Under Control (feat. Hurts)", "artist": "Calvin Harris & Alesso & Hurts"}
        refs = artist_refs_for_track(p)
        assert [r["slug"] for r in refs] == ["calvin-harris", "alesso", "hurts"]
        assert [r["role"] for r in refs] == ["main", "main", "feat"]

    def test_sqlite_mirror_arrays_recover_feat_role_from_title(self):
        # SQLite track_metadata mirrors extended artists/artist_slugs but has
        # no feat_artist_slugs column — the title parse must restore the role.
        p = {
            "title": "Bangarang (ft. Sirah)", "artist": "Skrillex",
            "artists": ["Skrillex", "Sirah"],
            "artist_slugs": ["skrillex", "sirah"],
        }
        refs = artist_refs_for_track(p)
        assert refs == [
            {"name": "Skrillex", "slug": "skrillex", "role": "main"},
            {"name": "Sirah", "slug": "sirah", "role": "feat"},
        ]

    def test_empty(self):
        assert artist_refs_for_track({}) == []


class TestDisplayTitleForTrack:
    def test_payload_title_display_wins(self):
        p = {"title": "Bangarang (ft. Sirah)", "title_display": "Bangarang"}
        assert display_title_for_track(p) == "Bangarang"

    def test_fallback_parses_title(self):
        p = {"title": "Bangarang (ft. Sirah)"}
        assert display_title_for_track(p) == "Bangarang"

    def test_clean_title_returns_none(self):
        assert display_title_for_track({"title": "Heartless"}) is None


# --- featured-only album detection (Atlas "featured on" rail) ---

from app.api.routes.artists import _featured_only
from app.domain.models import TrackMetadata


def _tm(title, artist, refs=None, primary=None):
    return TrackMetadata(
        track_id="x", title=title, artist=artist, duration_sec=1.0,
        file_path="/f", primary_artist_slug=primary, artist_refs=refs or [],
    )


class TestFeaturedOnlyAlbum:
    def test_feat_only_album_is_featured(self):
        tracks = [_tm(
            "Bangarang", "Skrillex", primary="skrillex",
            refs=[{"name": "Skrillex", "slug": "skrillex", "role": "main"},
                  {"name": "Sirah", "slug": "sirah", "role": "feat"}],
        )]
        assert _featured_only(tracks, "sirah") is True
        assert _featured_only(tracks, "skrillex") is False

    def test_main_credit_anywhere_makes_album_own(self):
        tracks = [
            _tm("T1", "A", primary="a",
                refs=[{"name": "A", "slug": "a", "role": "main"},
                      {"name": "B", "slug": "b", "role": "feat"}]),
            _tm("T2", "B", primary="a",
                refs=[{"name": "B", "slug": "b", "role": "main"}]),
        ]
        assert _featured_only(tracks, "b") is False

    def test_primary_artist_slug_match_is_own(self):
        tracks = [_tm("T", "A", primary="b", refs=[])]
        assert _featured_only(tracks, "b") is False

    def test_legacy_track_without_refs_is_own(self):
        # No per-participant refs → no role signal → never exile the album.
        tracks = [_tm("T", "A", primary="a", refs=[])]
        assert _featured_only(tracks, "b") is False

    def test_co_lead_duet_is_own(self):
        tracks = [_tm(
            "One Kiss", "Calvin Harris & Dua Lipa", primary="calvin-harris",
            refs=[{"name": "Calvin Harris", "slug": "calvin-harris", "role": "main"},
                  {"name": "Dua Lipa", "slug": "dua-lipa", "role": "main"}],
        )]
        assert _featured_only(tracks, "dua-lipa") is False


class TestDisplayNameForSlug:
    """display_name_for_slug never labels a slug with ANOTHER artist's name.

    The failure it guards: a track tagged «Kanye West» whose TITLE credits a
    guest («FML (ft. The Weeknd)») puts `the-weeknd` into artist_slugs while
    the raw tag names only Kanye. Falling back to that raw tag hands the guest
    a foreign name — and downstream the bio task researches the wrong person.
    """

    def test_participant_list_wins(self):
        assert display_name_for_slug(
            "the-weeknd",
            participants=["Kanye West", "The Weeknd"],
            raw="Kanye West",
        ) == "The Weeknd"

    def test_keeps_source_casing(self):
        assert display_name_for_slug(
            "babytron", participants=["Eminem", "BabyTron"], raw="Eminem",
        ) == "BabyTron"

    def test_falls_back_to_raw_tag_participant(self):
        assert display_name_for_slug(
            "dua-lipa", participants=None, raw="Dua Lipa x Angele",
        ) == "Dua Lipa"

    def test_never_returns_a_foreign_name(self):
        # Neither the participants nor the raw tag mention this slug's artist:
        # a title-cased slug is correct, the other artist's name never is.
        assert display_name_for_slug(
            "the-weeknd", participants=["Kanye West"], raw="Kanye West",
        ) == "The Weeknd"

    def test_alias_resolves_to_the_canonical_slug(self):
        assert display_name_for_slug(
            "kanye-west", participants=["Ye"], raw="Ye",
        ) == "Ye"

    def test_no_signal_at_all_titlecases_the_slug(self):
        assert display_name_for_slug("agents-of-time") == "Agents Of Time"
