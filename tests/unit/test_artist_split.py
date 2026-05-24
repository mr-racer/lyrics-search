"""Unit tests for the artist splitter (whole-slug matching, curated rules)."""

from app.services.artist_split import split_artists, artist_slugs, primary_artist


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
