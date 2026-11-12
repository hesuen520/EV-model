"""Independent verification of :mod:`evmodel.names`.

``player_key`` is checked against hand-written expected keys for real feed
spellings; ``NameMatcher`` against the matching order documented in its
docstring and against adversarial pairs of *different* real players, where the
only acceptable answers are the right player or ``None``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from evmodel.names import NameMatcher, player_key

# --------------------------------------------------------------------------
# player_key
# --------------------------------------------------------------------------
#: Spellings of one player per row; every spelling must give the first column.
EQUIVALENT_SPELLINGS: List[Tuple[str, List[str]]] = [
    ("nikola jokic", ["Nikola Jokić", "Nikola Jokic", "NIKOLA JOKIĆ", "nikola jokić", "Jokić, Nikola", " Nikola  Jokic "]),
    ("michael porter", ["Michael Porter Jr.", "Michael Porter Jr", "Michael Porter", "Michael Porter, Jr.",
                        "Porter, Michael Jr.", "Porter Jr., Michael", "MICHAEL PORTER JR."]),
    ("cj stroud", ["C.J. Stroud", "CJ Stroud", "C.J Stroud", "cj stroud", "Stroud, C.J.", "Stroud, CJ"]),
    ("amonra st brown", ["Amon-Ra St. Brown", "Amon-Ra St Brown", "AmonRa St. Brown", "St. Brown, Amon-Ra"]),
    ("royce oneale", ["Royce O'Neale", "Royce O’Neale", "Royce ONeale", "O'Neale, Royce"]),
    ("deaaron fox", ["De'Aaron Fox", "De’Aaron Fox", "DeAaron Fox", "Fox, De'Aaron"]),
    ("robert griffin", ["Robert Griffin III", "Robert Griffin", "Griffin III, Robert", "Griffin, Robert III"]),
    ("luka doncic", ["Luka Dončić", "Luka Doncic", "LUKA DONČIĆ", "Dončić, Luka"]),
    ("shai gilgeousalexander", ["Shai Gilgeous-Alexander", "SHAI GILGEOUS-ALEXANDER", "Gilgeous-Alexander, Shai"]),
    ("pj washington", ["P.J. Washington", "PJ Washington", "P.J. Washington Jr.", "Washington Jr., P.J."]),
    ("jaren jackson", ["Jaren Jackson Jr.", "Jaren Jackson, Jr", "Jaren Jackson Jr", "Jackson Jr., Jaren"]),
    ("dangelo russell", ["D'Angelo Russell", "D’Angelo Russell", "DAngelo Russell"]),
    ("jamarr chase", ["Ja'Marr Chase", "JaMarr Chase", "Chase, Ja'Marr"]),
    ("og anunoby", ["O.G. Anunoby", "OG Anunoby", "Og Anunoby", "Anunoby, O.G."]),
]


class TestPlayerKey:
    def test_docstring_examples(self) -> None:
        assert player_key("Nikola Jokić") == "nikola jokic"
        assert player_key("Michael Porter Jr.") == "michael porter"
        assert player_key("C.J. Stroud") == player_key("CJ Stroud") == "cj stroud"
        assert player_key("St. Brown") == "st brown"
        assert player_key("Amon-Ra") == "amonra"
        assert player_key("O'Neale") == "oneale"

    @pytest.mark.parametrize("expected, spellings", EQUIVALENT_SPELLINGS, ids=[e for e, _ in EQUIVALENT_SPELLINGS])
    def test_equivalent_spellings_share_one_key(self, expected: str, spellings: List[str]) -> None:
        for spelling in spellings:
            assert player_key(spelling) == expected, spelling

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Dennis Schröder", "dennis schroder"),
            ("Kristaps Porziņģis", "kristaps porzingis"),
            ("Ömer Aşık", "omer asik"),            # dotless i
            ("İlkay Gündoğan", "ilkay gundogan"),  # dotted capital I
            ("Łukasz Fabiański", "lukasz fabianski"),
            ("Đorđe Petrović", "dorde petrovic"),
            ("Jusuf Nurkić", "jusuf nurkic"),
            ("Dario Šarić", "dario saric"),
            ("Martin Ødegaard", "martin odegaard"),
            ("José Ramírez", "jose ramirez"),
            ("Ronald Acuña Jr.", "ronald acuna"),
            ("Vít Krejčí", "vit krejci"),
            ("Nenê", "nene"),
            ("Tomáš Hertl", "tomas hertl"),
            ("Teuvo Teräväinen", "teuvo teravainen"),
            ("Weiß", "weiss"),
            ("Þór Æsir", "thor aesir"),
            ("Cœur Œuvre", "coeur oeuvre"),
            ("Guðmundur", "gudmundur"),
        ],
    )
    def test_accents_and_special_letters(self, name: str, expected: str) -> None:
        assert player_key(name) == expected
        assert player_key(name.upper()) == expected
        assert player_key(name.lower()) == expected

    def test_composed_and_decomposed_unicode_agree(self) -> None:
        composed = "Luka Dončić"
        decomposed = "Luka Dončić"
        assert composed != decomposed
        assert player_key(composed) == player_key(decomposed) == "luka doncic"

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Marvin Harrison Jr.", "marvin harrison"),
            ("Marvin Harrison Jr", "marvin harrison"),
            ("Marvin Harrison JR", "marvin harrison"),
            ("Ken Griffey Sr.", "ken griffey"),
            ("Patrick Mahomes II", "patrick mahomes"),
            ("Kenneth Walker III", "kenneth walker"),
            ("Trey Murphy III", "trey murphy"),
            ("Glenn Robinson IV", "glenn robinson"),
            ("Dale Earnhardt V", "dale earnhardt"),
            ("Gary Payton II", "gary payton"),
            ("Robert Griffin III Jr.", "robert griffin"),
        ],
    )
    def test_generational_suffixes_are_removed(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Jr", "jr"),                      # a lone token is never stripped
            ("III", "iii"),
            ("Ivan Rabb", "ivan rabb"),        # 'iv' only as a whole trailing token
            ("Vince Williams Jr.", "vince williams"),
            ("V. Wembanyama", "v wembanyama"), # leading initial that looks like a suffix
            ("II Kim", "ii kim"),
            ("Isaiah Livers", "isaiah livers"),
            ("Jrue Holiday", "jrue holiday"),
            ("Srdjan Sr Petrovic", "srdjan sr petrovic"),
        ],
    )
    def test_suffix_lookalikes_are_kept(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("T.J. Watt", "tj watt"),
            ("TJ Watt", "tj watt"),
            ("A.J. Brown", "aj brown"),
            ("J.J. McCarthy", "jj mccarthy"),
            ("A. St. Brown", "a st brown"),
            ("E. St. Brown", "e st brown"),
            ("St. Brown, A.", "a st brown"),
            ("R.J. Barrett", "rj barrett"),
            ("Fred VanVleet", "fred vanvleet"),
            ("LeBron James*", "lebron james"),
            ("LeBron_James", "lebron james"),
            ("LeBron\tJames", "lebron james"),
            ("LeBron James", "lebron james"),    # non-breaking space
            ("​LeBron James", "lebron james"),   # zero-width space
            ("Karl-Anthony Towns", "karlanthony towns"),
            ("Hyun-Jin Ryu", "hyunjin ryu"),
            ("Ha-Seong Kim", "haseong kim"),
        ],
    )
    def test_punctuation_initials_hyphens_whitespace(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("James, LeBron", "lebron james"),
            ("JAMES, LEBRON", "lebron james"),
            ("James,LeBron", "lebron james"),
            ("  James ,  LeBron  ", "lebron james"),
            ("Wembanyama, Victor", "victor wembanyama"),
            ("Brown, A.J.", "aj brown"),
            ("Antetokounmpo, Giannis", "giannis antetokounmpo"),
            ("Towns, Karl-Anthony", "karlanthony towns"),
            ("Harrison Jr., Marvin", "marvin harrison"),
            ("Harrison, Marvin Jr.", "marvin harrison"),
            ("Marvin Harrison, Jr.", "marvin harrison"),
            ("Marvin Harrison, III", "marvin harrison"),
        ],
    )
    def test_last_comma_first(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name",
        ["", " ", "   \t\n", None, float("nan"), np.nan, np.float64("nan"), pd.NA, pd.NaT, "nan", "NaN", " NAN "],
    )
    def test_missing_values_give_an_empty_key(self, name: object) -> None:
        """None, NaN and the pandas sentinels must not become join keys such as 'na'."""
        assert player_key(name) == ""

    @pytest.mark.parametrize("name", ["...", "---", "'", ",", "?!", "王治郅"])
    def test_names_without_latin_letters_or_digits_give_an_empty_key(self, name: str) -> None:
        assert player_key(name) == ""

    def test_non_string_input_is_stringified(self) -> None:
        assert player_key(23) == "23"
        assert player_key(b"LeBron".decode()) == "lebron"

    def test_a_real_first_name_nan_is_not_mistaken_for_missing(self) -> None:
        assert player_key("Nan Smith") == "nan smith"

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Wembanyama, V.", "v wembanyama"),   # a lone V after a comma is an initial
            ("Williams, V", "v williams"),
            ("Jokic, N.", "n jokic"),
            ("Porter, Michael, Jr.", "michael porter"),
            ("St. Brown, Amon-Ra, Jr.", "amonra st brown"),
            ("John Smith, Sr.", "john smith"),
            ("LeBron James,", "lebron james"),
            (", LeBron James", "lebron james"),
        ],
    )
    def test_last_comma_first_edge_forms(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Amon\u2010Ra St. Brown", "amonra st brown"),   # HYPHEN
            ("Amon\u2011Ra St. Brown", "amonra st brown"),   # NON-BREAKING HYPHEN
            ("Amon\u2013Ra St. Brown", "amonra st brown"),   # EN DASH
            ("Shai Gilgeous\u2014Alexander", "shai gilgeousalexander"),  # EM DASH
            ("De\u2018Aaron Fox", "deaaron fox"),            # LEFT SINGLE QUOTATION MARK
            ("De\u02bcAaron Fox", "deaaron fox"),            # MODIFIER LETTER APOSTROPHE
            ("De`Aaron Fox", "deaaron fox"),
            ("De\u2032Aaron Fox", "deaaron fox"),            # PRIME
        ],
    )
    def test_typographic_hyphens_and_apostrophes(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [("D\u00b4Angelo Russell", "dangelo russell"), ("N\u00b4Keal Harry", "nkeal harry"), ("Jos\u00e9 Ram\u00edrez", "jose ramirez")],
    )
    def test_acute_accent_used_as_an_apostrophe(self, name: str, expected: str) -> None:
        """U+00B4 must be dropped before NFKD turns it into a space plus a combining mark."""
        assert player_key(name) == expected

    @pytest.mark.parametrize(
        "spaced, compact",
        [
            ("C. J. Stroud", "C.J. Stroud"),
            ("T. J. Watt", "TJ Watt"),
            ("A J Brown", "A.J. Brown"),
            ("Brown, A. J.", "AJ Brown"),
            ("P. J. Washington Jr.", "P.J. Washington"),
            ("K. C. Jones", "KC Jones"),
        ],
    )
    def test_spaced_initials_join_with_compact_initials(self, spaced: str, compact: str) -> None:
        assert player_key(spaced) == player_key(compact)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("A. St. Brown", "a st brown"),          # one initial: nothing to merge
            ("J Williams", "j williams"),
            ("Michael A. Jordan", "michael a jordan"),  # a middle initial is not a leading run
            ("A B", "a b"),                          # the last token is never swallowed
        ],
    )
    def test_initial_merging_leaves_other_names_alone(self, name: str, expected: str) -> None:
        assert player_key(name) == expected

    def test_key_is_canonical(self) -> None:
        """Lower-case ASCII letters/digits separated by single spaces, and idempotent."""
        corpus = [s for _, spellings in EQUIVALENT_SPELLINGS for s in spellings]
        corpus += ["Kristaps Porziņģis", "  Ömer   Aşık ", "T.J. Watt", "LeBron James #23", "Luguentz Dort (LAC)",
                   "Nenê", "St. Brown, A.", "O'Neale", "X Æ A-12", "Ja Morant\n"]
        pattern = re.compile(r"^[a-z0-9]+( [a-z0-9]+)*$")
        for name in corpus:
            key = player_key(name)
            assert pattern.match(key), (name, key)
            assert player_key(key) == key
            assert player_key(name.upper()) == key
            assert player_key(f"  {name}  ") == key

    @pytest.mark.parametrize(
        "first, second",
        [
            ("Jalen Williams", "Jaylin Williams"),
            ("Jaden McDaniels", "Jalen McDaniels"),
            ("Keon Johnson", "Keldon Johnson"),
            ("Nikola Jokić", "Nikola Jović"),
            ("Bogdan Bogdanović", "Bojan Bogdanović"),
            ("Marcus Morris Sr.", "Markieff Morris"),
            ("Seth Curry", "Stephen Curry"),
            ("T.J. Watt", "J.J. Watt"),
            ("Amon-Ra St. Brown", "Equanimeous St. Brown"),
            ("A.J. Brown", "A.J. Green"),
        ],
    )
    def test_different_players_keep_different_keys(self, first: str, second: str) -> None:
        assert player_key(first) != player_key(second)


# --------------------------------------------------------------------------
# NameMatcher
# --------------------------------------------------------------------------
ROSTER: List[Tuple[str, Optional[str]]] = [
    ("Nikola Jokić", "DEN"),
    ("Michael Porter Jr.", "DEN"),
    ("Amon-Ra St. Brown", "DET"),
    ("Equanimeous St. Brown", "CHI"),
    ("Nicolas Claxton", "BKN"),
    ("Jalen Williams", "OKC"),
    ("Jaylin Williams", "OKC"),
    ("Shai Gilgeous-Alexander", "OKC"),
    ("Donovan Mitchell", "CLE"),
    ("Davion Mitchell", "MIA"),
    ("Giannis Antetokounmpo", "MIL"),
    ("Thanasis Antetokounmpo", "MIL"),
    ("C.J. Stroud", "HOU"),
    ("Nah'Shon Hyland", "LAC"),
    ("Trae Young", "ATL"),
    ("Thaddeus Young", "TOR"),
    ("LeBron James", "LAL"),
    ("Nenê", "HOU"),
]
ALIASES: Dict[str, str] = {
    "Bones Hyland": "Nah'Shon Hyland",
    "The Joker": "NIKOLA JOKIC",
    "Greek Freak": "Giannis Antetokounmpo",
    "Ghost": "Nobody Here",
}


@pytest.fixture
def matcher() -> NameMatcher:
    return NameMatcher([n for n, _ in ROSTER], [t for _, t in ROSTER], aliases=ALIASES)


class TestMatcherExact:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("Nikola Jokić", "nikola jokic"),
            ("Nikola Jokic", "nikola jokic"),
            ("NIKOLA JOKIC", "nikola jokic"),
            ("Jokić, Nikola", "nikola jokic"),
            ("Michael Porter", "michael porter"),
            ("Michael Porter Jr", "michael porter"),
            ("CJ Stroud", "cj stroud"),
            ("Amon-Ra St Brown", "amonra st brown"),
            ("Shai Gilgeous-Alexander", "shai gilgeousalexander"),
            ("Nene", "nene"),
            ("NahShon Hyland", "nahshon hyland"),
        ],
    )
    def test_exact_key_match(self, matcher: NameMatcher, query: str, expected: str) -> None:
        assert matcher.match(query) == expected

    def test_every_roster_name_matches_itself(self, matcher: NameMatcher) -> None:
        for name, team in ROSTER:
            assert matcher.match(name) == player_key(name)
            assert matcher.match(name, team) == player_key(name)
            assert matcher.match(name.upper()) == player_key(name)

    def test_team_hint_vetoes_even_an_exact_match(self, matcher: NameMatcher) -> None:
        """Documented: a candidate whose known teams exclude the hint is never returned."""
        assert matcher.match("Trae Young", "ATL") == "trae young"
        assert matcher.match("Trae Young", "TOR") is None
        assert matcher.match("Nikola Jokić", "MIA") is None

    def test_team_hint_cannot_veto_a_player_without_a_known_team(self) -> None:
        m = NameMatcher(["Trae Young", "LeBron James"], [None, "LAL"])
        assert m.match("Trae Young", "TOR") == "trae young"
        assert m.match("T. Young", "TOR") == "trae young"
        assert m.match("LeBron James", "TOR") is None

    def test_contains(self, matcher: NameMatcher) -> None:
        assert "Nikola Jokic" in matcher
        assert "JOKIĆ, NIKOLA" in matcher
        assert "Jokic" not in matcher
        assert "Bones Hyland" not in matcher  # aliases are not roster members
        assert "" not in matcher

    def test_display_name_returns_the_registered_spelling(self, matcher: NameMatcher) -> None:
        assert matcher.display_name("nikola jokic") == "Nikola Jokić"
        assert matcher.display_name("michael porter") == "Michael Porter Jr."
        assert matcher.display_name("nobody") is None

    def test_result_is_always_a_roster_key_or_none(self, matcher: NameMatcher) -> None:
        roster_keys = {player_key(n) for n, _ in ROSTER}
        queries = ["J. Williams", "T. Young", "Zzzz Qqqq", "Nikola", "A. St. Brown", "Greek Freak", "LeBron Jmes",
                   "Ghost", "", "K. Durant", "Giannis", "N Jokic", "Bones Hyland", "D. Mitchell"]
        for query in queries:
            for team in (None, "OKC", "CLE", "XXX"):
                result = matcher.match(query, team)
                assert result is None or result in roster_keys


class TestMatcherAlias:
    def test_alias_resolves_to_the_roster_key(self, matcher: NameMatcher) -> None:
        assert matcher.match("Bones Hyland") == "nahshon hyland"
        assert matcher.match("bones hyland") == "nahshon hyland"
        assert matcher.match("BONES HYLAND Jr.") == "nahshon hyland"
        assert matcher.match("The Joker") == "nikola jokic"
        assert matcher.match("Greek Freak") == "giannis antetokounmpo"

    def test_alias_respects_the_team_hint(self, matcher: NameMatcher) -> None:
        assert matcher.match("Bones Hyland", "LAC") == "nahshon hyland"
        assert matcher.match("Bones Hyland", "lac") == "nahshon hyland"
        assert matcher.match("Bones Hyland", "DEN") is None

    def test_alias_to_an_unknown_player_is_ignored(self, matcher: NameMatcher) -> None:
        assert matcher.match("Ghost") is None

    def test_exact_match_beats_an_alias(self) -> None:
        m = NameMatcher(["Jalen Williams", "Jaylin Williams"], aliases={"Jalen Williams": "Jaylin Williams"})
        assert m.match("Jalen Williams") == "jalen williams"

    def test_alias_beats_looser_rules(self) -> None:
        """'J. Williams' is ambiguous by initial, an explicit alias settles it."""
        m = NameMatcher(["Jalen Williams", "Jaylin Williams"], aliases={"J. Williams": "Jaylin Williams"})
        assert m.match("J. Williams") == "jaylin williams"
        assert m.match("J Williams") == "jaylin williams"

    def test_alias_whose_target_is_missing_falls_through(self) -> None:
        m = NameMatcher(["Nicolas Claxton"], aliases={"Nic Claxton": "Somebody Else"})
        assert m.match("Nic Claxton") == "nicolas claxton"


class TestMatcherInitialAndPrefix:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("A. St. Brown", "amonra st brown"),
            ("E. St. Brown", "equanimeous st brown"),
            ("St. Brown, A.", "amonra st brown"),
            ("N. Claxton", "nicolas claxton"),
            ("N Jokic", "nikola jokic"),
            ("G. Antetokounmpo", "giannis antetokounmpo"),
            ("S. Gilgeous-Alexander", "shai gilgeousalexander"),
            ("L. James", "lebron james"),
            ("M. Porter Jr.", "michael porter"),
            ("Antetokounmpo, T.", "thanasis antetokounmpo"),  # G. and T. differ
        ],
    )
    def test_unique_initial(self, matcher: NameMatcher, query: str, expected: str) -> None:
        assert matcher.match(query) == expected

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("Nic Claxton", "nicolas claxton"),
            ("Nico Claxton", "nicolas claxton"),
            ("Don Mitchell", "donovan mitchell"),
            ("Dav Mitchell", "davion mitchell"),
            ("Mike Porter", None),         # 'mike' is not a prefix of 'michael'
            ("Thad Young", "thaddeus young"),
            ("Tr. Young", "trae young"),
        ],
    )
    def test_first_name_prefix(self, matcher: NameMatcher, query: str, expected: Optional[str]) -> None:
        assert matcher.match(query) == expected

    def test_prefix_works_in_both_directions(self) -> None:
        """The roster may hold the short form and the feed the long one."""
        m = NameMatcher(["Nic Claxton", "Cam Thomas"])
        assert m.match("Nicolas Claxton") == "nic claxton"
        assert m.match("Cameron Thomas") == "cam thomas"

    @pytest.mark.parametrize("query", ["J. Williams", "J Williams", "Ja Williams", "Williams, J.", "T. Young", "D. Mitchell"])
    def test_ambiguous_initial_returns_none(self, matcher: NameMatcher, query: str) -> None:
        assert matcher.match(query) is None

    def test_incompatible_first_name_does_not_match_on_surname_alone(self, matcher: NameMatcher) -> None:
        for query in ("K. Jokic", "Bronny James", "Seth Mitchell", "X. Claxton", "Kostas Antetokounmpo"):
            assert matcher.match(query) is None

    def test_surname_or_first_name_alone_does_not_match(self, matcher: NameMatcher) -> None:
        for query in ("Jokic", "James", "Antetokounmpo", "Nikola", "LeBron", "Williams", "SGA"):
            assert matcher.match(query) is None


class TestMatcherTeamHint:
    def test_team_hint_resolves_an_ambiguous_initial(self, matcher: NameMatcher) -> None:
        assert matcher.match("D. Mitchell") is None
        assert matcher.match("D. Mitchell", "CLE") == "donovan mitchell"
        assert matcher.match("D. Mitchell", "MIA") == "davion mitchell"
        assert matcher.match("T. Young", "ATL") == "trae young"
        assert matcher.match("T. Young", "TOR") == "thaddeus young"

    def test_team_hint_is_case_insensitive(self, matcher: NameMatcher) -> None:
        assert matcher.match("D. Mitchell", "cle") == "donovan mitchell"
        assert NameMatcher(["Donovan Mitchell", "Davion Mitchell"], ["cle", "mia"]).match("D. Mitchell", "CLE") == (
            "donovan mitchell"
        )

    def test_team_hint_cannot_split_team_mates(self, matcher: NameMatcher) -> None:
        """Jalen and Jaylin Williams both play for OKC: still ambiguous."""
        assert matcher.match("J. Williams", "OKC") is None

    def test_unknown_team_leaves_the_ambiguity(self, matcher: NameMatcher) -> None:
        assert matcher.match("D. Mitchell", "XXX") is None
        assert matcher.match("J. Williams", "NYK") is None

    @pytest.mark.parametrize("team", [None, "", "  ", float("nan"), np.nan, pd.NA, pd.NaT, "nan", "None"])
    def test_missing_team_hint_is_ignored(self, matcher: NameMatcher, team: object) -> None:
        """A missing value from a DataFrame must mean 'no hint', not 'team NAN'."""
        assert matcher.match("Nikola Jokić", team) == "nikola jokic"  # type: ignore[arg-type]
        assert matcher.match("N. Claxton", team) == "nicolas claxton"  # type: ignore[arg-type]
        assert matcher.match("D. Mitchell", team) is None  # type: ignore[arg-type]

    def test_team_hint_whitespace_and_case_are_ignored(self, matcher: NameMatcher) -> None:
        assert matcher.match("Nikola Jokić", " den ") == "nikola jokic"
        assert NameMatcher(["Trae Young"], [" atl "]).match("Trae Young", "ATL") == "trae young"
        assert NameMatcher(["Trae Young"], [" atl "]).match("Trae Young", "TOR") is None

    def test_a_player_can_be_registered_with_several_teams(self) -> None:
        m = NameMatcher(
            ["James Harden", "James Harden", "Jaden Hardy", "Jalen Harden"],
            ["LAC", "PHI", "DAL", "DAL"],
        )
        assert m.match("J. Harden", "PHI") == "james harden"
        assert m.match("J. Harden", "LAC") == "james harden"
        assert m.match("J. Harden", "DAL") == "jalen harden"
        assert m.match("J. Harden") is None
        assert m.display_name("james harden") == "James Harden"

    def test_missing_roster_teams_are_tolerated(self) -> None:
        m = NameMatcher(["Trae Young", "Thaddeus Young"], [None, float("nan")])  # type: ignore[list-item]
        assert m.match("T. Young", "ATL") is None
        assert m.match("Trae Young", "ATL") == "trae young"

    def test_team_hint_vetoes_prefix_candidates(self, matcher: NameMatcher) -> None:
        assert matcher.match("Nic Claxton", "BKN") == "nicolas claxton"
        assert matcher.match("Nic Claxton", "NYK") is None
        assert matcher.match("A. St. Brown", "DET") == "amonra st brown"
        assert matcher.match("A. St. Brown", "CHI") is None

    @pytest.mark.parametrize(
        "roster_name, roster_team, query, team",
        [
            ("Davion Mitchell", "MIA", "D. Mitchell", "CLE"),
            ("Jalen McDaniels", "CHA", "J. McDaniels", "MIN"),
            ("Thaddeus Young", "TOR", "T. Young", "ATL"),
        ],
    )
    def test_contradicting_team_hint_blocks_an_initial_match(
        self, roster_name: str, roster_team: str, query: str, team: str
    ) -> None:
        """Only Davion Mitchell (MIA) is known: 'D. Mitchell' of CLE must not become him."""
        m = NameMatcher([roster_name], [roster_team])
        assert m.match(query, team) is None
        assert m.match(query, roster_team) == player_key(roster_name)
        assert m.match(query) == player_key(roster_name)


class TestMatcherDoesNotGuess:
    """Documented: nothing fuzzier than initial/prefix compatibility is attempted."""

    @pytest.mark.parametrize(
        "query",
        [
            "Giannis Antetokoumpo",     # surname typo
            "Gianis Antetokounmpo",     # first-name typo
            "Nikola Jokicc",
            "Nicola Jokic",
            "Lebron Jamess",
            "Donavan Mitchell",
            "Nikola Jović",             # a different NBA player
        ],
    )
    def test_misspelled_or_look_alike_names_return_none(self, matcher: NameMatcher, query: str) -> None:
        assert matcher.match(query) is None

    @pytest.mark.parametrize("query", ["Zzzz Qqqq", "Kevin Durant", "Stephen Curry", "Luka Doncic", "Jayson Tatum"])
    def test_unrelated_names_return_none(self, matcher: NameMatcher, query: str) -> None:
        assert matcher.match(query) is None

    def test_an_alias_is_the_way_to_bridge_a_spelling_difference(self) -> None:
        m = NameMatcher(["Giannis Antetokounmpo"], aliases={"Giannis Antetokoumpo": "Giannis Antetokounmpo"})
        assert m.match("Giannis Antetokoumpo") == "giannis antetokounmpo"
        assert m.match("GIANNIS ANTETOKOUMPO") == "giannis antetokounmpo"

    def test_mononyms_are_only_matched_exactly(self, matcher: NameMatcher) -> None:
        assert matcher.match("Nenê") == "nene"
        assert matcher.match("Nenee") is None
        assert matcher.match("Nen") is None

    def test_spaced_initials_resolve_through_the_key(self, matcher: NameMatcher) -> None:
        assert matcher.match("C. J. Stroud") == "cj stroud"
        assert matcher.match("C J Stroud", "HOU") == "cj stroud"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (debatable): a hyphen written as a space is a routine feed difference, but "
            "player_key drops hyphens without a space ('gilgeousalexander') and keeps the "
            "space variant as two tokens ('gilgeous alexander'), and the matcher no longer "
            "has any step that bridges them: 'Shai Gilgeous Alexander', 'Karl Anthony Towns' "
            "and 'Amon Ra St. Brown' all come back None unless an alias is supplied for each"
        ),
    )
    @pytest.mark.parametrize(
        "roster_name, query",
        [
            ("Shai Gilgeous-Alexander", "Shai Gilgeous Alexander"),
            ("Karl-Anthony Towns", "Karl Anthony Towns"),
            ("Amon-Ra St. Brown", "Amon Ra St. Brown"),
            ("Kentavious Caldwell-Pope", "Kentavious Caldwell Pope"),
        ],
    )
    def test_hyphen_written_as_a_space_still_matches(self, roster_name: str, query: str) -> None:
        assert NameMatcher([roster_name]).match(query) == player_key(roster_name)


class TestMatcherConstruction:
    def test_empty_roster(self) -> None:
        m = NameMatcher([])
        assert m.match("LeBron James") is None
        assert "LeBron James" not in m

    @pytest.mark.parametrize("query", ["", "   ", None, float("nan"), "nan", "..."])
    def test_missing_query_returns_none(self, matcher: NameMatcher, query: object) -> None:
        assert matcher.match(query) is None  # type: ignore[arg-type]

    def test_blank_roster_entries_are_skipped(self) -> None:
        m = NameMatcher(["", None, float("nan"), "LeBron James"], ["A", "B", "C", "LAL"])  # type: ignore[list-item]
        assert m.match("LeBron James") == "lebron james"
        assert "" not in m
        assert m.match("L. James", "LAL") == "lebron james"

    def test_duplicate_spellings_collapse_and_keep_the_first(self) -> None:
        m = NameMatcher(["LeBron James", "Lebron James", "LEBRON JAMES Jr."], ["LAL", "CLE", None])
        assert m.display_name("lebron james") == "LeBron James"
        assert m.match("L. James", "CLE") == "lebron james"
        assert m.match("L. James", "LAL") == "lebron james"

    def test_names_and_teams_must_align(self) -> None:
        with pytest.raises(ValueError):
            NameMatcher(["A B", "C D"], ["X"])
        with pytest.raises(ValueError):
            NameMatcher(["A B"], ["X", "Y"])

    def test_accepts_any_iterable(self) -> None:
        m = NameMatcher((n for n in ["Trae Young", "LeBron James"]), iter(["ATL", "LAL"]))
        assert m.match("T. Young", "ATL") == "trae young"
        assert NameMatcher(pd.Series(["Trae Young"]), pd.Series(["ATL"])).match("Trae Young") == "trae young"


# --------------------------------------------------------------------------
# Adversarial pairs: different real players that look alike
# --------------------------------------------------------------------------
#: Different players sharing a surname (brothers, namesakes).
SAME_SURNAME_PAIRS: List[Tuple[str, str]] = [
    ("Jalen Williams", "Jaylin Williams"),
    ("Marcus Morris Sr.", "Markieff Morris"),
    ("Caleb Martin", "Cody Martin"),
    ("Tre Jones", "Tyus Jones"),
    ("Amen Thompson", "Ausar Thompson"),
    ("Franz Wagner", "Moritz Wagner"),
    ("Brook Lopez", "Robin Lopez"),
    ("Seth Curry", "Stephen Curry"),
    ("Giannis Antetokounmpo", "Thanasis Antetokounmpo"),
    ("Jrue Holiday", "Justin Holiday"),
    ("Jrue Holiday", "Aaron Holiday"),
    ("Keegan Murray", "Kris Murray"),
    ("Mikal Bridges", "Miles Bridges"),
    ("Lonzo Ball", "LaMelo Ball"),
    ("Evan Mobley", "Isaiah Mobley"),
    ("Kevin Porter Jr.", "Michael Porter Jr."),
    ("Jabari Smith Jr.", "Jalen Smith"),
    ("Jerami Grant", "Jerian Grant"),
    ("Mason Plumlee", "Miles Plumlee"),
    ("Travis Kelce", "Jason Kelce"),
    ("T.J. Watt", "J.J. Watt"),
    ("Trevon Diggs", "Stefon Diggs"),
    ("Shaquill Griffin", "Shaquem Griffin"),
    ("Nick Bosa", "Joey Bosa"),
    ("William Contreras", "Willson Contreras"),
    ("Aaron Nola", "Austin Nola"),
    ("Yandy Díaz", "Yainer Díaz"),
    ("Brady Tkachuk", "Matthew Tkachuk"),
    ("Jack Hughes", "Quinn Hughes"),
]

#: Different players whose full names are >= 0.9 similar as strings.
LOOK_ALIKE_PAIRS: List[Tuple[str, str]] = [
    ("Jaden McDaniels", "Jalen McDaniels"),      # brothers, both NBA
    ("Keon Johnson", "Keldon Johnson"),          # both NBA
    ("Bogdan Bogdanović", "Bojan Bogdanović"),   # both NBA
    ("Diontae Johnson", "Dontae Johnson"),       # NFL receiver / NFL cornerback
    ("Nikola Jović", "Nikola Jokić"),            # Miami forward / Denver centre
    ("Darius Slay", "Darius Slayton"),           # NFL cornerback / NFL receiver
    ("Jalen Hurts", "Jalen Hurd"),
]

#: Same first name and distinct surnames that differ by a few letters.
SIMILAR_SURNAME_PAIRS: List[Tuple[str, str]] = [
    ("Avery Williams", "Avery Williamson"),      # NFL returner / NFL linebacker
    ("Michael Harris II", "Michael Harrison"),
    ("Jose Hernandez", "Jose Fernandez"),
    ("Cam Johnson", "Cam Johnston"),             # NBA forward / NFL punter
]

_NAMESAKES = [pair for pair in SAME_SURNAME_PAIRS + LOOK_ALIKE_PAIRS if player_key(pair[0]).split()[-1] == player_key(pair[1]).split()[-1]]
#: Namesakes whose first names start with the same letter / with different letters.
SHARED_INITIAL_PAIRS = [(a, b) for a, b in _NAMESAKES if a[0] == b[0]]
DISTINCT_INITIAL_PAIRS = [(a, b) for a, b in _NAMESAKES if a[0] != b[0]]


class TestAdversarialPairs:
    @pytest.mark.parametrize("first, second", SAME_SURNAME_PAIRS + LOOK_ALIKE_PAIRS + SIMILAR_SURNAME_PAIRS)
    def test_both_on_the_roster_each_resolves_to_himself(self, first: str, second: str) -> None:
        m = NameMatcher([first, second])
        assert m.match(first) == player_key(first)
        assert m.match(second) == player_key(second)
        assert m.match(first.upper()) == player_key(first)
        assert player_key(first) != player_key(second)

    @pytest.mark.parametrize("first, second", SHARED_INITIAL_PAIRS)
    def test_shared_initial_is_ambiguous_when_both_are_on_the_roster(self, first: str, second: str) -> None:
        surname = player_key(first).split(" ", 1)[1]
        m = NameMatcher([first, second])
        assert m.match(f"{first[0]}. {surname}") is None
        assert m.match(f"{surname}, {first[0]}.") is None

    @pytest.mark.parametrize("first, second", DISTINCT_INITIAL_PAIRS)
    def test_distinct_initials_identify_each_namesake(self, first: str, second: str) -> None:
        surname = player_key(first).split(" ", 1)[1]
        m = NameMatcher([first, second])
        assert m.match(f"{first[0]}. {surname}") == player_key(first)
        assert m.match(f"{second[0]}. {surname}") == player_key(second)

    @pytest.mark.parametrize("first, second", SAME_SURNAME_PAIRS + LOOK_ALIKE_PAIRS + SIMILAR_SURNAME_PAIRS)
    def test_missing_player_is_not_matched_to_his_look_alike(self, first: str, second: str) -> None:
        """Only one of the two is on the roster: the other must come back as None."""
        assert NameMatcher([second]).match(first) is None
        assert NameMatcher([first]).match(second) is None
        assert NameMatcher([second], ["AAA"]).match(first, "AAA") is None
        assert NameMatcher([second], ["AAA"]).match(first, "BBB") is None

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (debatable, follows from documented rule 3): the prefix rule treats a "
            "first name that is a prefix of another as the same person, so a player missing "
            "from the roster resolves to a different real player when no team hint is "
            "given: with only Tyler Johnson (NFL receiver) known, 'Ty Johnson' (NFL running "
            "back) matches him; likewise Tre Brown / Trent Brown. A contradicting team hint "
            "does prevent it"
        ),
    )
    @pytest.mark.parametrize("first, second", [("Ty Johnson", "Tyler Johnson"), ("Tre Brown", "Trent Brown")])
    def test_prefix_rule_does_not_merge_different_players(self, first: str, second: str) -> None:
        assert NameMatcher([second]).match(first) is None
        assert NameMatcher([first]).match(second) is None

    @pytest.mark.parametrize("first, second", [("Ty Johnson", "Tyler Johnson"), ("Tre Brown", "Trent Brown")])
    def test_prefix_namesakes_are_safe_with_both_known_or_a_team_hint(self, first: str, second: str) -> None:
        both = NameMatcher([first, second])
        assert both.match(first) == player_key(first)
        assert both.match(second) == player_key(second)
        assert NameMatcher([second], ["LAR"]).match(first, "BUF") is None
