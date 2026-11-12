"""Independent verification of :mod:`evmodel.schema`.

The contracts are taken from the module docstring and the column comments:
canonical column order, dtypes, defaults, which rows are dropped and how
timestamps are normalised.  Dtype assertions are written so that they hold
for both the object-string (pandas 2) and the ``str`` (pandas 3) backends.
Fixed UTC offsets are used instead of named zones so the tests do not depend
on a tz database being installed.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from evmodel.schema import (
    GAMELOG_COLUMNS,
    OPPORTUNITY_COLUMNS,
    QUOTE_COLUMNS,
    empty_quotes,
    normalize_gamelogs,
    normalize_quotes,
    prop_id,
)

EST = dt.timezone(dt.timedelta(hours=-5))
QUOTE_TEXT_COLUMNS = ("sport", "event_id", "home_team", "away_team", "book", "market", "player", "player_key")
GAMELOG_TEXT_COLUMNS = ("sport", "game_id", "player_id", "player", "player_key", "team", "opponent")


def _quotes(n: int = 1, **columns: object) -> pd.DataFrame:
    """A valid quotes frame of ``n`` rows; keyword arguments replace or add columns."""
    data: Dict[str, object] = {
        "book": ["draftkings"] * n,
        "market": ["player_points"] * n,
        "player": ["LeBron James"] * n,
        "line": [24.5] * n,
        "over_price": [1.91] * n,
        "under_price": [1.91] * n,
    }
    data.update(columns)
    return pd.DataFrame(data)


def _gamelogs(n: int = 1, **columns: object) -> pd.DataFrame:
    """A valid game-log frame of ``n`` rows; keyword arguments replace or add columns."""
    data: Dict[str, object] = {
        "sport": ["nba"] * n,
        "game_date": [f"2025-01-{day:02d}" for day in range(1, n + 1)],
        "player": ["LeBron James"] * n,
    }
    data.update(columns)
    return pd.DataFrame(data)


def _is_text(series: pd.Series) -> bool:
    return bool(series.map(lambda v: isinstance(v, str)).all()) and (
        series.dtype == object or pd.api.types.is_string_dtype(series)
    )


def _assert_quote_dtypes(df: pd.DataFrame) -> None:
    for col in QUOTE_TEXT_COLUMNS:
        assert _is_text(df[col]), col
    for col in ("line", "over_price", "under_price"):
        assert df[col].dtype == np.float64, col
    for col in ("commence_time", "last_update"):
        assert isinstance(df[col].dtype, pd.DatetimeTZDtype), col
        assert str(df[col].dt.tz) == "UTC", col
    assert df["is_alternate"].dtype == bool


def _assert_gamelog_dtypes(df: pd.DataFrame) -> None:
    for col in GAMELOG_TEXT_COLUMNS:
        assert _is_text(df[col]), col
    assert pd.api.types.is_datetime64_dtype(df["game_date"])  # tz-naive
    assert df["is_home"].dtype == bool
    assert df["minutes"].dtype == np.float64
    assert pd.api.types.is_numeric_dtype(df["season"])


# --------------------------------------------------------------------------
# Column contracts
# --------------------------------------------------------------------------
class TestContracts:
    def test_quote_columns(self) -> None:
        assert QUOTE_COLUMNS == [
            "sport", "event_id", "commence_time", "home_team", "away_team", "book", "market", "player",
            "player_key", "line", "over_price", "under_price", "last_update", "is_alternate",
        ]

    def test_gamelog_columns(self) -> None:
        assert GAMELOG_COLUMNS == [
            "sport", "season", "game_id", "game_date", "player_id", "player", "player_key", "team",
            "opponent", "is_home", "minutes",
        ]

    def test_opportunity_columns_cover_what_a_bettor_needs(self) -> None:
        assert len(OPPORTUNITY_COLUMNS) == len(set(OPPORTUNITY_COLUMNS))
        for col in ("sport", "event_id", "player", "player_key", "market", "line", "side", "book", "price",
                    "fair_prob", "push_prob", "ev", "edge", "kelly", "stake_units", "flags"):
            assert col in OPPORTUNITY_COLUMNS

    def test_no_duplicate_columns(self) -> None:
        assert len(QUOTE_COLUMNS) == len(set(QUOTE_COLUMNS))
        assert len(GAMELOG_COLUMNS) == len(set(GAMELOG_COLUMNS))

    def test_prop_id(self) -> None:
        assert prop_id("e1", "nikola jokic", "player_points") == "e1|nikola jokic|player_points"
        assert prop_id(42, "a b", "m") == "42|a b|m"
        assert prop_id("e1", "a b", "m") != prop_id("e1", "a b", "n")


# --------------------------------------------------------------------------
# normalize_quotes
# --------------------------------------------------------------------------
class TestQuotesShape:
    def test_empty_quotes_is_correctly_typed(self) -> None:
        df = empty_quotes()
        assert list(df.columns) == QUOTE_COLUMNS
        assert len(df) == 0
        _assert_quote_dtypes(df)

    def test_empty_input_with_only_required_columns(self) -> None:
        df = normalize_quotes(pd.DataFrame(columns=["book", "market", "player", "line"]))
        assert list(df.columns) == QUOTE_COLUMNS
        assert len(df) == 0
        _assert_quote_dtypes(df)

    @pytest.mark.parametrize("missing", ["book", "market", "player", "line"])
    def test_missing_required_column_raises(self, missing: str) -> None:
        with pytest.raises(ValueError, match=missing):
            normalize_quotes(_quotes().drop(columns=[missing]))

    def test_no_columns_at_all_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_quotes(pd.DataFrame())

    def test_minimal_frame_gets_defaults(self) -> None:
        df = normalize_quotes(
            pd.DataFrame({"book": ["dk"], "market": ["m"], "player": ["A B"], "line": [1.5], "over_price": [1.9]})
        )
        assert list(df.columns) == QUOTE_COLUMNS
        _assert_quote_dtypes(df)
        row = df.iloc[0]
        assert (row["sport"], row["event_id"], row["home_team"], row["away_team"]) == ("", "", "", "")
        assert pd.isna(row["commence_time"]) and pd.isna(row["last_update"])
        assert row["over_price"] == 1.9 and np.isnan(row["under_price"])
        assert row["is_alternate"] is np.False_ or row["is_alternate"] is False
        assert row["player_key"] == "a b"

    def test_canonical_columns_come_first_and_extras_are_kept(self) -> None:
        raw = _quotes(2, zzz=[1, 2], aaa=["x", "y"])
        raw = raw[["aaa", "under_price", "player", "zzz", "line", "market", "over_price", "book"]]
        df = normalize_quotes(raw)
        assert list(df.columns) == QUOTE_COLUMNS + ["aaa", "zzz"]
        assert df["zzz"].tolist() == [1, 2]
        assert df["aaa"].tolist() == ["x", "y"]

    def test_returns_a_new_frame_with_a_fresh_index(self) -> None:
        raw = _quotes(4, player=["A B", "", "C D", "E F"])
        raw.index = pd.Index([10, 11, 12, 13])
        before = raw.copy(deep=True)
        df = normalize_quotes(raw)
        pd.testing.assert_frame_equal(raw, before)  # input untouched
        assert df.index.tolist() == [0, 1, 2]
        assert df["player"].tolist() == ["A B", "C D", "E F"]  # original order kept

    def test_duplicate_input_index_is_handled(self) -> None:
        raw = _quotes(3, player=["A B", "C D", "E F"])
        raw.index = pd.Index([7, 7, 7])
        assert normalize_quotes(raw)["player"].tolist() == ["A B", "C D", "E F"]

    def test_is_idempotent(self) -> None:
        raw = _quotes(
            3,
            sport=["NBA", "nba", None],
            player=["Nikola Jokić", " LeBron James ", "Michael Porter Jr."],
            line=["24.5", 25, 7.5],
            over_price=[1.91, "2.05", 1.0],
            commence_time=["2025-01-01T00:00:00Z"] * 3,
            is_alternate=[True, None, False],
            note=["a", "b", "c"],
        )
        once = normalize_quotes(raw)
        pd.testing.assert_frame_equal(normalize_quotes(once), once)


class TestQuotesText:
    def test_keys_are_lower_cased_and_everything_is_stripped(self) -> None:
        df = normalize_quotes(
            _quotes(
                sport=[" NBA "],
                book=["  DraftKings "],
                market=[" Player_Points"],
                player=["  Nikola Jokić  "],
                home_team=[" Denver Nuggets "],
                away_team=["Los Angeles Lakers  "],
                event_id=[" abc123 "],
            )
        )
        row = df.iloc[0]
        assert row["sport"] == "nba"
        assert row["book"] == "draftkings"
        assert row["market"] == "player_points"
        assert row["player"] == "Nikola Jokić"          # display name: stripped, case and accents kept
        assert row["home_team"] == "Denver Nuggets"
        assert row["away_team"] == "Los Angeles Lakers"
        assert row["event_id"] == "abc123"
        assert row["player_key"] == "nikola jokic"

    def test_missing_text_becomes_empty_string(self) -> None:
        df = normalize_quotes(_quotes(sport=[None], event_id=[np.nan], home_team=[None], away_team=[np.nan]))
        row = df.iloc[0]
        assert (row["sport"], row["event_id"], row["home_team"], row["away_team"]) == ("", "", "", "")
        _assert_quote_dtypes(df)

    def test_numeric_ids_become_strings(self) -> None:
        df = normalize_quotes(_quotes(2, event_id=[12345, 67890]))
        assert df["event_id"].tolist() == ["12345", "67890"]

    def test_categorical_and_nullable_string_inputs(self) -> None:
        raw = pd.DataFrame(
            {
                "book": pd.Categorical(["DK", "FD"]),
                "market": pd.array(["Player_Points", "player_points"], dtype="string"),
                "player": pd.Categorical(["A B", "C D"]),
                "line": pd.array([1.5, 2.5], dtype="Float64"),
                "over_price": pd.array([1.9, pd.NA], dtype="Float64"),
                "under_price": [np.nan, 1.8],
                "is_alternate": pd.array([True, pd.NA], dtype="boolean"),
            }
        )
        df = normalize_quotes(raw)
        _assert_quote_dtypes(df)
        assert df["book"].tolist() == ["dk", "fd"]
        assert df["market"].tolist() == ["player_points", "player_points"]
        assert df["is_alternate"].tolist() == [True, False]
        assert np.isnan(df["over_price"].iloc[1])

    def test_nullable_string_missing_values_drop_the_row(self) -> None:
        raw = _quotes(2, book=pd.array(["DK", pd.NA], dtype="string"), player=pd.array(["A B", "C D"], dtype="string"))
        df = normalize_quotes(raw)
        assert df["player"].tolist() == ["A B"]


class TestQuotesPlayerKey:
    def test_is_computed_when_the_column_is_absent(self) -> None:
        df = normalize_quotes(_quotes(3, player=["Nikola Jokić", "Michael Porter Jr.", "C.J. Stroud"]))
        assert df["player_key"].tolist() == ["nikola jokic", "michael porter", "cj stroud"]

    def test_blank_entries_are_filled_and_existing_ones_kept(self) -> None:
        raw = _quotes(
            4,
            player=["A B", "Č D Jr.", "E F", "G H"],
            player_key=["custom key", None, "", np.nan],
        )
        df = normalize_quotes(raw)
        assert df["player_key"].tolist() == ["custom key", "c d", "e f", "g h"]
        assert df["player"].tolist() == ["A B", "Č D Jr.", "E F", "G H"]

    def test_all_missing_key_column_is_filled(self) -> None:
        df = normalize_quotes(_quotes(2, player=["A B", "C D"], player_key=[np.nan, np.nan]))
        assert df["player_key"].tolist() == ["a b", "c d"]

    def test_row_without_any_player_identity_is_dropped(self) -> None:
        raw = _quotes(4, player=["A B", "", None, "   "], player_key=[None, None, "", np.nan])
        df = normalize_quotes(raw)
        assert df["player"].tolist() == ["A B"]

    def test_unusable_names_are_dropped(self) -> None:
        """Names with no letters produce an empty key and cannot be joined."""
        df = normalize_quotes(_quotes(3, player=["A B", "???", "nan"]))
        assert df["player"].tolist() == ["A B"]

    def test_supplied_keys_are_stripped_and_whitespace_only_counts_as_blank(self) -> None:
        df = normalize_quotes(_quotes(2, player=["A B", "C D"], player_key=["  ", " custom "]))
        assert df["player_key"].tolist() == ["a b", "custom"]

    def test_nullable_string_key_column(self) -> None:
        raw = _quotes(2, player=["A B", "C D"], player_key=pd.array([pd.NA, "x"], dtype="string"))
        assert normalize_quotes(raw)["player_key"].tolist() == ["a b", "x"]


class TestQuotesNumbers:
    def test_numeric_strings_are_coerced(self) -> None:
        df = normalize_quotes(_quotes(2, line=["24.5", 25], over_price=["1.91", 2], under_price=[" 1.95 ", "2.10"]))
        assert df["line"].tolist() == [24.5, 25.0]
        assert df["over_price"].tolist() == [1.91, 2.0]
        assert df["under_price"].tolist() == [1.95, 2.10]
        _assert_quote_dtypes(df)

    def test_integer_inputs_become_float(self) -> None:
        df = normalize_quotes(_quotes(2, line=[24, 25], over_price=[2, 3], under_price=[2, 2]))
        _assert_quote_dtypes(df)
        assert df["line"].tolist() == [24.0, 25.0]

    @pytest.mark.parametrize("bad", [1.0, 0.99, 0.0, -110, -1.5, "abc", None, np.nan, ""])
    def test_price_not_above_one_becomes_nan_but_the_other_side_survives(self, bad: object) -> None:
        df = normalize_quotes(_quotes(over_price=[bad], under_price=[1.87]))
        assert len(df) == 1
        assert np.isnan(df["over_price"].iloc[0])
        assert df["under_price"].iloc[0] == 1.87
        df = normalize_quotes(_quotes(over_price=[2.2], under_price=[bad]))
        assert len(df) == 1
        assert df["over_price"].iloc[0] == 2.2
        assert np.isnan(df["under_price"].iloc[0])

    @pytest.mark.parametrize("bad", [1.0, 0.5, -110, "abc", None, np.nan])
    def test_row_without_a_valid_price_on_either_side_is_dropped(self, bad: object) -> None:
        raw = _quotes(2, player=["A B", "C D"], over_price=[bad, 1.9], under_price=[bad, 1.9])
        assert normalize_quotes(raw)["player"].tolist() == ["C D"]

    def test_one_sided_quotes_are_kept(self) -> None:
        df = normalize_quotes(pd.DataFrame(
            {"book": ["dk"], "market": ["anytime_td"], "player": ["A B"], "line": [0.5], "over_price": [3.4]}
        ))
        assert len(df) == 1 and np.isnan(df["under_price"].iloc[0])
        df = normalize_quotes(pd.DataFrame(
            {"book": ["dk"], "market": ["anytime_td"], "player": ["A B"], "line": [0.5], "under_price": [1.3]}
        ))
        assert len(df) == 1 and np.isnan(df["over_price"].iloc[0])

    @pytest.mark.parametrize("bad_line", [None, np.nan, "abc", ""])
    def test_row_without_a_line_is_dropped(self, bad_line: object) -> None:
        raw = _quotes(2, player=["A B", "C D"], line=[bad_line, 3.5])
        assert normalize_quotes(raw)["player"].tolist() == ["C D"]

    def test_zero_and_negative_lines_are_valid(self) -> None:
        df = normalize_quotes(_quotes(3, line=[0, -2.5, 0.5]))
        assert df["line"].tolist() == [0.0, -2.5, 0.5]

    @pytest.mark.parametrize("column", ["book", "market"])
    @pytest.mark.parametrize("blank", ["", "   ", None, np.nan])
    def test_row_without_book_or_market_is_dropped(self, column: str, blank: object) -> None:
        raw = _quotes(2, player=["A B", "C D"])
        raw[column] = [blank, "x"]
        assert normalize_quotes(raw)["player"].tolist() == ["C D"]

    @pytest.mark.parametrize("bad", [np.inf, "inf", -np.inf, "1e400"])
    def test_infinite_price_is_not_a_valid_quote(self, bad: object) -> None:
        """Documented: a price must be a finite number > 1."""
        df = normalize_quotes(_quotes(over_price=[bad], under_price=[1.87]))
        assert np.isnan(df["over_price"].iloc[0])
        assert len(normalize_quotes(_quotes(over_price=[bad], under_price=[bad]))) == 0

    @pytest.mark.parametrize("bad_line", [np.inf, -np.inf, "inf"])
    def test_row_with_an_infinite_line_is_dropped(self, bad_line: object) -> None:
        raw = _quotes(2, player=["A B", "C D"], line=[bad_line, 3.5])
        assert normalize_quotes(raw)["player"].tolist() == ["C D"]


class TestQuotesTimes:
    def _commence(self, values: List[object]) -> pd.Series:
        df = normalize_quotes(_quotes(len(values), commence_time=values, last_update=values))
        _assert_quote_dtypes(df)
        pd.testing.assert_series_equal(
            df["commence_time"].astype("datetime64[ns, UTC]"),
            df["last_update"].astype("datetime64[ns, UTC]"),
            check_names=False,
        )
        return df["commence_time"]

    def test_utc_strings(self) -> None:
        times = self._commence(["2025-01-01T00:10:00Z", "2025-03-09T23:59:59Z"])
        assert times.iloc[0] == pd.Timestamp("2025-01-01 00:10:00", tz="UTC")
        assert times.iloc[1] == pd.Timestamp("2025-03-09 23:59:59", tz="UTC")

    def test_offset_strings_are_converted_to_utc(self) -> None:
        times = self._commence(["2025-01-01T19:00:00-05:00", "2025-01-02T19:00:00-08:00"])
        assert times.iloc[0] == pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
        assert times.iloc[1] == pd.Timestamp("2025-01-03 03:00:00", tz="UTC")

    def test_naive_values_are_taken_as_utc(self) -> None:
        times = self._commence(["2025-01-01 19:00:00", "2025-01-02 19:00:00"])
        assert times.iloc[0] == pd.Timestamp("2025-01-01 19:00:00", tz="UTC")
        times = self._commence([dt.datetime(2025, 1, 1, 19), dt.datetime(2025, 1, 2, 19)])
        assert times.iloc[1] == pd.Timestamp("2025-01-02 19:00:00", tz="UTC")

    def test_aware_python_datetimes_with_different_offsets(self) -> None:
        times = self._commence([dt.datetime(2025, 1, 1, 19, tzinfo=EST), dt.datetime(2025, 1, 2, 19, tzinfo=dt.timezone.utc)])
        assert times.iloc[0] == pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
        assert times.iloc[1] == pd.Timestamp("2025-01-02 19:00:00", tz="UTC")

    def test_tz_aware_column_is_converted(self) -> None:
        aware = pd.Series(pd.to_datetime(["2025-01-01 19:00", "2025-01-02 19:00"])).dt.tz_localize(EST)
        times = self._commence(list(aware))
        assert times.iloc[0] == pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
        df = normalize_quotes(_quotes(2, commence_time=aware))
        assert df["commence_time"].iloc[1] == pd.Timestamp("2025-01-03 00:00:00", tz="UTC")

    def test_missing_and_garbage_become_nat_without_dropping_the_row(self) -> None:
        times = self._commence([None, "2025-01-02T03:04:05Z", "not a time", pd.NaT])
        assert times.isna().tolist() == [True, False, True, True]
        assert times.iloc[1] == pd.Timestamp("2025-01-02 03:04:05", tz="UTC")

    def test_all_missing_times(self) -> None:
        times = self._commence([pd.NaT, None])
        assert times.isna().all()

    @pytest.mark.parametrize(
        "second, expected",
        [
            ("2025-01-02T03:04:05.123Z", "2025-01-02 03:04:05.123"),
            ("2025-01-02 03:04:05", "2025-01-02 03:04:05"),
            ("2025-01-02T03:04:05", "2025-01-02 03:04:05"),
            ("2025-01-02T03:04Z", "2025-01-02 03:04:00"),
            ("2025-01-02", "2025-01-02 00:00:00"),
            ("2025-01-01T22:04:05-05:00", "2025-01-02 03:04:05"),
        ],
    )
    def test_mixed_iso_formats_are_all_parsed(self, second: str, expected: str) -> None:
        """One feed's format must not turn another feed's timestamps into NaT."""
        df = normalize_quotes(_quotes(2, commence_time=["2025-01-01T00:00:00Z", second]))
        assert df["commence_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
        assert df["commence_time"].iloc[1] == pd.Timestamp(expected, tz="UTC")

    @pytest.mark.parametrize(
        "values",
        [
            [1735689600, 1735776000],                 # Unix seconds
            [1735689600000, 1735776000000],           # Unix milliseconds
            [1735689600.0, 1735776000.0],
            [np.int64(1735689600), "2025-01-02T00:00:00Z"],   # numbers and text in one column
        ],
    )
    def test_unix_epochs_are_seconds_or_milliseconds_never_nanoseconds(self, values: List[object]) -> None:
        """1735689600 is 2025-01-01T00:00:00Z; read as nanoseconds it would be 1970."""
        times = self._commence(values)
        assert times.tolist() == [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-02", tz="UTC")]

    @pytest.mark.parametrize("value", [5, 20250101, 1735689600000000, np.inf, -np.inf, 1e300, np.nan])
    def test_numbers_that_are_not_plausible_epochs_become_nat(self, value: float) -> None:
        times = self._commence([value, 1735689600])
        assert pd.isna(times.iloc[0])
        assert times.iloc[1] == pd.Timestamp("2025-01-01", tz="UTC")

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "BUG: an epoch column with pandas' nullable integer dtype and a missing value "
            "crashes normalize_quotes: _to_utc calls float(pd.NA) -> TypeError('float() "
            "argument must be a string or a real number, not NAType')"
        ),
    )
    def test_nullable_integer_epochs_with_a_gap(self) -> None:
        raw = _quotes(2, commence_time=pd.array([1735689600, pd.NA], dtype="Int64"))
        times = normalize_quotes(raw)["commence_time"]
        assert times.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC") and pd.isna(times.iloc[1])


class TestQuotesAlternateFlag:
    @pytest.mark.parametrize(
        "values, expected",
        [
            ([True, False], [True, False]),
            ([None, True], [False, True]),
            ([np.nan, True], [False, True]),
            ([0, 1], [False, True]),
            ([np.True_, np.False_], [True, False]),
        ],
    )
    def test_coerced_to_bool_with_missing_as_false(self, values: List[object], expected: List[bool]) -> None:
        df = normalize_quotes(_quotes(2, is_alternate=values))
        assert df["is_alternate"].tolist() == expected
        assert df["is_alternate"].dtype == bool

    def test_defaults_to_false(self) -> None:
        assert normalize_quotes(_quotes(2))["is_alternate"].tolist() == [False, False]

    @pytest.mark.parametrize(
        "values", [["False", "True"], ["false", "true"], ["0", "1"], ["no", "yes"], [" N ", " Y "], ["", "t"]]
    )
    def test_text_flags_are_parsed(self, values: List[str]) -> None:
        """astype(bool) would turn the string 'false' into True."""
        df = normalize_quotes(_quotes(2, is_alternate=values))
        assert df["is_alternate"].tolist() == [False, True]
        assert df["is_alternate"].dtype == bool

    def test_mixed_flag_representations(self) -> None:
        df = normalize_quotes(_quotes(6, is_alternate=[True, "false", 0, None, 1.0, np.nan]))
        assert df["is_alternate"].tolist() == [True, False, False, False, True, False]


class TestQuotesIds:
    def test_integer_ids_with_a_gap_keep_their_integer_spelling(self) -> None:
        """An int column with a missing value is float in pandas; 12345.0 must stay '12345'."""
        df = normalize_quotes(_quotes(2, event_id=[12345, np.nan]))
        assert df["event_id"].tolist() == ["12345", ""]
        df = normalize_quotes(_quotes(2, event_id=pd.array([7, pd.NA], dtype="Int64")))
        assert df["event_id"].tolist() == ["7", ""]

    def test_text_ids_are_kept_verbatim(self) -> None:
        df = normalize_quotes(_quotes(3, event_id=["0022400001", " abc ", "1e5"]))
        assert df["event_id"].tolist() == ["0022400001", "abc", "1e5"]


# --------------------------------------------------------------------------
# normalize_gamelogs
# --------------------------------------------------------------------------
class TestGamelogsShape:
    @pytest.mark.parametrize("missing", ["sport", "game_date", "player"])
    def test_missing_required_column_raises(self, missing: str) -> None:
        with pytest.raises(ValueError, match=missing):
            normalize_gamelogs(_gamelogs().drop(columns=[missing]))

    def test_minimal_frame_gets_defaults(self) -> None:
        df = normalize_gamelogs(_gamelogs())
        assert list(df.columns) == GAMELOG_COLUMNS
        _assert_gamelog_dtypes(df)
        row = df.iloc[0]
        assert (row["game_id"], row["player_id"], row["team"], row["opponent"]) == ("", "", "", "")
        assert np.isnan(row["season"]) and np.isnan(row["minutes"])
        assert not row["is_home"]
        assert row["player_key"] == "lebron james"
        assert row["game_date"] == pd.Timestamp("2025-01-01")

    @pytest.mark.parametrize("columns", [GAMELOG_COLUMNS, ["sport", "game_date", "player"]])
    def test_empty_input(self, columns: List[str]) -> None:
        df = normalize_gamelogs(pd.DataFrame(columns=columns))
        assert list(df.columns) == GAMELOG_COLUMNS
        assert len(df) == 0
        assert pd.api.types.is_datetime64_dtype(df["game_date"])
        assert df["is_home"].dtype == bool
        assert df["minutes"].dtype == np.float64

    def test_identity_columns_first_then_stats_in_input_order(self) -> None:
        raw = _gamelogs(2, rebounds=[5, 6], points=[20, 30], team=["LAL", "LAL"])
        raw = raw[["rebounds", "player", "points", "team", "game_date", "sport"]]
        df = normalize_gamelogs(raw)
        assert list(df.columns) == GAMELOG_COLUMNS + ["rebounds", "points"]

    def test_returns_a_new_frame_with_a_fresh_index(self) -> None:
        raw = _gamelogs(3, points=["1", "2", "3"])
        raw.index = pd.Index([5, 5, 9])
        before = raw.copy(deep=True)
        df = normalize_gamelogs(raw)
        pd.testing.assert_frame_equal(raw, before)
        assert df.index.tolist() == [0, 1, 2]

    def test_is_idempotent(self) -> None:
        raw = _gamelogs(
            4,
            player=["Nikola Jokić", "LeBron James", "Nikola Jokić", "LeBron James"],
            game_date=["2025-01-03", "2025-01-02", "2025-01-01", "2025-01-04"],
            season=[2024] * 4,
            player_id=[203999, 2544, 203999, 2544],
            is_home=[True, None, False, True],
            minutes=["34.5", 33, "DNP", 30],
            points=["30", 25, None, 10],
        )
        once = normalize_gamelogs(raw)
        pd.testing.assert_frame_equal(normalize_gamelogs(once), once)


class TestGamelogsCoercion:
    def test_text_columns(self) -> None:
        df = normalize_gamelogs(
            _gamelogs(
                sport=["NBA"],
                player=["  Nikola Jokić "],
                team=[" DEN "],
                opponent=[None],
                player_id=[203999],
                game_id=[22400123],
            )
        )
        _assert_gamelog_dtypes(df)
        row = df.iloc[0]
        assert row["sport"] == "nba"
        assert row["player"] == "Nikola Jokić"
        assert row["player_key"] == "nikola jokic"
        assert row["team"] == "DEN"            # stripped, case kept
        assert row["opponent"] == ""
        assert row["player_id"] == "203999"
        assert row["game_id"] == "22400123"

    def test_minutes_season_and_home_flag(self) -> None:
        df = normalize_gamelogs(
            _gamelogs(
                4,
                minutes=["34.5", 33, "DNP", None],
                season=["2024", 2024, "x", None],
                is_home=[True, None, False, np.nan],
            )
        )
        assert df["minutes"].tolist()[:2] == [34.5, 33.0]
        assert df["minutes"].isna().tolist() == [False, False, True, True]
        assert df["season"].tolist()[:2] == [2024, 2024]
        assert df["season"].isna().tolist() == [False, False, True, True]
        assert df["is_home"].tolist() == [True, False, False, False]
        _assert_gamelog_dtypes(df)

    def test_numeric_extra_columns_become_float_stats_by_default(self) -> None:
        raw = _gamelogs(
            3,
            points=[30, 25, 12],
            assists=[1.5, None, 2.0],
            double_double=pd.array([1, pd.NA, 0], dtype="Int64"),
            started=[True, False, True],
            position=["C", "F", "G"],
        )
        df = normalize_gamelogs(raw)
        for col in ("points", "assists", "double_double"):
            assert df[col].dtype == np.float64, col
        assert df["points"].tolist() == [30.0, 25.0, 12.0]
        assert df["double_double"].tolist()[0] == 1.0 and np.isnan(df["double_double"].iloc[1])
        assert df["started"].tolist() == [True, False, True]   # flags are not stats
        assert df["position"].tolist() == ["C", "F", "G"]      # text is left alone

    @pytest.mark.parametrize("values", [["30", "25", "12"], ["30", 25, None], [" 7 ", "", None], ["1e1", "2.5", "3"]])
    def test_numeric_looking_text_columns_become_float_stats_by_default(self, values: List[object]) -> None:
        """Stats read as text (a CSV column, a JSON API) are still stats."""
        df = normalize_gamelogs(_gamelogs(3, points=values))
        assert df["points"].dtype == np.float64
        assert df["points"].iloc[0] == float(str(values[0]).strip())

    @pytest.mark.parametrize("values", [[30, "DNP", 12], ["1,234", "2", "3"], ["C", "F", "G"], ["", None, " "]])
    def test_columns_with_non_numeric_text_are_left_alone_by_default(self, values: List[object]) -> None:
        """One marker such as 'DNP' makes a column non-numeric: name it in stat_columns to coerce it."""
        df = normalize_gamelogs(_gamelogs(3, extra=values))
        assert df["extra"].dtype != np.float64
        assert df["extra"].tolist()[0] == values[0]
        coerced = normalize_gamelogs(_gamelogs(3, extra=values), stat_columns=["extra"])
        assert coerced["extra"].dtype == np.float64

    def test_explicit_stat_columns_coerce_text_and_markers(self) -> None:
        raw = _gamelogs(3, points=["30", "DNP", None], started=[True, False, True])
        df = normalize_gamelogs(raw, stat_columns=["points", "started"])
        assert df["points"].dtype == np.float64 and df["started"].dtype == np.float64
        assert df["points"].iloc[0] == 30.0 and df["points"].isna().tolist() == [False, True, True]
        assert df["started"].tolist() == [1.0, 0.0, 1.0]

    def test_explicit_stat_columns_leave_other_extras_alone(self) -> None:
        raw = _gamelogs(2, points=["30", "25"], position=["C", "F"], rebounds=["10", "12"])
        df = normalize_gamelogs(raw, stat_columns=["points", "not_a_column"])
        assert df["points"].dtype == np.float64
        assert df["points"].tolist() == [30.0, 25.0]
        assert df["position"].tolist() == ["C", "F"]
        assert df["rebounds"].tolist() == ["10", "12"]
        assert "not_a_column" not in df.columns

    def test_stat_columns_accepts_any_iterable(self) -> None:
        df = normalize_gamelogs(_gamelogs(2, points=["1", "2"], note=["a", "b"]), stat_columns=(c for c in ["points"]))
        assert df["points"].tolist() == [1.0, 2.0]
        assert df["note"].tolist() == ["a", "b"]

    def test_identity_columns_are_never_treated_as_stats(self) -> None:
        df = normalize_gamelogs(_gamelogs(2, team=["LAL", "LAL"], points=[1, 2]))
        assert df["team"].tolist() == ["LAL", "LAL"]
        assert df["player"].tolist() == ["LeBron James", "LeBron James"]


class TestGamelogsRows:
    def test_rows_without_a_date_or_a_player_are_dropped(self) -> None:
        raw = _gamelogs(
            6,
            game_date=["2025-01-01", "bad", None, "2025-01-04", "2025-01-05", "2025-01-06"],
            player=["A B", "A B", "A B", "", None, "   "],
            points=[1, 2, 3, 4, 5, 6],
        )
        df = normalize_gamelogs(raw)
        assert df["points"].tolist() == [1.0]
        assert df.index.tolist() == [0]

    def test_sorted_by_player_then_date_then_game(self) -> None:
        raw = _gamelogs(
            6,
            player=["B B", "B B", "A A", "A A", "B B", "Á A"],
            game_date=["2025-01-02", "2025-01-01", "2025-01-01", "2025-01-03", "2025-01-01", "2025-01-02"],
            game_id=["g3", "g2", "g1", "g9", "g1", "g5"],
            points=[1, 2, 3, 4, 5, 6],
        )
        df = normalize_gamelogs(raw)
        assert df["player_key"].tolist() == ["a a", "a a", "a a", "b b", "b b", "b b"]
        assert df["points"].tolist() == [3.0, 6.0, 4.0, 5.0, 2.0, 1.0]
        assert df["game_date"].is_monotonic_increasing is False  # sorted within player, not globally
        for _, group in df.groupby("player_key"):
            assert group["game_date"].is_monotonic_increasing

    def test_sort_is_stable_for_full_ties(self) -> None:
        raw = _gamelogs(3, game_date=["2025-01-01"] * 3, game_id=["g", "g", "g"], points=[7, 8, 9])
        assert normalize_gamelogs(raw)["points"].tolist() == [7.0, 8.0, 9.0]

    def test_player_key_is_computed_from_the_display_name(self) -> None:
        df = normalize_gamelogs(_gamelogs(3, player=["Nikola Jokić", "Michael Porter Jr.", "C.J. Stroud"]))
        assert sorted(df["player_key"]) == ["cj stroud", "michael porter", "nikola jokic"]

    def test_existing_player_keys_are_kept(self) -> None:
        df = normalize_gamelogs(_gamelogs(2, player=["A B", "C D"], player_key=["custom one", "custom two"]))
        assert df["player_key"].tolist() == ["custom one", "custom two"]

    def test_blank_player_keys_are_filled_from_the_player_name(self) -> None:
        """Same rule as normalize_quotes: a blank or missing key is derived, never kept."""
        raw = _gamelogs(4, player=["A B", "C D", "E F", "G H"], player_key=["custom", None, "", "  "])
        df = normalize_gamelogs(raw)
        assert sorted(df["player_key"].tolist()) == ["c d", "custom", "e f", "g h"]
        assert _is_text(df["player_key"])


class TestGamelogsDates:
    def _dates(self, values: List[object]) -> pd.Series:
        df = normalize_gamelogs(_gamelogs(len(values), game_date=values, game_id=[f"g{i}" for i in range(len(values))]))
        assert pd.api.types.is_datetime64_dtype(df["game_date"])  # tz-naive datetime64
        return df.sort_values("game_id")["game_date"].reset_index(drop=True)

    @pytest.mark.parametrize(
        "values",
        [
            ["2025-01-01", "2025-01-02"],
            ["01/01/2025", "01/02/2025"],
            ["20250101", "20250102"],
            ["Jan 1, 2025", "Jan 2, 2025"],
            [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
            [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)],
            [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")],
        ],
    )
    def test_common_date_representations(self, values: List[object]) -> None:
        dates = self._dates(values)
        assert dates.tolist() == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]

    def test_utc_timestamps_become_naive(self) -> None:
        dates = self._dates(["2025-01-01T00:30:00Z", "2025-01-02T00:30:00Z"])
        assert dates.tolist() == [pd.Timestamp("2025-01-01 00:30:00"), pd.Timestamp("2025-01-02 00:30:00")]

    def test_tz_aware_column_becomes_naive(self) -> None:
        aware = pd.Series(pd.to_datetime(["2025-01-01 19:30", "2025-01-02 19:30"])).dt.tz_localize("UTC")
        df = normalize_gamelogs(_gamelogs(2, game_date=aware))
        assert pd.api.types.is_datetime64_dtype(df["game_date"])
        assert df["game_date"].tolist() == [pd.Timestamp("2025-01-01 19:30"), pd.Timestamp("2025-01-02 19:30")]

    def test_offsets_are_dropped_so_an_evening_game_keeps_its_local_date(self) -> None:
        """Documented: the offset is dropped, not converted (19:30 -05:00 stays on Jan 1)."""
        dates = self._dates(["2025-01-01T19:30:00-05:00", "2025-01-02T19:30:00-05:00"])
        assert dates.tolist() == [pd.Timestamp("2025-01-01 19:30:00"), pd.Timestamp("2025-01-02 19:30:00")]
        aware = [dt.datetime(2025, 1, 1, 19, 30, tzinfo=EST), dt.datetime(2025, 1, 2, 19, 30, tzinfo=EST)]
        assert self._dates(aware).tolist() == dates.tolist()

    @pytest.mark.parametrize(
        "second, expected",
        [
            ("2025-01-02T00:00:00", "2025-01-02 00:00:00"),
            ("2025-01-02 19:30:00", "2025-01-02 19:30:00"),
            ("01/02/2025", "2025-01-02 00:00:00"),
            ("Jan 2, 2025", "2025-01-02 00:00:00"),
            ("20250102", "2025-01-02 00:00:00"),
        ],
    )
    def test_mixed_date_formats_keep_every_row(self, second: str, expected: str) -> None:
        """Rows must not vanish because their date is spelled differently from row one."""
        dates = self._dates(["2025-01-01", second])
        assert dates.tolist() == [pd.Timestamp("2025-01-01"), pd.Timestamp(expected)]

    def test_mixed_utc_offsets_across_a_daylight_saving_change(self) -> None:
        dates = self._dates(["2025-03-08T19:30:00-05:00", "2025-03-10T19:30:00-04:00", "2025-03-12"])
        assert dates.tolist() == [
            pd.Timestamp("2025-03-08 19:30:00"), pd.Timestamp("2025-03-10 19:30:00"), pd.Timestamp("2025-03-12"),
        ]

    def test_mixed_python_date_objects(self) -> None:
        values = [dt.date(2025, 1, 1), dt.datetime(2025, 1, 2, 19, 30), pd.Timestamp("2025-01-03", tz="UTC"),
                  np.datetime64("2025-01-04")]
        assert self._dates(values).tolist() == [
            pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02 19:30"), pd.Timestamp("2025-01-03"),
            pd.Timestamp("2025-01-04"),
        ]

    @pytest.mark.parametrize("bad", ["bad", "", "2025-13-45", "2025-02-30", None, np.nan])
    def test_unparseable_dates_drop_only_their_own_row(self, bad: object) -> None:
        df = normalize_gamelogs(_gamelogs(3, game_date=["2025-01-01", bad, "2025-01-03"], points=[1, 2, 3]))
        assert df["points"].tolist() == [1.0, 3.0]

    @pytest.mark.parametrize(
        "values",
        [[20250101, 20250102], [20250101.0, 20250102.0], [np.int64(20250101), "2025-01-02"], ["20250101", "20250102"]],
    )
    def test_integer_yyyymmdd_dates(self, values: List[object]) -> None:
        """20250101 is a date, not 20,250,101 nanoseconds after 1970."""
        assert self._dates(values).tolist() == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]

    @pytest.mark.parametrize("bad", [1735689600, 2025, 20251340, 20250230, 20250101.5, 0, -20250101])
    def test_numbers_that_are_not_yyyymmdd_drop_their_row(self, bad: float) -> None:
        df = normalize_gamelogs(_gamelogs(2, game_date=[bad, 20250102], points=[1, 2]))
        assert df["points"].tolist() == [2.0]
        assert df["game_date"].tolist() == [pd.Timestamp("2025-01-02")]

    def test_nullable_integer_dates_with_a_gap(self) -> None:
        df = normalize_gamelogs(_gamelogs(2, game_date=pd.array([20250101, pd.NA], dtype="Int64"), points=[1, 2]))
        assert df["points"].tolist() == [1.0]


class TestGamelogsIdsAndFlags:
    def test_integer_ids_with_a_gap_keep_their_integer_spelling(self) -> None:
        df = normalize_gamelogs(_gamelogs(3, player_id=[2544, np.nan, 203999], game_id=["0022400001", 22400002, None]))
        assert sorted(df["player_id"].tolist()) == ["", "203999", "2544"]
        assert sorted(df["game_id"].tolist()) == ["", "0022400001", "22400002"]

    @pytest.mark.parametrize("values", [["False", "True"], ["false", "true"], ["0", "1"], ["no", "yes"]])
    def test_text_home_flags_are_parsed(self, values: List[str]) -> None:
        df = normalize_gamelogs(_gamelogs(2, is_home=values))
        assert df["is_home"].tolist() == [False, True]

    def test_missing_sport_becomes_empty_text(self) -> None:
        df = normalize_gamelogs(_gamelogs(3, sport=[None, " NFL ", np.nan]))
        assert df["sport"].tolist() == ["", "nfl", ""]
        assert _is_text(df["sport"])
