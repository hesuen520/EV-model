"""EV module: price maths checked by hand, eligibility, settlement rules,
paper tracking, and the wall between the prediction engine and prices."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sp_helpers import prediction
from sportspred.config import Settings
from sportspred.ev import (
    FEES,
    RULES,
    PriceImportError,
    comparable,
    evaluate,
    import_prices,
    open_paper_positions,
    paper_report,
    settle_outcome,
    settle_paper_positions,
)
from sportspred.ev.odds_api import parse_odds_api

NOW = "2030-01-01T23:50:00Z"
SLATE = date(2030, 1, 1)                         # 2030-01-02T03:30Z is 5:30 PM on Jan 1 in Honolulu
HEAD = ("league,event_id,home,away,date,market,selection,line,venue_kind,venue,price,ask,bid,quoted_at,rules_key,"
        "fee_key,is_executable\n")


def _quotes(tmp_path, body, name="prices.csv"):
    path = tmp_path / name
    path.write_text(HEAD + body)
    return path


def _final_pick(store, scheduled_event, two_teams, prob=0.66, **over):
    fields = {"rec_status": "FINAL", "validation_status": "validated", **over}
    return store.insert_prediction(prediction(
        scheduled_event, two_teams, prob=prob, probs={two_teams[0]: prob, two_teams[1]: 1 - prob}, **fields))


def test_sportsbook_maths_matches_hand_calculation(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams, prob=0.66)
    path = _quotes(tmp_path,
                   "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"
                   "nba,401,,,,winner,Los Angeles Lakers,,sportsbook,book,+130,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n")
    assert import_prices(store, path) == {"quotes_new": 2, "quotes_already_on_file": 0}
    assert import_prices(store, path)["quotes_new"] == 0                       # idempotent
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]
    # -150 -> decimal 1.6667 -> raw implied 0.6000; +130 -> 2.30 -> 0.43478; no-vig = 0.6 / 1.03478
    assert row["raw_implied_prob"] == pytest.approx(0.6)
    assert row["no_vig_prob"] == pytest.approx(0.6 / (0.6 + 1 / 2.3))
    assert row["fair_odds_decimal"] == pytest.approx(1 / 0.66)
    assert row["edge"] == pytest.approx(0.06)
    assert row["ev_per_unit"] == pytest.approx(0.66 * (1 / 0.6 - 1) - 0.34)    # +0.10 per unit
    assert row["eligible"] and row["status"] == "evaluated"
    assert row["raw_implied_prob"] != pytest.approx(row["no_vig_prob"])        # the two are reported separately


def test_missing_price_means_ev_unavailable_never_an_estimate(store, two_teams, scheduled_event):
    _final_pick(store, scheduled_event, two_teams)
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]
    assert row["status"] == "EV unavailable" and not row["eligible"] and "ev_per_unit" not in row.dropna().index


def test_one_sided_quote_has_no_no_vig_figure(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams)
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"))
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]
    assert pd.isna(row["no_vig_prob"]) and row["raw_implied_prob"] == pytest.approx(0.6)


def test_band_preset_applies_to_nba_winner_only_and_is_editable(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams, prob=0.85)
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-300,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"))
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]                          # raw implied 0.75 > 0.70
    assert not row["eligible"] and any("outside the us_major_team_win band" in r for r in row["reasons"])
    wide = Settings()
    wide.ev.presets[0].raw_implied_max = 0.80                                  # the preset is editable
    assert evaluate(store, settings=wide, day=SLATE, now=NOW).iloc[0]["eligible"]
    other = Settings()
    other.ev.presets[0].leagues = ["nhl", "nfl"]                               # not applied to other leagues
    assert evaluate(store, settings=other, day=SLATE, now=NOW).iloc[0]["eligible"]
    assert Settings().ev.presets[0].markets == ["winner"]                      # never props or other markets


def test_stale_quote_late_quote_and_non_final_prediction_are_ineligible(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams, rec_status="PROVISIONAL")
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01T20:00:00Z,moneyline_incl_ot,sportsbook,1\n"))
    reasons = " | ".join(evaluate(store, day=SLATE, now=NOW).iloc[0]["reasons"])
    assert "minutes old" in reasons and "not FINAL" in reasons
    # A quote from the future relative to "now" is invisible.
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book2,-110,,,2030-01-02T01:00:00Z,moneyline_incl_ot,sportsbook,1\n",
                  "later.csv"))
    assert set(evaluate(store, day=SLATE, now=NOW)["venue"]) == {"book"}


def test_prediction_market_cost_includes_the_fee(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams, prob=0.66)
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,prediction_market,kalshi,,0.60,0.58,2030-01-01T23:45:00Z,"
                  "prediction_market_winner_incl_ot,kalshi_taker,1\n"))
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]
    # fee for one contract: 0.07 * 0.6 * 0.4 = 0.0168, rounded UP to the cent = 0.02 -> cost 0.62
    assert FEES["kalshi_taker"].per_contract(0.60) == pytest.approx(0.02)
    assert FEES["kalshi_taker"].per_contract(0.60, contracts=100) == pytest.approx(0.0168)
    assert FEES["polymarket_us_taker"].per_contract(0.50) == pytest.approx(0.0695 * 0.25)
    assert row["raw_implied_prob"] == pytest.approx(0.62)
    assert row["ev_per_unit"] == pytest.approx((0.66 - 0.62) / 0.62) and row["edge"] == pytest.approx(0.04)


def test_regulation_three_way_quote_is_not_comparable_with_a_winner_prediction(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams)
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-120,,,2030-01-01T23:40:00Z,three_way_regulation,sportsbook,1\n"))
    row = evaluate(store, day=SLATE, now=NOW).iloc[0]
    assert row["status"] == "not comparable" and not row["eligible"]
    assert comparable("winner", RULES["moneyline_incl_ot"])[0]
    assert not comparable("winner_3way", RULES["moneyline_incl_ot"])[0]
    assert comparable("player_prop:points", RULES["player_prop_void_if_dnp"])[0]
    # Sport-specific rule sets are not comparable with another sport's prediction.
    assert comparable("winner", RULES["tennis_void_on_retirement"], "tennis")[0]
    ok, why = comparable("winner", RULES["tennis_void_on_retirement"], "basketball")
    assert not ok and "written for tennis" in why
    assert comparable("winner", RULES["moneyline_incl_ot"], "basketball")[0]


def test_bad_price_rows_are_rejected_with_line_numbers(store, two_teams, scheduled_event, tmp_path):
    path = _quotes(tmp_path,
                   "nba,999,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"
                   "nba,401,,,,winner,Chicago Bulls,,sportsbook,book,-150,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"
                   "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01 23:40,moneyline_incl_ot,sportsbook,1\n"
                   "nba,401,,,,winner,Boston Celtics,,prediction_market,kalshi,,1.20,,2030-01-01T23:40:00Z,nope,kalshi_taker,1\n")
    with pytest.raises(PriceImportError) as err:
        import_prices(store, path)
    text = "\n".join(err.value.errors)
    assert "line 2: 0 events match" in text and "line 3: selection 'Chicago Bulls'" in text
    assert "line 4: quoted_at" in text and "line 5: rules_key 'nope'" in text
    assert store.query("SELECT COUNT(*) AS n FROM prices")["n"][0] == 0


EVENT = {"status": "final", "home_id": "H", "away_id": "A", "winner_id": "H", "is_draw": 0, "home_score": 3.0,
         "away_score": 2.0, "went_overtime": 1, "went_shootout": 0, "reg_home_score": None, "reg_away_score": None,
         "method": None}


@pytest.mark.parametrize("rules, market, selection, line, change, expected", [
    ("moneyline_incl_ot", "winner", "H", None, {}, "win"),
    ("moneyline_incl_ot", "winner", "A", None, {}, "loss"),
    ("three_way_regulation", "winner_3way", "H", None, {}, "loss"),          # needed overtime: level after regulation
    ("three_way_regulation", "winner_3way", "draw", None, {}, "win"),
    ("moneyline_incl_ot", "winner", "H", None, {"is_draw": 1, "winner_id": None}, "push"),
    ("moneyline_incl_ot_tie_loses", "winner", "H", None, {"is_draw": 1, "winner_id": None}, "loss"),
    ("prediction_market_winner_incl_ot", "winner", "H", None, {"is_draw": 1, "winner_id": None}, "half"),
    ("spread_incl_ot", "spread", "H", -1.0, {}, "push"),                     # won by exactly the line
    ("spread_incl_ot", "spread", "H", -1.5, {}, "loss"),
    ("spread_incl_ot", "spread", "A", -1.5, {}, "win"),
    ("total_incl_ot", "total", "over", 5.0, {}, "push"),
    ("total_incl_ot", "total", "under", 5.5, {}, "win"),
    ("moneyline_incl_ot", "winner", "H", None, {"status": "postponed"}, "void"),
    ("prediction_market_winner_incl_ot", "winner", "H", None, {"status": "canceled"}, "half"),
    ("tennis_void_on_retirement", "winner", "H", None, {"method": "RET"}, "void"),
    ("tennis_action_on_retirement", "winner", "H", None, {"method": "RET"}, "win"),
    ("tennis_action_after_first_set", "winner", "H", None, {"method": "RET"}, "unsettled"),
    ("tennis_action_after_first_set", "winner", "H", None, {"method": "RET", "detail_json": '{"sets_completed": 0}'}, "void"),
    ("tennis_action_after_first_set", "winner", "H", None, {"method": "RET", "detail_json": '{"sets_completed": 1}'}, "win"),
    ("tennis_action_on_retirement", "winner", "H", None, {"method": "W/O"}, "void"),
    ("mma_draw_void", "winner", "H", None, {"method": "NC"}, "void"),
    ("moneyline_incl_ot", "winner", "H", None, {"status": "scheduled"}, "unsettled"),
])
def test_settlement_rules(rules, market, selection, line, change, expected):
    assert settle_outcome(RULES[rules], market, selection, line, {**EVENT, **change})[0] == expected


def test_paper_positions_settle_resettle_and_report_separately(store, two_teams, scheduled_event, tmp_path):
    _final_pick(store, scheduled_event, two_teams, prob=0.66)
    import_prices(store, _quotes(tmp_path,
                  "nba,401,,,,winner,Boston Celtics,,sportsbook,book,-150,,,2030-01-01T23:40:00Z,moneyline_incl_ot,sportsbook,1\n"))
    evals = evaluate(store, day=SLATE, now=NOW)
    assert open_paper_positions(store, evals, now=NOW) == 1 and open_paper_positions(store, evals, now=NOW) == 0
    assert store.query("SELECT mode FROM paper_positions")["mode"].tolist() == ["paper"]
    assert settle_paper_positions(store).empty                              # no result yet
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 99, "away_score": 104}]))
    first = settle_paper_positions(store)
    assert first.iloc[0]["outcome"] == "loss" and first.iloc[0]["pnl"] == pytest.approx(-1.0)
    # The result is corrected: a new settlement is appended and linked; the old one stays.
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 105, "away_score": 104}]))
    second = settle_paper_positions(store)
    assert second.iloc[0]["outcome"] == "win" and second.iloc[0]["resettled"]
    assert second.iloc[0]["pnl"] == pytest.approx(1 / 0.6 - 1)
    chain = store.query("SELECT * FROM settlements ORDER BY settlement_id")
    assert len(chain) == 2 and chain.iloc[1]["supersedes_settlement_id"] == chain.iloc[0]["settlement_id"]
    report = paper_report(store)
    assert report["mode"] == "paper" and report["settled"] == 1 and report["roi"] == pytest.approx(1 / 0.6 - 1)
    assert report["expected_roi_at_entry"] == pytest.approx(0.10)
    with pytest.raises(Exception):                                          # the schema refuses any non-paper mode
        with store.conn() as con:
            con.execute("UPDATE paper_positions SET mode='live'")


def test_odds_api_parser_follows_the_documented_shape():
    payload = [{"id": "x", "sport_key": "basketball_nba", "commence_time": "2030-01-02T03:30:00Z",
                "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers",
                "bookmakers": [{"key": "book", "last_update": "2030-01-01T23:40:00Z", "markets": [
                    {"key": "h2h", "last_update": "2030-01-01T23:41:00Z",
                     "outcomes": [{"name": "Boston Celtics", "price": -150}, {"name": "Los Angeles Lakers", "price": 130}]},
                    {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": -110, "point": -3.5},
                                                    {"name": "Los Angeles Lakers", "price": -110, "point": 3.5}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 221.5},
                                                   {"name": "Under", "price": -115, "point": 221.5}]}]}]}]
    df = parse_odds_api(payload, "nba")
    assert len(df) == 6 and set(df["market"]) == {"winner", "spread", "total"}
    spread = df[df["market"] == "spread"]
    assert set(spread["line"]) == {-3.5}                                    # both sides share the home handicap
    assert set(df[df["market"] == "total"]["selection"]) == {"over", "under"}
    assert df[df["market"] == "winner"]["quoted_at"].iloc[0] == "2030-01-01T23:41:00Z"
    soccer = parse_odds_api([{"commence_time": "2030-01-02T15:00:00Z", "home_team": "A", "away_team": "B",
                              "bookmakers": [{"key": "b", "last_update": "2030-01-02T10:00:00Z", "markets": [
                                  {"key": "h2h", "outcomes": [{"name": "A", "price": 120}, {"name": "Draw", "price": 230},
                                                              {"name": "B", "price": 240}]}]}]}], "epl")
    assert set(soccer["market"]) == {"winner_3way"} and set(soccer["rules_key"]) == {"three_way_regulation"}


def test_prediction_side_never_imports_prices():
    """The engine that produces probabilities must not be able to see prices."""
    root = Path(__file__).resolve().parents[2] / "src" / "sportspred"
    offenders = []
    for package in ("predict", "models", "features", "validation", "sports", "combos", "ledger", "core", "storage"):
        for path in (root / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [("." * node.level) + (node.module or "")]
                for name in names:
                    if name.startswith("evmodel") or name.endswith(".ev") or ".ev." in name or name == "sportspred.ev":
                        offenders.append(f"{path.relative_to(root)}: {name}")
    assert not offenders, offenders
    for module in ("pipeline.py", "coverage.py", "config.py"):
        assert "evmodel" not in (root / module).read_text() and "from .ev" not in (root / module).read_text()


def test_a_feature_frame_that_lists_a_price_as_a_model_input_is_refused():
    """Validation and training both call ``FeatureFrame.validate``; a price
    column among the inputs stops them, whatever a plugin author intended."""
    import pandas as pd
    import pytest

    from sportspred.core.enums import OutcomeType
    from sportspred.sports.base import FeatureFrame

    def frame(features, extra=None, retro=()):
        df = pd.DataFrame({
            "event_id": ["e1", "e2"], "start_time_utc": pd.to_datetime(["2030-01-01T00:00:00Z", "2030-01-02T00:00:00Z"]),
            "block": ["2030", "2030"], "y": [0.0, 1.0], "side_a_id": ["a", "a"], "side_b_id": ["b", "b"],
            **{c: [0.1, 0.2] for c in list(features) + list(retro)},
        })
        return FeatureFrame("nfl", "winner", OutcomeType.BINARY, ("away", "home"), df, list(features),
                            retrospective_cols=list(retro), extra=extra or {})

    frame(["elo_diff", "rest_diff"]).validate()                                  # ordinary inputs pass
    for bad in ("home_moneyline", "implied_prob_home", "closing_odds", "market_prob"):
        with pytest.raises(ValueError, match="price columns may not be model inputs"):
            frame(["elo_diff", bad]).validate()
    with pytest.raises(ValueError, match="b365_home"):                           # declared by the plugin as a price
        frame(["elo_diff", "b365_home"], extra={"price_cols": ["b365_home"]}).validate()
    with pytest.raises(ValueError, match="price columns"):                       # retrospective inputs are checked too
        frame(["elo_diff"], retro=["spread_odds"]).validate()
