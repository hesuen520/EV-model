"""Check the core maths against independently computed reference values.

``fixtures/golden_vectors.json`` was produced outside this package with
50-digit arithmetic (mpmath), from the defining equations of each method, by
a separate implementation.  Agreement here means two independent derivations
of the odds, de-vig, EV, Kelly and line-translation maths give the same
numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evmodel.devig import METHODS, devig, worst_case
from evmodel.distributions import DiscretizedNormal, FamilySpec
from evmodel.ev import bet_variance, expected_value, expected_log_growth, fractional_kelly, kelly_fraction
from evmodel.odds import american_to_decimal, american_to_prob, breakeven_prob, decimal_to_american

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden_vectors.json").read_text())
TOL = 1e-9

_DEVIG = [v for v in GOLDEN["devig"] if v["method"] in METHODS and v["valid"]]
_LOGIT = [v for v in GOLDEN["devig"] if v["method"] == "logit_shift" and v["valid"]]
_WORST = [v for v in GOLDEN["devig"] if v["method"] == "worst_case"]
_INVALID = [v for v in GOLDEN["devig"] if not v["valid"]]


def _implied(vector):
    return [american_to_prob(a) for a in vector["american"]]


@pytest.mark.parametrize("row", GOLDEN["conversions"]["american_to_decimal_prob"], ids=lambda r: str(r["american"]))
def test_american_conversions(row):
    decimal = american_to_decimal(row["american"])
    assert decimal == pytest.approx(row["decimal"], rel=1e-11)
    assert american_to_prob(row["american"]) == pytest.approx(row["implied_prob"], abs=1e-11)
    assert decimal_to_american(decimal) == pytest.approx(row["canonical_american"], rel=1e-9)


@pytest.mark.parametrize("vector", _DEVIG, ids=lambda v: v["id"])
def test_devig_matches_reference(vector):
    fair = devig(_implied(vector), vector["method"])
    assert fair == pytest.approx(vector["fair"], abs=TOL)
    assert sum(fair) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("vector", _LOGIT, ids=lambda v: v["id"])
def test_logit_shift_is_the_odds_ratio_method(vector):
    assert devig(_implied(vector), "odds_ratio") == pytest.approx(vector["fair"], abs=TOL)


@pytest.mark.parametrize("vector", _WORST, ids=lambda v: v["id"])
def test_worst_case_matches_reference(vector):
    implied = _implied(vector)
    got = [worst_case(implied, i) for i in range(len(implied))]
    assert got == pytest.approx(vector["fair"], abs=TOL)


@pytest.mark.parametrize("vector", _INVALID, ids=lambda v: v["id"])
def test_additive_outside_unit_interval_falls_back_to_multiplicative(vector):
    """The reference flags these as invalid; the package documents a fallback."""
    assert vector["method"] == "additive"
    assert min(vector["fair"]) <= 0.0
    implied = _implied(vector)
    assert devig(implied, "additive") == pytest.approx(devig(implied, "multiplicative"), abs=1e-15)


@pytest.mark.parametrize("row", GOLDEN["ev"], ids=lambda r: f"{r['p_win']}@{r['american']}")
def test_expected_value(row):
    decimal = american_to_decimal(row["american"])
    assert expected_value(row["p_win"], decimal, row["p_push"]) == pytest.approx(row["ev_per_unit"], abs=TOL)
    assert breakeven_prob(decimal, row["p_push"]) == pytest.approx(row["breakeven_p_win"], abs=TOL)
    assert bet_variance(row["p_win"], decimal, row["p_push"]) == pytest.approx(row["variance_per_unit"], abs=TOL)


@pytest.mark.parametrize("row", GOLDEN["kelly"], ids=lambda r: f"{r['p_win']}/{r['p_push']}@{r['american']}")
def test_kelly(row):
    decimal = american_to_decimal(row["american"])
    full = kelly_fraction(row["p_win"], decimal, row["p_push"])
    assert full == pytest.approx(row["kelly_full_stake"], abs=TOL)
    assert fractional_kelly(row["p_win"], decimal, row["p_push"], 0.5) == pytest.approx(row["kelly_half_stake"], abs=TOL)
    assert fractional_kelly(row["p_win"], decimal, row["p_push"], 0.25) == pytest.approx(row["kelly_quarter_stake"], abs=TOL)
    if 0.0 < full < 1.0:
        growth = expected_log_growth(full, row["p_win"], decimal, row["p_push"])
        assert growth == pytest.approx(row["log_growth_at_full_kelly"], abs=TOL)


def _spec(vector):
    family, params = vector["family"], vector["params"]
    if family == "poisson":
        return FamilySpec("poisson")
    if family == "nb1":  # var = phi * mu
        return FamilySpec("negbin", phi=params["phi"], alpha=0.0)
    if family == "nb2":  # var = mu + mu^2 / r
        return FamilySpec("negbin", phi=1.0, alpha=1.0 / params["r"])
    raise AssertionError(family)


_COUNT_TRANSLATIONS = [v for v in GOLDEN["line_translation"] if v["family"] != "normal"]
_NORMAL_TRANSLATIONS = [v for v in GOLDEN["line_translation"] if v["family"] == "normal"]


def _id(vector):
    return f"{vector['family']}-{vector['sharp_line']}-{vector['sharp_fair_p_over_excl_push']}"


@pytest.mark.parametrize("vector", _COUNT_TRANSLATIONS, ids=_id)
def test_line_translation_for_count_families(vector):
    spec = _spec(vector)
    mean = spec.solve_mean(vector["sharp_line"], vector["sharp_fair_p_over_excl_push"])
    assert mean == pytest.approx(vector["mu"], abs=1e-7)
    dist = spec.dist(mean)
    sharp = vector["sharp_line_probs"]
    assert dist.probs(vector["sharp_line"]) == pytest.approx(
        (sharp["p_over"], sharp["p_push"], sharp["p_under"]), abs=1e-8
    )
    for target in vector["targets"]:
        over, push, under = spec.translate(
            vector["sharp_line"], vector["sharp_fair_p_over_excl_push"], target["line"]
        )
        assert (over, push, under) == pytest.approx(
            (target["p_over"], target["p_push"], target["p_under"]), abs=1e-8
        )


@pytest.mark.parametrize("vector", _NORMAL_TRANSLATIONS, ids=_id)
def test_line_translation_for_the_discretised_normal(vector):
    """Fixed-sigma normal: checked at the distribution level."""
    dist = DiscretizedNormal(vector["mu"], vector["params"]["sigma"])
    assert dist.prob_over_no_push(vector["sharp_line"]) == pytest.approx(
        vector["sharp_fair_p_over_excl_push"], abs=1e-9
    )
    for target in vector["targets"]:
        assert dist.probs(target["line"]) == pytest.approx(
            (target["p_over"], target["p_push"], target["p_under"]), abs=1e-9
        )
