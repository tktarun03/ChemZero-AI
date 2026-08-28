"""Tests for the toy scenario ranking."""

from __future__ import annotations

import pytest

from src.optimization import MAX_POWER_PRICE_PER_KWH, SCENARIOS, WEIGHTS, rank_scenarios


def test_returns_every_scenario():
    ranked = rank_scenarios(0.65, 0.10)
    assert len(ranked) == len(SCENARIOS)
    assert {row["name"] for row in ranked} == {s["name"] for s in SCENARIOS}


def test_sorted_best_first():
    scores = [row["score"] for row in rank_scenarios(0.65, 0.10)]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_sequential_from_one():
    ranked = rank_scenarios(0.65, 0.10)
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))


def test_score_matches_its_reported_terms():
    """The breakdown shown in the dashboard must actually add up to the score."""
    for row in rank_scenarios(0.7, 0.12):
        expected = (
            WEIGHTS["recovery_fit"] * row["recovery_fit"]
            + WEIGHTS["energy_value"] * row["energy_value"]
            - row["penalty"]
        )
        assert row["score"] == pytest.approx(expected, abs=1e-3)


def test_recovery_fit_peaks_at_the_scenario_design_point():
    flare = next(r for r in rank_scenarios(0.0, 0.10) if r["name"] == "Flare")
    assert flare["recovery_fit"] == pytest.approx(1.0)


def test_higher_power_price_raises_energy_value():
    cheap = {r["name"]: r for r in rank_scenarios(0.65, 0.04)}
    dear = {r["name"]: r for r in rank_scenarios(0.65, 0.18)}
    # Flare converts nothing, so its energy value stays pinned at zero.
    assert dear["Recover / Reuse"]["energy_value"] > cheap["Recover / Reuse"]["energy_value"]
    assert dear["Flare"]["energy_value"] == cheap["Flare"]["energy_value"] == 0.0


def test_power_price_normalisation_saturates():
    at_cap = rank_scenarios(0.65, MAX_POWER_PRICE_PER_KWH)
    way_over = rank_scenarios(0.65, MAX_POWER_PRICE_PER_KWH * 10)
    assert [r["score"] for r in at_cap] == [r["score"] for r in way_over]


@pytest.mark.parametrize("recovery", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_out_of_range_recovery_is_clamped_not_raised(recovery):
    ranked = rank_scenarios(recovery, 0.10)
    assert all(0.0 <= row["recovery_fit"] <= 1.0 for row in ranked)


def test_custom_weights_change_the_outcome():
    """Zeroing the capex penalty must not leave every score untouched."""
    default = rank_scenarios(0.65, 0.10)
    no_capex = rank_scenarios(0.65, 0.10, weights={"capex": 0.0})
    assert [r["score"] for r in default] != [r["score"] for r in no_capex]


def test_partial_weight_override_keeps_the_other_defaults():
    overridden = rank_scenarios(0.65, 0.10, weights={"capex": 0.0})
    row = next(r for r in overridden if r["name"] == "Recover / Reuse")
    scenario = next(s for s in SCENARIOS if s["name"] == "Recover / Reuse")
    assert row["penalty"] == pytest.approx(WEIGHTS["complexity"] * scenario["complexity"])


def test_source_scenarios_are_not_mutated():
    """rank_scenarios must copy: the module constant is shared state."""
    before = [dict(s) for s in SCENARIOS]
    rank_scenarios(0.65, 0.10)
    assert [dict(s) for s in SCENARIOS] == before
    assert all("score" not in s for s in SCENARIOS)
