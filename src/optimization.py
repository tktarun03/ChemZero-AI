"""Toy scenario ranking for recovered gas.

The score is a transparent weighted sum over four hand-written scenarios. It is
a *discussion device*: it shows how weighting recovery fit, energy value, capital
cost, and complexity changes which option looks attractive. The weights are
guesses, the scenario attributes are guesses, and none of it is an economic
evaluation. A real selection needs vendor quotes, a cash-flow model, site
constraints, and an options study.
"""

from __future__ import annotations

from typing import Any, TypedDict

__all__ = [
    "SCENARIOS",
    "WEIGHTS",
    "MAX_POWER_PRICE_PER_KWH",
    "Scenario",
    "rank_scenarios",
]


class Scenario(TypedDict):
    """A fixed end-use option for the recovered gas.

    Keys ``recovery``/``efficiency``/``capex``/``complexity`` are unitless 0-1
    stand-ins, not measured values.
    """

    name: str
    recovery: float
    efficiency: float
    capex: float
    complexity: float


SCENARIOS: list[Scenario] = [
    {"name": "Flare", "recovery": 0.00, "efficiency": 0.00, "capex": 0.05, "complexity": 0.10},
    {"name": "Recover / Reuse", "recovery": 0.75, "efficiency": 0.85, "capex": 0.60, "complexity": 0.55},
    {"name": "Generate Power", "recovery": 0.65, "efficiency": 0.35, "capex": 0.70, "complexity": 0.65},
    {"name": "Compress / Store", "recovery": 0.70, "efficiency": 0.70, "capex": 0.80, "complexity": 0.75},
]

#: Score weights. Positive terms reward, negative terms penalise. Exposed as a
#: module constant so a learner can change them and watch the ranking move.
WEIGHTS: dict[str, float] = {
    "recovery_fit": 0.55,
    "energy_value": 0.45,
    "capex": 0.35,
    "complexity": 0.25,
}

#: Power price treated as "full value" when normalising, $/kWh.
MAX_POWER_PRICE_PER_KWH: float = 0.18


def rank_scenarios(
    predicted_recovery_fraction: float,
    power_price_per_kwh: float,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score and rank the scenarios for one operating point.

    Args:
        predicted_recovery_fraction: Recovery fraction to match against, 0-1.
        power_price_per_kwh: Power price used to value recovered energy, $/kWh.
        weights: Optional override of :data:`WEIGHTS`; missing keys fall back to
            the defaults.

    Returns:
        Scenario dicts sorted best-first. Each carries its ``score``, a 1-based
        ``rank``, and the individual ``recovery_fit`` / ``energy_value`` /
        ``penalty`` terms so the dashboard can show *why* it scored that way.
    """
    w = {**WEIGHTS, **(weights or {})}
    recovery = min(max(float(predicted_recovery_fraction), 0.0), 1.0)
    normalised_price = min(max(float(power_price_per_kwh) / MAX_POWER_PRICE_PER_KWH, 0.0), 1.0)

    results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        # How closely this scenario's design point matches the recovery we expect.
        recovery_fit = 1.0 - abs(recovery - scenario["recovery"])
        # Value of the energy this scenario converts, scaled by power price.
        energy_value = normalised_price * scenario["efficiency"]
        penalty = w["capex"] * scenario["capex"] + w["complexity"] * scenario["complexity"]
        score = w["recovery_fit"] * recovery_fit + w["energy_value"] * energy_value - penalty

        results.append(
            {
                **scenario,
                "recovery_fit": round(recovery_fit, 4),
                "energy_value": round(energy_value, 4),
                "penalty": round(penalty, 4),
                "score": round(score, 4),
            }
        )

    results.sort(key=lambda row: row["score"], reverse=True)
    for position, row in enumerate(results, start=1):
        row["rank"] = position
    return results
