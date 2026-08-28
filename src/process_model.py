"""Simplified gas-recovery mass and energy balance.

The model is deliberately first-order: recovered methane is treated as
``flow x methane_fraction x recovery_fraction`` with no separation efficiency,
no pressure-drop term, no compositional carry-over, and no equipment model.
It exists to make the arithmetic of a recovery project visible, not to size one.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    CO2_KG_PER_NM3_METHANE_COMBUSTED,
    DEFAULT_OPERATING_HOURS_PER_YEAR,
    METHANE_LHV_KWH_PER_NM3,
)

__all__ = [
    "ProcessInput",
    "ProcessResult",
    "evaluate_process",
    "METHANE_LHV_KWH_PER_NM3",
    "CO2_KG_PER_NM3_METHANE_COMBUSTED",
]


def _clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``.

    Inputs arrive from sliders and model predictions, so out-of-range values are
    clamped rather than raised on: a demo should degrade to a sensible number
    instead of crashing the dashboard.
    """
    return min(max(float(value), low), high)


@dataclass(frozen=True)
class ProcessInput:
    """One synthetic operating point.

    Args:
        flow_nm3_h: Total raw gas flow, Nm³/h. Negative values clamp to 0.
        methane_fraction: Methane mole/volume fraction, clamped to [0, 1].
        recovery_fraction: Fraction of the methane actually captured,
            clamped to [0, 1].
        operating_hours_per_year: Annual on-stream hours, clamped to [0, 8766].
        power_price_per_kwh: Value assigned to recovered energy, $/kWh.
    """

    flow_nm3_h: float
    methane_fraction: float
    recovery_fraction: float
    operating_hours_per_year: float = DEFAULT_OPERATING_HOURS_PER_YEAR
    power_price_per_kwh: float = 0.0


@dataclass(frozen=True)
class ProcessResult:
    """Derived quantities for a single operating point."""

    methane_flow_nm3_h: float
    recovered_gas_nm3_h: float
    recovered_energy_kwh_h: float
    annual_recovered_gas_nm3: float
    annual_recovered_energy_mwh: float
    estimated_avoided_co2_t_y: float
    annual_energy_value_usd: float

    def as_dict(self) -> dict[str, float]:
        """Return the result as a plain dict (handy for DataFrames)."""
        return {
            "methane_flow_nm3_h": self.methane_flow_nm3_h,
            "recovered_gas_nm3_h": self.recovered_gas_nm3_h,
            "recovered_energy_kwh_h": self.recovered_energy_kwh_h,
            "annual_recovered_gas_nm3": self.annual_recovered_gas_nm3,
            "annual_recovered_energy_mwh": self.annual_recovered_energy_mwh,
            "estimated_avoided_co2_t_y": self.estimated_avoided_co2_t_y,
            "annual_energy_value_usd": self.annual_energy_value_usd,
        }

    # Dict-style access keeps older call sites (and the tests) working.
    def __getitem__(self, key: str) -> float:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc


def evaluate_process(inp: ProcessInput) -> ProcessResult:
    """Run the simplified balance for one operating point.

    Args:
        inp: The operating point to evaluate.

    Returns:
        A :class:`ProcessResult`. It also supports ``result["key"]`` lookup so
        it can be used wherever the older dict return value was expected.

    Note:
        ``estimated_avoided_co2_t_y`` is the CO₂ from *combusting* the recovered
        methane, i.e. the fossil CO₂ displaced if that energy replaces an
        equally carbon-intense source one-for-one. It is an illustrative
        upper bound, not an accounting-grade emissions credit: a real figure
        needs the displaced fuel's actual intensity, methane slip, parasitic
        load, and a recognised accounting boundary.
    """
    flow = max(float(inp.flow_nm3_h), 0.0)
    methane_fraction = _clamp(inp.methane_fraction, 0.0, 1.0)
    recovery = _clamp(inp.recovery_fraction, 0.0, 1.0)
    hours = _clamp(inp.operating_hours_per_year, 0.0, 8766.0)
    price = max(float(inp.power_price_per_kwh), 0.0)

    methane_flow = flow * methane_fraction
    recovered_gas = methane_flow * recovery
    recovered_energy_kwh_h = recovered_gas * METHANE_LHV_KWH_PER_NM3

    annual_recovered_gas = recovered_gas * hours
    annual_energy_kwh = recovered_energy_kwh_h * hours

    return ProcessResult(
        methane_flow_nm3_h=methane_flow,
        recovered_gas_nm3_h=recovered_gas,
        recovered_energy_kwh_h=recovered_energy_kwh_h,
        annual_recovered_gas_nm3=annual_recovered_gas,
        annual_recovered_energy_mwh=annual_energy_kwh / 1000.0,
        estimated_avoided_co2_t_y=(
            annual_recovered_gas * CO2_KG_PER_NM3_METHANE_COMBUSTED / 1000.0
        ),
        annual_energy_value_usd=annual_energy_kwh * price,
    )
