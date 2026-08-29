"""Tests for the simplified mass/energy balance."""

from __future__ import annotations

import pytest

from src.config import CO2_KG_PER_NM3_METHANE_COMBUSTED, METHANE_LHV_KWH_PER_NM3
from src.process_model import ProcessInput, evaluate_process

# --- original behaviour, kept as a regression guard -----------------------


def test_zero_recovery():
    r = evaluate_process(ProcessInput(1000, 0.8, 0.0))
    assert r["recovered_gas_nm3_h"] == 0


def test_recovery_clamped_to_one():
    r = evaluate_process(ProcessInput(1000, 0.8, 2.0))
    assert r["recovered_gas_nm3_h"] == 800


def test_negative_flow_clamped():
    r = evaluate_process(ProcessInput(-100, 0.8, 0.5))
    assert r["recovered_gas_nm3_h"] == 0


# --- clamping ------------------------------------------------------------


@pytest.mark.parametrize("methane, expected", [(-0.5, 0.0), (0.0, 0.0), (1.5, 1000.0)])
def test_methane_fraction_clamped(methane, expected):
    r = evaluate_process(ProcessInput(1000, methane, 1.0))
    assert r["recovered_gas_nm3_h"] == expected


def test_negative_recovery_clamped():
    r = evaluate_process(ProcessInput(1000, 0.8, -1.0))
    assert r["recovered_gas_nm3_h"] == 0


def test_operating_hours_clamped_to_a_year():
    """More than 8766 h/y is impossible, so the annual figures must not inflate."""
    a = evaluate_process(ProcessInput(1000, 0.8, 0.5, operating_hours_per_year=99_000))
    b = evaluate_process(ProcessInput(1000, 0.8, 0.5, operating_hours_per_year=8766))
    assert a["annual_recovered_energy_mwh"] == b["annual_recovered_energy_mwh"]


def test_negative_power_price_clamped():
    r = evaluate_process(ProcessInput(1000, 0.8, 0.5, power_price_per_kwh=-5.0))
    assert r["annual_energy_value_usd"] == 0.0


# --- arithmetic ----------------------------------------------------------


def test_energy_follows_lower_heating_value():
    r = evaluate_process(ProcessInput(1000, 1.0, 1.0))
    assert r["recovered_energy_kwh_h"] == pytest.approx(1000 * METHANE_LHV_KWH_PER_NM3)


def test_avoided_co2_matches_the_stated_factor():
    r = evaluate_process(ProcessInput(1000, 1.0, 1.0, operating_hours_per_year=1000))
    expected_t = 1000 * 1000 * CO2_KG_PER_NM3_METHANE_COMBUSTED / 1000
    assert r["estimated_avoided_co2_t_y"] == pytest.approx(expected_t)


def test_annual_energy_is_hourly_times_hours():
    r = evaluate_process(ProcessInput(1500, 0.8, 0.65, operating_hours_per_year=8000))
    assert r["annual_recovered_energy_mwh"] == pytest.approx(
        r["recovered_energy_kwh_h"] * 8000 / 1000
    )


def test_energy_value_uses_power_price():
    r = evaluate_process(
        ProcessInput(1000, 1.0, 1.0, operating_hours_per_year=1000, power_price_per_kwh=0.10)
    )
    assert r["annual_energy_value_usd"] == pytest.approx(r["recovered_energy_kwh_h"] * 1000 * 0.10)


def test_recovery_scales_output_linearly():
    half = evaluate_process(ProcessInput(1000, 0.8, 0.25))
    full = evaluate_process(ProcessInput(1000, 0.8, 0.50))
    assert full["recovered_gas_nm3_h"] == pytest.approx(2 * half["recovered_gas_nm3_h"])


# --- result object -------------------------------------------------------


def test_result_supports_both_dict_and_attribute_access():
    r = evaluate_process(ProcessInput(1000, 0.8, 0.5))
    assert r["recovered_gas_nm3_h"] == r.recovered_gas_nm3_h


def test_result_as_dict_is_complete():
    r = evaluate_process(ProcessInput(1000, 0.8, 0.5)).as_dict()
    assert set(r) == {
        "methane_flow_nm3_h",
        "recovered_gas_nm3_h",
        "recovered_energy_kwh_h",
        "annual_recovered_gas_nm3",
        "annual_recovered_energy_mwh",
        "estimated_avoided_co2_t_y",
        "annual_energy_value_usd",
    }


def test_unknown_key_raises_keyerror():
    r = evaluate_process(ProcessInput(1000, 0.8, 0.5))
    with pytest.raises(KeyError):
        r["not_a_field"]
