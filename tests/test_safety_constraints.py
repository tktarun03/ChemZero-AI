"""Tests for the educational input screening."""

from __future__ import annotations

import pytest

from src.safety_constraints import (
    H2S_ELEVATED_PPM,
    H2S_NOTICEABLE_PPM,
    HIGH_CO2_FRACTION,
    HIGH_PRESSURE_BAR,
    LOW_METHANE_FRACTION,
    Severity,
    assess_demo_constraints,
    evaluate_demo_safety,
)


# --- rejection of impossible inputs --------------------------------------


@pytest.mark.parametrize("pressure", [0.0, -1.0])
def test_non_positive_pressure_rejected(pressure):
    assert evaluate_demo_safety(pressure, 10.0, 0.8).is_valid is False


@pytest.mark.parametrize("methane", [-0.1, 1.1])
def test_methane_fraction_out_of_range_rejected(methane):
    assert evaluate_demo_safety(5.0, 10.0, methane).is_valid is False


def test_negative_h2s_rejected():
    assert evaluate_demo_safety(5.0, -1.0, 0.8).is_valid is False


def test_co2_out_of_range_rejected_when_supplied():
    assert evaluate_demo_safety(5.0, 10.0, 0.8, co2_fraction=1.5).is_valid is False


def test_co2_is_optional():
    assert evaluate_demo_safety(5.0, 10.0, 0.8).is_valid is True


def test_rejected_input_carries_no_flags():
    """A rejected point was never screened, so it must not imply anything."""
    assert evaluate_demo_safety(0.0, 10.0, 0.8).flags == []


# --- educational flags ---------------------------------------------------


def test_clean_midrange_point_has_no_flags():
    result = evaluate_demo_safety(6.0, 10.0, 0.80, 0.05)
    assert result.is_valid and result.flags == []
    assert result.highest_severity is None


def test_elevated_h2s_flagged_as_serious():
    result = evaluate_demo_safety(6.0, H2S_ELEVATED_PPM, 0.80, 0.05)
    assert result.highest_severity is Severity.SERIOUS


def test_moderate_h2s_flagged_as_warning_only():
    result = evaluate_demo_safety(6.0, H2S_NOTICEABLE_PPM, 0.80, 0.05)
    assert result.highest_severity is Severity.WARNING


def test_h2s_flag_is_not_double_counted():
    """The elevated and noticeable bands are exclusive, not additive."""
    result = evaluate_demo_safety(6.0, 900.0, 0.80, 0.05)
    h2s_flags = [f for f in result.flags if "H₂S" in f.title]
    assert len(h2s_flags) == 1


def test_high_pressure_flagged():
    result = evaluate_demo_safety(HIGH_PRESSURE_BAR, 10.0, 0.80, 0.05)
    assert any("pressure" in f.title.lower() for f in result.flags)


def test_high_co2_flagged():
    result = evaluate_demo_safety(6.0, 10.0, 0.80, HIGH_CO2_FRACTION)
    assert any("CO₂" in f.title for f in result.flags)


def test_lean_gas_flagged_as_info():
    result = evaluate_demo_safety(6.0, 10.0, LOW_METHANE_FRACTION, 0.05)
    assert any(f.severity is Severity.INFO for f in result.flags)


def test_highest_severity_wins_over_lesser_flags():
    """Serious must outrank the info/warning flags raised alongside it."""
    result = evaluate_demo_safety(18.0, 800.0, 0.35, 0.20)
    assert len(result.flags) > 1
    assert result.highest_severity is Severity.SERIOUS


def test_flags_are_valid_but_never_an_approval():
    """A flagged point stays valid: flags teach, they do not gate."""
    result = evaluate_demo_safety(18.0, 800.0, 0.35, 0.20)
    assert result.is_valid is True


# --- backward compatibility ----------------------------------------------


def test_legacy_wrapper_returns_a_tuple():
    ok, message = assess_demo_constraints(6.0, 50.0, 0.8)
    assert ok is True
    assert "educational" in message.lower()


def test_legacy_wrapper_rejects_bad_pressure():
    ok, message = assess_demo_constraints(0.0, 50.0, 0.8)
    assert ok is False
    assert "pressure" in message.lower()
