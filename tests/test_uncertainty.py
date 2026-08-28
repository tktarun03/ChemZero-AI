"""Tests for split conformal prediction.

The headline property is coverage: a 90% interval should contain the truth
about 90% of the time. That is checked here on held-out data, because a
calibration procedure that is never coverage-tested is just arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import FEATURES, TARGET
from src.uncertainty import calibrate, empirical_coverage, interval_for


@pytest.fixture(scope="module")
def split(sample_df):
    """Train / calibration / test split, disjoint by construction."""
    train = sample_df.iloc[:150]
    calibration = sample_df.iloc[150:220]
    test = sample_df.iloc[220:]
    return train, calibration, test


@pytest.fixture(scope="module")
def fitted(split):
    from src.train_model import build_model

    train, _, _ = split
    return build_model(n_estimators=20).fit(train[FEATURES], train[TARGET])


def test_quantile_is_positive(fitted, split):
    _, cal, _ = split
    assert calibrate(fitted, cal, cal[TARGET]).quantile > 0


def test_confidence_is_one_minus_alpha(fitted, split):
    _, cal, _ = split
    assert calibrate(fitted, cal, cal[TARGET], alpha=0.05).confidence == pytest.approx(0.95)


def test_coverage_lands_near_the_target(fitted, split):
    """The whole point of the procedure."""
    _, cal, test = split
    calibration = calibrate(fitted, cal, cal[TARGET], alpha=0.1)
    coverage = empirical_coverage(fitted, test, test[TARGET], calibration)
    assert 0.80 <= coverage <= 1.0


def test_tighter_alpha_widens_the_interval(fitted, split):
    """Demanding more confidence must cost interval width, never less.

    Both alphas are chosen to be supportable by this calibration set: a 99%
    interval would need ~99 points and is rejected outright - see
    test_calibration_set_too_small_raises.
    """
    _, cal, _ = split
    wide = calibrate(fitted, cal, cal[TARGET], alpha=0.05)
    narrow = calibrate(fitted, cal, cal[TARGET], alpha=0.2)
    assert wide.quantile > narrow.quantile


def test_interval_is_centred_on_the_prediction(fitted, split):
    _, cal, _ = split
    calibration = calibrate(fitted, cal, cal[TARGET])
    low, high = interval_for(0.5, calibration)
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-9)


def test_interval_clips_to_physical_bounds(fitted, split):
    """A recovery fraction cannot be negative or above 1, whatever the maths says."""
    _, cal, _ = split
    calibration = calibrate(fitted, cal, cal[TARGET])
    assert interval_for(0.0, calibration)[0] == 0.0
    assert interval_for(1.0, calibration)[1] == 1.0


def test_interval_respects_custom_bounds(fitted, split):
    _, cal, _ = split
    calibration = calibrate(fitted, cal, cal[TARGET])
    low, high = interval_for(0.5, calibration, lower_bound=0.49, upper_bound=0.51)
    assert (low, high) == (0.49, 0.51)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_invalid_alpha_raises(fitted, split, alpha):
    _, cal, _ = split
    with pytest.raises(ValueError, match="alpha must be in"):
        calibrate(fitted, cal, cal[TARGET], alpha=alpha)


def test_calibration_set_too_small_raises(fitted, split):
    """A 99% interval cannot be certified from five points."""
    _, cal, _ = split
    with pytest.raises(ValueError, match="too small"):
        calibrate(fitted, cal.head(5), cal[TARGET].head(5), alpha=0.01)


def test_records_the_calibration_size(fitted, split):
    _, cal, _ = split
    assert calibrate(fitted, cal, cal[TARGET]).n_calibration == len(cal)


def test_finite_sample_correction_is_conservative(fitted, split):
    """The rank-based quantile must be >= the plain empirical quantile.

    Using the plain quantile would make intervals slightly too narrow on small
    calibration sets, which is exactly where the guarantee matters most.
    """
    _, cal, _ = split
    calibration = calibrate(fitted, cal, cal[TARGET], alpha=0.1)
    residuals = np.abs(np.asarray(cal[TARGET]) - fitted.predict(cal[FEATURES]))
    assert calibration.quantile >= np.quantile(residuals, 0.9) - 1e-9


def test_as_dict_is_json_ready(fitted, split):
    import json

    _, cal, _ = split
    payload = calibrate(fitted, cal, cal[TARGET]).as_dict()
    assert json.loads(json.dumps(payload)) == payload
