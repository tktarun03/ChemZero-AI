"""Split conformal prediction: turning point estimates into honest intervals.

A point prediction of "66.6% recovery" invites more confidence than any model
has earned. This module attaches an interval to it.

**How split conformal works.** Hold out a calibration set the model never saw.
Measure the absolute residual on every calibration point. Take the
(1 - alpha) quantile of those residuals, call it ``q``. Then for any new input,
``[prediction - q, prediction + q]`` contains the true value with probability at
least 1 - alpha.

**Why it is worth teaching.** The guarantee is *distribution-free* and
*model-agnostic*: it needs no assumption of normal errors, and it works
identically for a linear model or a forest. The only requirement is
exchangeability - calibration data drawn like future data. That assumption is
exactly what breaks when a model meets an operating regime it was never
trained on, which makes it a precise, honest way to state a model's limits.

Reference: Vovk et al., *Algorithmic Learning in a Random World* (2005);
Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import FEATURES

__all__ = ["ConformalCalibration", "calibrate", "interval_for", "empirical_coverage"]


@dataclass(frozen=True)
class ConformalCalibration:
    """The single number that turns predictions into intervals.

    Attributes:
        quantile: Half-width of the interval, in target units.
        alpha: Miscoverage rate; 0.1 means a 90% interval.
        n_calibration: How many points produced the quantile.
    """

    quantile: float
    alpha: float
    n_calibration: int

    @property
    def confidence(self) -> float:
        """Nominal coverage, e.g. 0.9 for a 90% interval."""
        return 1.0 - self.alpha

    def as_dict(self) -> dict[str, float | int]:
        return {
            "quantile": self.quantile,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
        }


def calibrate(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Compute the conformal quantile from a held-out calibration set.

    Args:
        model: A fitted estimator. It must NOT have seen ``X_cal`` in training,
            or the interval will be too narrow.
        X_cal: Calibration features.
        y_cal: Calibration targets.
        alpha: Miscoverage rate. 0.1 gives a 90% interval.

    Returns:
        The calibration to use with :func:`interval_for`.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1) or the calibration set is
            too small to support the requested level.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(X_cal)
    # The finite-sample guarantee needs ceil((n+1)(1-alpha)) <= n, i.e. the
    # requested quantile must actually exist within the calibration sample.
    if n < 1 / alpha - 1:
        raise ValueError(
            f"Calibration set of {n} points is too small for alpha={alpha}; "
            f"need at least {int(np.ceil(1 / alpha - 1))}."
        )

    residuals = np.abs(np.asarray(y_cal) - model.predict(X_cal[FEATURES]))

    # The finite-sample correction: use the ceil((n+1)(1-alpha))-th smallest
    # residual rather than the plain empirical quantile. Without it the interval
    # is very slightly too narrow on small calibration sets.
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(rank, n)
    quantile = float(np.sort(residuals)[rank - 1])

    return ConformalCalibration(
        quantile=round(quantile, 6), alpha=alpha, n_calibration=n
    )


def interval_for(
    prediction: float,
    calibration: ConformalCalibration,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float]:
    """Turn a point prediction into a conformal interval.

    Args:
        prediction: The point prediction.
        calibration: Output of :func:`calibrate`.
        lower_bound: Physical floor to clip to (recovery cannot go below 0).
        upper_bound: Physical ceiling to clip to (recovery cannot exceed 1).

    Returns:
        ``(low, high)``, clipped to the physical bounds.

    Note:
        Clipping to physical bounds can only *shrink* the interval, so the
        coverage guarantee still holds whenever the true value also respects
        those bounds - which, for a recovery fraction, it must.
    """
    low = max(prediction - calibration.quantile, lower_bound)
    high = min(prediction + calibration.quantile, upper_bound)
    return low, high


def empirical_coverage(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    calibration: ConformalCalibration,
) -> float:
    """Fraction of test points whose true value falls inside the interval.

    This is the honesty check on the whole procedure: it should land near
    ``calibration.confidence``. A materially lower number means the
    exchangeability assumption is violated.

    Args:
        model: The fitted estimator.
        X_test: Test features, disjoint from the calibration set.
        y_test: True test targets.
        calibration: Output of :func:`calibrate`.

    Returns:
        Empirical coverage in [0, 1].
    """
    predictions = model.predict(X_test[FEATURES])
    truth = np.asarray(y_test)
    inside = np.abs(truth - predictions) <= calibration.quantile
    return round(float(inside.mean()), 4)
