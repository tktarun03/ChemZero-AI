"""Load the trained model and predict recovery fraction for new operating points.

This is the piece that connects training to the dashboard: without it the model
would be trained and then never used. Predictions are clamped to [0, 1] because
a regressor has no idea that a recovery fraction is bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config import FEATURES, METRICS_PATH, MODEL_PATH
from src.uncertainty import ConformalCalibration, interval_for

__all__ = [
    "ModelNotTrainedError",
    "load_model",
    "load_metrics",
    "calibration_from_metrics",
    "predict_recovery",
    "predict_with_interval",
    "predict_recovery_batch",
    "sweep_feature",
]


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the serialised model is missing.

    Carries the exact command needed to fix it, so the dashboard can show a
    useful message instead of a stack trace.
    """


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load the fitted estimator from disk.

    Args:
        path: Path to the joblib file.

    Returns:
        The deserialised scikit-learn estimator.

    Raises:
        ModelNotTrainedError: If the file does not exist.
    """
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model at {path}. Train one first with: "
            "python src/train_model.py"
        )

    import joblib  # imported lazily so `--help` paths stay fast

    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> dict[str, Any] | None:
    """Load the metrics JSON written by training, or None if absent/unreadable.

    Metrics are informational, so a missing or corrupt file degrades to None
    rather than raising - the dashboard simply hides that panel.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _to_frame(inputs: Mapping[str, float]) -> pd.DataFrame:
    """Build a one-row frame with columns in the exact training order.

    Raises:
        KeyError: If any required feature is missing.
    """
    missing = [name for name in FEATURES if name not in inputs]
    if missing:
        raise KeyError(f"Missing required features: {missing}")
    return pd.DataFrame([[float(inputs[name]) for name in FEATURES]], columns=FEATURES)


def predict_recovery(model: Any, inputs: Mapping[str, float]) -> float:
    """Predict the recovery fraction for a single operating point.

    Args:
        model: A fitted estimator from :func:`load_model`.
        inputs: Mapping containing every name in :data:`src.config.FEATURES`.

    Returns:
        The predicted recovery fraction, clamped to [0, 1].

    Raises:
        KeyError: If a required feature is missing.
    """
    prediction = float(model.predict(_to_frame(inputs))[0])
    return min(max(prediction, 0.0), 1.0)


def predict_recovery_batch(model: Any, df: pd.DataFrame) -> pd.Series:
    """Predict recovery fractions for many rows at once.

    Args:
        model: A fitted estimator.
        df: Frame containing at least the feature columns.

    Returns:
        A Series of predictions clamped to [0, 1], indexed like ``df``.

    Raises:
        KeyError: If a required feature column is missing.
    """
    missing = [name for name in FEATURES if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    predictions = model.predict(df[FEATURES])
    return pd.Series(predictions, index=df.index).clip(0.0, 1.0)


def sweep_feature(
    model: Any,
    base_inputs: Mapping[str, float],
    feature: str,
    values: list[float],
) -> pd.DataFrame:
    """Vary one feature over a range, holding the others fixed.

    This is a one-at-a-time sensitivity sweep. It shows how the *model* responds,
    which is not the same as how a plant would respond, and it deliberately
    ignores interactions between variables.

    Args:
        model: A fitted estimator.
        base_inputs: The operating point to hold fixed.
        feature: Which feature to vary; must be a known feature name.
        values: The values to sweep through.

    Returns:
        A frame with the swept ``feature`` column and a
        ``predicted_recovery_fraction`` column.

    Raises:
        ValueError: If ``feature`` is not a known feature name.
        KeyError: If ``base_inputs`` is missing a required feature.
    """
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}. Expected one of: {FEATURES}")

    rows = [{**base_inputs, feature: value} for value in values]
    frame = pd.DataFrame(rows, columns=FEATURES)
    return pd.DataFrame(
        {
            feature: frame[feature],
            "predicted_recovery_fraction": predict_recovery_batch(model, frame),
        }
    )


def calibration_from_metrics(metrics: dict[str, Any] | None) -> ConformalCalibration | None:
    """Rebuild the conformal calibration recorded during training.

    The quantile lives in ``model_metrics.json`` rather than inside the model
    file, which keeps ``recovery_model.joblib`` a plain scikit-learn estimator
    that anything can load.

    Args:
        metrics: Output of :func:`load_metrics`, or None.

    Returns:
        The calibration, or None when training recorded no conformal block
        (for example when ``--calibration-size 0`` was used).
    """
    if not metrics:
        return None
    block = metrics.get("conformal") or {}
    if "quantile" not in block:
        return None
    return ConformalCalibration(
        quantile=float(block["quantile"]),
        alpha=float(block["alpha"]),
        n_calibration=int(block["n_calibration"]),
    )


def predict_with_interval(
    model: Any,
    inputs: Mapping[str, float],
    calibration: ConformalCalibration | None,
) -> tuple[float, float | None, float | None]:
    """Predict a recovery fraction with a conformal interval.

    Args:
        model: A fitted estimator from :func:`load_model`.
        inputs: Mapping containing every name in :data:`src.config.FEATURES`.
        calibration: From :func:`calibration_from_metrics`; None disables the
            interval.

    Returns:
        ``(prediction, low, high)``. ``low`` and ``high`` are None when no
        calibration is available - an unknown interval is reported as unknown
        rather than silently as zero width.

    Raises:
        KeyError: If a required feature is missing.
    """
    prediction = predict_recovery(model, inputs)
    if calibration is None:
        return prediction, None, None
    low, high = interval_for(prediction, calibration)
    return prediction, low, high
