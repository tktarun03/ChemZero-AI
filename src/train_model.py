"""Train the recovery-fraction regressor on the synthetic dataset.

The metrics this prints describe how well the model reproduces the *invented*
correlation in ``generate_data.py`` - nothing more. A strong R2 here is not
evidence about any real process; it only says the model learned the formula we
wrote. Treat the numbers as a demonstration of the training workflow.

Three things here exist to keep that demonstration honest:

* a **baseline ladder** (:mod:`src.baselines`), so "is the complex model worth
  it?" is answered every run rather than assumed;
* **permutation importance** alongside the forest's built-in impurity
  importance, because the built-in one is biased toward continuous features;
* **split conformal calibration** (:mod:`src.uncertainty`), so predictions ship
  with an interval instead of false precision.

The data is split three ways - train / calibration / test. The calibration
split exists only for conformal prediction and is never trained on.

Run as either::

    python src/train_model.py
    python -m src.train_model --model linear --n-estimators 400
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow `python src/train_model.py` as well as `python -m src.train_model`.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import compare_baselines
from src.config import DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET
from src.uncertainty import calibrate, empirical_coverage

__all__ = [
    "MODEL_KINDS",
    "MissingColumnsError",
    "TrainingMetrics",
    "build_estimator",
    "build_model",
    "load_dataset",
    "main",
    "train_and_evaluate",
]

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MIN_SAMPLES_LEAF = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 5
DEFAULT_ALPHA = 0.1

MODEL_KINDS = ("rf", "linear", "ridge")


@dataclass(frozen=True)
class TrainingMetrics:
    """Everything worth recording about one training run."""

    model_kind: str
    mae: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float
    n_train: int
    n_calibration: int
    n_test: int
    n_estimators: int
    min_samples_leaf: int
    test_size: float
    seed: int
    permutation_importances: dict[str, float]
    trained_at_utc: str
    impurity_importances: dict[str, float] = field(default_factory=dict)
    baselines: list[dict[str, Any]] = field(default_factory=list)
    conformal: dict[str, Any] = field(default_factory=dict)


class MissingColumnsError(ValueError):
    """Raised when the dataset lacks the columns the model needs."""


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the training CSV and verify it has the expected columns.

    Args:
        path: CSV to load.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist, with a hint on how to
            create it.
        MissingColumnsError: If required feature/target columns are absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it first with: python src/generate_data.py"
        )

    df = pd.read_csv(path)
    missing = [column for column in [*FEATURES, TARGET] if column not in df.columns]
    if missing:
        raise MissingColumnsError(f"Dataset {path} is missing columns: {missing}")
    return df


def build_model(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    seed: int = DEFAULT_SEED,
) -> RandomForestRegressor:
    """Construct the (unfitted) random forest.

    A forest needs no feature scaling and is hard to misconfigure, which makes
    it a convenient teaching default. It is *not* the best choice for this
    dataset - see :mod:`src.baselines`, where linear regression beats it,
    because the target was generated from a nearly linear formula.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def build_estimator(
    kind: str = "rf",
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    seed: int = DEFAULT_SEED,
):
    """Construct an unfitted estimator by name.

    Args:
        kind: One of :data:`MODEL_KINDS`.
        n_estimators: Trees, for the forest only.
        min_samples_leaf: Leaf size, for the forest only.
        seed: Random seed.

    Returns:
        An unfitted scikit-learn estimator or pipeline.

    Raises:
        ValueError: If ``kind`` is not recognised.
    """
    if kind == "rf":
        return build_model(n_estimators, min_samples_leaf, seed)
    if kind == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(f"Unknown model kind {kind!r}. Expected one of: {MODEL_KINDS}")


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    cv_folds: int = DEFAULT_CV_FOLDS,
    model_kind: str = "rf",
    calibration_size: float = DEFAULT_CALIBRATION_SIZE,
    alpha: float = DEFAULT_ALPHA,
    run_baselines: bool = True,
) -> tuple[Any, TrainingMetrics]:
    """Fit, calibrate, and score a model.

    The data is split three ways: the model trains on the first split, the
    conformal quantile is computed on the second (never trained on), and every
    reported score comes from the third.

    Args:
        df: Dataset containing the feature and target columns.
        test_size: Held-out test fraction of the whole dataset.
        seed: Seed for splits and the forest.
        n_estimators: Number of trees (forest only).
        min_samples_leaf: Minimum samples per leaf (forest only).
        cv_folds: Cross-validation folds; pass 0 or 1 to skip CV.
        model_kind: One of :data:`MODEL_KINDS`.
        calibration_size: Fraction of the non-test data reserved for conformal
            calibration. Pass 0 to skip calibration entirely.
        alpha: Conformal miscoverage rate; 0.1 gives a 90% interval.
        run_baselines: Whether to run the baseline ladder (the slow part).

    Returns:
        The fitted model and its :class:`TrainingMetrics`.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size, random_state=seed
    )

    if calibration_size > 0:
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_rest, y_rest, test_size=calibration_size, random_state=seed
        )
    else:
        X_train, y_train = X_rest, y_rest
        X_cal = y_cal = None

    model = build_estimator(model_kind, n_estimators, min_samples_leaf, seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Cross-validation gives a stability read the single split cannot: a wide
    # spread across folds means the held-out score was partly luck.
    if cv_folds and cv_folds > 1:
        cv_scores = cross_val_score(
            build_estimator(model_kind, n_estimators, min_samples_leaf, seed),
            df[FEATURES],
            df[TARGET],
            cv=cv_folds,
            scoring="r2",
        )
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean = cv_std = float("nan")

    # Permutation importance is measured on held-out data and works for any
    # estimator. Unlike the forest's impurity importance it does not inflate
    # continuous features that the target never depended on.
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed)
    permutation_importances = {
        name: round(float(value), 4)
        for name, value in sorted(
            zip(FEATURES, perm.importances_mean, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }

    impurity_importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        impurity_importances = {
            name: round(float(value), 4)
            for name, value in sorted(
                zip(FEATURES, model.feature_importances_, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        }

    conformal: dict[str, Any] = {}
    if X_cal is not None and len(X_cal) > 0:
        calibration = calibrate(model, X_cal, y_cal, alpha=alpha)
        conformal = {
            **calibration.as_dict(),
            "confidence": calibration.confidence,
            "empirical_coverage": empirical_coverage(model, X_test, y_test, calibration),
        }

    baselines = (
        [score.as_dict() for score in compare_baselines(df, max(cv_folds, 2), seed)]
        if run_baselines
        else []
    )

    metrics = TrainingMetrics(
        model_kind=model_kind,
        mae=round(float(mean_absolute_error(y_test, pred)), 4),
        rmse=round(float(root_mean_squared_error(y_test, pred)), 4),
        r2=round(float(r2_score(y_test, pred)), 4),
        cv_r2_mean=round(cv_mean, 4),
        cv_r2_std=round(cv_std, 4),
        n_train=len(X_train),
        n_calibration=len(X_cal) if X_cal is not None else 0,
        n_test=len(X_test),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        test_size=test_size,
        seed=seed,
        permutation_importances=permutation_importances,
        impurity_importances=impurity_importances,
        baselines=baselines,
        conformal=conformal,
        trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return model, metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train the recovery-fraction model.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="input CSV")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="model path")
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH, help="metrics JSON")
    parser.add_argument("--model", choices=MODEL_KINDS, default="rf", help="estimator to fit")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--calibration-size", type=float, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="conformal miscoverage rate (0.1 = 90%% interval)",
    )
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-baselines", action="store_true", help="skip the baseline ladder")
    args = parser.parse_args(argv)

    try:
        df = load_dataset(args.data)
    except (FileNotFoundError, MissingColumnsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model, metrics = train_and_evaluate(
        df,
        test_size=args.test_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        cv_folds=args.cv_folds,
        model_kind=args.model,
        calibration_size=args.calibration_size,
        alpha=args.alpha,
        run_baselines=not args.skip_baselines,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    args.metrics_out.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    print(f"Model        : {metrics.model_kind}")
    print(
        f"Rows         : {len(df)} ({metrics.n_train} train / "
        f"{metrics.n_calibration} calibration / {metrics.n_test} test)"
    )
    print(f"MAE          : {metrics.mae}")
    print(f"RMSE         : {metrics.rmse}")
    print(f"R2           : {metrics.r2}")
    print(f"CV R2        : {metrics.cv_r2_mean} +/- {metrics.cv_r2_std} ({args.cv_folds} folds)")

    if metrics.conformal:
        conf = metrics.conformal
        pct = int(conf["confidence"] * 100)
        print(
            f"Conformal    : +/-{conf['quantile']:.4f} for {pct}% coverage "
            f"(measured {conf['empirical_coverage']:.1%} on the test split)"
        )

    print(f"Top drivers  : {', '.join(list(metrics.permutation_importances)[:3])}")

    if metrics.baselines:
        print("\nBaseline ladder (cross-validated R2, best first):")
        best = metrics.baselines[0]["name"]
        for row in metrics.baselines:
            marker = "  <-- best" if row["name"] == best else ""
            print(
                f"  {row['name']:<20s} {row['cv_r2_mean']:+.4f} +/- {row['cv_r2_std']:.4f}{marker}"
            )
        if best != "Random forest" and metrics.model_kind == "rf":
            print(
                f"\n  NOTE: '{best}' scores higher than the random forest you just "
                "trained.\n        The target was generated from a nearly linear "
                "formula, so the\n        simpler model fits it better. Try: "
                "--model linear"
            )

    print(f"\nSaved model  : {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")
    print("Reminder     : these scores describe synthetic data, not a real process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
