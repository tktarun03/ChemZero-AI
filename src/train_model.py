"""Train the recovery-fraction regressor on the synthetic dataset.

The metrics this prints describe how well the model reproduces the *invented*
correlation in ``generate_data.py`` - nothing more. A strong R2 here is not
evidence about any real process; it only says the model learned the formula we
wrote. Treat the numbers as a demonstration of the training workflow.

Run as either::

    python src/train_model.py
    python -m src.train_model --n-estimators 400 --test-size 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

# Allow `python src/train_model.py` as well as `python -m src.train_model`.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET  # noqa: E402

__all__ = [
    "TrainingMetrics",
    "MissingColumnsError",
    "load_dataset",
    "build_model",
    "train_and_evaluate",
    "main",
]

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MIN_SAMPLES_LEAF = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 5


@dataclass(frozen=True)
class TrainingMetrics:
    """Everything worth recording about one training run."""

    mae: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float
    n_train: int
    n_test: int
    n_estimators: int
    min_samples_leaf: int
    test_size: float
    seed: int
    feature_importances: dict[str, float]
    trained_at_utc: str


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
            f"Dataset not found at {path}. Generate it first with: "
            "python src/generate_data.py"
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
    """Construct the (unfitted) estimator.

    A random forest is used because it needs no feature scaling and is hard to
    misconfigure - suitable for a teaching repo. It is not a considered choice
    for this problem, which is almost linear by construction.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> tuple[RandomForestRegressor, TrainingMetrics]:
    """Fit the model on a train split and score it on the held-out split.

    Args:
        df: Dataset containing the feature and target columns.
        test_size: Held-out fraction.
        seed: Seed for the split and the forest.
        n_estimators: Number of trees.
        min_samples_leaf: Minimum samples per leaf.
        cv_folds: Cross-validation folds; pass 0 or 1 to skip CV.

    Returns:
        The fitted model and its :class:`TrainingMetrics`.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size, random_state=seed
    )

    model = build_model(n_estimators, min_samples_leaf, seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Cross-validation gives a stability read the single split cannot: a wide
    # spread across folds means the held-out score was partly luck.
    if cv_folds and cv_folds > 1:
        cv_scores = cross_val_score(
            build_model(n_estimators, min_samples_leaf, seed),
            df[FEATURES],
            df[TARGET],
            cv=cv_folds,
            scoring="r2",
        )
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean = cv_std = float("nan")

    metrics = TrainingMetrics(
        mae=round(float(mean_absolute_error(y_test, pred)), 4),
        rmse=round(float(root_mean_squared_error(y_test, pred)), 4),
        r2=round(float(r2_score(y_test, pred)), 4),
        cv_r2_mean=round(cv_mean, 4),
        cv_r2_std=round(cv_std, 4),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        test_size=test_size,
        seed=seed,
        feature_importances={
            name: round(float(value), 4)
            for name, value in sorted(
                zip(FEATURES, model.feature_importances_),
                key=lambda pair: pair[1],
                reverse=True,
            )
        },
        trained_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return model, metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train the recovery-fraction model.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="input CSV")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="model path")
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH, help="metrics JSON")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    args.metrics_out.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    top_drivers = ", ".join(list(metrics.feature_importances)[:3])
    print(f"Rows         : {len(df)} ({metrics.n_train} train / {metrics.n_test} test)")
    print(f"MAE          : {metrics.mae}")
    print(f"RMSE         : {metrics.rmse}")
    print(f"R2           : {metrics.r2}")
    print(f"CV R2        : {metrics.cv_r2_mean} +/- {metrics.cv_r2_std} ({args.cv_folds} folds)")
    print(f"Top drivers  : {top_drivers}")
    print(f"Saved model  : {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")
    print("Reminder     : these scores describe synthetic data, not a real process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
