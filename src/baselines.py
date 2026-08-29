"""Baseline model comparison.

The point of this module is to make one question unavoidable: **is the
complicated model earning its keep?**

Reaching for a random forest without checking a linear fit first is the most
common self-inflicted wound in applied ML. On this repo's synthetic data the
honest answer is *no* - the data-generating process is linear plus a ``log1p``
term, so linear regression beats the forest. The ladder below surfaces that
every time you train, instead of leaving it for someone to discover later.

Every model is scored the same way, with cross-validation on the full dataset,
so the numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, TARGET

__all__ = ["BaselineScore", "build_baselines", "compare_baselines"]


@dataclass(frozen=True)
class BaselineScore:
    """Cross-validated score for one candidate model."""

    name: str
    cv_r2_mean: float
    cv_r2_std: float
    note: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
            "note": self.note,
        }


def build_baselines(seed: int = 42) -> dict[str, tuple[object, str]]:
    """Return the candidate ladder as ``{name: (estimator, note)}``.

    Ordered simplest to most complex, which is the order they should be tried
    in. Ridge and linear regression are scaled because regularisation is not
    scale-invariant and the features span very different magnitudes.
    """
    from src.train_model import build_model  # local import avoids a cycle

    return {
        "Mean (no model)": (
            DummyRegressor(strategy="mean"),
            "Predicts the training mean. R2 near 0 by construction - the floor "
            "any real model must clear.",
        ),
        "Linear regression": (
            make_pipeline(StandardScaler(), LinearRegression()),
            "Eight coefficients and an intercept. Nearly the true model here, "
            "since the target was generated from a linear formula.",
        ),
        "Ridge (alpha=1)": (
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Linear regression with L2 regularisation - insurance against "
            "collinearity that this dataset does not actually need.",
        ),
        "Random forest": (
            build_model(seed=seed),
            "Hundreds of trees. Flexible enough for interactions and "
            "non-linearity, at the cost of interpretability and size.",
        ),
    }


def compare_baselines(df: pd.DataFrame, cv_folds: int = 5, seed: int = 42) -> list[BaselineScore]:
    """Cross-validate every candidate on the same data and folds.

    Args:
        df: Dataset containing the feature and target columns.
        cv_folds: Number of cross-validation folds. Must be at least 2.
        seed: Seed passed to the forest.

    Returns:
        Scores sorted best-first. A model that cannot beat "Mean (no model)"
        has learned nothing.

    Raises:
        ValueError: If ``cv_folds`` is less than 2.
    """
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be at least 2, got {cv_folds}")

    X, y = df[FEATURES], df[TARGET]
    scores: list[BaselineScore] = []

    for name, (estimator, note) in build_baselines(seed).items():
        cv = cross_val_score(estimator, X, y, cv=cv_folds, scoring="r2")
        scores.append(
            BaselineScore(
                name=name,
                cv_r2_mean=round(float(cv.mean()), 4),
                cv_r2_std=round(float(cv.std()), 4),
                note=note,
            )
        )

    scores.sort(key=lambda score: score.cv_r2_mean, reverse=True)
    return scores
