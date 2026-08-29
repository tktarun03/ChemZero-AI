"""Shared fixtures.

Tests never touch the checked-in ``data/`` artifacts: each fixture builds its
own small dataset and model in a temporary directory, so the suite passes on a
fresh clone where nothing has been generated yet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.generate_data import generate_synthetic_data
from src.train_model import train_and_evaluate


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """A small reproducible synthetic dataset."""
    return generate_synthetic_data(n=300, seed=0)


@pytest.fixture(scope="session")
def trained(sample_df: pd.DataFrame):
    """A model trained on the small dataset, plus its metrics.

    Kept small (20 trees, no CV, no baseline ladder) so the whole suite stays
    fast. The ladder is exercised separately in test_baselines.py.
    """
    return train_and_evaluate(sample_df, n_estimators=20, cv_folds=0, run_baselines=False)


@pytest.fixture(scope="session")
def model(trained):
    """Just the fitted estimator."""
    return trained[0]


@pytest.fixture
def base_inputs() -> dict[str, float]:
    """A mid-range operating point covering every model feature."""
    return {
        "flow_nm3_h": 1500.0,
        "pressure_bar": 6.0,
        "temperature_c": 55.0,
        "methane_fraction": 0.80,
        "co2_fraction": 0.08,
        "h2s_ppm": 100.0,
        "availability": 0.95,
        "power_price_per_kwh": 0.10,
    }
