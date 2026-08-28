"""Tests for synthetic data generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.generate_data import generate_synthetic_data, main


def test_row_count_and_columns(sample_df: pd.DataFrame):
    assert len(sample_df) == 300
    assert list(sample_df.columns) == [*FEATURES, TARGET]


def test_is_reproducible_for_a_given_seed():
    a = generate_synthetic_data(n=50, seed=7)
    b = generate_synthetic_data(n=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = generate_synthetic_data(n=50, seed=1)
    b = generate_synthetic_data(n=50, seed=2)
    assert not a[TARGET].equals(b[TARGET])


@pytest.mark.parametrize("n", [0, -10])
def test_non_positive_row_count_raises(n):
    with pytest.raises(ValueError, match="must be positive"):
        generate_synthetic_data(n=n)


def test_target_stays_inside_its_physical_bounds(sample_df: pd.DataFrame):
    assert sample_df[TARGET].between(0.05, 0.95).all()


def test_no_missing_values(sample_df: pd.DataFrame):
    assert not sample_df.isna().any().any()


@pytest.mark.parametrize(
    "column, low, high",
    [
        ("flow_nm3_h", 200, 5000),
        ("pressure_bar", 1.2, 18.0),
        ("methane_fraction", 0.55, 0.95),
        ("co2_fraction", 0.01, 0.20),
        ("h2s_ppm", 0, 1500),
        ("availability", 0.80, 0.995),
    ],
)
def test_features_stay_in_their_documented_ranges(sample_df, column, low, high):
    assert sample_df[column].between(low, high).all()


def test_target_correlates_with_methane_as_designed(sample_df: pd.DataFrame):
    """Methane carries the largest positive coefficient in the generator."""
    assert sample_df["methane_fraction"].corr(sample_df[TARGET]) > 0.3


def test_target_falls_with_co2_as_designed(sample_df: pd.DataFrame):
    assert sample_df["co2_fraction"].corr(sample_df[TARGET]) < 0


def test_cli_writes_a_readable_csv(tmp_path):
    out = tmp_path / "nested" / "data.csv"
    exit_code = main(["--rows", "25", "--seed", "3", "--out", str(out)])
    assert exit_code == 0
    written = pd.read_csv(out)
    assert len(written) == 25
    assert list(written.columns) == [*FEATURES, TARGET]
