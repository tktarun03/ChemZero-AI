"""Tests for the baseline ladder.

The ladder exists to answer one question honestly, so these tests pin that
answer: on this synthetic data a linear model must beat the random forest,
because the target was generated from a nearly linear formula. If that ever
stops being true, either the generator changed or the comparison broke - and
either way someone should look.
"""

from __future__ import annotations

import pytest

from src.baselines import build_baselines, compare_baselines


@pytest.fixture(scope="module")
def ladder(sample_df):
    """The ladder computed once - cross-validating four models is the slow part."""
    return compare_baselines(sample_df, cv_folds=3)


@pytest.fixture(scope="module")
def by_name(ladder):
    return {score.name: score for score in ladder}


def test_every_candidate_is_scored(ladder):
    assert len(ladder) == len(build_baselines())


def test_sorted_best_first(ladder):
    values = [score.cv_r2_mean for score in ladder]
    assert values == sorted(values, reverse=True)


def test_dummy_scores_near_zero(by_name):
    """The mean predictor defines R2 = 0 by construction."""
    assert by_name["Mean (no model)"].cv_r2_mean == pytest.approx(0.0, abs=0.15)


def test_real_models_beat_the_dummy(by_name):
    dummy = by_name["Mean (no model)"].cv_r2_mean
    for name in ("Linear regression", "Ridge (alpha=1)", "Random forest"):
        assert by_name[name].cv_r2_mean > dummy


def test_linear_beats_the_forest_on_this_data(by_name):
    """The repo's central teaching claim, pinned as a test.

    The generator builds the target from a linear combination plus a log term,
    so linear regression is close to the true model and the forest cannot win.
    """
    assert by_name["Linear regression"].cv_r2_mean > by_name["Random forest"].cv_r2_mean


def test_every_candidate_carries_an_explanatory_note(ladder):
    for score in ladder:
        assert score.note and len(score.note) > 20


def test_std_is_never_negative(ladder):
    assert all(score.cv_r2_std >= 0 for score in ladder)


@pytest.mark.parametrize("folds", [0, 1, -3])
def test_too_few_folds_raises(sample_df, folds):
    with pytest.raises(ValueError, match="at least 2"):
        compare_baselines(sample_df, cv_folds=folds)


def test_as_dict_is_json_ready(ladder):
    import json

    payload = [score.as_dict() for score in ladder]
    assert json.loads(json.dumps(payload)) == payload
