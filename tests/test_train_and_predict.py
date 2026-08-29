"""Tests for training, serialisation, and prediction."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.config import FEATURES, FEATURES_NOT_IN_TARGET, TARGET
from src.predict import (
    ModelNotTrainedError,
    calibration_from_metrics,
    load_metrics,
    load_model,
    predict_recovery,
    predict_recovery_batch,
    predict_with_interval,
    sweep_feature,
)
from src.train_model import (
    MissingColumnsError,
    build_estimator,
    load_dataset,
    main,
    train_and_evaluate,
)
from src.uncertainty import calibrate

# --- training ------------------------------------------------------------


def test_metrics_are_populated(trained):
    _, metrics = trained
    assert 0.0 <= metrics.r2 <= 1.0
    assert metrics.mae > 0
    assert metrics.rmse >= metrics.mae  # RMSE can never be below MAE
    # Three-way split: train + calibration + test accounts for every row.
    assert metrics.n_train + metrics.n_calibration + metrics.n_test == 300
    assert metrics.n_calibration > 0


def test_model_beats_predicting_the_mean(trained):
    """A positive R2 is the minimum bar: the model must add something."""
    _, metrics = trained
    assert metrics.r2 > 0.0


def test_impurity_importances_cover_every_feature_and_sum_to_one(trained):
    _, metrics = trained
    assert set(metrics.impurity_importances) == set(FEATURES)
    assert sum(metrics.impurity_importances.values()) == pytest.approx(1.0, abs=1e-3)


def test_both_importance_kinds_are_sorted_descending(trained):
    _, metrics = trained
    for importances in (metrics.impurity_importances, metrics.permutation_importances):
        values = list(importances.values())
        assert values == sorted(values, reverse=True)


def test_permutation_importances_cover_every_feature(trained):
    _, metrics = trained
    assert set(metrics.permutation_importances) == set(FEATURES)


def test_methane_is_a_top_driver(trained):
    """The generator gives methane the biggest coefficient; the model should see it."""
    _, metrics = trained
    assert next(iter(metrics.permutation_importances)) == "methane_fraction"


def test_permutation_scores_absent_features_near_zero(trained):
    """The control group: variables the target formula never used.

    Permutation importance should place them at essentially zero. This is the
    check that justifies preferring it over the forest's built-in importance.
    """
    _, metrics = trained
    for feature in FEATURES_NOT_IN_TARGET:
        assert metrics.permutation_importances[feature] < 0.02


def test_linear_model_has_no_impurity_importances(sample_df):
    """Only tree models expose feature_importances_; the field stays empty."""
    _, metrics = train_and_evaluate(sample_df, model_kind="linear", cv_folds=0, run_baselines=False)
    assert metrics.impurity_importances == {}
    assert set(metrics.permutation_importances) == set(FEATURES)


def test_training_is_reproducible(sample_df):
    """Same seed, same predictions.

    Compared with a tolerance rather than exactly: the forest trains across
    threads, so the order floats are summed in can differ by the last bit.
    """
    a, _ = train_and_evaluate(sample_df, n_estimators=10, cv_folds=0)
    b, _ = train_and_evaluate(sample_df, n_estimators=10, cv_folds=0)
    row = sample_df[FEATURES].head(5)
    assert list(a.predict(row)) == pytest.approx(list(b.predict(row)))


def test_cross_validation_runs_when_requested(sample_df):
    _, metrics = train_and_evaluate(sample_df, n_estimators=10, cv_folds=3)
    assert metrics.cv_r2_mean == metrics.cv_r2_mean  # not NaN
    assert metrics.cv_r2_std >= 0


# --- dataset loading -----------------------------------------------------


def test_load_dataset_missing_file_explains_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data"):
        load_dataset(tmp_path / "absent.csv")


def test_load_dataset_rejects_wrong_columns(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(MissingColumnsError, match="missing columns"):
        load_dataset(path)


# --- prediction ----------------------------------------------------------


def test_prediction_is_a_fraction(model, base_inputs):
    assert 0.0 <= predict_recovery(model, base_inputs) <= 1.0


def test_prediction_ignores_extra_keys(model, base_inputs):
    """The dashboard passes a superset of the features; that must be fine.

    Predictions are compared with a tolerance throughout: the forest predicts
    across threads, so two identical calls can differ in the last bit.
    """
    noisy = {**base_inputs, "unused_column": 123.0}
    assert predict_recovery(model, noisy) == pytest.approx(predict_recovery(model, base_inputs))


def test_missing_feature_raises_keyerror(model, base_inputs):
    del base_inputs["methane_fraction"]
    with pytest.raises(KeyError, match="methane_fraction"):
        predict_recovery(model, base_inputs)


def test_key_order_does_not_change_the_prediction(model, base_inputs):
    """Columns are matched by position, so order-independence is worth pinning."""
    reversed_inputs = dict(reversed(list(base_inputs.items())))
    assert predict_recovery(model, reversed_inputs) == pytest.approx(
        predict_recovery(model, base_inputs)
    )


def test_richer_gas_predicts_higher_recovery(model, base_inputs):
    lean = predict_recovery(model, {**base_inputs, "methane_fraction": 0.60})
    rich = predict_recovery(model, {**base_inputs, "methane_fraction": 0.92})
    assert rich > lean


def test_more_co2_predicts_lower_recovery(model, base_inputs):
    clean = predict_recovery(model, {**base_inputs, "co2_fraction": 0.02})
    sour = predict_recovery(model, {**base_inputs, "co2_fraction": 0.19})
    assert sour < clean


def test_batch_prediction_matches_single(model, sample_df):
    batch = predict_recovery_batch(model, sample_df.head(5))
    singles = [predict_recovery(model, row) for _, row in sample_df.head(5).iterrows()]
    assert list(batch) == pytest.approx(singles)


def test_batch_prediction_preserves_the_index(model, sample_df):
    subset = sample_df.iloc[10:15]
    assert list(predict_recovery_batch(model, subset).index) == list(subset.index)


def test_batch_rejects_a_frame_without_the_features(model):
    with pytest.raises(KeyError, match="Missing required feature columns"):
        predict_recovery_batch(model, pd.DataFrame({"nope": [1, 2]}))


# --- sweeps --------------------------------------------------------------


def test_sweep_returns_one_row_per_value(model, base_inputs):
    values = [0.6, 0.7, 0.8, 0.9]
    swept = sweep_feature(model, base_inputs, "methane_fraction", values)
    assert len(swept) == len(values)
    assert list(swept["methane_fraction"]) == values


def test_sweep_predictions_stay_in_range(model, base_inputs):
    swept = sweep_feature(model, base_inputs, "h2s_ppm", [0.0, 750.0, 1500.0])
    assert swept["predicted_recovery_fraction"].between(0.0, 1.0).all()


def test_sweep_rejects_an_unknown_feature(model, base_inputs):
    with pytest.raises(ValueError, match="Unknown feature"):
        sweep_feature(model, base_inputs, "not_a_feature", [1.0])


# --- persistence ---------------------------------------------------------


def test_missing_model_explains_the_fix(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="train_model"):
        load_model(tmp_path / "absent.joblib")


def test_metrics_absent_returns_none(tmp_path):
    assert load_metrics(tmp_path / "absent.json") is None


def test_corrupt_metrics_returns_none_rather_than_raising(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_metrics(path) is None


def test_roundtrip_through_disk_preserves_predictions(model, tmp_path, base_inputs):
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert predict_recovery(load_model(path), base_inputs) == pytest.approx(
        predict_recovery(model, base_inputs)
    )


def test_cli_writes_model_and_metrics(tmp_path, sample_df):
    data = tmp_path / "data.csv"
    sample_df.to_csv(data, index=False)
    model_out, metrics_out = tmp_path / "m.joblib", tmp_path / "m.json"

    exit_code = main(
        [
            "--data",
            str(data),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
            "--n-estimators",
            "10",
            "--cv-folds",
            "0",
            "--skip-baselines",
        ]
    )

    assert exit_code == 0
    assert model_out.exists()
    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["n_estimators"] == 10
    assert set(payload["permutation_importances"]) == set(FEATURES)
    assert payload["conformal"]["quantile"] > 0


def test_cli_rejects_an_unknown_model_kind(tmp_path, sample_df, capsys):
    data = tmp_path / "data.csv"
    sample_df.to_csv(data, index=False)
    with pytest.raises(SystemExit):
        main(["--data", str(data), "--model", "not_a_model"])


def test_cli_reports_a_missing_dataset_without_traceback(tmp_path, capsys):
    exit_code = main(["--data", str(tmp_path / "absent.csv")])
    assert exit_code == 1
    assert "generate_data" in capsys.readouterr().err


# --- conformal intervals through the public API --------------------------


def test_calibration_round_trips_through_metrics(tmp_path, sample_df):
    """The dashboard rebuilds the calibration from JSON; that path must work."""
    data = tmp_path / "data.csv"
    sample_df.to_csv(data, index=False)
    metrics_out = tmp_path / "m.json"

    main(
        [
            "--data",
            str(data),
            "--model-out",
            str(tmp_path / "m.joblib"),
            "--metrics-out",
            str(metrics_out),
            "--n-estimators",
            "10",
            "--cv-folds",
            "0",
            "--skip-baselines",
        ]
    )

    calibration = calibration_from_metrics(load_metrics(metrics_out))
    assert calibration is not None
    assert calibration.quantile > 0
    assert calibration.confidence == pytest.approx(0.9)


def test_no_calibration_block_yields_no_interval(model, base_inputs):
    prediction, low, high = predict_with_interval(model, base_inputs, None)
    assert low is None and high is None
    assert 0.0 <= prediction <= 1.0


def test_interval_brackets_the_prediction(model, base_inputs, sample_df):
    calibration = calibrate(model, sample_df.head(60), sample_df[TARGET].head(60))
    prediction, low, high = predict_with_interval(model, base_inputs, calibration)
    assert low is not None and high is not None
    assert low <= prediction <= high


def test_calibration_from_metrics_handles_missing_input():
    assert calibration_from_metrics(None) is None
    assert calibration_from_metrics({}) is None
    assert calibration_from_metrics({"conformal": {}}) is None


# --- estimator selection and optional stages -----------------------------


@pytest.mark.parametrize("kind", ["rf", "linear", "ridge"])
def test_build_estimator_accepts_every_declared_kind(kind):
    assert build_estimator(kind) is not None


def test_build_estimator_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_estimator("gradient_boosted_wishful_thinking")


def test_calibration_can_be_disabled(sample_df):
    """--calibration-size 0 trades the interval for more training data."""
    _, metrics = train_and_evaluate(
        sample_df, calibration_size=0, cv_folds=0, run_baselines=False, n_estimators=10
    )
    assert metrics.conformal == {}
    assert metrics.n_calibration == 0
    assert metrics.n_train + metrics.n_test == 300


def test_cli_prints_the_baseline_ladder(tmp_path, sample_df, capsys):
    """The ladder is the point of the run; it must reach stdout."""
    data = tmp_path / "data.csv"
    sample_df.to_csv(data, index=False)

    exit_code = main(
        [
            "--data",
            str(data),
            "--model-out",
            str(tmp_path / "m.joblib"),
            "--metrics-out",
            str(tmp_path / "m.json"),
            "--n-estimators",
            "10",
            "--cv-folds",
            "2",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Baseline ladder" in out
    assert "Linear regression" in out
    assert "Mean (no model)" in out
    # On this data linear wins, so the run must say so and name the flag.
    assert "--model linear" in out
