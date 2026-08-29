"""Smoke tests for the Streamlit dashboard.

The dashboard is a script, not a module with a typed API, so it is verified by
*running* it. Streamlit's AppTest executes the script headlessly and surfaces
any exception the real app would have shown a user.

These caught a real regression once already: renaming a field in the metrics
JSON broke the Model & data tab, and nothing else in the suite would have
noticed.
"""

from __future__ import annotations

import pytest

from src.config import DATASET_PATH, MODEL_PATH

pytest.importorskip("streamlit.testing.v1", reason="Streamlit testing API required")

from streamlit.testing.v1 import AppTest

APP_PATH = str((DATASET_PATH.parents[1] / "app" / "streamlit_app.py").resolve())

#: The dashboard degrades gracefully without them, but the interesting paths
#: (prediction, intervals, baselines) need a trained model to exercise.
needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="no trained model; run python src/train_model.py",
)


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def test_app_runs_without_exceptions(app):
    assert not app.exception


def test_app_renders_every_tab(app):
    assert len(app.tabs) == 5


def test_app_leads_with_the_safety_notice(app):
    """The disclaimer is the first thing on the page and must not drift."""
    assert any("Educational simulation only" in w.value for w in app.warning)


def test_app_shows_the_headline_metrics(app):
    labels = [metric.label for metric in app.metric]
    assert "Recovery fraction" in labels
    assert "Recovered gas" in labels


@needs_model
def test_manual_override_switches_the_source(app):
    """Toggling the model off must fall back to the slider, not crash."""
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert app.metric[0].value == "65.0%"


@needs_model
def test_changing_an_input_moves_the_prediction(app):
    before = app.metric[0].value
    app.sidebar.slider[3].set_value(0.95).run()  # methane fraction
    assert not app.exception
    assert app.metric[0].value != before


@needs_model
def test_high_h2s_raises_an_educational_flag(app):
    app.sidebar.slider[5].set_value(1400.0).run()  # H2S ppm
    assert not app.exception
    assert len(app.expander) >= 1


def test_extreme_inputs_do_not_crash(app):
    """Every slider at its limit is still a valid page, not a stack trace."""
    for slider in app.sidebar.slider:
        slider.set_value(slider.max)
    app.run()
    assert not app.exception
