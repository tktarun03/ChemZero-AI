"""Tests for the chart builders.

Charts are tested through the Vega-Lite spec they compile to. That catches the
things that actually break a chart - a wrong encoding, a colour that ignores the
theme, an emphasis rule highlighting the wrong bar - without needing a browser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.viz import (
    DARK,
    LIGHT,
    emphasis_bar,
    importance_bar,
    scenario_bar,
    sensitivity_line,
    theme_for,
)


@pytest.fixture
def line_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.0, 0.5, 1.0], "y": [10.0, 50.0, 90.0]})


@pytest.fixture
def importances() -> dict[str, float]:
    return {"methane_fraction": 0.9, "co2_fraction": 0.25, "h2s_ppm": 0.2}


@pytest.fixture
def ranked() -> list[dict[str, object]]:
    return [
        {"name": "Recover / Reuse", "score": 0.36, "rank": 1},
        {"name": "Generate Power", "score": 0.23, "rank": 2},
        {"name": "Flare", "score": 0.15, "rank": 3},
    ]


# --- themes --------------------------------------------------------------


def test_theme_for_selects_by_mode():
    assert theme_for(True) is DARK
    assert theme_for(False) is LIGHT


def test_themes_define_every_role():
    for theme in (LIGHT, DARK):
        for role in ("series", "muted_series", "text_primary", "text_muted", "grid", "axis"):
            value = getattr(theme, role)
            assert value.startswith("#") and len(value) == 7


def test_series_and_muted_differ_in_both_themes():
    """The emphasis form collapses if the accent and the de-emphasis grey match."""
    for theme in (LIGHT, DARK):
        assert theme.series != theme.muted_series


# --- every builder compiles ---------------------------------------------


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_all_builders_compile_in_both_themes(theme, line_df, importances, ranked):
    specs = [
        sensitivity_line(line_df, "x", "y", "X", "Y", theme).to_dict(),
        importance_bar(importances, {}, theme).to_dict(),
        emphasis_bar(["a", "b"], [1.0, 2.0], 0, "Score", theme).to_dict(),
        scenario_bar(ranked, theme).to_dict(),
    ]
    for spec in specs:
        assert spec["layer"]


# --- sensitivity line ----------------------------------------------------


def test_line_uses_the_theme_accent(line_df):
    spec = sensitivity_line(line_df, "x", "y", "X", "Y", LIGHT).to_dict()
    colours = [
        layer["mark"].get("color") for layer in spec["layer"] if isinstance(layer["mark"], dict)
    ]
    assert LIGHT.series in colours


def test_line_carries_axis_titles(line_df):
    spec = sensitivity_line(line_df, "x", "y", "Recovery", "Energy", LIGHT).to_dict()
    encoding = spec["layer"][1]["encoding"]
    assert encoding["x"]["title"] == "Recovery"
    assert encoding["y"]["title"] == "Energy"


def test_line_has_a_hover_selection(line_df):
    """The crosshair is the interaction; losing it silently would be easy.

    Altair hoists selection params to the top level of the spec rather than
    nesting them in the layer that declared them.
    """
    spec = sensitivity_line(line_df, "x", "y", "X", "Y", LIGHT).to_dict()
    selections = [param["select"] for param in spec["params"] if "select" in param]
    assert any(
        sel["type"] == "point" and sel.get("nearest") and sel.get("on") == "pointerover"
        for sel in selections
    )


def test_line_y_axis_includes_zero(line_df):
    """A truncated y-axis exaggerates differences; magnitude charts start at zero."""
    spec = sensitivity_line(line_df, "x", "y", "X", "Y", LIGHT).to_dict()
    assert spec["layer"][1]["encoding"]["y"]["scale"]["zero"] is True


# --- importance bars -----------------------------------------------------


def test_importance_uses_one_hue_for_every_bar(importances):
    """Nominal categories must not get a value ramp - length already encodes size."""
    spec = importance_bar(importances, {}, LIGHT).to_dict()
    bar_layer = spec["layer"][0]
    assert bar_layer["mark"]["color"] == LIGHT.series
    assert "color" not in bar_layer.get("encoding", {})


def test_importance_applies_display_labels(importances):
    spec = importance_bar(importances, {"methane_fraction": "Methane fraction"}, LIGHT).to_dict()
    names = [row["feature"] for row in spec["datasets"][spec["data"]["name"]]]
    assert "Methane fraction" in names
    assert "methane_fraction" not in names


def test_importance_falls_back_to_raw_names(importances):
    spec = importance_bar(importances, {}, LIGHT).to_dict()
    names = [row["feature"] for row in spec["datasets"][spec["data"]["name"]]]
    assert "methane_fraction" in names


def test_importance_sorted_descending(importances):
    spec = importance_bar(importances, {}, LIGHT).to_dict()
    values = [row["importance"] for row in spec["datasets"][spec["data"]["name"]]]
    assert values == sorted(values, reverse=True)


def test_importance_direct_labels_present(importances):
    """Values sit beside their bars, so no axis-reading is required."""
    spec = importance_bar(importances, {}, LIGHT).to_dict()
    assert spec["layer"][1]["mark"]["type"] == "text"


# --- emphasis bars -------------------------------------------------------


def test_emphasis_highlights_exactly_one_bar():
    spec = emphasis_bar(["a", "b", "c"], [3.0, 2.0, 1.0], 1, "Score", LIGHT).to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    assert [row["leader"] for row in rows] == [False, True, False]


def test_emphasis_uses_accent_for_leader_and_grey_for_the_rest():
    spec = emphasis_bar(["a", "b"], [1.0, 2.0], 0, "Score", LIGHT).to_dict()
    condition = spec["layer"][0]["encoding"]["color"]
    assert condition["condition"]["value"] == LIGHT.series
    assert condition["value"] == LIGHT.muted_series


def test_emphasis_preserves_the_given_order():
    """Order is the caller's decision; the chart must not re-sort behind them."""
    spec = emphasis_bar(["z", "m", "a"], [1.0, 2.0, 3.0], 0, "Score", LIGHT).to_dict()
    assert spec["layer"][0]["encoding"]["y"]["sort"] == ["z", "m", "a"]


def test_emphasis_handles_negative_values():
    """Cross-validated R2 can be negative; the chart must still build."""
    spec = emphasis_bar(["good", "bad"], [0.7, -0.3], 0, "CV R2", LIGHT).to_dict()
    values = [row["value"] for row in spec["datasets"][spec["data"]["name"]]]
    assert -0.3 in values


# --- scenario bars -------------------------------------------------------


def test_scenario_emphasises_the_rank_one_row(ranked):
    spec = scenario_bar(ranked, LIGHT).to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    leader = next(row for row in rows if row["leader"])
    assert leader["label"] == "Recover / Reuse"


def test_scenario_emphasises_rank_one_even_when_listed_late():
    """Emphasis follows the rank field, not the row position."""
    out_of_order = [
        {"name": "Second", "score": 0.2, "rank": 2},
        {"name": "Winner", "score": 0.9, "rank": 1},
    ]
    spec = scenario_bar(out_of_order, LIGHT).to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    assert next(row for row in rows if row["leader"])["label"] == "Winner"


def test_scenario_without_rank_falls_back_to_the_first_row():
    spec = scenario_bar([{"name": "Only", "score": 0.5}], LIGHT).to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    assert rows[0]["leader"] is True
