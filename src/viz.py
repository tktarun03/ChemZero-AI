"""Altair chart builders for the dashboard.

Design rules applied here, so the charts read as one system:

* **One hue per chart.** Feature importance and sensitivity are *magnitude*
  questions, so they use a single blue at one step - not a value ramp across
  nominal categories, which would double-encode bar length as colour.
* **Emphasis for the ranking.** The story is "this option scores highest", so
  the leader is blue and the rest recede to grey rather than each taking its own
  categorical hue.
* **Never two y-scales on one plot.** Energy and CO2 have different units, so
  they are two charts side by side, never a dual axis.
* **Recessive chrome.** Hairline solid gridlines, muted axis ink, thin marks.

Colours come from a validated palette; the blue/grey pair separates at CVD
delta-E 15.9 (the safety floor is 8), so the emphasis reads for colourblind
viewers too. Both light and dark steps are chosen for their own surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import altair as alt
import pandas as pd

__all__ = ["Theme", "LIGHT", "DARK", "theme_for", "sensitivity_line", "importance_bar", "scenario_bar"]


@dataclass(frozen=True)
class Theme:
    """Colour roles for one rendering surface."""

    series: str
    muted_series: str
    text_primary: str
    text_muted: str
    grid: str
    axis: str


LIGHT = Theme(
    series="#2a78d6",
    muted_series="#898781",
    text_primary="#0b0b0b",
    text_muted="#52514e",
    grid="#e1e0d9",
    axis="#c3c2b7",
)

DARK = Theme(
    series="#3987e5",
    muted_series="#898781",
    text_primary="#ffffff",
    text_muted="#c3c2b7",
    grid="#2c2c2a",
    axis="#383835",
)


def theme_for(is_dark: bool) -> Theme:
    """Return the colour roles for the current surface."""
    return DARK if is_dark else LIGHT


def _style(chart: alt.Chart, theme: Theme) -> alt.Chart:
    """Apply the shared chrome: hairline solid grid, muted labels, no border."""
    return (
        chart.configure_axis(
            grid=True,
            gridColor=theme.grid,
            gridWidth=1,
            domainColor=theme.axis,
            tickColor=theme.axis,
            labelColor=theme.text_muted,
            titleColor=theme.text_muted,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_view(strokeWidth=0)
        .configure_legend(disable=True)
    )


def sensitivity_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    theme: Theme,
    y_format: str = ",.0f",
    height: int = 240,
) -> alt.Chart:
    """A single-series line with a hover crosshair.

    One series, so no legend: the axis title names the measure.

    Args:
        df: Source data.
        x: Column for the horizontal axis.
        y: Column for the vertical axis.
        x_title: Axis label for ``x``.
        y_title: Axis label for ``y``.
        theme: Colour roles for the current surface.
        y_format: d3 number format for the value in the tooltip.
        height: Chart height in pixels.

    Returns:
        A configured Altair chart.
    """
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(grid=False)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=True)),
    )

    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    tooltip = [
        alt.Tooltip(f"{x}:Q", title=x_title, format=".3f"),
        alt.Tooltip(f"{y}:Q", title=y_title, format=y_format),
    ]

    line = base.mark_line(color=theme.series, strokeWidth=2)
    # Invisible wide targets make the crosshair easy to hit.
    targets = base.mark_point(size=200, opacity=0).encode(tooltip=tooltip).add_params(hover)
    rule = (
        base.mark_rule(color=theme.axis, strokeWidth=1)
        .encode(y=alt.value(0))
        .transform_filter(hover)
    )
    marker = base.mark_point(
        size=80, filled=True, color=theme.series, stroke="white", strokeWidth=2
    ).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))

    layered = alt.layer(rule, line, marker, targets).properties(height=height, width="container")
    return _style(layered, theme)


def importance_bar(
    importances: Mapping[str, float],
    labels: Mapping[str, str],
    theme: Theme,
    height: int = 260,
) -> alt.Chart:
    """Horizontal bars for feature importance.

    Features are nominal, so every bar takes the same hue - the length already
    encodes magnitude. Horizontal keeps the long feature names readable.

    Args:
        importances: Feature name -> importance, any order.
        labels: Feature name -> display label.
        theme: Colour roles for the current surface.
        height: Chart height in pixels.

    Returns:
        A configured Altair chart.
    """
    df = pd.DataFrame(
        {
            "feature": [labels.get(name, name) for name in importances],
            "importance": list(importances.values()),
        }
    ).sort_values("importance", ascending=False)

    base = alt.Chart(df).encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X(
            "importance:Q",
            title="Relative importance",
            axis=alt.Axis(format=".0%", grid=True),
        ),
    )
    bars = base.mark_bar(color=theme.series, height=14, cornerRadiusEnd=4).encode(
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("importance:Q", title="Importance", format=".1%"),
        ]
    )
    # Direct labels: the value sits beside its bar, so no axis-reading needed.
    text = base.mark_text(align="left", dx=6, color=theme.text_muted, fontSize=11).encode(
        text=alt.Text("importance:Q", format=".1%")
    )
    return _style(alt.layer(bars, text).properties(height=height, width="container"), theme)


def scenario_bar(
    ranked: Sequence[Mapping[str, Any]],
    theme: Theme,
    height: int = 200,
) -> alt.Chart:
    """Horizontal bars for scenario scores, with the leader emphasised.

    The top-ranked scenario carries the accent hue and the rest recede to grey -
    the point of the chart is which one leads, not four competing identities.

    Args:
        ranked: Output of :func:`src.optimization.rank_scenarios`.
        theme: Colour roles for the current surface.
        height: Chart height in pixels.

    Returns:
        A configured Altair chart.
    """
    df = pd.DataFrame(
        {
            "name": [row["name"] for row in ranked],
            "score": [row["score"] for row in ranked],
            "leader": [row.get("rank") == 1 for row in ranked],
        }
    )

    base = alt.Chart(df).encode(
        y=alt.Y("name:N", sort=alt.EncodingSortField("score", order="descending"), title=None),
        x=alt.X("score:Q", title="Toy score (higher = better fit to the weights)"),
    )
    bars = base.mark_bar(height=16, cornerRadiusEnd=4).encode(
        color=alt.condition(
            alt.datum.leader, alt.value(theme.series), alt.value(theme.muted_series)
        ),
        tooltip=[
            alt.Tooltip("name:N", title="Scenario"),
            alt.Tooltip("score:Q", title="Score", format=".3f"),
        ],
    )
    text = base.mark_text(align="left", dx=6, color=theme.text_muted, fontSize=11).encode(
        text=alt.Text("score:Q", format=".2f")
    )
    return _style(alt.layer(bars, text).properties(height=height, width="container"), theme)
