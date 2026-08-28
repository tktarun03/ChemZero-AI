"""ChemZero-AI dashboard.

Ties the pieces together: synthetic inputs -> educational screening -> trained
model prediction -> simplified balance -> scenario comparison.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATASET_PATH,
    DEFAULT_OPERATING_HOURS_PER_YEAR,
    FEATURE_LABELS,
    FEATURES,
    FEATURES_NOT_IN_TARGET,
    SAFETY_NOTICE,
    TARGET,
)
from src.optimization import MAX_POWER_PRICE_PER_KWH, WEIGHTS, rank_scenarios  # noqa: E402
from src.predict import (  # noqa: E402
    ModelNotTrainedError,
    calibration_from_metrics,
    load_metrics,
    load_model,
    predict_with_interval,
    sweep_feature,
)
from src.process_model import ProcessInput, evaluate_process  # noqa: E402
from src.safety_constraints import Severity, evaluate_demo_safety  # noqa: E402
from src.viz import (  # noqa: E402
    emphasis_bar,
    importance_bar,
    scenario_bar,
    sensitivity_line,
    theme_for,
)

st.set_page_config(page_title="ChemZero-AI", page_icon="⚗️", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per session. Returns (model, error_message)."""
    try:
        return load_model(), None
    except ModelNotTrainedError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict | None:
    """Load training metrics, or None when training has not been run."""
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame | None:
    """Load the synthetic dataset, or None when it has not been generated."""
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


def is_dark_theme() -> bool:
    """Best-effort detection of the viewer's Streamlit theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # pragma: no cover - older Streamlit or bare mode
        return False


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("ChemZero-AI")
st.caption("Educational Chemical Engineering + AI proof of concept")
st.warning(SAFETY_NOTICE, icon="⚠️")

model, model_error = get_model()
metrics = get_metrics()
dataset = get_dataset()
theme = theme_for(is_dark_theme())

# --------------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Synthetic process inputs")
    flow = st.slider("Gas flow (Nm³/h)", 100.0, 5000.0, 1500.0, 50.0)
    pressure = st.slider("Pressure (bar)", 1.0, 20.0, 6.0, 0.5)
    temperature = st.slider("Temperature (°C)", 20.0, 120.0, 55.0, 1.0)
    methane = st.slider("Methane fraction", 0.50, 0.98, 0.80, 0.01)
    co2 = st.slider("CO₂ fraction", 0.0, 0.25, 0.08, 0.01)
    h2s = st.slider("H₂S (ppm)", 0.0, 1500.0, 100.0, 25.0)
    availability = st.slider("Availability", 0.70, 1.00, 0.95, 0.01)
    power_price = st.slider("Power price ($/kWh)", 0.03, 0.20, 0.10, 0.01)

    st.divider()
    st.header("Recovery fraction")
    use_model = st.toggle(
        "Predict with the trained model",
        value=model is not None,
        disabled=model is None,
        help="Off = set the recovery fraction by hand and compare it with the model.",
    )
    manual_recovery = st.slider("Manual recovery fraction", 0.0, 0.95, 0.65, 0.05)

    st.divider()
    operating_hours = st.number_input(
        "Operating hours per year",
        min_value=0.0,
        max_value=8766.0,
        value=DEFAULT_OPERATING_HOURS_PER_YEAR,
        step=100.0,
    )

feature_inputs: dict[str, float] = {
    "flow_nm3_h": flow,
    "pressure_bar": pressure,
    "temperature_c": temperature,
    "methane_fraction": methane,
    "co2_fraction": co2,
    "h2s_ppm": h2s,
    "availability": availability,
    "power_price_per_kwh": power_price,
}

# --------------------------------------------------------------------------
# Screening (educational only)
# --------------------------------------------------------------------------

assessment = evaluate_demo_safety(pressure, h2s, methane, co2)
if not assessment.is_valid:
    st.error(assessment.message)
    st.stop()

if model_error:
    st.info(
        f"{model_error}\n\nThe dashboard is running on the manual recovery slider "
        "until a model is trained.",
        icon="ℹ️",
    )

# --------------------------------------------------------------------------
# Recovery fraction: model prediction or manual
# --------------------------------------------------------------------------

calibration = calibration_from_metrics(metrics)
if model is not None:
    predicted_recovery, interval_low, interval_high = predict_with_interval(
        model, feature_inputs, calibration
    )
else:
    predicted_recovery = interval_low = interval_high = None

recovery = predicted_recovery if (use_model and predicted_recovery is not None) else manual_recovery

result = evaluate_process(
    ProcessInput(
        flow_nm3_h=flow,
        methane_fraction=methane,
        recovery_fraction=recovery,
        operating_hours_per_year=operating_hours,
        power_price_per_kwh=power_price,
    )
)
ranked = rank_scenarios(recovery, power_price)

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------

source_label = "model prediction" if (use_model and predicted_recovery is not None) else "manual input"
delta = None
if predicted_recovery is not None and not use_model:
    delta = f"{(manual_recovery - predicted_recovery):+.1%} vs model"

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Recovery fraction",
    f"{recovery:.1%}",
    delta=delta,
    delta_color="off",
    help=f"Source: {source_label}",
)
if use_model and interval_low is not None:
    confidence = int((1 - calibration.alpha) * 100)
    k1.caption(f"{confidence}% interval: {interval_low:.1%} – {interval_high:.1%}")
k2.metric("Recovered gas", f"{result['recovered_gas_nm3_h']:,.0f} Nm³/h")
k3.metric("Recovered energy", f"{result['annual_recovered_energy_mwh']:,.0f} MWh/y")
k4.metric("Illustrative avoided CO₂", f"{result['estimated_avoided_co2_t_y']:,.0f} t/y")

if assessment.flags:
    icons = {Severity.SERIOUS: "🔴", Severity.WARNING: "🟠", Severity.INFO: "🔵"}
    label = f"Educational flags on these inputs ({len(assessment.flags)})"
    with st.expander(label, expanded=assessment.highest_severity == Severity.SERIOUS):
        st.caption(
            "Teaching prompts about what these values would mean in practice. "
            "They are not a hazard assessment and carry no engineering authority."
        )
        for flag in assessment.flags:
            st.markdown(f"{icons[flag.severity]} **{flag.title}** — {flag.detail}")

overview, scenarios_tab, sensitivity_tab, model_tab, learn_tab = st.tabs(
    ["Overview", "Scenarios", "Sensitivity", "Model & data", "How this works"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with overview:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Calculated results")
        results_df = pd.DataFrame(
            [
                ("Methane flow", f"{result['methane_flow_nm3_h']:,.0f}", "Nm³/h"),
                ("Recovered gas", f"{result['recovered_gas_nm3_h']:,.0f}", "Nm³/h"),
                ("Recovered energy", f"{result['recovered_energy_kwh_h']:,.0f}", "kWh/h"),
                ("Annual recovered gas", f"{result['annual_recovered_gas_nm3']:,.0f}", "Nm³/y"),
                ("Annual recovered energy", f"{result['annual_recovered_energy_mwh']:,.0f}", "MWh/y"),
                ("Illustrative energy value", f"{result['annual_energy_value_usd']:,.0f}", "$/y"),
                ("Illustrative avoided CO₂", f"{result['estimated_avoided_co2_t_y']:,.0f}", "t/y"),
            ],
            columns=["Quantity", "Value", "Unit"],
        )
        st.dataframe(results_df, hide_index=True, width="stretch")

        export = pd.DataFrame([{**feature_inputs, "recovery_fraction": recovery, **result.as_dict()}])
        st.download_button(
            "Download this operating point (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="chemzero_operating_point.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Where the recovery fraction came from")
        if predicted_recovery is not None:
            st.markdown(
                f"- **Model prediction:** {predicted_recovery:.1%}\n"
                f"- **Manual slider:** {manual_recovery:.1%}\n"
                f"- **In use:** {recovery:.1%} ({source_label})"
            )
        else:
            st.markdown(f"- **Manual slider:** {manual_recovery:.1%} (no trained model loaded)")

        st.caption(
            "The energy figure assumes every recovered Nm³ of methane is converted "
            "at its lower heating value with no losses, and the CO₂ figure assumes "
            "it displaces an equally carbon-intense fuel one-for-one. Both are "
            "upper bounds, not project numbers."
        )

# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

with scenarios_tab:
    st.subheader("Scenario comparison")
    st.altair_chart(scenario_bar(ranked, theme), width="stretch")

    st.caption(
        f"Score = {WEIGHTS['recovery_fit']} x recovery fit "
        f"+ {WEIGHTS['energy_value']} x energy value "
        f"- {WEIGHTS['capex']} x capex - {WEIGHTS['complexity']} x complexity. "
        f"Power price is normalised against ${MAX_POWER_PRICE_PER_KWH}/kWh. "
        "The weights and the scenario attributes are invented for teaching."
    )

    st.dataframe(
        pd.DataFrame(ranked)[
            ["rank", "name", "score", "recovery_fit", "energy_value", "penalty",
             "recovery", "efficiency", "capex", "complexity"]
        ],
        hide_index=True,
        width="stretch",
    )

    st.info(
        f"Toy-model ranking: **{ranked[0]['name']}** scores highest at this operating "
        "point. This is a weighted-sum exercise, not an engineering recommendation.",
        icon="🧮",
    )

# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------

with sensitivity_tab:
    st.subheader("One-at-a-time sensitivity")

    if model is None:
        st.info("Train a model to enable the model sensitivity sweep.", icon="ℹ️")
    else:
        swept = st.selectbox(
            "Vary this input",
            options=FEATURES,
            index=FEATURES.index("methane_fraction"),
            format_func=lambda name: FEATURE_LABELS.get(name, name),
        )
        ranges = {
            "flow_nm3_h": (100.0, 5000.0),
            "pressure_bar": (1.0, 20.0),
            "temperature_c": (20.0, 120.0),
            "methane_fraction": (0.50, 0.98),
            "co2_fraction": (0.0, 0.25),
            "h2s_ppm": (0.0, 1500.0),
            "availability": (0.70, 1.00),
            "power_price_per_kwh": (0.03, 0.20),
        }
        low, high = ranges[swept]
        values = [low + (high - low) * i / 40 for i in range(41)]
        sweep_df = sweep_feature(model, feature_inputs, swept, values)

        st.altair_chart(
            sensitivity_line(
                sweep_df,
                swept,
                "predicted_recovery_fraction",
                FEATURE_LABELS.get(swept, swept),
                "Predicted recovery fraction",
                theme,
                y_format=".1%",
            ),
            width="stretch",
        )
        st.caption(
            "Every other input is held at its sidebar value. A flat line means the "
            "model learned no dependence on that input — expected for flow and "
            "temperature, which the data generator leaves out of the target formula."
        )

    st.divider()
    st.subheader("What recovery fraction is worth")
    st.caption(
        "Two measures on two different scales, so they get two charts rather than "
        "two y-axes on one plot."
    )

    recovery_grid = [i / 40 for i in range(41)]
    outputs = [
        evaluate_process(
            ProcessInput(flow, methane, value, operating_hours, power_price)
        ).as_dict()
        for value in recovery_grid
    ]
    value_df = pd.DataFrame(outputs).assign(recovery_fraction=recovery_grid)

    c_left, c_right = st.columns(2)
    with c_left:
        st.altair_chart(
            sensitivity_line(
                value_df, "recovery_fraction", "annual_recovered_energy_mwh",
                "Recovery fraction", "Recovered energy (MWh/y)", theme,
            ),
            width="stretch",
        )
    with c_right:
        st.altair_chart(
            sensitivity_line(
                value_df, "recovery_fraction", "estimated_avoided_co2_t_y",
                "Recovery fraction", "Illustrative avoided CO₂ (t/y)", theme,
            ),
            width="stretch",
        )

# --------------------------------------------------------------------------
# Model & data
# --------------------------------------------------------------------------

with model_tab:
    st.subheader("Trained model")

    if metrics is None:
        st.info(
            "No metrics found. Run `python src/train_model.py` to train the model "
            "and write `data/model_metrics.json`.",
            icon="ℹ️",
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (held-out)", f"{metrics['r2']:.3f}")
        m2.metric("MAE", f"{metrics['mae']:.4f}")
        m3.metric("RMSE", f"{metrics['rmse']:.4f}")
        m4.metric("CV R²", f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
        st.caption(
            f"Estimator `{metrics.get('model_kind', 'rf')}`, trained on "
            f"{metrics['n_train']} rows, calibrated on {metrics['n_calibration']}, "
            f"scored on {metrics['n_test']} held out ({metrics['trained_at_utc']}). "
            "These scores say the model reproduced the formula that generated the "
            "data. They say nothing about a real process."
        )

        # -- Is the complexity earning its keep? ---------------------------
        baselines = metrics.get("baselines") or []
        if baselines:
            st.divider()
            st.subheader("Is the complexity earning its keep?")
            best = baselines[0]
            st.altair_chart(
                emphasis_bar(
                    labels=[row["name"] for row in baselines],
                    values=[row["cv_r2_mean"] for row in baselines],
                    leader_index=0,
                    x_title="Cross-validated R² (higher is better)",
                    theme=theme,
                    value_format="+.3f",
                ),
                width="stretch",
            )

            if best["name"] != "Random forest" and metrics.get("model_kind") == "rf":
                st.warning(
                    f"**{best['name']}** scores higher than the random forest "
                    "currently loaded. The target was generated from a nearly "
                    "linear formula, so the simpler model fits it better — and it "
                    "is smaller, faster, and directly interpretable. Retrain with "
                    "`python src/train_model.py --model linear` to switch.",
                    icon="📉",
                )

            st.dataframe(
                pd.DataFrame(baselines)[["name", "cv_r2_mean", "cv_r2_std", "note"]].rename(
                    columns={
                        "name": "Model",
                        "cv_r2_mean": "CV R²",
                        "cv_r2_std": "± std",
                        "note": "What it is",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Every candidate is scored the same way, on the same folds. A model "
                "that cannot beat \"Mean (no model)\" has learned nothing at all."
            )

        # -- Uncertainty ----------------------------------------------------
        conformal = metrics.get("conformal") or {}
        if conformal:
            st.divider()
            st.subheader("How much to trust one prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric("Interval half-width", f"±{conformal['quantile']:.4f}")
            c2.metric("Target coverage", f"{conformal['confidence']:.0%}")
            c3.metric("Measured coverage", f"{conformal['empirical_coverage']:.1%}")
            st.caption(
                f"Split conformal prediction, calibrated on {conformal['n_calibration']} "
                "points the model never trained on. Measured coverage is the fraction "
                "of held-out points that actually landed inside the interval — it "
                "should sit near the target. The guarantee needs no assumption about "
                "the shape of the errors, but it does assume new inputs resemble the "
                "training data, which is exactly what fails when a model meets an "
                "operating regime it has never seen."
            )

        # -- What the model leans on ---------------------------------------
        st.divider()
        st.subheader("What the model leans on")

        permutation = metrics.get("permutation_importances") or {}
        impurity = metrics.get("impurity_importances") or {}

        if permutation:
            st.altair_chart(importance_bar(permutation, FEATURE_LABELS, theme), width="stretch")
            st.caption(
                "**Permutation importance**, measured on held-out data: how much "
                "worse the model gets when one feature is shuffled. It works for any "
                "estimator and does not reward a feature merely for having many "
                "possible split points."
            )

        if impurity and permutation:
            comparison = pd.DataFrame(
                {
                    "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
                    "Permutation": [permutation.get(f, 0.0) for f in FEATURES],
                    "Impurity (built-in)": [impurity.get(f, 0.0) for f in FEATURES],
                    "In the target formula?": [
                        "no" if f in FEATURES_NOT_IN_TARGET else "yes" for f in FEATURES
                    ],
                }
            ).sort_values("Permutation", ascending=False)
            st.dataframe(comparison, hide_index=True, width="stretch")

            inflated = [
                FEATURE_LABELS.get(f, f)
                for f in FEATURES_NOT_IN_TARGET
                if impurity.get(f, 0.0) > 0.01
            ]
            if inflated:
                st.info(
                    "**Why two columns?** The forest's built-in impurity importance "
                    f"assigns non-trivial weight to {', '.join(inflated)} — variables "
                    "the data generator never put in the target formula. Impurity "
                    "importance is biased toward continuous, high-cardinality "
                    "features, which get many split points to fit noise with. "
                    "Permutation importance correctly scores them at roughly zero. "
                    "This repo has a rare luxury: we wrote the formula, so we know "
                    "which column is right.",
                    icon="🔍",
                )

    st.divider()
    st.subheader("Synthetic dataset")
    if dataset is None:
        st.info("No dataset found. Run `python src/generate_data.py`.", icon="ℹ️")
    else:
        st.caption(f"{len(dataset):,} rows x {dataset.shape[1]} columns from `{DATASET_PATH.name}`.")
        st.dataframe(dataset.head(20), hide_index=True, width="stretch")
        with st.expander("Column statistics"):
            st.dataframe(dataset[[*FEATURES, TARGET]].describe().T, width="stretch")

# --------------------------------------------------------------------------
# How this works
# --------------------------------------------------------------------------

with learn_tab:
    st.subheader("The pipeline behind this page")
    st.markdown(
        """
1. **`src/generate_data.py`** draws synthetic operating points and computes a
   target recovery fraction from a hand-written correlation plus noise.
2. **`src/train_model.py`** fits a model, compares it against a ladder of
   simpler baselines, and splits off a calibration set it never trains on.
3. **`src/baselines.py`** answers "is the complex model worth it?" every run.
4. **`src/uncertainty.py`** turns the point prediction into a conformal
   interval using that calibration set.
5. **`src/predict.py`** loads the fitted model and predicts a recovery fraction,
   with its interval, for whatever the sidebar is set to.
6. **`src/process_model.py`** turns that fraction into recovered gas, energy,
   and an illustrative avoided-CO₂ figure.
7. **`src/optimization.py`** scores four end-use scenarios with a transparent
   weighted sum.
8. **`src/safety_constraints.py`** range-checks the inputs and raises teaching
   flags.
        """
    )

    st.subheader("What this demo cannot tell you")
    st.markdown(
        """
- **The data is invented.** The model can only rediscover the formula in
  `generate_data.py`; a high R² here is circular, not evidence.
- **The balance is first-order.** No separation efficiency, pressure drop,
  compression duty, heat integration, or equipment model.
- **The economics are a sketch.** No capital cost, operating cost, financing,
  or cash-flow model — the scenario weights are guesses.
- **The screening is not process safety.** No hazard model, no site data, no
  standards. Real screening is HAZOP/LOPA plus qualified review.
- **Sensitivity is one-at-a-time.** Interactions between variables are ignored.
- **The interval is conditional on the data looking familiar.** Conformal
  coverage holds while new inputs resemble the calibration set. Push the sliders
  somewhere the training data never went and the guarantee quietly lapses -
  without the interval widening to tell you.
        """
    )
    st.warning(
        "Use this to practise the workflow — generate, train, predict, compare, "
        "question. Do not use it to design, operate, optimise, or control anything real.",
        icon="⚠️",
    )
