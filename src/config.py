"""Shared paths, feature names, and physical constants.

Centralising these means the data generator, trainer, predictor, and dashboard
can never silently disagree about column names or file locations.
"""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"

DATASET_PATH: Path = DATA_DIR / "synthetic_process_data.csv"
MODEL_PATH: Path = DATA_DIR / "recovery_model.joblib"
METRICS_PATH: Path = DATA_DIR / "model_metrics.json"

#: Model input columns, in the exact order the estimator was fitted on.
#: scikit-learn matches columns by position, so this order is load-bearing.
FEATURES: list[str] = [
    "flow_nm3_h",
    "pressure_bar",
    "temperature_c",
    "methane_fraction",
    "co2_fraction",
    "h2s_ppm",
    "availability",
    "power_price_per_kwh",
]

TARGET: str = "target_recovery_fraction"

#: Human-readable labels for the dashboard and charts.
FEATURE_LABELS: dict[str, str] = {
    "flow_nm3_h": "Gas flow (Nm³/h)",
    "pressure_bar": "Pressure (bar)",
    "temperature_c": "Temperature (°C)",
    "methane_fraction": "Methane fraction",
    "co2_fraction": "CO₂ fraction",
    "h2s_ppm": "H₂S (ppm)",
    "availability": "Availability",
    "power_price_per_kwh": "Power price ($/kWh)",
}

#: Lower heating value of methane at normal conditions, kWh per Nm³.
METHANE_LHV_KWH_PER_NM3: float = 9.94

#: Stoichiometric CO₂ released per Nm³ of methane combusted, kg.
CO2_KG_PER_NM3_METHANE_COMBUSTED: float = 1.90

#: Default plant operating hours per year used across the demo.
DEFAULT_OPERATING_HOURS_PER_YEAR: float = 8000.0

SAFETY_NOTICE: str = (
    "Educational simulation only. Not for real plant design, operation, control, "
    "process safety, or production decision-making."
)
