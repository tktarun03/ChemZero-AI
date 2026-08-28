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

#: Features the data generator deliberately leaves OUT of the target formula.
#: Any importance a model assigns to these is an artefact, not a finding - which
#: makes them the built-in control group for comparing importance methods.
FEATURES_NOT_IN_TARGET: list[str] = [
    "flow_nm3_h",
    "temperature_c",
    "power_price_per_kwh",
]

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

# ---------------------------------------------------------------------------
# Physical constants
#
# REFERENCE CONDITIONS: 0 °C and 101.325 kPa ("normal" conditions, Nm³), where
# the ideal-gas molar volume is 22.414 L/mol. Both constants below are stated on
# this same basis. Mixing bases is a classic source of quiet error - a 15 °C
# basis (23.645 L/mol) would change the CO₂ factor by about 5%.
# ---------------------------------------------------------------------------

#: Lower heating value of methane, kWh per Nm³.
#: 35.8 MJ/Nm³ / 3.6 MJ/kWh = 9.94 kWh/Nm³ at 0 °C, 101.325 kPa.
METHANE_LHV_KWH_PER_NM3: float = 9.94

#: Stoichiometric CO₂ released per Nm³ of methane combusted, kg.
#:
#: From CH₄ + 2 O₂ -> CO₂ + 2 H₂O, one mole of methane yields one mole of CO₂:
#:     (1000 L/Nm³ / 22.414 L/mol) x 44.009 g/mol / 1000 = 1.963 kg/Nm³
#:
#: This repo previously carried 1.90, which matches no standard reference
#: condition and was 3.2% low against the 0 °C basis its LHV already assumed.
CO2_KG_PER_NM3_METHANE_COMBUSTED: float = 1.963

#: Ideal-gas molar volume at the reference conditions above, L/mol.
MOLAR_VOLUME_L_PER_MOL: float = 22.414

#: Molar mass of CO₂, g/mol.
MW_CO2_G_PER_MOL: float = 44.009

#: Default plant operating hours per year used across the demo.
DEFAULT_OPERATING_HOURS_PER_YEAR: float = 8000.0

SAFETY_NOTICE: str = (
    "Educational simulation only. Not for real plant design, operation, control, "
    "process safety, or production decision-making."
)
