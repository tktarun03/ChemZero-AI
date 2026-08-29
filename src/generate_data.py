"""Generate the synthetic process dataset.

There is no real plant data here. Feature values are drawn from independent
uniform distributions over plausible ranges, and the target is a hand-written
correlation plus Gaussian noise. Because the "truth" is a formula we chose, any
model trained on it can only rediscover that formula - which is exactly the
point: it lets you practise the ML workflow without pretending to have learned
anything about a real process.

Run as either::

    python src/generate_data.py
    python -m src.generate_data --rows 5000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python src/generate_data.py` as well as `python -m src.generate_data`.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH, FEATURES, TARGET

__all__ = ["generate_synthetic_data", "main"]

DEFAULT_ROWS = 1500
DEFAULT_SEED = 42


def generate_synthetic_data(n: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Build the synthetic dataset.

    Args:
        n: Number of rows to generate. Must be positive.
        seed: Seed for the random generator, so runs are reproducible.

    Returns:
        A DataFrame with :data:`src.config.FEATURES` plus
        :data:`src.config.TARGET`.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    rng = np.random.default_rng(seed)
    flow = rng.uniform(200, 5000, n)
    pressure = rng.uniform(1.2, 18.0, n)
    temperature = rng.uniform(20, 120, n)
    methane = rng.uniform(0.55, 0.95, n)
    co2 = rng.uniform(0.01, 0.20, n)
    h2s = rng.uniform(0, 1500, n)
    availability = rng.uniform(0.80, 0.995, n)
    power_price = rng.uniform(0.04, 0.18, n)

    # The invented "ground truth": richer, drier, higher-pressure, higher-uptime
    # gas recovers better; CO2 and H2S drag it down. Noise stands in for the
    # measurement and operating scatter a real dataset would carry.
    recovery = (
        0.30
        + 0.35 * methane
        + 0.015 * np.log1p(pressure)
        - 0.40 * co2
        - 0.00005 * h2s
        + 0.10 * availability
        + rng.normal(0, 0.03, n)
    )
    recovery = np.clip(recovery, 0.05, 0.95)

    return pd.DataFrame(
        {
            "flow_nm3_h": flow,
            "pressure_bar": pressure,
            "temperature_c": temperature,
            "methane_fraction": methane,
            "co2_fraction": co2,
            "h2s_ppm": h2s,
            "availability": availability,
            "power_price_per_kwh": power_price,
            TARGET: recovery,
        },
        columns=[*FEATURES, TARGET],
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="rows to generate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--out", type=Path, default=DATASET_PATH, help="output CSV path")
    args = parser.parse_args(argv)

    df = generate_synthetic_data(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
