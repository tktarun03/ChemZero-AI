# ChemZero-AI

An educational proof of concept that walks through a complete Chemical Engineering + AI
workflow: generate synthetic process data, train a model on it, predict a recovery
fraction, turn that into a mass and energy balance, and compare end-use scenarios — all
in one Streamlit dashboard.

> ### ⚠️ Safety notice
>
> **This repository is for education and demonstration only.** Do not use it to design,
> operate, optimize, or control a real industrial plant. Every number it produces comes
> from invented data and deliberately simplified formulas. Real process decisions require
> validated plant-specific data, formal process-safety analysis (HAZOP/LOPA), approved
> engineering models, instrumentation, controls, and qualified review.
>
> AI can help engineers spot patterns and compare options. It does not replace validated
> process-safety systems or qualified engineering judgement.

---

## Table of contents

1. [What it demonstrates](#what-it-demonstrates)
2. [How the pieces fit together](#how-the-pieces-fit-together)
3. [Requirements](#requirements)
4. [Step-by-step setup](#step-by-step-setup)
   - [Step 1 — Get the code](#step-1--get-the-code)
   - [Step 2 — Create a virtual environment](#step-2--create-a-virtual-environment)
   - [Step 3 — Install the dependencies](#step-3--install-the-dependencies)
   - [Step 4 — Generate the synthetic dataset](#step-4--generate-the-synthetic-dataset)
   - [Step 5 — Train the model](#step-5--train-the-model)
   - [Step 6 — Run the tests](#step-6--run-the-tests)
   - [Step 7 — Launch the dashboard](#step-7--launch-the-dashboard)
5. [Using the dashboard](#using-the-dashboard)
6. [Project structure](#project-structure)
7. [Command-line reference](#command-line-reference)
8. [Things to try](#things-to-try)
9. [Troubleshooting](#troubleshooting)
10. [What this demo cannot tell you](#what-this-demo-cannot-tell-you)
11. [Learning path](#learning-path)

---

## What it demonstrates

| Capability | Where it lives |
| --- | --- |
| Synthetic process data generation | [`src/generate_data.py`](src/generate_data.py) |
| Simplified gas-recovery mass & energy balance | [`src/process_model.py`](src/process_model.py) |
| Educational input screening with graded flags | [`src/safety_constraints.py`](src/safety_constraints.py) |
| Scenario comparison with a transparent weighted score | [`src/optimization.py`](src/optimization.py) |
| Model training, cross-validation, metrics export | [`src/train_model.py`](src/train_model.py) |
| Loading the model and predicting on new inputs | [`src/predict.py`](src/predict.py) |
| Chart builders (single-hue, emphasis, no dual axes) | [`src/viz.py`](src/viz.py) |
| Interactive dashboard | [`app/streamlit_app.py`](app/streamlit_app.py) |
| Test suite (94 tests) | [`tests/`](tests/) |

## How the pieces fit together

```
  generate_data.py          train_model.py             predict.py
  ────────────────          ──────────────             ──────────
  draws random              fits a random forest       loads the model,
  operating points          on that CSV, writes        predicts a recovery
  and computes a       ──>  recovery_model.joblib  ──> fraction for whatever
  target recovery           + model_metrics.json       the sidebar is set to
  from a hand-written
  correlation + noise                                        │
         │                                                   │
         └──> data/synthetic_process_data.csv                ▼
                                                   process_model.py
                                                   ────────────────
                                                   recovery fraction ->
                                                   recovered gas, energy,
                                                   illustrative avoided CO₂
                                                             │
                        safety_constraints.py                ▼
                        ─────────────────────         optimization.py
                        range-checks inputs,          ───────────────
                        raises teaching flags   ──>   ranks 4 end-use
                                                      scenarios by a
                                                      weighted score
                                                             │
                                                             ▼
                                                   app/streamlit_app.py
```

The important thing to notice: **the "ground truth" is a formula we wrote ourselves.**
The model can only rediscover that formula. A high R² here is circular — it is a
demonstration of the workflow, not evidence about any real process.

---

## Requirements

- **Python 3.10 or newer** (developed against 3.12)
- About 400 MB of disk space for the virtual environment
- No GPU, no API keys, no network access after installation

Check your Python version:

```bash
python --version
```

If that prints something below 3.10, install a newer Python before continuing.

---

## Step-by-step setup

Every step below shows the command and what you should see. Commands are given for
**Windows PowerShell** and **macOS / Linux** where they differ.

### Step 1 — Get the code

```bash
git clone <repository-url>
cd ChemZero-AI
```

If you already have the folder, just `cd` into it. Every command that follows is run
from the **repository root** (the folder containing `README.md`).

### Step 2 — Create a virtual environment

A virtual environment keeps this project's packages separate from your system Python.

```bash
python -m venv .venv
```

Now **activate** it — this step is easy to skip and causes most setup problems:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

✅ **Check:** your prompt should now start with `(.venv)`. Confirm the right Python is
in use:

```bash
python -c "import sys; print(sys.executable)"
```

The path printed must point inside `.venv`. If it does not, activation did not work.

> **PowerShell blocks the activation script?** Run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal first.
> It applies only to the current window.

### Step 3 — Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This pulls in pandas, numpy, scikit-learn, Streamlit, Altair, joblib, and pytest.
Expect it to take a minute or two.

✅ **Check:**

```bash
python -c "import pandas, sklearn, streamlit; print('ok')"
```

### Step 4 — Generate the synthetic dataset

```bash
python src/generate_data.py
```

Expected output:

```
Wrote 1500 rows x 9 columns to ...\data\synthetic_process_data.csv
```

This creates 1500 random operating points — flow, pressure, temperature, composition,
availability, power price — and computes a target recovery fraction from a hand-written
correlation plus Gaussian noise. The seed is fixed, so you get the same data every time.

Want a bigger or different dataset?

```bash
python src/generate_data.py --rows 5000 --seed 7
```

### Step 5 — Train the model

```bash
python src/train_model.py
```

Expected output (your numbers will match if you used the default dataset):

```
Rows         : 1500 (1200 train / 300 test)
MAE          : 0.0237
RMSE         : 0.0304
R2           : 0.7464
CV R2        : 0.7218 +/- 0.0163 (5 folds)
Top drivers  : methane_fraction, co2_fraction, h2s_ppm
Saved model  : ...\data\recovery_model.joblib
Saved metrics: ...\data\model_metrics.json
Reminder     : these scores describe synthetic data, not a real process.
```

This fits a random forest, scores it on a held-out split **and** with 5-fold
cross-validation, then saves two files:

- `data/recovery_model.joblib` — the fitted model the dashboard loads
- `data/model_metrics.json` — scores and feature importances the dashboard displays

Both are gitignored, so anyone cloning fresh runs Steps 4 and 5 themselves.

> **Read the metrics critically.** `Top drivers` lists methane fraction and CO₂ first
> because those carry the largest coefficients in the generating formula. The model
> recovered the formula we gave it. That is the expected result, not a discovery.

### Step 6 — Run the tests

```bash
python -m pytest
```

Expected output:

```
94 passed
```

The tests build their own small dataset and model in a temporary directory, so they pass
on a fresh clone even before Steps 4 and 5. They run from any directory in the repo.

For more detail:

```bash
python -m pytest -v
```

### Step 7 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Expected output:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If not, visit **http://localhost:8501**.

Press `Ctrl+C` in the terminal to stop it.

> The dashboard still runs if you skipped Steps 4 and 5 — it falls back to the manual
> recovery slider and tells you which command to run to enable the model features.

---

## Using the dashboard

**Sidebar** — set the synthetic operating point: flow, pressure, temperature, methane
and CO₂ fractions, H₂S, availability, power price, and operating hours. The
**"Predict with the trained model"** toggle switches between the model's predicted
recovery fraction and your own manual value; with it off, the KPI shows how far your
guess sits from the model's.

| Tab | What it shows |
| --- | --- |
| **Overview** | The full calculated balance, where the recovery fraction came from, and a CSV download of the current operating point |
| **Scenarios** | The four end-use options ranked by score, with the leader emphasised and the score broken into its terms |
| **Sensitivity** | Sweep any single input and watch the model's prediction respond; plus what recovery fraction is worth in energy and CO₂ |
| **Model & data** | Held-out and cross-validated scores, feature importances, and the underlying dataset with column statistics |
| **How this works** | The pipeline in six steps, and an explicit list of what the demo cannot tell you |

**Educational flags** appear above the tabs when an input crosses a teaching threshold —
elevated H₂S, high pressure, high CO₂, or lean gas. They are prompts to think about what
the value would mean in practice. They are not a hazard assessment and carry no
engineering authority.

---

## Project structure

```
ChemZero-AI/
├── app/
│   └── streamlit_app.py        # dashboard: inputs -> screening -> prediction -> results
├── data/
│   ├── synthetic_process_data.csv   # created by Step 4
│   ├── recovery_model.joblib        # created by Step 5 (gitignored)
│   └── model_metrics.json           # created by Step 5 (gitignored)
├── src/
│   ├── config.py               # shared paths, feature names, physical constants
│   ├── generate_data.py        # synthetic dataset generator
│   ├── process_model.py        # simplified mass & energy balance
│   ├── safety_constraints.py   # educational input screening
│   ├── optimization.py         # scenario scoring and ranking
│   ├── train_model.py          # model training, evaluation, export
│   ├── predict.py              # model loading and prediction
│   └── viz.py                  # Altair chart builders
├── tests/
│   ├── conftest.py             # shared fixtures (own dataset + model, no shared state)
│   ├── test_generate_data.py
│   ├── test_optimization.py
│   ├── test_process_model.py
│   ├── test_safety_constraints.py
│   └── test_train_and_predict.py
├── pyproject.toml              # project metadata + pytest configuration
├── requirements.txt
└── README.md
```

`src/config.py` is the single source of truth for column names and file paths, so the
generator, trainer, predictor, and dashboard can never silently disagree about them.

---

## Command-line reference

Both scripts work as a path or as a module (`python -m src.generate_data`).

### `generate_data.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rows` | `1500` | Number of operating points to generate |
| `--seed` | `42` | Random seed, for reproducibility |
| `--out` | `data/synthetic_process_data.csv` | Output CSV path |

### `train_model.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/synthetic_process_data.csv` | Input CSV |
| `--model-out` | `data/recovery_model.joblib` | Where to save the fitted model |
| `--metrics-out` | `data/model_metrics.json` | Where to save scores and importances |
| `--n-estimators` | `200` | Number of trees in the forest |
| `--min-samples-leaf` | `3` | Minimum samples per leaf |
| `--test-size` | `0.2` | Held-out fraction |
| `--cv-folds` | `5` | Cross-validation folds (`0` skips CV) |
| `--seed` | `42` | Random seed |

Example — a larger forest on a bigger dataset:

```bash
python src/generate_data.py --rows 10000 --seed 1
python src/train_model.py --n-estimators 500 --test-size 0.25
```

---

## Things to try

These are the edits that teach the most, roughly in order of difficulty.

1. **Break the model on purpose.** Regenerate with `--rows 30` and retrain. R² drops
   from about 0.75 to *below zero* (worse than predicting the mean) and the
   cross-validation spread widens from ±0.02 to roughly ±0.46. Small data, unstable
   model — and a negative R² is the clearest possible signal to distrust it.
2. **Compare your intuition with the model.** Turn the prediction toggle off, guess a
   recovery fraction, and read the delta on the KPI.
3. **Find the inputs the model ignores.** In the Sensitivity tab, sweep flow and
   temperature, then sweep methane fraction. Flow and temperature are nearly flat
   (a range of roughly 0.003–0.007 in predicted recovery) while methane spans about
   0.13 — because flow and temperature were never in the generating formula. The small
   residual wiggle is the forest fitting noise, which is itself worth noticing: an
   unimportant variable rarely reads as *exactly* zero. This is the clearest
   illustration in the repo that the model only knows what the data contains.
4. **Change the scenario weights.** Edit `WEIGHTS` in [`src/optimization.py`](src/optimization.py)
   and reload. Lower the capex penalty and watch the capital-heavy options climb.
5. **Change the physics.** The correlation lives in `generate_synthetic_data()` in
   [`src/generate_data.py`](src/generate_data.py). Make pressure matter more, regenerate,
   retrain, and confirm the feature importances move.
6. **Add a feature.** Add a column to the generator, list it in `FEATURES` in
   [`src/config.py`](src/config.py), add a slider in the app, regenerate and retrain.
   The config module means those are the only places you need to touch.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` | The virtual environment is not activated, or you installed into a different Python. Redo [Step 2](#step-2--create-a-virtual-environment) and confirm `sys.executable` points inside `.venv`. |
| `ModuleNotFoundError: No module named 'src'` | You are running from outside the repository root. `cd` to the folder containing `README.md`. |
| `Dataset not found at ...` | Run `python src/generate_data.py` ([Step 4](#step-4--generate-the-synthetic-dataset)). |
| `No trained model at ...` | Run `python src/train_model.py` ([Step 5](#step-5--train-the-model)). The dashboard still works without it, using the manual slider. |
| `streamlit: command not found` | Either the environment is not activated, or use `python -m streamlit run app/streamlit_app.py`. |
| PowerShell: *"running scripts is disabled on this system"* | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Applies to that window only. |
| Port 8501 already in use | `streamlit run app/streamlit_app.py --server.port 8502` |
| Dashboard shows stale numbers after you retrain | Streamlit caches the model per session. Press `R` in the browser, or use "Clear cache" in the ⋮ menu. |
| `UnicodeEncodeError` when printing on Windows | The console codepage cannot render `³`/`₂`. Use `set PYTHONIOENCODING=utf-8` before the command. Does not affect the dashboard. |

---

## What this demo cannot tell you

Stated plainly, because a polished dashboard invites more trust than this model has earned:

- **The data is invented.** Features are independent uniform draws; the target is a
  formula plus noise. The model can only rediscover that formula.
- **The balance is first-order.** No separation efficiency, pressure drop, compression
  duty, heat integration, phase behaviour, or equipment model.
- **The avoided-CO₂ figure is an upper bound**, not an emissions credit. It assumes
  recovered methane displaces an equally carbon-intense fuel one-for-one, and ignores
  methane slip, parasitic load, and any accounting boundary.
- **The economics are a sketch.** No capital cost, operating cost, financing, or
  cash-flow model. The scenario attributes and weights are guesses.
- **The screening is not process safety.** Thresholds are round teaching numbers, not
  limits from any standard, code, or datasheet. Real screening means HAZOP/LOPA,
  applicable codes, and qualified engineering review.
- **Sensitivity is one-at-a-time.** Interactions between variables are ignored entirely.

---

## Learning path

1. Chemical Engineering fundamentals — mass and energy balances first
2. Python + pandas — get comfortable manipulating tabular process data
3. Process simulation thinking — what a model includes, and what it leaves out
4. ML basics — train/test splits, cross-validation, over-fitting, feature importance
5. Optimization — objective functions, weights, constraints, trade-offs
6. Safety and uncertainty — where models stop and engineering judgement starts
7. Digital-twin concepts — validated models tied to real plant data

---

## Important principle

AI can help engineers identify patterns and compare options.
**AI does not replace validated process-safety systems or qualified engineering judgement.**
