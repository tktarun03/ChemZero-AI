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
9. [Developing on this repo](#developing-on-this-repo)
10. [Running with Docker](#running-with-docker)
11. [Troubleshooting](#troubleshooting)
12. [What this demo cannot tell you](#what-this-demo-cannot-tell-you)
13. [Learning path](#learning-path)

---

## What it demonstrates

| Capability | Where it lives |
| --- | --- |
| Synthetic process data generation | [`src/generate_data.py`](src/generate_data.py) |
| Simplified gas-recovery mass & energy balance | [`src/process_model.py`](src/process_model.py) |
| Educational input screening with graded flags | [`src/safety_constraints.py`](src/safety_constraints.py) |
| Scenario comparison with a transparent weighted score | [`src/optimization.py`](src/optimization.py) |
| Model training, cross-validation, metrics export | [`src/train_model.py`](src/train_model.py) |
| Baseline ladder — *is the complex model worth it?* | [`src/baselines.py`](src/baselines.py) |
| Conformal prediction intervals | [`src/uncertainty.py`](src/uncertainty.py) |
| Loading the model and predicting on new inputs | [`src/predict.py`](src/predict.py) |
| Chart builders (single-hue, emphasis, no dual axes) | [`src/viz.py`](src/viz.py) |
| Interactive dashboard | [`app/streamlit_app.py`](app/streamlit_app.py) |
| Test suite (155 tests, 99% coverage) | [`tests/`](tests/) |

## How the pieces fit together

```
  generate_data.py          train_model.py             predict.py
  ────────────────          ──────────────             ──────────
  draws random              splits train/calib/test,   loads the model,
  operating points          fits a model, ranks it     predicts a recovery
  and computes a       ──>  against simpler        ──> fraction + a conformal
  target recovery           baselines, calibrates       interval for whatever
  from a hand-written       a conformal interval        the sidebar is set to
  correlation + noise
         │                  baselines.py  uncertainty.py      │
         │                  ────────────  ──────────────      │
         │                  is complexity  turns a point      │
         │                  worth it?      into an interval   │
         │                                                    │
         └──> data/synthetic_process_data.csv                 ▼
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

- **Python 3.11 or newer** (developed and CI-tested on 3.11, 3.12, and 3.13)
- About 400 MB of disk space for the virtual environment
- No GPU, no API keys, no network access after installation

Check your Python version:

```bash
python --version
```

If that prints something below 3.11, install a newer Python before continuing.
pandas, scikit-learn, and numpy all set the floor here, not this project's own code.

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
Model        : rf
Rows         : 1500 (960 train / 240 calibration / 300 test)
MAE          : 0.0241
RMSE         : 0.0307
R2           : 0.7397
CV R2        : 0.7218 +/- 0.0163 (5 folds)
Conformal    : +/-0.0560 for 90% coverage (measured 92.7% on the test split)
Top drivers  : methane_fraction, co2_fraction, h2s_ppm

Baseline ladder (cross-validated R2, best first):
  Linear regression    +0.7551 +/- 0.0178  <-- best
  Ridge (alpha=1)      +0.7551 +/- 0.0177
  Random forest        +0.7218 +/- 0.0163
  Mean (no model)      -0.0022 +/- 0.0026

  NOTE: 'Linear regression' scores higher than the random forest you just trained.
        The target was generated from a nearly linear formula, so the
        simpler model fits it better. Try: --model linear

Saved model  : ...\data\recovery_model.joblib
Saved metrics: ...\data\model_metrics.json
Reminder     : these scores describe synthetic data, not a real process.
```

The data is split **three** ways. The model trains on the first split; the second
is used only to calibrate the prediction interval and is never trained on; every
reported score comes from the third.

Two files are saved, both gitignored so a fresh clone runs Steps 4 and 5 itself:

- `data/recovery_model.joblib` — the fitted model the dashboard loads
- `data/model_metrics.json` — scores, both kinds of feature importance, the
  baseline ladder, and the conformal quantile

Three things in that output are worth pausing on:

> **1. The simplest model wins.** Linear regression beats the random forest, and the
> run tells you so. That is not a bug — the target was generated from a nearly linear
> formula, so a linear model is close to the *true* model. Reaching for a forest and
> never checking is the most common self-inflicted wound in applied ML; the ladder is
> here so this repo cannot commit it silently. Retrain with `--model linear` and R²
> rises to about 0.78 while the interval *narrows* to ±0.049.
>
> **2. `Mean (no model)` is the floor.** It scores ≈ 0 by construction. Anything that
> cannot beat it has learned nothing at all.
>
> **3. `Top drivers` is a rediscovery, not a discovery.** Methane fraction and CO₂ lead
> because they carry the largest coefficients in the formula we wrote.

### Step 6 — Run the tests

```bash
python -m pytest
```

Expected output:

```
155 passed
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
| **Model & data** | The baseline ladder, interval coverage, *both* kinds of feature importance, and the dataset with column statistics |
| **How this works** | The pipeline in six steps, and an explicit list of what the demo cannot tell you |

The **recovery fraction** KPI carries a 90% conformal interval whenever the model is
driving it. The interval is calibrated on data the model never trained on, and it holds
without assuming the errors are normally distributed — but only while new inputs
resemble the training data.

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
│   ├── baselines.py            # baseline ladder: is the complexity worth it?
│   ├── train_model.py          # model training, evaluation, export
│   ├── uncertainty.py          # split conformal prediction intervals
│   ├── predict.py              # model loading and prediction
│   └── viz.py                  # Altair chart builders
├── tests/
│   ├── conftest.py             # shared fixtures (own dataset + model, no shared state)
│   ├── test_app_smoke.py       # runs the dashboard headlessly via AppTest
│   ├── test_baselines.py
│   ├── test_generate_data.py
│   ├── test_optimization.py
│   ├── test_process_model.py
│   ├── test_safety_constraints.py
│   ├── test_train_and_predict.py
│   ├── test_uncertainty.py
│   └── test_viz.py
├── .github/
│   ├── workflows/ci.yml        # lint, types, tests on 3.11-3.13, fresh-clone
│   └── dependabot.yml
├── .pre-commit-config.yaml     # the same gates, before the commit exists
├── Dockerfile                  # known-good environment, non-root, health check
├── pyproject.toml              # metadata + pytest, ruff, mypy, coverage config
├── requirements.txt            # what the project needs
├── constraints.txt             # exact pins that are known to work together
├── LICENSE                     # MIT, plus an educational-use disclaimer
└── README.md
```

`src/config.py` is the single source of truth for column names and file paths, so the
generator, trainer, predictor, and dashboard can never silently disagree about them.

It is also where the physical constants live, each stated **on the same reference
basis** (0 °C, 101.325 kPa) and shown with its derivation. Mixing bases is a classic
quiet error: this repo previously paired a 0 °C heating value with a CO₂ factor of
1.90 kg/Nm³, which matches no standard condition and was 3.2% low against the
stoichiometric 1.963 kg/Nm³.

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
| `--model` | `rf` | Estimator: `rf`, `linear`, or `ridge` |
| `--n-estimators` | `200` | Number of trees (forest only) |
| `--min-samples-leaf` | `3` | Minimum samples per leaf (forest only) |
| `--test-size` | `0.2` | Held-out test fraction |
| `--calibration-size` | `0.2` | Fraction of non-test data reserved for the conformal interval (`0` disables it) |
| `--alpha` | `0.1` | Conformal miscoverage rate; `0.1` gives a 90% interval |
| `--cv-folds` | `5` | Cross-validation folds (`0` skips CV) |
| `--skip-baselines` | off | Skip the baseline ladder (the slow part) |
| `--seed` | `42` | Random seed |

Example — a larger forest on a bigger dataset:

```bash
python src/generate_data.py --rows 10000 --seed 1
python src/train_model.py --n-estimators 500 --test-size 0.25
```

Example — take the ladder's advice, and ask for a 95% interval:

```bash
python src/train_model.py --model linear --alpha 0.05
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
4. **Take the ladder's advice.** Retrain with `--model linear` and watch three things
   improve at once: R² rises (0.74 → 0.78), the conformal interval narrows
   (±0.056 → ±0.049), and the model becomes eight coefficients you can read. Simpler
   was better on every axis.
5. **Read the two importance columns against each other.** In the Model & data tab,
   compare permutation against the forest's built-in impurity importance. Impurity
   gives non-zero credit to flow, temperature, and power price — none of which appear
   in the generating formula. Permutation puts them at zero. You know which is right
   because you have the formula; in real work you never do, which is the point.
6. **Push the model outside what it has seen.** Set methane fraction to its extreme
   and watch the interval *not* widen. Conformal coverage assumes new inputs resemble
   the calibration data; when they stop, the guarantee lapses silently. That failure
   mode is the single most important thing on this page.
7. **Change the scenario weights.** Edit `WEIGHTS` in [`src/optimization.py`](src/optimization.py)
   and reload. Lower the capex penalty and watch the capital-heavy options climb.
8. **Change the physics.** The correlation lives in `generate_synthetic_data()` in
   [`src/generate_data.py`](src/generate_data.py). Make pressure matter more, regenerate,
   retrain, and confirm the feature importances move. Add a genuine interaction term and
   watch the random forest finally overtake linear regression.
9. **Add a feature.** Add a column to the generator, list it in `FEATURES` in
   [`src/config.py`](src/config.py), add a slider in the app, regenerate and retrain.
   The config module means those are the only places you need to touch.

---

## Developing on this repo

Four gates run on every push and pull request via
[GitHub Actions](.github/workflows/ci.yml). All four run locally with the same
commands, so nothing should ever fail first in CI.

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

| Gate | Command | What it enforces |
| --- | --- | --- |
| Lint | `ruff check .` | Import order, likely bugs, modern syntax |
| Format | `ruff format .` | One formatting style, not negotiable per file |
| Types | `mypy` | Type errors across `src/` and `tests/` |
| Tests | `pytest --cov` | 155 tests **and** a 95% coverage floor |

The coverage floor is a ratchet: raise it when coverage climbs, never lower it
to turn a red build green. If a change drops coverage, the fix is a test.

CI runs the suite on Python 3.11, 3.12, and 3.13. Two jobs are worth knowing
about:

- **`test`** runs the pipeline end to end from a clean checkout — generate,
  train, then test — in the order this README tells a newcomer to run it. If the
  documented steps break, CI fails.
- **`fresh-clone`** runs the suite with *no* dataset and *no* model present,
  because someone who clones the repo and types `pytest` first should not see a
  wall of red.

To catch all of this before a commit exists:

```bash
python -m pip install pre-commit
pre-commit install
```

### Reproducible installs

`requirements.txt` says what the project needs; `constraints.txt` pins exactly
what was last known to work together:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

Those pins are Python 3.12-specific — numpy 2.5 requires 3.12+. CI deliberately
installs *without* them so each Python in the matrix resolves what it actually
supports.

## Running with Docker

The image generates the dataset and trains the model at build time, so the
container starts with a working dashboard rather than an empty one:

```bash
docker build -t chemzero-ai .
docker run --rm -p 8501:8501 chemzero-ai
```

Then open **http://localhost:8501**. It runs as a non-root user and exposes a
health check on `/_stcore/health`.

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
| `ImportError: cannot import name ... from 'src.config'` while the app is running | Streamlit hot-reloads the script but keeps imported modules cached, so a *newly added* name in `src/` is not picked up. Stop the server (`Ctrl+C`) and start it again. |
| `ruff format` wants to change files you did not touch | Run `ruff format .` once and commit the result; the formatter is the source of truth, not a suggestion. |
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
- **The interval is conditional.** Conformal coverage holds while new inputs resemble
  the calibration data. Push the sliders somewhere the training data never went and the
  guarantee lapses — without the interval widening to warn you.

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

## License

MIT — see [LICENSE](LICENSE). The license carries an additional notice: the warranty
disclaimer applies with particular force to any use of this code in an engineering,
process-safety, or production context.

## Important principle

AI can help engineers identify patterns and compare options.
**AI does not replace validated process-safety systems or qualified engineering judgement.**
