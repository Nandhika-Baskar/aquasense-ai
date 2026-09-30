# AquaSense AI

**AI-Based Water Quality Prediction and Contamination Risk Classification using IoT Sensor Data**

A local Streamlit application that classifies water-quality sensor readings as
**LOW / MEDIUM / HIGH** contamination risk using a Random Forest classifier, and
shows how confident the model is, why it decided that, and how well it performs.

Everything runs locally. No external APIs, no database, no authentication, no
container runtime.

---

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\generate_dataset.py     # only needed once
python scripts\train_model.py          # only needed once
streamlit run app\main.py
```

Open <http://localhost:8501>.

> **Use Python 3.12.** The Python 3.14 install at `C:\Python314` on this machine
> is missing its standard library folder and is broken. Always activate `.venv`
> first.

---

## Pages

| Page | What it does |
|------|--------------|
| **Dashboard** | Five sensor cards, the live risk card, system status, trend and scatter charts |
| **Live Monitoring** | Simulated sensor stream, guideline check, reading log, CSV download |
| **AI Prediction** | Manual sensor entry, risk class with probability, contributing factors, explanation |
| **Analytics** | Trends, per-sensor distributions, correlation matrix, class mix, feature importance |
| **ML Performance** | Accuracy / precision / recall / F1, confusion matrix, feature importance, classification report |
| **Dataset Explorer** | Dataset statistics, preview, descriptive stats, CSV upload |
| **System** | Architecture, ML pipeline, limitations, future IoT integration |

---

## The model

**Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)

| Setting | Value |
|---------|-------|
| Trees | 300 |
| class_weight | `balanced` |
| Features | `pH`, `TDS`, `Turbidity`, `Temperature`, `Conductivity` |
| Target | `contamination_risk` (LOW / MEDIUM / HIGH) |
| Split | 80 / 20 stratified, `random_state=42` |
| Training samples | 940 |
| Testing samples | 236 |

The preprocessing path — median imputation, then standard scaling — lives
**inside** the scikit-learn pipeline, so training and inference cannot drift
apart and a missing sensor reading cannot break a prediction.

### Measured performance

Computed on the held-out test set of 236 samples. Regenerate with
`python scripts\train_model.py`.

| Metric | Macro-averaged | Weighted |
|--------|----------------|----------|
| Accuracy | **90.25%** | — |
| Precision | 89.44% | 90.26% |
| Recall | 89.37% | 90.25% |
| F1 | 89.37% | 90.22% |

5-fold cross-validated accuracy on the training split: **92.45% ± 0.92%**.
Macro averaging is reported as the headline because the three classes are not
perfectly balanced.

Confusion matrix (rows = actual, columns = predicted):

|  | LOW | MEDIUM | HIGH |
|---|---|---|---|
| **LOW** | 92 | 0 | 0 |
| **MEDIUM** | 1 | 61 | 9 |
| **HIGH** | 0 | 13 | 60 |

Feature importance (Gini):

| Sensor | Importance |
|--------|-----------|
| Turbidity | 36.6% |
| TDS | 27.8% |
| Conductivity | 21.2% |
| pH | 10.3% |
| Temperature | 4.0% |

Full metrics, including the classification report, are written to
`models/metrics.json`. Every number in the UI is read from there or recomputed
at runtime — nothing is hard-coded.

---

## The dataset

> **Synthetic demonstration data.** Generated programmatically by
> `scripts/generate_dataset.py` with a fixed random seed. This is **not**
> real-world laboratory or field measurement data, and no result from this
> project should be presented as real-world evidence. It exists so the pipeline
> and dashboard can be demonstrated end to end.

`data/water_quality.csv` — 1176 rows, 8 columns, 0 missing values.

| Class | Count | Share |
|-------|-------|-------|
| LOW | 461 | 39.2% |
| MEDIUM | 352 | 29.9% |
| HIGH | 363 | 30.9% |

| Sensor | Min | Max | Mean | Guideline band |
|--------|-----|-----|------|----------------|
| pH | 4.50 | 9.21 | 6.93 | 6.5 – 8.5 |
| TDS | 30.00 | 2419.49 | 751.79 | ≤ 500 mg/L |
| Turbidity | 0.20 | 106.59 | 20.31 | ≤ 5 NTU |
| Temperature | 5.05 | 45.00 | 26.76 | 15 – 35 °C |
| Conductivity | 41.47 | 2500.00 | 1189.66 | 200 – 800 µS/cm |

**How labels are produced.** Readings come from a mixture of three hand-tuned
water-body archetypes, clipped to plausible limits. Conductivity is *derived*
from TDS rather than drawn independently, because the two are physically coupled
(correlation 0.98). The label is then assigned by an explicit rule in
`app/data_gen.py`:

1. Per-sensor severity — 0 inside the guideline band, growing outside it.
   Distance is measured in "guideline widths" and passed through `r / (1 + r)`,
   which saturates but never loses ordering.
2. Weighted risk score — turbidity 0.30, TDS 0.26, pH 0.22, conductivity 0.12,
   temperature 0.10. `< 0.28` → LOW, `< 0.50` → MEDIUM, else HIGH.
3. **Safety floor** — guidelines are not averages, so one grossly failed sensor
   overrides the score: past `0.25` guideline widths → at least MEDIUM, past
   `6.0` widths → HIGH.
4. Controlled overlap — Gaussian noise (sd 0.07) so ~36% of rows sit near a
   class boundary and the task is not trivially separable.

Regeneration is bit-for-bit reproducible (verified by SHA-256):

```powershell
python scripts\generate_dataset.py
python scripts\generate_dataset.py --rows 2000 --seed 7
```

The generator validates itself on every run — all compliant readings must be
LOW, no LOW row may be out of band, no MEDIUM row may be grossly failed, and
risk must increase monotonically — and only writes the CSV if every check passes.

---

## Project structure

```
water-quality-ai/
├── app/
│   ├── main.py              # entry point: sidebar nav + routing
│   ├── config.py            # branding, palette, sensor bands, label rules, paths
│   ├── data_gen.py          # synthetic dataset generator + verification
│   ├── preprocessing.py     # imputer + scaler + classifier pipeline
│   ├── modeling.py          # train, evaluate, predict, explain, persist
│   ├── simulator.py         # simulated "live" readings
│   ├── charts.py            # Plotly figure builders
│   ├── components.py        # reusable HTML cards and status blocks
│   ├── theme.py             # all custom CSS
│   ├── context.py           # AppContext shared by the pages
│   └── views/               # one module per page
├── data/
│   └── water_quality.csv    # the synthetic dataset
├── models/
│   ├── random_forest_model.joblib
│   └── metrics.json
├── scripts/
│   ├── generate_dataset.py
│   └── train_model.py
├── tests/
│   ├── smoke_test.py        # app boots and renders the dashboard
│   ├── test_modeling.py     # metrics, prediction, validation
│   └── test_pages.py        # every page renders, metrics match metrics.json
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

---

## Tests

```powershell
python tests\smoke_test.py       # app boots, dashboard renders
python tests\test_modeling.py    # metrics, prediction, input validation
python tests\test_pages.py       # all 7 pages, metrics tied to metrics.json
```

`test_pages.py` renders every page through Streamlit's own test harness and
fails on any exception. It also asserts that the accuracy, precision, recall and
F1 values rendered on the ML Performance page match `models/metrics.json`, so
the dashboard cannot drift away from the real model output.

---

## Limitations

- **Trained on synthetic data.** The 90.25% accuracy measures how well the model
  learned the generating rule, not how it would perform on real water.
- **Risk classification is not potability.** LOW / MEDIUM / HIGH describe
  relative risk against the guideline bands used here. They are not a health
  assessment and not a certification that water is safe to drink.
- **Five sensors only.** Real drinking-water standards also cover pathogens,
  heavy metals, pesticides and disinfection by-products, none of which these
  sensors can observe.
- **No live hardware.** The monitoring page replays a simulated stream derived
  from the dataset. There is no serial, MQTT or network ingestion.
- **Correlated inputs.** Conductivity is derived from TDS, so two of the five
  features carry partly the same information.
- **Single holdout.** The headline accuracy comes from one stratified 20% test
  split; the 5-fold cross-validated score is shown next to it for context.