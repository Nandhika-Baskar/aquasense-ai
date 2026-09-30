"""
Project-wide configuration for:

"AI-Based Water Quality Prediction and Contamination Risk Classification
 using IoT Sensor Data"

Everything a beginner needs to tweak in one place: paths, sensor schema,
the label rules, and the fixed random seed.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_CSV = DATA_DIR / "water_quality.csv"

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 1200

# --- Columns ----------------------------------------------------------------
ID_COL = "sample_id"
TIME_COL = "timestamp"

FEATURES = ["pH", "TDS", "Turbidity", "Temperature", "Conductivity"]
TARGET = "contamination_risk"
CLASSES = ["LOW", "MEDIUM", "HIGH"]

EXPECTED_COLUMNS = [ID_COL, TIME_COL, *FEATURES, TARGET]

# --- Sensor metadata --------------------------------------------------------
SENSOR_UNITS = {
    "pH": "pH units",
    "TDS": "mg/L",
    "Turbidity": "NTU",
    "Temperature": "\u00b0C",
    "Conductivity": "\u00b5S/cm",
}

# Drinking-water guideline bands used by the labelling rule.
SAFE_LOW = {
    "pH": 6.5,
    "TDS": 0.0,  # only an upper bound matters
    "Turbidity": 0.0,  # only an upper bound matters
    "Temperature": 15.0,
    "Conductivity": 200.0,
}
SAFE_HIGH = {
    "pH": 8.5,
    "TDS": 500.0,
    "Turbidity": 5.0,
    "Temperature": 35.0,
    "Conductivity": 800.0,
}

# Physical plausibility limits used to clip generated values.
CLIP_LOW = {
    "pH": 4.5,
    "TDS": 30.0,
    "Turbidity": 0.2,
    "Temperature": 3.0,
    "Conductivity": 25.0,
}
CLIP_HIGH = {
    "pH": 10.5,
    "TDS": 2500.0,
    "Turbidity": 150.0,
    "Temperature": 45.0,
    "Conductivity": 2500.0,
}

# --- Labelling rule ---------------------------------------------------------
# Severity of each sensor: 0.0 inside the safe band, growing as the value
# leaves it. The scale is the size of one "unit of violation" and is chosen
# relative to the drinking-water guideline itself, so a reading of one
# guideline-width beyond the limit gives severity ~1.
#   TDS 814 mg/L -> 0.63 units past the 500 mg/L limit  -> MEDIUM, not LOW
SEVERITY_SCALE = {
    "pH": 1.0,
    "TDS": 500.0,
    "Turbidity": 5.0,
    "Temperature": 7.5,
    "Conductivity": 600.0,
}

# Relative importance of each sensor in the overall risk score.
FEATURE_WEIGHTS = {
    "Turbidity": 0.30,
    "TDS": 0.26,
    "pH": 0.22,
    "Conductivity": 0.12,
    "Temperature": 0.10,
}

# Score thresholds -> class. A single sensor far outside its band can never
# pull the weighted average down to LOW, because the transform below keeps the
# order of badness instead of capping it.
LOW_MAX_SCORE = 0.28
HIGH_MIN_SCORE = 0.50

# Safety floor, applied after the score. Drinking-water guidelines are not
# averages: one grossly failed sensor is unacceptable no matter how clean the
# other readings are. These are measured in "guideline widths" past the limit.
#   turbidity  6.3 NTU  -> 0.25 widths over -> at least MEDIUM
#   turbidity 35   NTU  -> 6.0  widths over -> HIGH (gross failure)
FLOOR_MEDIUM_RATIO = 0.25
FLOOR_HIGH_RATIO = 6.0

# Gaussian noise added to the score so that borderline samples overlap.
# This is what stops the classification problem from being trivially perfect.
LABEL_NOISE_SD = 0.07

# Share of readings where one sensor drops out (simulated sensor failure).
# These rows are removed before the CSV is written.
SENSOR_DROPOUT_RATE = 0.02

# --- Presentation -----------------------------------------------------------
CLASS_COLORS = {
    "LOW": "#2B8A3E",
    "MEDIUM": "#E8590C",
    "HIGH": "#C92A2A",
}

DATASET_NOTE = (
    "SYNTHETIC DEMONSTRATION DATA - generated programmatically with a fixed "
    "random seed. Not real-world laboratory or field measurements."
)

# --- Product branding -------------------------------------------------------
APP_NAME = "AquaSense AI"
APP_SUBTITLE = "Intelligent Water Quality & Contamination Intelligence"
APP_VERSION = "1.0.0"
APP_MODEL_NAME = "Random Forest Classifier"

# --- Palette ----------------------------------------------------------------
COLOR_PRIMARY = "#0B7285"
COLOR_PRIMARY_DARK = "#095C6B"
COLOR_INK = "#0F1B23"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E3E8EC"
COLOR_SURFACE = "#FFFFFF"
COLOR_CANVAS = "#F4F7F9"
COLOR_SIDEBAR = "#0C2530"

COLOR_OK = "#1F9D55"
COLOR_WARN = "#D98324"
COLOR_DANGER = "#D64545"
COLOR_NEUTRAL = "#64748B"

RISK_COLORS = {
    "LOW": "#1F9D55",
    "MEDIUM": "#D98324",
    "HIGH": "#D64545",
}
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]

CHART_PALETTE = ["#0B7285", "#D98324", "#1F9D55", "#D64545", "#5B7C99", "#7A5C99"]

PLOTLY_TEMPLATE = "plotly_white"

# --- Model artefacts --------------------------------------------------------
MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

TEST_SIZE = 0.20

# --- Navigation -------------------------------------------------------------
NAV_ITEMS = [
    ("Dashboard", "◉", "app.views.dashboard"),
    ("Live Monitoring", "◈", "app.views.monitoring"),
    ("AI Prediction", "◐", "app.views.prediction"),
    ("Analytics", "◑", "app.views.analytics"),
    ("ML Performance", "◒", "app.views.performance"),
    ("Dataset Explorer", "◓", "app.views.explorer"),
    ("System", "◔", "app.views.system"),
]

# --- Simulation -------------------------------------------------------------
# The "live" readings are simulated from the demonstration dataset. They are
# never presented as coming from a physical device.
SIMULATION_NOTE = (
    "Simulated sensor stream generated from demonstration data. "
    "No physical IoT hardware is connected."
)
SIM_SEED_BASE = 7_000

# --- Trend chart smoothing ---------------------------------------------------
# Readings are ~37 minutes apart, so a 12-reading centred rolling mean spans
# roughly 7 hours. That is long enough to read as a trend and short enough to
# still follow a real excursion outside a guideline band.
TREND_ROLLING_DEFAULT = 12
TREND_ROLLING_MAX = 48