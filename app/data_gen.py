"""
Synthetic DEMONSTRATION dataset generator.

IMPORTANT
---------
Everything produced here is SIMULATED. It is not real-world laboratory or field
data. Values are drawn from hand-tuned distributions that loosely follow
drinking-water guidelines, and the risk labels are produced by an explicit,
readable rule (see `risk_score` and `assign_labels`).

Run it with:  python scripts/generate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------
# Each archetype describes a plausible water body. Samples are drawn from a
# mixture of archetypes, then labelled by the rule below -- the archetype is
# never used as the label itself.
ARCHETYPES: dict[str, dict] = {
    "clean": {
        "weight": 0.40,
        "pH": (7.4, 0.45),
        "TDS": (230.0, 90.0),
        "Turbidity": (2.2, 1.1),
        "Temperature": (24.0, 3.5),
    },
    "moderate": {
        "weight": 0.35,
        "pH": (6.9, 0.75),
        "TDS": (900.0, 220.0),
        "Turbidity": (22.0, 10.0),
        "Temperature": (27.0, 5.0),
    },
    "contaminated": {
        "weight": 0.25,
        "pH": (6.4, 1.00),
        "TDS": (1350.0, 380.0),
        "Turbidity": (45.0, 22.0),
        "Temperature": (30.0, 6.0),
    },
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _draw_sensors(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw n rows of sensor readings from the archetype mixture."""
    names = list(ARCHETYPES)
    weights = np.array([ARCHETYPES[k]["weight"] for k in names], dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(len(names), size=n, p=weights)

    data = {}
    for feature in ("pH", "TDS", "Turbidity", "Temperature"):
        means = np.array([ARCHETYPES[k][feature][0] for k in names])[chosen]
        sds = np.array([ARCHETYPES[k][feature][1] for k in names])[chosen]
        values = rng.normal(means, sds)
        data[feature] = np.clip(values, cfg.CLIP_LOW[feature], cfg.CLIP_HIGH[feature])

    # Conductivity is physically coupled to dissolved solids
    # (EC in uS/cm ~ TDS in mg/L / 0.5-0.7), so derive it instead of
    # drawing it independently. Real sensor pairs are strongly correlated.
    factor = rng.normal(0.62, 0.06, size=n)
    factor = np.clip(factor, 0.45, 0.80)
    ec = data["TDS"] / factor
    data["Conductivity"] = np.clip(ec, cfg.CLIP_LOW["Conductivity"], cfg.CLIP_HIGH["Conductivity"])

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Labelling rule
# ---------------------------------------------------------------------------
def sensor_severity(name: str, values: np.ndarray) -> np.ndarray:
    """Severity of one sensor, on 0.0 (safe) to 1.0 (extremely bad).

    Inside the guideline band the severity is 0. Outside it, the distance is
    measured in "guideline widths" and passed through r / (1 + r), which
    saturates towards 1 but -- unlike np.clip -- never loses the ordering, so a
    worse reading always scores higher than a mildly bad one.
    """
    lo, hi = cfg.SAFE_LOW[name], cfg.SAFE_HIGH[name]
    exceed = np.maximum(np.maximum(lo - values, values - hi), 0.0)
    ratio = exceed / cfg.SEVERITY_SCALE[name]
    return ratio / (1.0 + ratio)


def exceedance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor distance outside the safe band, in guideline widths.

    0.0 means the reading is compliant. 0.25 means a quarter of a guideline
    width past the limit; 1.0 means one full guideline width past it.
    """
    ratios = {}
    for feature in cfg.FEATURES:
        lo, hi = cfg.SAFE_LOW[feature], cfg.SAFE_HIGH[feature]
        values = df[feature].to_numpy()
        exceed = np.maximum(np.maximum(lo - values, values - hi), 0.0)
        ratios[feature] = exceed / cfg.SEVERITY_SCALE[feature]
    return pd.DataFrame(ratios, index=df.index)


def risk_score(df: pd.DataFrame) -> np.ndarray:
    """Weighted sum of per-sensor severities. 0 = clean, 1 = worst case."""
    score = np.zeros(len(df), dtype=float)
    for feature, weight in cfg.FEATURE_WEIGHTS.items():
        score += weight * sensor_severity(feature, df[feature].to_numpy())
    return score


def _floor_class(worst_ratio: np.ndarray) -> np.ndarray:
    """Worst-sensor override, so one failed sensor is never averaged away.

    Returns a class *rank* per row: 0 = no override, 1 = at least MEDIUM,
    2 = HIGH.
    """
    return np.where(
        worst_ratio >= cfg.FLOOR_HIGH_RATIO,
        2,
        np.where(worst_ratio >= cfg.FLOOR_MEDIUM_RATIO, 1, 0),
    )


def assign_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Turn the sensor readings into LOW / MEDIUM / HIGH.

    Two stages:
      1. weighted severity score -> base class (with a little noise so that
         borderline readings genuinely overlap);
      2. worst-sensor safety floor -> a grossly failed sensor upgrades the class.
    """
    noisy = risk_score(df) + rng.normal(0.0, cfg.LABEL_NOISE_SD, size=len(df))
    base_rank = np.where(
        noisy < cfg.LOW_MAX_SCORE,
        0,
        np.where(noisy < cfg.HIGH_MIN_SCORE, 1, 2),
    )

    worst = exceedance_ratios(df).max(axis=1).to_numpy()
    final_rank = np.maximum(base_rank, _floor_class(worst))

    return pd.Series(np.array(cfg.CLASSES)[final_rank], index=df.index, name=cfg.TARGET)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with any missing value in any expected column.

    Returns (clean_df, n_dropped_rows).
    """
    before = len(df)
    clean = df.dropna(subset=cfg.EXPECTED_COLUMNS).reset_index(drop=True)
    return clean, before - len(clean)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def generate_dataset(
    n_samples: int = cfg.N_SAMPLES,
    seed: int = cfg.RANDOM_SEED,
    simulate_dropouts: bool = True,
) -> pd.DataFrame:
    """Build the full synthetic dataset (including the labelled dropouts)."""
    rng = np.random.default_rng(seed)

    sensors = _draw_sensors(rng, n_samples)
    sensors.insert(0, cfg.ID_COL, [f"WQ-{i:05d}" for i in range(1, n_samples + 1)])

    # Simulated 15-minute readings spread over 30 days.
    start = pd.Timestamp("2026-09-01 00:00:00")
    offsets = np.sort(rng.integers(0, 30 * 24 * 4, size=n_samples))
    sensors.insert(1, cfg.TIME_COL, start + pd.to_timedelta(offsets * 15, unit="m"))

    sensors[cfg.TARGET] = assign_labels(sensors, rng)

    if simulate_dropouts:
        n_hit = int(round(n_samples * cfg.SENSOR_DROPOUT_RATE))
        rows = rng.choice(len(sensors), size=n_hit, replace=False)
        for row in rows:
            feature = cfg.FEATURES[rng.integers(0, len(cfg.FEATURES))]
            sensors.iat[row, sensors.columns.get_loc(feature)] = np.nan

    return sensors


def build_clean_dataset(n_samples: int = cfg.N_SAMPLES, seed: int = cfg.RANDOM_SEED):
    """Generate, then clean. Returns (clean_df, n_dropped_rows)."""
    return clean_dataset(generate_dataset(n_samples, seed, simulate_dropouts=True))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_dataset(df: pd.DataFrame) -> dict:
    """Run every check needed before the data is trusted for training."""
    missing_cols = [c for c in cfg.EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in cfg.EXPECTED_COLUMNS]

    counts = df[cfg.TARGET].value_counts().reindex(cfg.CLASSES, fill_value=0)
    shares = counts / len(df)
    imbalance_ratio = counts.max() / max(counts.min(), 1)

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "missing_columns": missing_cols,
        "unexpected_columns": extra_cols,
        "n_missing_values": int(df.isna().sum().sum()),
        "nulls_per_column": df.isna().sum().to_dict(),
        "duplicate_ids": int(df[cfg.ID_COL].duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": counts.to_dict(),
        "class_shares": {k: round(v, 4) for k, v in shares.items()},
        "imbalance_ratio": round(float(imbalance_ratio), 2),
        "sensor_min": df[cfg.FEATURES].min().round(3).to_dict(),
        "sensor_max": df[cfg.FEATURES].max().round(3).to_dict(),
        "sensor_mean": df[cfg.FEATURES].mean().round(3).to_dict(),
        "describe": df[cfg.FEATURES].describe().round(3),
    }


def verify_labels(df: pd.DataFrame) -> dict:
    """Sanity-check that the labels are physically defensible.

    These catch labelling bugs that look fine in a class-distribution printout
    but are nonsense to a human reviewer -- for example a reading with
    turbidity far past the limit being labelled LOW.
    """
    ratios = exceedance_ratios(df)
    worst = ratios.max(axis=1)
    labels = df[cfg.TARGET]

    compliant = worst == 0
    low = labels == cfg.CLASSES[0]
    medium = labels == cfg.CLASSES[1]

    medians = {
        cls: round(float(worst[labels == cls].median()), 3) for cls in cfg.CLASSES
    }

    return {
        "fully_compliant_rows": int(compliant.sum()),
        "compliant_not_low": int((labels[compliant] != cfg.CLASSES[0]).sum()),
        "low_beyond_floor": int((low & (worst >= cfg.FLOOR_MEDIUM_RATIO)).sum()),
        "medium_beyond_floor": int((medium & (worst >= cfg.FLOOR_HIGH_RATIO)).sum()),
        "max_worst_ratio_low": round(float(worst[low].max()), 3),
        "median_worst_by_class": medians,
        "classes_ordered": medians[cfg.CLASSES[0]] <= medians[cfg.CLASSES[1]] <= medians[cfg.CLASSES[2]],
        "class_sensor_max": df.loc[low, cfg.FEATURES].max().round(2).to_dict(),
    }


if __name__ == "__main__":
    data, dropped = build_clean_dataset()
    report = verify_dataset(data)
    print(f"rows={report['n_rows']} cols={report['n_cols']} dropped={dropped}")
    print(report["class_counts"])