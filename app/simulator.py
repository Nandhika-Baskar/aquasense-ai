"""
Simulated sensor readings for the Live Monitoring view.

These readings are produced from the demonstration dataset. There is no
physical device in this project, and nothing here should be read as a live
hardware feed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config as cfg


def simulated_readings(df: pd.DataFrame, nonce: int = 0) -> dict[str, float]:
    """
    A plausible "current" reading set.

    The baseline is a real row drawn from the most recent part of the simulated
    deployment window, so the reading is always a combination the dataset
    actually contains rather than an arbitrary synthetic value. `nonce` advances
    the selection and adds a small amount of jitter on refresh.
    """
    rng = np.random.default_rng(cfg.SIM_SEED_BASE + int(nonce))

    ordered = df.sort_values(cfg.TIME_COL) if cfg.TIME_COL in df.columns else df
    recent = ordered.tail(max(len(ordered) // 10, 1))
    baseline = recent.iloc[rng.integers(len(recent))]

    values: dict[str, float] = {}
    for feature in cfg.FEATURES:
        spread = float(df[feature].std()) * 0.02
        jittered = float(baseline[feature]) + rng.normal(0.0, spread)
        values[feature] = float(np.clip(jittered, cfg.CLIP_LOW[feature], cfg.CLIP_HIGH[feature]))
    return values


def recent_window(df: pd.DataFrame, points: int = 72) -> pd.DataFrame:
    """The most recent `points` readings, in time order."""
    ordered = df.sort_values(cfg.TIME_COL) if cfg.TIME_COL in df.columns else df
    return ordered.tail(points).reset_index(drop=True)


def sample_interval(df: pd.DataFrame) -> pd.Timedelta:
    """Median spacing between consecutive readings."""
    if cfg.TIME_COL not in df.columns or len(df) < 2:
        return pd.Timedelta(0)
    delta = df[cfg.TIME_COL].sort_values().diff().median()
    return delta if pd.notna(delta) else pd.Timedelta(0)


def smoothing_span(df: pd.DataFrame, window: int) -> str:
    """
    Describe a rolling window in words, for the chart captions.

    Reported as approximate because the simulated timestamps are not perfectly
    evenly spaced, so the window covers a slightly varying span of real time.
    """
    size = int(window or 1)
    if size <= 1:
        return "no smoothing, every reading is plotted"
    hours = sample_interval(df).total_seconds() * size / 3600
    if hours < 48:
        return f"{size} readings averaged, roughly {hours:.0f} hours"
    return f"{size} readings averaged, roughly {hours / 24:.1f} days"


def trend_caption(df: pd.DataFrame, window: int) -> str:
    """
    The note printed under a smoothed trend chart.

    It says plainly what the smoothing does and what it cannot claim. In this
    demonstration dataset consecutive readings are independent draws from an
    archetype mixture (lag-1 autocorrelation is ~0), so the line shows the local
    average level, not a physical drift path. Saying so here stops the smoothed
    curve from being misread as a physical excursion.
    """
    return (
        f"Trend smoothing: {smoothing_span(df, window)}. The shaded band marks "
        "readings inside the guideline. The line is a centred moving average used "
        "for legibility only: it shows the local average level of each sensor, and "
        "in this synthetic dataset consecutive readings are independent draws "
        "rather than a continuous physical signal. The model, the reported metrics "
        "and the readings log all use the unaltered data, and no reading is "
        "discarded or altered anywhere else."
    )