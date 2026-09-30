"""Shared application state passed to every view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app import config as cfg


@dataclass
class AppContext:
    """Everything a view needs, loaded once per app run."""

    df: pd.DataFrame
    pipeline: Any
    metrics: dict
    feature_importance: dict
    split: dict
    model: dict
    dataset: dict
    trained_at: str

    def metric(self, name: str) -> float:
        return float(self.metrics[name])

    def class_counts(self) -> dict[str, int]:
        return {cls: int((self.df[cfg.TARGET] == cls).sum()) for cls in cfg.RISK_ORDER}