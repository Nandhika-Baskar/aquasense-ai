"""
Preprocessing for the AquaSense AI classifier.

The whole path from raw sensor values to a prediction lives in a single
scikit-learn Pipeline, so training and inference cannot drift apart:

    SimpleImputer(median)  ->  StandardScaler  ->  RandomForestClassifier

A Random Forest does not need scaling to work, but keeping the scaler in the
pipeline means the same code path is ready if a linear or distance-based model
is swapped in later.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import config as cfg


def build_preprocessor() -> SimpleImputer:
    """Median imputation, so a dropped sensor reading cannot break a prediction."""
    return SimpleImputer(strategy="median")


def build_classifier() -> RandomForestClassifier:
    """
    Random Forest classifier.

    class_weight="balanced" stops the majority class (LOW) from dominating,
    which matters because the three classes are not perfectly equal.
    """
    return RandomForestClassifier(**cfg.MODEL_PARAMS)


def build_pipeline() -> Pipeline:
    """Full preprocessing + model pipeline."""
    return Pipeline(
        [
            ("imputer", build_preprocessor()),
            ("scaler", StandardScaler()),
            ("model", build_classifier()),
        ]
    )


def get_estimator(pipeline: Pipeline) -> RandomForestClassifier:
    """Unwrap the fitted classifier from a pipeline."""
    return pipeline.named_steps["model"]