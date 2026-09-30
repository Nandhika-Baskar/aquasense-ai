"""
Training, evaluation and inference for the AquaSense AI risk classifier.

Every number reported by the dashboard comes from the functions in this module.
Nothing is hard-coded or estimated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app import config as cfg
from app.preprocessing import build_pipeline, get_estimator


class DatasetError(ValueError):
    """Raised when the dataset is missing columns or unusable."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset(df: pd.DataFrame) -> None:
    """Fail loudly rather than training on something broken."""
    missing = [c for c in [*cfg.FEATURES, cfg.TARGET] if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required column(s): {', '.join(missing)}")

    if df.empty:
        raise DatasetError("Dataset is empty.")

    if df[cfg.TARGET].isna().any():
        raise DatasetError(f"Column '{cfg.TARGET}' contains missing values.")

    unexpected = sorted(set(df[cfg.TARGET].unique()) - set(cfg.RISK_ORDER))
    if unexpected:
        raise DatasetError(f"Unexpected risk label(s) in target: {unexpected}")

    if df[cfg.FEATURES].isna().any().any():
        raise DatasetError(
            "Sensor columns contain missing values. Clean the dataset before training."
        )

    # Stratified splitting needs enough rows per class to hold out every label.
    min_rows = len(cfg.RISK_ORDER) * 10
    if len(df) < min_rows:
        raise DatasetError(
            f"Only {len(df)} rows available; at least {min_rows} are needed to train and "
            f"validate a {len(cfg.RISK_ORDER)}-class model."
        )


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read and validate the demonstration dataset."""
    path = Path(path or cfg.DATASET_CSV)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Generate it first with: python scripts/generate_dataset.py"
        )
    df = pd.read_csv(path)
    if cfg.TIME_COL in df.columns:
        df[cfg.TIME_COL] = pd.to_datetime(df[cfg.TIME_COL], errors="coerce")
    validate_dataset(df)
    return df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[cfg.FEATURES].copy(), df[cfg.TARGET].copy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame, test_size: float = cfg.TEST_SIZE, seed: int = cfg.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split, so all three classes keep their proportion."""
    X, y = features_and_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "f1_weighted": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = cfg.TEST_SIZE, seed: int = cfg.RANDOM_SEED
) -> dict:
    """
    Train the Random Forest and evaluate it on a held-out stratified test set.

    Returns a bundle dict holding the fitted pipeline and every metric the
    dashboard displays.
    """
    validate_dataset(df)
    labels = cfg.RISK_ORDER

    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = _classification_metrics(y_test, y_pred, labels)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    estimator = get_estimator(pipeline)
    importance = dict(zip(cfg.FEATURES, map(float, estimator.feature_importances_)))

    # Robustness check: the test set is one split, this shows the spread.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, n_jobs=-1)

    return {
        "pipeline": pipeline,
        "metrics": {
            **metrics,
            "confusion_labels": labels,
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=labels, zero_division=0, digits=4
            ),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_folds": len(cv_scores),
        },
        "feature_importance": importance,
        "split": {
            "test_size": test_size,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_total": int(len(df)),
            "features": list(cfg.FEATURES),
            "classes": labels,
        },
        "model": {
            "name": cfg.APP_MODEL_NAME,
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "params": dict(cfg.MODEL_PARAMS),
            "n_trees": int(estimator.n_estimators),
            "classes": list(estimator.classes_),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def train_from_file(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Load data, train, evaluate, then persist the model and its metrics."""
    dataset_path = Path(dataset_path or cfg.DATASET_CSV)
    model_path = Path(model_path or cfg.MODEL_PATH)
    metrics_path = Path(metrics_path or cfg.METRICS_PATH)

    df = load_dataset(dataset_path)
    bundle = train_and_evaluate(df)

    bundle["dataset"] = {
        "path": str(dataset_path.relative_to(cfg.PROJECT_ROOT)),
        "rows": int(len(df)),
        "sha256": file_sha256(dataset_path),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["pipeline"], model_path)
    metrics_path.write_text(json.dumps(serialisable(bundle), indent=2), encoding="utf-8")

    if verbose:
        m = bundle["metrics"]
        print(f"Saved model    -> {model_path}")
        print(f"Saved metrics  -> {metrics_path}")
        print(
            f"Train/test     -> {bundle['split']['n_train']} / {bundle['split']['n_test']} "
            f"(test_size={cfg.TEST_SIZE})"
        )
        print(f"Accuracy       -> {m['accuracy']:.4f}")
        print(
            "Macro scores   -> "
            f"P {m['precision_macro']:.4f} | R {m['recall_macro']:.4f} | F1 {m['f1_macro']:.4f}"
        )
        print(
            f"5-fold CV      -> {m['cv_accuracy_mean']:.4f} +/- {m['cv_accuracy_std']:.4f}"
        )

    return bundle


def serialisable(bundle: dict) -> dict:
    """Bundle without the fitted estimator, safe to write as JSON."""
    return {k: v for k, v in bundle.items() if k != "pipeline"}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_bundle(model_path: Path | None = None, metrics_path: Path | None = None) -> dict:
    """
    Load the saved model plus its metrics.

    Raises DatasetError with an actionable message when artefacts are missing,
    so the dashboard can show a clean empty state instead of a stack trace.
    """
    model_path = Path(model_path or cfg.MODEL_PATH)
    metrics_path = Path(metrics_path or cfg.METRICS_PATH)

    if not model_path.exists():
        raise DatasetError(
            "No trained model found. Train one first with:  python scripts\\train_model.py"
        )
    if not metrics_path.exists():
        raise DatasetError(
            "Model metrics file is missing. Re-train with:  python scripts\\train_model.py"
        )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["pipeline"] = joblib.load(model_path)
    return payload


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_risk(pipeline, readings: dict[str, float]) -> dict:
    """
    Classify one set of sensor readings.

    `probabilities` is keyed by risk class so the dashboard never has to guess
    the column order of predict_proba (sklearn sorts classes alphabetically,
    which is HIGH / LOW / MEDIUM -- not the display order).
    """
    frame = pd.DataFrame([{f: float(readings[f]) for f in cfg.FEATURES}])

    predicted = str(pipeline.predict(frame)[0])
    raw = pipeline.predict_proba(frame)[0]
    classes = [str(c) for c in pipeline.classes_]
    probabilities = dict(zip(classes, map(float, raw)))

    ordered = sorted(cfg.RISK_ORDER, key=lambda c: probabilities.get(c, 0.0), reverse=True)

    return {
        "risk": predicted,
        "confidence": probabilities[predicted],
        "probabilities": probabilities,
        "ranked": ordered,
        "runner_up": ordered[1] if len(ordered) > 1 else None,
        "readings": {f: float(readings[f]) for f in cfg.FEATURES},
    }


def sensor_exceedance(name: str, value: float) -> dict:
    """How far one reading sits outside its guideline band, in guideline widths."""
    lo, hi = cfg.SAFE_LOW[name], cfg.SAFE_HIGH[name]
    exceed = max(lo - value, value - hi, 0.0)
    ratio = exceed / cfg.SEVERITY_SCALE[name]
    return {
        "sensor": name,
        "value": float(value),
        "unit": cfg.SENSOR_UNITS[name],
        "safe_low": lo,
        "safe_high": hi,
        "ratio": float(ratio),
        "in_band": bool(exceed == 0.0),
    }


def explain_prediction(
    result: dict, feature_importance: dict[str, float], top_n: int = 3
) -> list[dict]:
    """
    Build a short, factual explanation from the actual readings and model output.

    Every statement is derived from the numbers supplied and from the fitted
    model's feature importances. It describes relative risk classification
    only, and never claims the water is safe or unsafe to drink.
    """
    notes: list[dict] = []
    readings = result["readings"]

    exceedances = [sensor_exceedance(f, readings[f]) for f in cfg.FEATURES]
    outside = sorted([e for e in exceedances if not e["in_band"]], key=lambda e: -e["ratio"])

    ranked_importance = sorted(feature_importance.items(), key=lambda kv: -kv[1])

    for exc in outside[:top_n]:
        direction = "above" if exc["value"] > exc["safe_high"] else "below"
        limit = exc["safe_high"] if direction == "above" else exc["safe_low"]
        ratio_txt = (
            f"{exc['ratio']:.1f}x the guideline margin"
            if exc["ratio"] >= 1
            else f"{exc['ratio'] * 100:.0f}% past the guideline"
        )
        notes.append(
            {
                "title": f"{exc['sensor']} {direction} guideline",
                "detail": (
                    f"Reading {exc['value']:.1f} {exc['unit']} against a guideline of "
                    f"{limit:g} {exc['unit']} ({ratio_txt})."
                ),
                "level": "danger" if exc["ratio"] >= cfg.FLOOR_MEDIUM_RATIO else "warn",
            }
        )

    if not outside:
        notes.append(
            {
                "title": "All sensors within guideline bands",
                "detail": "Every reading sits inside its reference band for this model.",
                "level": "ok",
            }
        )

    drivers = ", ".join(f"{name} ({weight * 100:.0f}%)" for name, weight in ranked_importance[:top_n])
    notes.append(
        {
            "title": "Model drivers",
            "detail": (
                f"This classifier weights {drivers} most heavily when deciding risk."
            ),
            "level": "info",
        }
    )

    notes.append(
        {
            "title": "Class probabilities",
            "detail": "  |  ".join(
                f"{cls} {result['probabilities'].get(cls, 0.0) * 100:.1f}%"
                for cls in result["ranked"]
            ),
            "level": "info",
        }
    )

    notes.append(
        {
            "title": "Interpretation",
            "detail": (
                "This is a relative risk classification produced by a model trained on "
                "synthetic demonstration data. It is not a potability or health assessment."
            ),
            "level": "info",
        }
    )

    return notes


def sensor_status(name: str, value: float) -> tuple[str, str]:
    """Return (status_label, colour) for a single reading."""
    exc = sensor_exceedance(name, value)
    if exc["in_band"]:
        return "NOMINAL", cfg.COLOR_OK
    if exc["ratio"] >= cfg.FLOOR_MEDIUM_RATIO:
        return "OUT OF BAND", cfg.COLOR_DANGER
    return "MARGINAL", cfg.COLOR_WARN