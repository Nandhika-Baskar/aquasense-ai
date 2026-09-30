"""
Unit tests for the ML layer.

Run with:  python tests\\test_modeling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg  # noqa: E402
from app import modeling  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main() -> int:
    print("=" * 68)
    print("AquaSense AI -- model tests")
    print("=" * 68)

    df = modeling.load_dataset()
    bundle = modeling.load_bundle()
    pipeline = bundle["pipeline"]
    metrics = bundle["metrics"]

    print("\n-- dataset --")
    check("dataset loads and validates", len(df) > 0, f"{len(df)} rows")
    check("no missing sensor values", df[cfg.FEATURES].isna().sum().sum() == 0)
    check(
        "target has exactly the 3 risk classes",
        sorted(df[cfg.TARGET].unique()) == sorted(cfg.RISK_ORDER),
    )

    print("\n-- metrics are real and persisted --")
    check(
        "accuracy in range",
        0.0 <= metrics["accuracy"] <= 1.0,
        f"{metrics['accuracy']:.4f}",
    )
    check("precision/recall/F1 present", all(
        k in metrics for k in ("precision_macro", "recall_macro", "f1_macro")
    ))
    check(
        "train + test equals dataset size",
        metrics["confusion_matrix"] and (
            bundle["split"]["n_train"] + bundle["split"]["n_test"] == len(df)
        ),
    )
    check("5 features recorded", len(bundle["split"]["features"]) == 5)
    check("300 trees", bundle["model"]["n_trees"] == 300)

    # Confusion matrix must agree with the reported accuracy.
    matrix = metrics["confusion_matrix"]
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(metrics["confusion_labels"])))
    check(
        "confusion matrix totals equal the test-set size",
        total == bundle["split"]["n_test"],
        f"{total} == {bundle['split']['n_test']}",
    )
    check(
        "accuracy matches the confusion matrix diagonal",
        abs(correct / total - metrics["accuracy"]) < 1e-9,
        f"{correct}/{total} = {correct / total:.4f}",
    )

    print("\n-- feature importance --")
    importance = bundle["feature_importance"]
    check("importance covers every feature", set(importance) == set(cfg.FEATURES))
    check(
        "importances sum to 1.0",
        abs(sum(importance.values()) - 1.0) < 1e-6,
        f"{sum(importance.values()):.6f}",
    )

    print("\n-- prediction --")
    readings = {"pH": 7.2, "TDS": 180.0, "Turbidity": 2.0, "Temperature": 24.0, "Conductivity": 300.0}
    clean = modeling.predict_risk(pipeline, readings)
    check("clean readings classify as LOW", clean["risk"] == "LOW", clean["risk"])
    check(
        "probabilities cover all classes",
        set(clean["probabilities"]) == set(cfg.RISK_ORDER),
    )
    check(
        "probabilities sum to 1",
        abs(sum(clean["probabilities"].values()) - 1.0) < 1e-9,
    )
    check(
        "confidence equals the predicted class probability",
        abs(clean["confidence"] - clean["probabilities"][clean["risk"]]) < 1e-12,
    )
    check(
        "ranked list is sorted high to low",
        clean["ranked"][0] == clean["risk"],
        str(clean["ranked"]),
    )

    dirty = modeling.predict_risk(
        pipeline,
        {"pH": 5.2, "TDS": 2200.0, "Turbidity": 90.0, "Temperature": 31.0, "Conductivity": 2400.0},
    )
    check("severely out-of-band readings classify as HIGH", dirty["risk"] == "HIGH", dirty["risk"])

    print("\n-- explanation --")
    notes = modeling.explain_prediction(dirty, importance)
    check("explanation returns notes", len(notes) >= 3, f"{len(notes)} notes")
    check("notes have title and detail", all("title" in n and "detail" in n for n in notes))
    check(
        "no potability claim in the explanation",
        not any(
            word in n["detail"].lower()
            for n in notes
            for word in ("safe to drink", "potable", "drinkable")
        ),
    )

    print("\n-- sensor status --")
    check("in-band reading is NOMINAL", modeling.sensor_status("TDS", 200.0)[0] == "NOMINAL")
    check(
        "mildly out-of-band reading is MARGINAL",
        modeling.sensor_status("TDS", 560.0)[0] == "MARGINAL",
    )
    check(
        "grossly out-of-band reading is flagged",
        modeling.sensor_status("Turbidity", 80.0)[0] == "OUT OF BAND",
    )

    print("\n-- input validation --")
    bad_schema = pd.DataFrame(
        {"pH": [7.0] * 40, "TDS": [200.0] * 40, cfg.TARGET: ["LOW"] * 40}
    )
    try:
        modeling.validate_dataset(bad_schema)
    except modeling.DatasetError as exc:
        check("missing columns raise DatasetError", "missing required column" in str(exc).lower())
    else:
        check("missing columns raise DatasetError", False, "no error raised")

    bad_label = pd.DataFrame(
        {**{f: [7.0] * 40 for f in cfg.FEATURES}, cfg.TARGET: ["EXTREME"] * 40}
    )
    try:
        modeling.validate_dataset(bad_label)
    except modeling.DatasetError as exc:
        check("bad label raises DatasetError", "unexpected risk label" in str(exc).lower())
    else:
        check("bad label raises DatasetError", False, "no error raised")

    tiny = pd.DataFrame(
        {**{f: [7.0] for f in cfg.FEATURES}, cfg.TARGET: ["LOW"]}
    )
    try:
        modeling.validate_dataset(tiny)
    except modeling.DatasetError as exc:
        check("tiny dataset raises DatasetError", "at least" in str(exc).lower())
    else:
        check("tiny dataset raises DatasetError", False, "no error raised")

    gappy = pd.DataFrame(
        {
            **{f: [7.0] * 40 for f in cfg.FEATURES},
            cfg.TARGET: ["LOW"] * 40,
        }
    )
    gappy.loc[3, "pH"] = None
    try:
        modeling.validate_dataset(gappy)
    except modeling.DatasetError as exc:
        check("missing sensor values raise DatasetError", "missing values" in str(exc).lower())
    else:
        check("missing sensor values raise DatasetError", False, "no error raised")

    print("\n-- batch prediction --")
    batch = df[cfg.FEATURES].head(200)
    preds = pipeline.predict(batch)
    check("batch predict returns one label per row", len(preds) == len(batch))
    check("all batch labels are valid classes", set(preds) <= set(cfg.RISK_ORDER))

    print("\nRESULT: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())