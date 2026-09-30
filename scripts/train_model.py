"""
Train the AquaSense AI Random Forest classifier and save it with Joblib.

Usage:
    python scripts\\train_model.py
    python scripts\\train_model.py --test-size 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as cfg  # noqa: E402
from app.modeling import DatasetError, load_bundle, load_dataset, train_from_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=cfg.DATASET_CSV)
    parser.add_argument("--model-out", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=cfg.METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=cfg.TEST_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 68)
    print(f"{cfg.APP_NAME}  --  model training")
    print("=" * 68)

    try:
        df = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Dataset        -> {args.dataset.name} ({len(df)} rows)")
    print("Class balance  -> " + ", ".join(
        f"{cls}: {int((df[cfg.TARGET] == cls).sum())}" for cls in cfg.RISK_ORDER
    ))

    bundle = train_from_file(
        dataset_path=args.dataset,
        model_path=args.model_out,
        metrics_path=args.metrics_out,
    )

    split = bundle["split"]
    print(f"Features       -> {', '.join(split['features'])} ({len(split['features'])})")
    print(f"Classes        -> {', '.join(split['classes'])}")
    print(f"Trees          -> {bundle['model']['n_trees']}")

    print("\nPer-class performance on the held-out test set:")
    for cls, row in bundle["metrics"]["per_class"].items():
        print(
            f"  {cls:<7} precision {row['precision']:.3f} | "
            f"recall {row['recall']:.3f} | f1 {row['f1']:.3f} | support {row['support']}"
        )

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    labels = bundle["metrics"]["confusion_labels"]
    matrix = bundle["metrics"]["confusion_matrix"]
    print("            " + "".join(f"{c:>9}" for c in labels))
    for label, row in zip(labels, matrix):
        print(f"  {label:<10}" + "".join(f"{v:>9}" for v in row))

    print("\nFeature importance:")
    for name, weight in sorted(bundle["feature_importance"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {weight:.4f}")

    # Prove the saved artefact loads and predicts.
    reloaded = load_bundle(args.model_out, args.metrics_out)
    sanity = pd.DataFrame([{f: float(df.iloc[0][f]) for f in cfg.FEATURES}])
    prediction = reloaded["pipeline"].predict(sanity)[0]
    print(f"\nReload check  -> OK (row 0 predicted {prediction})")
    print("\nRESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())