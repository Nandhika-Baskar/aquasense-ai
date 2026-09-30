"""
Generate and verify data/water_quality.csv  (synthetic demonstration data).

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --rows 2000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import config as cfg  # noqa: E402
from app.data_gen import build_clean_dataset, risk_score, verify_dataset, verify_labels  # noqa: E402

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=cfg.N_SAMPLES, help="number of readings to simulate")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="random seed for reproducibility")
    parser.add_argument("--out", type=Path, default=cfg.DATASET_CSV, help="output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(cfg.DATASET_NOTE)
    print("=" * 72)

    raw, n_dropped = build_clean_dataset(n_samples=args.rows, seed=args.seed)
    report = verify_dataset(raw)

    print(f"\nGenerating {args.rows} readings (seed={args.seed}) ...")
    print(f"Simulated sensor dropouts removed: {n_dropped} rows")

    # --- Checks -------------------------------------------------------------
    print("\n--- CHECKS ---")
    ok = True

    if report["n_rows"] < 1000:
        print(f"  [FAIL] only {report['n_rows']} rows (need at least 1000)")
        ok = False
    else:
        print(f"  [OK]   row count = {report['n_rows']} (>= 1000)")

    if report["missing_columns"]:
        print(f"  [FAIL] missing columns: {report['missing_columns']}")
        ok = False
    else:
        print(f"  [OK]   all expected columns present: {report['columns']}")

    if report["unexpected_columns"]:
        print(f"  [WARN] unexpected columns: {report['unexpected_columns']}")

    if report["n_missing_values"] > 0:
        print(f"  [FAIL] {report['n_missing_values']} missing values: {report['nulls_per_column']}")
        ok = False
    else:
        print("  [OK]   no missing values anywhere")

    if report["duplicate_ids"] or report["duplicate_rows"]:
        print(f"  [FAIL] duplicates found (ids={report['duplicate_ids']}, rows={report['duplicate_rows']})")
        ok = False
    else:
        print("  [OK]   no duplicate ids or rows")

    unexpected_classes = set(raw[cfg.TARGET].unique()) - set(cfg.CLASSES)
    if unexpected_classes:
        print(f"  [FAIL] unexpected class labels: {unexpected_classes}")
        ok = False
    else:
        print("  [OK]   target labels are exactly LOW / MEDIUM / HIGH")

    if report["imbalance_ratio"] > 3:
        print(f"  [WARN] class imbalance ratio {report['imbalance_ratio']} (> 3)")
    else:
        print(f"  [OK]   class imbalance ratio {report['imbalance_ratio']} (<= 3, acceptable)")

    # How many readings sit near a decision boundary? Confirms the task is
    # learnable but not trivially separable. Uses the label rule only -- no
    # model is trained here.
    scores = risk_score(raw)
    band = cfg.LABEL_NOISE_SD
    near_boundary = (
        (np.abs(scores - cfg.LOW_MAX_SCORE) < band) | (np.abs(scores - cfg.HIGH_MIN_SCORE) < band)
    ).sum()
    band_pct = 100 * near_boundary / len(raw)
    if band_pct < 5:
        print(f"  [WARN] only {band_pct:.1f}% of rows near a class boundary (problem may be too easy)")
    else:
        print(f"  [OK]   {band_pct:.1f}% of rows sit within 1 sd of a class boundary (non-trivial problem)")

    # Labels must be physically defensible, not just well distributed.
    print("\n--- LABEL SANITY ---")
    lab = verify_labels(raw)

    if lab["compliant_not_low"] == 0:
        print(f"  [OK]   all {lab['fully_compliant_rows']} fully compliant readings are LOW")
    else:
        print(f"  [FAIL] {lab['compliant_not_low']} compliant readings are not labelled LOW")
        ok = False

    if lab["low_beyond_floor"] == 0:
        print(f"  [OK]   no LOW row is out of band (worst exceedance {lab['max_worst_ratio_low']} widths)")
    else:
        print(f"  [FAIL] {lab['low_beyond_floor']} LOW rows exceed the {cfg.FLOOR_MEDIUM_RATIO}-width floor")
        ok = False

    if lab["medium_beyond_floor"] == 0:
        print(f"  [OK]   no MEDIUM row is grossly failed (>{cfg.FLOOR_HIGH_RATIO} guideline widths)")
    else:
        print(f"  [FAIL] {lab['medium_beyond_floor']} MEDIUM rows exceed the {cfg.FLOOR_HIGH_RATIO}-width floor")
        ok = False

    if lab["classes_ordered"]:
        print(f"  [OK]   risk increases LOW -> MEDIUM -> HIGH {lab['median_worst_by_class']}")
    else:
        print(f"  [FAIL] class ordering is not monotonic: {lab['median_worst_by_class']}")
        ok = False

    # --- Report -------------------------------------------------------------
    print("\n--- DATASET SUMMARY ---")
    print(f"Rows              : {report['n_rows']}")
    print(f"Columns           : {report['n_cols']}  -> {', '.join(report['columns'])}")
    print(f"Missing values    : {report['n_missing_values']}")

    print("\nClass distribution:")
    dist = pd.DataFrame(
        {
            "count": pd.Series(report["class_counts"]),
            "share_pct": (pd.Series(report["class_shares"]) * 100).round(2),
        }
    )
    print(dist.to_string())

    print("\nSensor summary (min / mean / max):")
    stats = pd.DataFrame(
        {
            "min": pd.Series(report["sensor_min"]),
            "mean": pd.Series(report["sensor_mean"]),
            "max": pd.Series(report["sensor_max"]),
            "unit": pd.Series(cfg.SENSOR_UNITS),
            "safe_band": pd.Series(
                {k: f"{cfg.SAFE_LOW[k]:g} - {cfg.SAFE_HIGH[k]:g}" for k in cfg.FEATURES}
            ),
        }
    )
    print(stats.to_string())

    print("\nClass means (shows the classes are related but NOT trivially separable):")
    print(raw.groupby(cfg.TARGET)[cfg.FEATURES].mean().round(2).to_string())

    print("\nHighest reading inside the LOW class (LOW must stay close to the safe bands):")
    print(pd.Series(lab["class_sensor_max"]).to_string())

    if not ok:
        print("\nRESULT: FAILED -- CSV not written.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out.relative_to(cfg.PROJECT_ROOT)}  ({args.out.stat().st_size:,} bytes)")
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())