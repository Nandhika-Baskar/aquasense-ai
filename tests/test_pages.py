"""
Render every page through Streamlit's test harness and fail on any exception.

Run with:  python tests\\test_pages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

from app import config as cfg  # noqa: E402
from app import modeling  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"
TIMEOUT = 180


def render_page(page: str, session_state: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.session_state["active_page"] = page
    for key, value in (session_state or {}).items():
        at.session_state[key] = value
    return at.run()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 68)
    print("AquaSense AI -- page render tests")
    print("=" * 68)

    failures: list[str] = []

    for name, _, _ in cfg.NAV_ITEMS:
        try:
            at = render_page(name)
            exceptions = [str(e.value) for e in at.exception]
            if exceptions:
                failures.append(f"{name}: {exceptions[0]}")
                print(f"  [FAIL] {name:<18} {exceptions[0][:90]}")
                continue

            charts = len(at.get("plotly_chart")) + len(at.get("vega_lite_chart"))
            buttons = len(at.button)
            tables = len(at.dataframe)
            print(
                f"  [PASS] {name:<18} rendered "
                f"({charts} charts, {tables} tables, {buttons} buttons)"
            )
        except Exception as exc:  # noqa: BLE001 - reported below
            failures.append(f"{name}: {exc}")
            print(f"  [FAIL] {name:<18} {exc}")

    # --- navigation is present and the sidebar status renders ----------------
    print("\n-- navigation --")
    at = render_page(cfg.NAV_ITEMS[0][0])
    for name, _, _ in cfg.NAV_ITEMS:
        key = f"nav_{name}"
        found = any(b.key == key for b in at.button)
        if not found:
            failures.append(f"nav button missing: {name}")
        print(f"  [{'PASS' if found else 'FAIL'}] nav button '{key}'")

    # --- prediction flow ----------------------------------------------------
    print("\n-- prediction flow --")
    at = render_page("AI Prediction")
    if at.exception:
        failures.append(f"prediction page exception: {at.exception[0].value}")
        print(f"  [FAIL] prediction page -- {at.exception[0].value[:90]}")
    else:
        empty_state = "No analysis yet" in "\n".join(m.value for m in at.markdown)
        print(f"  [{'PASS' if empty_state else 'WARN'}] empty state shown before any run")

    submitted = False
    for button in at.button:
        if button.label == "Predict Water Quality":
            button.click().run()
            submitted = True
            break

    if not submitted:
        failures.append("'Predict Water Quality' button not found")
        print("  [FAIL] 'Predict Water Quality' button not found")
    elif at.exception:
        failures.append(f"prediction submit raised: {at.exception[0].value}")
        print(f"  [FAIL] submitting the form -- {at.exception[0].value[:90]}")
    else:
        body = "\n".join(m.value for m in at.markdown)
        has_risk = "CONTAMINATION RISK" in body
        has_factors = "Key Contributing Factors" in body
        has_percent = "%" in body
        ok = has_risk and has_factors and has_percent
        if not ok:
            failures.append("prediction result incomplete")
        print(
            f"  [{'PASS' if ok else 'FAIL'}] prediction result rendered "
            f"(risk card={has_risk}, factors={has_factors}, probability={has_percent})"
        )

    # --- metrics are surfaced on the performance page ------------------------
    print("\n-- metrics surfaced --")
    at = render_page("ML Performance")
    body = "\n".join(m.value for m in at.markdown)

    # The page renders custom HTML cards rather than st.metric, so assert on the
    # rendered text -- and tie the displayed values back to models/metrics.json.
    bundle = modeling.load_bundle()
    metrics = bundle["metrics"]

    def as_pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    expected = [
        ("accuracy card", "ACCURACY" in body),
        ("precision card", "PRECISION" in body),
        ("recall card", "RECALL" in body),
        ("f1 card", "F1 SCORE" in body),
        ("accuracy value matches metrics.json", as_pct(metrics["accuracy"]) in body),
        ("precision value matches metrics.json", as_pct(metrics["precision_macro"]) in body),
        ("recall value matches metrics.json", as_pct(metrics["recall_macro"]) in body),
        ("f1 value matches metrics.json", as_pct(metrics["f1_macro"]) in body),
        ("model name", "Random Forest" in body),
        ("training sample count", f"{bundle['split']['n_train']:,}" in body),
        ("testing sample count", f"{bundle['split']['n_test']:,}" in body),
        ("feature count", str(len(bundle["split"]["features"])) in body),
        ("confusion matrix", "Confusion Matrix" in body),
        ("classification report", "Classification Report" in body),
    ]

    for label, ok in expected:
        if not ok:
            failures.append(f"performance page missing: {label}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("RESULT: ALL PAGES RENDERED WITHOUT ERRORS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())