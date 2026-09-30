"""
Quick boot check: does the app start, load the model and render the dashboard?

Run with:  python tests\\smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config as cfg  # noqa: E402

APP = PROJECT_ROOT / "app" / "main.py"


def main() -> int:
    print(f"Booting {cfg.APP_NAME} ...")

    at = AppTest.from_file(str(APP), default_timeout=180).run()

    if at.exception:
        print(f"[FAIL] app raised {len(at.exception)} exception(s):")
        for exc in at.exception:
            print(f"  {exc.value}")
        return 1

    body = "\n".join(m.value for m in at.markdown)

    checks = {
        "no exceptions": True,
        "dashboard rendered": "WATER QUALITY OVERVIEW" in body,
        "branding present": cfg.APP_NAME in body,
        "risk card present": "CONTAMINATION RISK" in body,
        "system status present": "System Status" in body,
        "all five sensors shown": all(f in body for f in cfg.FEATURES),
    }

    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        print(f"\nRESULT: FAILED ({', '.join(failed)})")
        return 1

    print("\nRESULT: SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())