"""
AquaSense AI
AI-Based Water Quality Prediction and Contamination Risk Classification
using IoT Sensor Data

Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import components, config as cfg, modeling, theme  # noqa: E402
from app.context import AppContext  # noqa: E402

PAGE_KEY = "active_page"


@st.cache_resource(show_spinner="Loading the trained Random Forest model ...")
def load_context() -> AppContext:
    """Load the dataset, model and metrics once per server session."""
    df = modeling.load_dataset()
    bundle = modeling.load_bundle()

    return AppContext(
        df=df,
        pipeline=bundle["pipeline"],
        metrics=bundle["metrics"],
        feature_importance=bundle["feature_importance"],
        split=bundle["split"],
        model=bundle["model"],
        dataset=bundle.get("dataset", {}),
        trained_at=bundle.get("trained_at", "unknown"),
    )


def render_sidebar() -> str:
    """Brand, navigation and status. Returns the selected page name."""
    with st.sidebar:
        components.brand_block()
        components.nav_heading()

        current = st.session_state.get(PAGE_KEY, cfg.NAV_ITEMS[0][0])
        selected = current

        for name, glyph, _ in cfg.NAV_ITEMS:
            if st.button(
                f"{glyph}  {name}",
                key=f"nav_{name}",
                type="primary" if name == current else "secondary",
                width="stretch",
            ):
                selected = name

        st.session_state[PAGE_KEY] = selected

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        components.sidebar_status()
        theme.sidebar_css_note()

    return selected


def main() -> None:
    theme.page_setup()
    theme.inject_css()

    page = render_sidebar()

    try:
        ctx = load_context()
    except modeling.DatasetError as exc:
        components.page_header(cfg.APP_NAME, "Setup required")
        components.empty_state(
            str(exc),
            "The application cannot run until the data and model exist.",
            "python scripts\\generate_dataset.py   then   python scripts\\train_model.py",
        )
        return

    module_name = next(item[2] for item in cfg.NAV_ITEMS if item[0] == page)
    module = importlib.import_module(module_name)
    module.render(ctx)


if __name__ == "__main__":
    main()