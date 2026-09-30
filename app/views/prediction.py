"""AI Prediction -- classify a set of sensor readings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, components, config as cfg, modeling
from app.context import AppContext

INPUT_STEP = {
    "pH": 0.05,
    "TDS": 5.0,
    "Turbidity": 0.1,
    "Temperature": 0.1,
    "Conductivity": 5.0,
}

INPUT_DECIMALS = {"pH": 2, "TDS": 0, "Turbidity": 1, "Temperature": 1, "Conductivity": 0}

RESULT_KEY = "last_prediction"


def class_presets(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """A representative reading for each class, taken as that class's medians."""
    presets = {}
    for cls in cfg.RISK_ORDER:
        subset = df.loc[df[cfg.TARGET] == cls, cfg.FEATURES]
        if subset.empty:
            continue
        presets[cls] = {f: float(subset[f].median()) for f in cfg.FEATURES}
    return presets


def validate(readings: dict[str, float]) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings) for a set of readings."""
    errors, warnings = [], []
    for feature, value in readings.items():
        low, high = cfg.CLIP_LOW[feature], cfg.CLIP_HIGH[feature]
        if not low <= value <= high:
            errors.append(f"{feature} = {value:g} is outside the plausible range {low:g}-{high:g}.")
    for feature, value in readings.items():
        status, _ = modeling.sensor_status(feature, value)
        if status == "MARGINAL":
            warnings.append(f"{feature} is marginally outside its guideline band.")
    return errors, warnings


def render(ctx: AppContext) -> None:
    components.page_header(
        "AI PREDICTION",
        "Enter five sensor readings and the Random Forest returns a contamination "
        "risk class with its probability.",
    )

    presets = class_presets(ctx.df)
    defaults = presets.get("MEDIUM") or {f: 0.0 for f in cfg.FEATURES}

    # --- form ---------------------------------------------------------------
    with st.form("prediction_form", border=False):
        top = st.columns([1, 1, 1])
        top[0].markdown(
            f'<div class="qs-eyebrow">SENSOR INPUTS</div>'
            f'<div style="font-size:.78rem;color:{cfg.COLOR_MUTED};margin-bottom:.35rem">'
            "Adjust any value to re-run the analysis.</div>",
            unsafe_allow_html=True,
        )

        with top[1]:
            preset_col = st.selectbox(
                "Load example",
                ["Custom"] + list(presets.keys()),
                help="Each option fills the form with that class's median reading.",
            )
        with top[2]:
            st.markdown(
                f'<div class="qs-eyebrow" style="margin-top:.35rem">GUIDELINE REFERENCE</div>'
                f'<div style="font-size:.75rem;color:{cfg.COLOR_MUTED}">'
                + " &middot; ".join(
                    f"{f} {cfg.SAFE_LOW[f]:g}-{cfg.SAFE_HIGH[f]:g}"
                    for f in cfg.FEATURES
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if preset_col != "Custom":
            defaults = presets[preset_col]

        cols = st.columns(len(cfg.FEATURES))
        readings: dict[str, float] = {}
        for col, feature in zip(cols, cfg.FEATURES):
            value = col.number_input(
                feature,
                min_value=float(cfg.CLIP_LOW[feature]),
                max_value=float(cfg.CLIP_HIGH[feature]),
                value=float(np_clip(defaults.get(feature, 0.0), feature)),
                step=INPUT_STEP[feature],
                format=f"%.{INPUT_DECIMALS[feature]}f",
                help=f"Unit: {cfg.SENSOR_UNITS[feature]}. "
                f"Guideline {cfg.SAFE_LOW[feature]:g}-{cfg.SAFE_HIGH[feature]:g}.",
            )
            readings[feature] = float(value)

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Water Quality", type="primary")

    if submitted:
        errors, warnings = validate(readings)
        if errors:
            for message in errors:
                st.error(message)
        else:
            result = modeling.predict_risk(ctx.pipeline, readings)
            result["notes"] = modeling.explain_prediction(result, ctx.feature_importance)
            result["warnings"] = warnings
            st.session_state[RESULT_KEY] = result

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        components.empty_state(
            "No analysis yet",
            "Set the five sensor values above and run the analysis to see a classification.",
            "Results are computed by models/random_forest_model.joblib.",
        )
        return

    # --- results ------------------------------------------------------------
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([2, 1], gap="large")

    with left:
        components.risk_hero(
            result["risk"],
            result["confidence"],
            result["probabilities"],
            components.now_label(),
        )
    with right:
        components.section("Class Probabilities")
        st.plotly_chart(charts.probability_bars(result["probabilities"]))

    for message in result.get("warnings", []):
        st.warning(message)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        components.section(
            "Key Contributing Factors",
            "Feature importance from the trained Random Forest, alongside what this "
            "specific reading did to the guideline bands.",
        )
        components.explanation_list(result["notes"])
    with right:
        components.section("Model Feature Importance")
        st.plotly_chart(charts.feature_importance(ctx.feature_importance, height=300))

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    components.section("Input Readings")
    components.metric_grid(
        [
            {
                "label": f,
                "value": f"{result['readings'][f]:,.{INPUT_DECIMALS[f]}f}",
                "unit": cfg.SENSOR_UNITS[f],
                "status": modeling.sensor_status(f, result["readings"][f])[0],
                "color": modeling.sensor_status(f, result["readings"][f])[1],
            }
            for f in cfg.FEATURES
        ]
    )


def np_clip(value: float, feature: str) -> float:
    """Keep a preset inside the widget's own bounds."""
    return float(min(max(value, cfg.CLIP_LOW[feature]), cfg.CLIP_HIGH[feature]))