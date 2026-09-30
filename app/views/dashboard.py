"""Dashboard -- the at-a-glance operations overview."""

from __future__ import annotations

import streamlit as st

from app import charts, components, config as cfg, modeling, simulator
from app.context import AppContext

SENSOR_DESCRIPTIONS = {
    "pH": "Acidity / alkalinity balance",
    "TDS": "Total dissolved solids",
    "Turbidity": "Suspended particle load",
    "Temperature": "Water temperature",
    "Conductivity": "Electrical conductivity",
}


def render(ctx: AppContext) -> None:
    nonce = st.session_state.get("sim_nonce", 0)
    readings = simulator.simulated_readings(ctx.df, nonce)
    prediction = modeling.predict_risk(ctx.pipeline, readings)

    components.page_header(
        "WATER QUALITY OVERVIEW",
        f"{cfg.APP_SUBTITLE} -- synthetic demonstration dataset, "
        f"{len(ctx.df):,} readings analysed",
    )

    # --- sensor cards -------------------------------------------------------
    cards = []
    for feature in cfg.FEATURES:
        status, color = modeling.sensor_status(feature, readings[feature])
        cards.append(
            {
                "label": feature,
                "value": f"{readings[feature]:,.2f}",
                "unit": cfg.SENSOR_UNITS[feature],
                "description": SENSOR_DESCRIPTIONS[feature],
                "status": status,
                "color": color,
            }
        )
    components.metric_grid(cards)

    # --- risk + system status ----------------------------------------------
    st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([2, 1], gap="large")

    with left:
        components.risk_hero(
            prediction["risk"],
            prediction["confidence"],
            prediction["probabilities"],
            components.now_label(),
        )

    with right:
        components.section("System Status")
        components.status_list(
            [
                ("Data Pipeline", "Operational", cfg.COLOR_OK),
                ("ML Model", "Operational", cfg.COLOR_OK),
                ("Prediction Engine", "Operational", cfg.COLOR_OK),
                ("Dashboard", "Operational", cfg.COLOR_OK),
            ]
        )
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        components.stat_grid(
            [
                ("Model", ctx.model["name"].replace(" Classifier", "")),
                ("Trees", f"{ctx.model['n_trees']:,}"),
                ("Accuracy", components.pct(ctx.metrics["accuracy"])),
            ]
        )

    # --- charts -------------------------------------------------------------
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    trend_heading, trend_control = st.columns([3, 2], gap="large")
    with trend_heading:
        components.section(
            "Sensor Trends",
            "Each series is scaled to its own guideline band, so sensors with very "
            "different units can share one axis. 0-1 means inside the band.",
        )
    with trend_control:
        st.markdown(
            f'<div class="qs-eyebrow" style="margin-top:.35rem">TREND SMOOTHING</div>'
            f'<div style="font-size:.75rem;color:{cfg.COLOR_MUTED};margin-bottom:.3rem">'
            "Moving average applied to the readings.</div>",
            unsafe_allow_html=True,
        )
        rolling = st.slider(
            "Readings per moving average",
            min_value=1,
            max_value=cfg.TREND_ROLLING_MAX,
            value=cfg.TREND_ROLLING_DEFAULT,
            help="Averages each sensor over this many consecutive readings. "
            "Set to 1 to plot every raw reading.",
        )

    st.plotly_chart(
        charts.sensor_trend_normalised(ctx.df, cfg.FEATURES, rolling=rolling)
    )
    st.caption(simulator.trend_caption(ctx.df, rolling))

    a, b = st.columns([3, 2], gap="large")
    with a:
        components.section("TDS vs Turbidity", "Dominated by TDS, so the x axis is logarithmic.")
        st.plotly_chart(charts.scatter_by_class(ctx.df, "TDS", "Turbidity", log_x=True))
    with b:
        components.section("Risk Class Mix")
        st.plotly_chart(charts.class_distribution(ctx.df))

    # --- latest readings ----------------------------------------------------
    components.section("Latest Readings", cfg.SIMULATION_NOTE)
    latest = simulator.recent_window(ctx.df, 8)
    st.dataframe(
        latest[[cfg.TIME_COL, *cfg.FEATURES, cfg.TARGET]].rename(
            columns={cfg.TARGET: "Predicted Risk"}
        ),
        width="stretch",
        hide_index=True,
        height=290,
    )