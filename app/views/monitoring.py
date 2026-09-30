"""Live Monitoring -- simulated sensor stream."""

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
    components.page_header(
        "LIVE MONITORING",
        "Simulated sensor stream built from the demonstration dataset.",
    )
    components.note(cfg.SIMULATION_NOTE, kind="warn")

    if st.button("Refresh simulated readings", type="primary"):
        st.session_state["sim_nonce"] = st.session_state.get("sim_nonce", 0) + 1

    nonce = st.session_state.get("sim_nonce", 0)
    readings = simulator.simulated_readings(ctx.df, nonce)
    prediction = modeling.predict_risk(ctx.pipeline, readings)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # --- current readings ---------------------------------------------------
    components.section("Current Readings", f"Stream revision {nonce}")
    components.metric_grid(
        [
            {
                "label": feature,
                "value": f"{readings[feature]:,.2f}",
                "unit": cfg.SENSOR_UNITS[feature],
                "description": SENSOR_DESCRIPTIONS[feature],
                "status": modeling.sensor_status(feature, readings[feature])[0],
                "color": modeling.sensor_status(feature, readings[feature])[1],
            }
            for feature in cfg.FEATURES
        ]
    )

    # --- risk + guidance ----------------------------------------------------
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
        components.section("Guideline Check")
        components.status_list(
            [
                (
                    feature,
                    f"{cfg.SAFE_LOW[feature]:g} - {cfg.SAFE_HIGH[feature]:g} {cfg.SENSOR_UNITS[feature]}",
                    modeling.sensor_status(feature, readings[feature])[1],
                )
                for feature in cfg.FEATURES
            ]
        )

    # --- time series --------------------------------------------------------
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    window = simulator.recent_window(ctx.df, 90)

    tab_norm, tab_raw, tab_risk = st.tabs(["Normalised Trend", "Raw Readings", "Turbidity Focus"])

    with tab_norm:
        components.section(
            "Recent Sensor Position",
            "Each sensor divided by the width of its guideline band. Outside 0-1 "
            "means the reading has left the band.",
        )
        st.plotly_chart(
            charts.sensor_trend_normalised(window, cfg.FEATURES, height=340, rolling=5)
        )
        st.caption(
            "Smoothed with a 5-reading moving average so short excursions stay "
            "visible. The Reading Log below holds the unaltered values."
        )

    with tab_raw:
        components.section("Recent TDS and Turbidity", "Raw units, plotted on separate axes.")
        st.plotly_chart(charts.sensor_trend(window, ["TDS", "Turbidity"], height=340))

    with tab_risk:
        components.section("Recent Turbidity", "The sensor with the highest model importance.")
        st.plotly_chart(charts.sensor_trend(window, ["Turbidity"], height=340))

    # --- log table ----------------------------------------------------------
    components.section("Reading Log", f"Most recent {len(window)} simulated readings")
    table = window[[cfg.TIME_COL, cfg.ID_COL, *cfg.FEATURES, cfg.TARGET]].rename(
        columns={cfg.TARGET: "Risk Label"}
    )
    st.dataframe(table, width="stretch", hide_index=True, height=320)

    st.download_button(
        "Download simulated readings (CSV)",
        window.to_csv(index=False),
        file_name="simulated_readings.csv",
        mime="text/csv",
    )