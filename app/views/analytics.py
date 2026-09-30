"""Analytics -- exploratory views over the dataset."""

from __future__ import annotations

import streamlit as st

from app import charts, components, config as cfg, simulator
from app.context import AppContext


def render(ctx: AppContext) -> None:
    components.page_header(
        "ANALYTICS",
        f"Exploratory views over {len(ctx.df):,} simulated sensor readings.",
    )

    overview_tab, trend_tab, distribution_tab, importance_tab = st.tabs(
        ["Overview", "Trends", "Distributions", "Feature Importance"]
    )

    with overview_tab:
        counts = ctx.class_counts()
        components.metric_grid(
            [
                {"label": "Readings", "value": f"{len(ctx.df):,}", "description": "rows in the dataset"},
                {
                    "label": "Features",
                    "value": str(len(cfg.FEATURES)),
                    "description": "sensor inputs",
                },
                {
                    "label": "Classes",
                    "value": str(len(cfg.RISK_ORDER)),
                    "description": "LOW / MEDIUM / HIGH",
                },
                {
                    "label": "Missing values",
                    "value": str(int(ctx.df.isna().sum().sum())),
                    "description": "dataset completeness",
                    "status": "Complete",
                    "color": cfg.COLOR_OK,
                },
            ]
        )

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        left, right = st.columns([2, 3], gap="large")
        with left:
            components.section("Risk Class Distribution")
            st.plotly_chart(charts.class_distribution(ctx.df), height=330)
        with right:
            components.section("Correlation Matrix")
            st.plotly_chart(charts.correlation_heatmap(ctx.df, cfg.FEATURES), height=330)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        components.note(
            "TDS and conductivity correlate strongly (~0.98) because conductivity is "
            "derived from dissolved solids in this simulated dataset. The model therefore "
            "sees partly redundant information, which is why they split most of the "
            "importance between them.",
            kind="warn",
        )

    with trend_tab:
        components.section(
            "Sensor Trend",
            "Readings over the 30-day simulated deployment window.",
        )
        rolling = st.slider(
            "Readings per moving average",
            min_value=1,
            max_value=cfg.TREND_ROLLING_MAX,
            value=cfg.TREND_ROLLING_DEFAULT,
            key="analytics_trend_rolling",
            help="Averages each sensor over this many consecutive readings. "
            "Set to 1 to plot every raw reading.",
        )
        st.caption(simulator.trend_caption(ctx.df, rolling))
        selected = st.multiselect(
            "Sensors to plot",
            cfg.FEATURES,
            default=cfg.FEATURES,
            help="Sensors have different units; use the raw view for one sensor at a time.",
        )
        if selected:
            st.plotly_chart(
                charts.sensor_trend_normalised(ctx.df, selected, height=380, rolling=rolling),
            )
        else:
            components.empty_state("No sensors selected", "Choose at least one sensor to plot.")

        components.section("Raw TDS and Turbidity")
        st.plotly_chart(
            charts.sensor_trend(ctx.df, ["TDS", "Turbidity"], height=320, rolling=rolling),
        )

    with distribution_tab:
        feature = st.selectbox("Sensor", cfg.FEATURES, index=2)
        left, right = st.columns([3, 2], gap="large")
        with left:
            components.section(
                f"{feature} Distribution",
                f"Shaded area marks the guideline band "
                f"({cfg.SAFE_LOW[feature]:g} - {cfg.SAFE_HIGH[feature]:g} {cfg.SENSOR_UNITS[feature]}).",
            )
            st.plotly_chart(charts.distribution(ctx.df, feature), height=380)
        with right:
            components.section("pH vs Conductivity")
            st.plotly_chart(charts.scatter_by_class(ctx.df, "pH", "Conductivity", height=380))

    with importance_tab:
        components.section(
            "Feature Importance",
            "Mean decrease in Gini impurity across the 300 trees. These values come "
            "from the fitted model, not from the labelling rule.",
        )
        st.plotly_chart(charts.feature_importance(ctx.feature_importance, height=400))

        ranked = sorted(ctx.feature_importance.items(), key=lambda kv: -kv[1])
        components.metric_grid(
            [
                {
                    "label": f"Most important -- {ranked[0][0]}",
                    "value": f"{ranked[0][1] * 100:.1f}%",
                    "description": "share of total importance",
                },
                {
                    "label": f"Least important -- {ranked[-1][0]}",
                    "value": f"{ranked[-1][1] * 100:.1f}%",
                    "description": "share of total importance",
                },
            ]
        )
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        components.note(
            "Feature importance shows how much the model relied on each sensor, not "
            "whether the sensor caused the risk. It is a correlation-based measure and "
            "should not be read causally.",
        )