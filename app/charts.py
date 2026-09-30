"""
Plotly chart builders for AquaSense AI.

One shared layout style keeps every chart on the dashboard consistent, and each
function returns a Plotly figure so pages stay declarative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import config as cfg


def _base_layout(fig: go.Figure, height: int = 300, showlegend: bool | None = None) -> go.Figure:
    fig.update_layout(
        template=cfg.PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=cfg.COLOR_INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#EEF2F5", zeroline=False, linecolor="#E3E8EC", tickfont_size=11)
    fig.update_yaxes(gridcolor="#EEF2F5", zeroline=False, linecolor="#E3E8EC", tickfont_size=11)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Centred rolling mean used to take the sample-level jitter out of a series.

    `window` is a count of samples, not a time span, because the readings are
    already evenly spaced in the demonstration dataset. A window of 1 (or less)
    returns the series untouched, so callers can always pass the control value
    through without special-casing it.
    """
    size = int(window or 1)
    if size <= 1 or series.empty:
        return series
    size = min(size, len(series))
    return series.rolling(window=size, min_periods=1, center=True).mean()


def smooth_frame(frame: pd.DataFrame, features: list[str], window: int) -> pd.DataFrame:
    """Apply `rolling_mean` to each sensor column. Returns a copy, never mutates."""
    smoothed = frame.copy()
    for feature in features:
        smoothed[feature] = rolling_mean(smoothed[feature], window)
    return smoothed


def sensor_trend(
    df: pd.DataFrame,
    features: list[str],
    timestamp_col: str = cfg.TIME_COL,
    height: int = 320,
    title: str | None = None,
    rolling: int = 1,
) -> go.Figure:
    """
    Multi-series line chart of sensor readings over time.

    `rolling` averages each series over that many consecutive samples. It only
    changes how the existing readings are drawn; no value is dropped or invented.
    """
    frame = smooth_frame(df[[timestamp_col, *features]], features, rolling).melt(
        id_vars=timestamp_col, var_name="Sensor", value_name="Reading"
    )
    fig = px.line(
        frame,
        x=timestamp_col,
        y="Reading",
        color="Sensor",
        color_discrete_sequence=cfg.CHART_PALETTE,
        title=title,
    )
    if timestamp_col in frame.columns:
        fig.update_xaxes(title_text="", showticklabels=False)
    return _base_layout(fig, height=height)


def sensor_trend_normalised(
    df: pd.DataFrame, features: list[str], height: int = 320, rolling: int = 1
) -> go.Figure:
    """
    Same trend, but each sensor scaled to its guideline band.

    Raw TDS (~1200) and pH (~7) cannot share one axis, and normalising by the
    band width is what makes the series comparable.

    The band itself is shaded between 0 and 1 so "inside the guideline" is
    readable at a glance instead of having to be inferred from the axis, and
    `rolling` applies the same centred moving average as `sensor_trend` so a
    long series reads as a trend line rather than a solid block of spikes.
    """
    frame = df[[cfg.TIME_COL, *features]].copy()
    for feature in features:
        width = max(cfg.SAFE_HIGH[feature] - cfg.SAFE_LOW[feature], 1e-9)
        frame[feature] = (frame[feature] - cfg.SAFE_LOW[feature]) / width

    frame = smooth_frame(frame, features, rolling)

    melted = frame.melt(id_vars=cfg.TIME_COL, var_name="Sensor", value_name="Normalised")
    fig = px.line(
        melted,
        x=cfg.TIME_COL,
        y="Normalised",
        color="Sensor",
        color_discrete_sequence=cfg.CHART_PALETTE,
        title="Sensor trend (normalised to guideline band; 0-1 = inside band)",
    )
    fig.update_xaxes(title_text="", showticklabels=False)
    fig.update_yaxes(title_text="Band position")
    # Shaded guideline band, drawn under the series.
    fig.add_hrect(
        y0=0.0,
        y1=1.0,
        fillcolor=cfg.COLOR_OK,
        opacity=0.07,
        line_width=0,
        layer="below",
    )
    fig.add_hline(y=0.0, line_color="#C8D2D8", line_width=1)
    fig.add_hline(y=1.0, line_color="#C8D2D8", line_width=1)
    return _base_layout(fig, height=height)


def distribution(df: pd.DataFrame, feature: str, risk_col: str = cfg.TARGET, height: int = 300) -> go.Figure:
    """Histogram of one sensor, split by risk class."""
    fig = px.histogram(
        df,
        x=feature,
        color=risk_col,
        color_discrete_map={c: cfg.RISK_COLORS[c] for c in cfg.RISK_ORDER},
        barmode="overlay",
        opacity=0.72,
        nbins=34,
        marginal="box",
        title=f"{feature} distribution by risk class",
    )
    lo, hi = cfg.SAFE_LOW[feature], cfg.SAFE_HIGH[feature]
    if lo > 0 or hi > 0:
        fig.add_vrect(x0=max(lo, 0), x1=hi, fillcolor="#1F9D55", opacity=0.07, line_width=0)
    fig.update_yaxes(title_text="Readings")
    return _base_layout(fig, height=height, showlegend=True)


def correlation_heatmap(df: pd.DataFrame, features: list[str], height: int = 380) -> go.Figure:
    corr = df[features].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=features,
            y=features,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            xgap=2,
            ygap=2,
            colorbar=dict(title="r", thickness=12, len=0.72, outlinewidth=0),
            text=np.round(corr.to_numpy(), 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Sensor correlation matrix")
    return _base_layout(fig, height=height, showlegend=False)


def class_distribution(df: pd.DataFrame, height: int = 300) -> go.Figure:
    counts = df[cfg.TARGET].value_counts().reindex(cfg.RISK_ORDER)
    fig = go.Figure(
        go.Bar(
            x=cfg.RISK_ORDER,
            y=counts.to_numpy(),
            marker_color=[cfg.RISK_COLORS[c] for c in cfg.RISK_ORDER],
            text=[f"{v:,}" for v in counts.to_numpy()],
            textposition="outside",
            hovertemplate="%{x}: %{y} readings<extra></extra>",
        )
    )
    fig.update_layout(title="Risk class distribution")
    fig.update_yaxes(title_text="Readings", rangemode="tozero")
    fig.update_xaxes(title_text="")
    return _base_layout(fig, height=height, showlegend=False)


def feature_importance(importance: dict[str, float], height: int = 320) -> go.Figure:
    ordered = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in ordered],
            y=[k for k, _ in ordered],
            orientation="h",
            marker_color=cfg.COLOR_PRIMARY,
            text=[f"{v * 100:.1f}%" for _, v in ordered],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title="Random Forest feature importance")
    fig.update_xaxes(title_text="Gini importance", rangemode="tozero")
    fig.update_yaxes(title_text="")
    return _base_layout(fig, height=height, showlegend=False)


def confusion_matrix_heatmap(labels: list[str], matrix: list[list[int]], height: int = 360) -> go.Figure:
    total = np.array(matrix, dtype=float)
    total[total == 0] = 1
    normalised = np.array(matrix, dtype=float) / total * 100

    text = [
        [f"{matrix[i][j]}<br><span style='font-size:10px'>{normalised[i][j]:.1f}%</span>"
         for j in range(len(labels))]
        for i in range(len(labels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            xgap=3,
            ygap=3,
            colorscale=[[0, "#FFFFFF"], [1, "#0B7285"]],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorbar=dict(title="Count", thickness=12, len=0.75, outlinewidth=0),
            hovertemplate="actual %{y} / predicted %{x}<br>%{z} samples<extra></extra>",
        )
    )
    fig.update_layout(title="Confusion matrix (rows = actual, columns = predicted)")
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return _base_layout(fig, height=height, showlegend=False)


def probability_bars(probabilities: dict[str, float], height: int = 210) -> go.Figure:
    """Horizontal bars for one prediction's class probabilities."""
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in ordered],
            y=[k for k, _ in ordered],
            orientation="h",
            marker_color=[cfg.RISK_COLORS[k] for k, _ in ordered],
            text=[f"{v * 100:.1f}%" for _, v in ordered],
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Probability", range=[0, 1.08], tickformat=".0%")
    fig.update_yaxes(title_text="")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=height, showlegend=False)


def scatter_by_class(
    df: pd.DataFrame, x: str, y: str, height: int = 340, log_x: bool = False
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=cfg.TARGET,
        color_discrete_map={c: cfg.RISK_COLORS[c] for c in cfg.RISK_ORDER},
        opacity=0.72,
        title=f"{x} vs {y} by risk class",
    )
    if log_x:
        fig.update_xaxes(type="log")
    return _base_layout(fig, height=height)