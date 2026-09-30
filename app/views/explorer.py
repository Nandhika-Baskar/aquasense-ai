"""Dataset Explorer -- inspect the data, or load your own CSV."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app import charts, components, config as cfg
from app.context import AppContext

MAX_UPLOAD_MB = 20


def _read_upload(uploaded) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse an uploaded CSV and validate it against the expected schema."""
    problems: list[str] = []
    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        return None, [f"Could not read the file as CSV: {exc}"]

    if frame.empty:
        return None, ["The uploaded file contains no rows."]

    missing = [c for c in [*cfg.FEATURES, cfg.TARGET] if c not in frame.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}")

    if cfg.TARGET in frame.columns:
        unexpected = sorted(set(frame[cfg.TARGET].dropna().unique()) - set(cfg.RISK_ORDER))
        if unexpected:
            problems.append(f"Unexpected label value(s) in {cfg.TARGET}: {unexpected}")

    if not missing and frame[cfg.FEATURES].isna().any().any():
        problems.append("Sensor columns contain missing values.")

    non_numeric = [
        c for c in cfg.FEATURES
        if c in frame.columns and not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if non_numeric:
        problems.append(f"Non-numeric sensor column(s): {', '.join(non_numeric)}")

    return frame, problems


def render(ctx: AppContext) -> None:
    components.page_header(
        "DATASET EXPLORER",
        f"Synthetic demonstration dataset -- {ctx.dataset['path']}",
    )
    components.note(cfg.DATASET_NOTE, kind="warn")

    source = st.radio(
        "Data source",
        ["Bundled demonstration dataset", "Upload a CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source.startswith("Upload"):
        _render_upload()
    else:
        _render_dataset(ctx.df, downloadable=True, original_bytes=None)


def _render_upload() -> None:
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        components.empty_state(
            "No file selected",
            "Upload a CSV to inspect it. Required columns: "
            + ", ".join([*cfg.FEATURES, cfg.TARGET]),
            "The bundled demonstration dataset is on the other tab.",
        )
        return

    size_mb = uploaded.size / 1_048_576
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File is {size_mb:.1f} MB. The limit is {MAX_UPLOAD_MB} MB.")
        return

    frame, problems = _read_upload(uploaded)

    if frame is None:
        st.error(problems[0])
        return

    for problem in problems:
        st.error(problem)

    if not problems:
        st.success(
            f"Loaded {len(frame):,} rows and {len(frame.columns)} columns from "
            f"{uploaded.name}."
        )
        _render_dataset(frame, downloadable=True, original_bytes=uploaded.getvalue())


def _render_dataset(df: pd.DataFrame, downloadable: bool, original_bytes: bytes | None) -> None:
    missing_total = int(df.isna().sum().sum())

    components.section("Dataset Summary")
    components.metric_grid(
        [
            {"label": "Rows", "value": f"{len(df):,}", "description": "total readings"},
            {
                "label": "Columns",
                "value": str(len(df.columns)),
                "description": "features + identifiers + target",
            },
            {
                "label": "Missing values",
                "value": f"{missing_total:,}",
                "description": "across all columns",
                "status": "None" if missing_total == 0 else "Present",
                "color": cfg.COLOR_OK if missing_total == 0 else cfg.COLOR_DANGER,
            },
            {
                "label": "Classes",
                "value": str(df[cfg.TARGET].nunique()) if cfg.TARGET in df.columns else "n/a",
                "description": "distinct risk labels",
            },
        ]
    )

    if cfg.TARGET in df.columns and df[cfg.TARGET].nunique() <= 6:
        components.section("Class Distribution")
        st.plotly_chart(charts.class_distribution(df), height=300)

    tab_preview, tab_stats, tab_columns = st.tabs(
        ["Preview", "Descriptive Statistics", "Columns"]
    )

    with tab_preview:
        rows = st.slider("Rows to preview", 5, min(200, max(len(df), 5)), 25, step=5)
        st.dataframe(df.head(rows), width="stretch", height=380)

    with tab_stats:
        numeric = df.select_dtypes("number")
        if numeric.empty:
            components.empty_state("No numeric columns", "Descriptive statistics are unavailable.")
        else:
            st.dataframe(
                numeric.describe().T.round(3), width="stretch", height=420
            )

    with tab_columns:
        info = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": [str(t) for t in df.dtypes],
                "Non-null": [int(df[c].notna().sum()) for c in df.columns],
                "Missing": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        st.dataframe(info, width="stretch", hide_index=True, height=420)

    if downloadable:
        payload = original_bytes if original_bytes is not None else df.to_csv(index=False)
        st.download_button(
            "Download this dataset (CSV)",
            payload,
            file_name="water_quality.csv",
            mime="text/csv",
        )