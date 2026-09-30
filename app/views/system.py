"""System -- architecture, pipeline and limitations."""

from __future__ import annotations

import streamlit as st

from app import components, config as cfg
from app.context import AppContext

TECH_STACK = ["Python", "Streamlit", "Scikit-learn", "Pandas", "NumPy", "Plotly", "Joblib"]

PIPELINE_STEPS = [
    "Sensor Data",
    "Data Validation",
    "Preprocessing",
    "Machine Learning Model",
    "Risk Classification",
    "Visualization / Dashboard",
]

PIPELINE_DETAIL = [
    "Five readings per sample: pH, TDS, Turbidity, Temperature, Conductivity, plus an "
    "identifier and a timestamp. Generated synthetically for this demonstration.",
    "Column presence, label vocabulary and value ranges are checked before anything is "
    "trained or predicted, so invalid input raises a clear error instead of silently "
    "returning a wrong answer.",
    "Median imputation for any missing sensor reading, then standard scaling. Both steps "
    "live inside the scikit-learn pipeline, so training and inference cannot drift apart.",
    f"{cfg.APP_MODEL_NAME} with {cfg.MODEL_PARAMS['n_estimators']} trees and "
    "class_weight='balanced', so the majority class cannot dominate the fit.",
    "predict_proba returns a LOW / MEDIUM / HIGH class with its probability. The column "
    "order is remapped from sklearn's alphabetical ordering so classes cannot be mixed up.",
    "Streamlit renders the metrics and Plotly renders the charts. The fitted pipeline is "
    "loaded once per session with st.cache_resource.",
]

FUTURE_ITEMS = [
    "Device layer -- read the five sensors from a microcontroller (for example ESP32 over "
    "Modbus) and publish readings over MQTT or a local REST endpoint.",
    "Ingestion -- a small local service that validates units and ranges, stamps each "
    "reading with a site id, and appends to a time-series store instead of a CSV file.",
    "Site awareness -- add location, depth and season as features, and train per-site "
    "models wherever water chemistry differs between sources.",
    "Drift monitoring -- track feature distributions and prediction confidence over time, "
    "and flag when they move away from the training distribution.",
    "Alerting -- route HIGH classifications to a notification channel with the underlying "
    "readings attached, so a person can judge them.",
    "Retraining -- retrain periodically against labelled field data once real laboratory "
    "results exist to provide those labels.",
]


def limitations(ctx: AppContext) -> list[dict]:
    return [
        {
            "title": "Synthetic training data",
            "detail": (
                "The model is trained on programmatically generated demonstration data, not "
                "on measured water samples. Its accuracy measures how well it learned the "
                "generating rule and says nothing about real water."
            ),
            "level": "info",
        },
        {
            "title": "Risk classification is not potability",
            "detail": (
                "LOW / MEDIUM / HIGH describe relative contamination risk against the "
                "guideline bands used in this project. They are not a health assessment and "
                "not a certification that water is safe to drink."
            ),
            "level": "info",
        },
        {
            "title": "Guideline bands are a simplification",
            "detail": (
                "Drinking-water standards also cover pathogens, heavy metals, pesticides "
                "and disinfection by-products. None of those are observable from these "
                "five sensors."
            ),
            "level": "info",
        },
        {
            "title": "No live hardware",
            "detail": (
                "The monitoring page replays a simulated stream derived from the dataset. "
                "There is no serial, MQTT or network ingestion in this build."
            ),
            "level": "info",
        },
        {
            "title": "Single holdout evaluation",
            "detail": (
                f"Headline accuracy comes from one stratified "
                f"{int(ctx.split['test_size'] * 100)}% test split "
                f"({ctx.split['n_test']:,} samples). The {ctx.metrics['cv_folds']}-fold "
                f"cross-validated score is shown next to it, because a single split alone "
                "can flatter a model."
            ),
            "level": "info",
        },
        {
            "title": "Correlated inputs",
            "detail": (
                "Conductivity is derived from TDS in this dataset, so two of the five "
                "inputs carry partly the same information and share the importance between "
                "them."
            ),
            "level": "info",
        },
    ]


def render(ctx: AppContext) -> None:
    components.page_header(
        "SYSTEM",
        "Architecture, machine learning pipeline and known limitations.",
    )

    components.section("Technology Stack")
    components.badges(TECH_STACK)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([2, 3], gap="large")

    with left:
        components.section("Architecture", "Path from a reading to a classification")
        components.flow_diagram(PIPELINE_STEPS)

    with right:
        components.section("ML Pipeline", "What each stage actually does")
        rows = "".join(
            f'<div style="display:flex;gap:.7rem;padding:.6rem 0;'
            f'border-bottom:1px solid {cfg.COLOR_BORDER}">'
            f'<div style="flex:0 0 auto;font-size:.63rem;font-weight:700;color:{cfg.COLOR_PRIMARY};'
            f'background:#E7F2F5;border-radius:5px;padding:.12rem .34rem;height:fit-content">'
            f"{i:02d}</div>"
            f'<div><div style="font-size:.82rem;font-weight:650">{name}</div>'
            f'<div style="font-size:.775rem;color:{cfg.COLOR_MUTED};line-height:1.5;'
            f'margin-top:.1rem">{detail}</div></div></div>'
            for i, (name, detail) in enumerate(zip(PIPELINE_STEPS, PIPELINE_DETAIL), start=1)
        )
        st.markdown(rows, unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    components.section("Model Artefacts")
    components.stat_grid(
        [
            ("Model", ctx.model["name"]),
            ("Trained at", ctx.trained_at),
            ("Training rows", f"{ctx.split['n_train']:,}"),
            ("Test rows", f"{ctx.split['n_test']:,}"),
            ("Dataset SHA-256", ctx.dataset.get("sha256", "n/a")[:16] + "..."),
            ("Test accuracy", components.pct(ctx.metrics["accuracy"], 2)),
        ]
    )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    limits_tab, future_tab = st.tabs(["Limitations", "Future IoT Integration"])

    with limits_tab:
        components.section(
            "Known Limitations",
            "Stated plainly, because they bound what the reported results mean.",
        )
        components.explanation_list(limitations(ctx))

    with future_tab:
        components.section(
            "Future IoT Integration",
            "Not implemented in this build. Listed so the extension path is clear.",
        )
        components.explanation_list(
            [{"title": "Planned", "detail": item, "level": "info"} for item in FUTURE_ITEMS]
        )
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        components.note(
            "Everything in this project runs locally. There are no external APIs, no "
            "authentication layer, no database and no container runtime."
        )