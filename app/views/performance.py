"""ML Performance -- evaluation of the trained Random Forest."""

from __future__ import annotations

import streamlit as st

from app import charts, components
from app.context import AppContext


def render(ctx: AppContext) -> None:
    m = ctx.metrics

    components.page_header(
        "MODEL PERFORMANCE",
        f"{ctx.model['name']} evaluated on a held-out stratified test set of "
        f"{ctx.split['n_test']} readings.",
    )

    # --- headline metrics ---------------------------------------------------
    components.metric_grid(
        [
            {
                "label": "Accuracy",
                "value": components.pct(m["accuracy"], 2),
                "description": "overall correct classifications",
            },
            {
                "label": "Precision",
                "value": components.pct(m["precision_macro"], 2),
                "description": "macro-averaged across 3 classes",
            },
            {
                "label": "Recall",
                "value": components.pct(m["recall_macro"], 2),
                "description": "macro-averaged across 3 classes",
            },
            {
                "label": "F1 Score",
                "value": components.pct(m["f1_macro"], 2),
                "description": "macro harmonic mean",
            },
        ]
    )
    st.caption(
        "Macro averaging weights LOW, MEDIUM and HIGH equally, which matters because "
        "the classes are not perfectly balanced. Weighted averages: "
        f"precision {components.pct(m['precision_weighted'], 2)}, "
        f"recall {components.pct(m['recall_weighted'], 2)}, "
        f"F1 {components.pct(m['f1_weighted'], 2)}."
    )

    # --- model identity -----------------------------------------------------
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    components.section("Model Configuration")
    components.stat_grid(
        [
            ("Model", ctx.model["name"]),
            ("Training samples", f"{ctx.split['n_train']:,}"),
            ("Testing samples", f"{ctx.split['n_test']:,}"),
            ("Features", str(len(ctx.split["features"]))),
            ("Trees", f"{ctx.model['n_trees']:,}"),
            ("Test split", f"{ctx.split['test_size'] * 100:.0f}%"),
        ]
    )
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    components.metric_grid(
        [
            {
                "label": f"{m['cv_folds']}-fold CV accuracy",
                "value": components.pct(m["cv_accuracy_mean"], 2),
                "description": f"+/- {m['cv_accuracy_std'] * 100:.2f}% standard deviation",
            },
            {
                "label": "Test-set accuracy",
                "value": components.pct(m["accuracy"], 2),
                "description": "single stratified holdout",
            },
        ]
    )
    components.note(
        "The cross-validated score is computed on the training split and the accuracy "
        "card is computed on the untouched test set. They are separate measurements of "
        "the same model, not two names for the same number.",
    )

    # --- confusion matrix ---------------------------------------------------
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        components.section(
            "Confusion Matrix",
            "Rows are the actual class, columns the predicted class. Cell text shows "
            "the count and the row percentage.",
        )
        st.plotly_chart(
            charts.confusion_matrix_heatmap(
                m["confusion_labels"], m["confusion_matrix"], height=400
            )
        )

    with right:
        components.section("Feature Importance")
        st.plotly_chart(charts.feature_importance(ctx.feature_importance, height=400))

    # --- per class ----------------------------------------------------------
    components.section("Per-Class Performance", "Held-out test set")
    per_class = m["per_class"]
    st.dataframe(
        [
            {
                "Class": cls,
                "Precision": components.pct(row["precision"], 2),
                "Recall": components.pct(row["recall"], 2),
                "F1": components.pct(row["f1"], 2),
                "Support": row["support"],
            }
            for cls, row in per_class.items()
        ],
        width="stretch",
        hide_index=True,
    )

    # --- report -------------------------------------------------------------
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    components.section("Classification Report", "sklearn.metrics.classification_report")
    st.code(m["classification_report"].strip(), language="text")

    components.note(
        "Every metric on this page was computed from the model in "
        "models/random_forest_model.joblib against the held-out test split. Nothing "
        "here is estimated or hand-written.",
    )