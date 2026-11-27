"""
Streamlit UI for interactive testing and error analysis of two sentiment models.

All model loading, inference, and routing live in predictor.py; all metric
computations live in metrics.py.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from metrics import (
    compute_classification_metrics,
    compute_comparison_table,
    compute_context_robustness,
    compute_disagreement_guard_sweep,
    compute_routing_metrics,
    compute_slice_analysis,
    normalize_and_filter_labels,
)
from predictor import (
    LABELS,
    LINEAR_MODEL_PATH,
    NEUTRAL_CONTEXT,
    RANDOM_SEED,
    TRANSFORMER_DIR,
    VERSIONS_PATH,
    SentimentModels,
    apply_routing_batch,
    apply_routing_row,
    clean_text,
    has_non_empty_context,
    load_models,
    predict_linear_batch,
    predict_transformer_batch,
)


# Configure wide layout and clean page title.
st.set_page_config(
    page_title="Sentiment Model Test Bench",
    layout="wide",
)


@st.cache_resource
def get_cached_models() -> SentimentModels:
    """
    Load and cache both sentiment models once per Streamlit server session.

    Returns:
        Loaded SentimentModels container.
    """
    # Delegate offline loading to predictor.py.
    return load_models()


def render_confusion_matrix(cm_pct_df: pd.DataFrame) -> plt.Figure:
    """
    Plot a row-normalized percentage confusion matrix heatmap.

    Args:
        cm_pct_df: DataFrame of percentages with true labels as index and predicted as columns.

    Returns:
        Matplotlib Figure object.
    """
    # Create a compact, readable heatmap figure.
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(
        cm_pct_df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        cbar_kws={"label": "Row %"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Expected label")
    ax.set_title("Row-normalized confusion matrix (%)")
    plt.tight_layout()
    return fig


def render_transformer_probs_chart(probs: np.ndarray) -> plt.Figure:
    """
    Plot a bar chart of the four transformer class probabilities.

    Args:
        probs: 1D array of 4 probabilities aligned with LABELS.

    Returns:
        Matplotlib Figure object.
    """
    # Plot probabilities in the fixed canonical label order.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.bar(LABELS, probs, color="#2b6cb0")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title("Transformer class probabilities")
    for bar, val in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    return fig


def render_robustness_chart(robustness_df: pd.DataFrame) -> plt.Figure:
    """
    Plot accuracy and macro-F1 across context robustness conditions with transformer reference.

    Args:
        robustness_df: Output DataFrame from compute_context_robustness.

    Returns:
        Matplotlib Figure object.
    """
    # Separate the three linear conditions from the transformer reference row.
    linear_rows = robustness_df.loc[
        robustness_df["condition"].str.startswith("Linear:")
    ].copy()
    ref_row = robustness_df.loc[
        robustness_df["condition"].str.startswith("Transformer")
    ]
    ref_acc = float(ref_row["accuracy"].iloc[0]) if not ref_row.empty else None

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(linear_rows))
    width = 0.35

    ax.bar(x - width / 2, linear_rows["accuracy"], width, label="Accuracy", color="#2b6cb0")
    ax.bar(x + width / 2, linear_rows["macro_f1"], width, label="Macro-F1", color="#38a169")

    # Draw transformer-alone accuracy as a horizontal dashed reference line.
    if ref_acc is not None:
        ax.axhline(
            ref_acc,
            color="#dd6b20",
            linestyle="--",
            linewidth=1.5,
            label=f"Transformer alone accuracy ({ref_acc:.3f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("Linear: ", "") for c in linear_rows["condition"]],
        rotation=0,
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Linear model robustness under context perturbation")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    return fig


def render_sidebar(models: SentimentModels | None, load_error: str | None) -> tuple[float, float, bool]:
    """
    Render model status, version checks, and routing control widgets in the sidebar.

    Args:
        models: Loaded SentimentModels or None if loading failed.
        load_error: Error string if model loading failed.

    Returns:
        Tuple of (low_conf, guard_thr, context_verified).
    """
    st.sidebar.header("Model & Routing Controls")

    # Show model loading status, device, and library version comparison.
    if load_error is not None or models is None:
        st.sidebar.error(f"Model status: Error\n\n{load_error}")
    else:
        st.sidebar.success("Model status: Loaded")
        device_label = "GPU (CUDA)" if models.device.type == "cuda" else "CPU"
        st.sidebar.write(f"Device: `{device_label}`")

        # Display installed vs training versions from versions.json.
        v_info = models.versions_info
        expected = v_info["expected"]
        installed = v_info["installed"]

        version_rows = [
            {
                "library": lib,
                "installed": installed.get(lib, ""),
                "versions.json": str(expected.get(lib, "")),
            }
            for lib in ("scikit-learn", "transformers", "torch")
        ]
        st.sidebar.dataframe(
            pd.DataFrame(version_rows),
            hide_index=True,
            use_container_width=True,
        )

        # Warn visibly when scikit-learn differs because joblib pickles are version-sensitive.
        if v_info["sklearn_mismatch"]:
            st.sidebar.warning(
                f"scikit-learn version mismatch: installed {installed['scikit-learn']} "
                f"vs versions.json {expected.get('scikit-learn')}. "
                "Joblib pipelines can break across versions."
            )
        if v_info["transformers_mismatch"] or v_info["torch_mismatch"]:
            st.sidebar.info(
                "Note: transformers or torch runtime version differs from versions.json."
            )

    st.sidebar.divider()

    # Routing threshold sliders and context verification toggle.
    low_conf = st.sidebar.slider(
        "low_conf (transformer-only review threshold)",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )
    guard_thr = st.sidebar.slider(
        "guard_thr (disagreement guard threshold)",
        min_value=0.00,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )
    context_verified = st.sidebar.checkbox(
        "Treat supplied context as verified",
        value=True,
    )

    return low_conf, guard_thr, context_verified


def render_single_message_tab(
    models: SentimentModels,
    low_conf: float,
    guard_thr: float,
    context_verified: bool,
) -> None:
    """
    Render Tab 1 for single-message inference and three-context comparison.

    Args:
        models: Loaded SentimentModels container.
        low_conf: Low-confidence threshold.
        guard_thr: Disagreement-guard threshold.
        context_verified: Whether supplied context is treated as verified.
    """
    st.subheader("Single Message Evaluation")
    st.caption(
        "Inspect routed predictions, model agreement, class probabilities, and context sensitivity."
    )

    # Input widgets for inquiry, optional context, and optional expected label.
    inquiry_input = st.text_area(
        "Customer inquiry",
        height=90,
        placeholder="Enter customer message...",
        key="single_inquiry_input",
    )
    context_input = st.text_area(
        "Conversation context (optional)",
        height=75,
        placeholder="Optional prior interaction summary...",
        key="single_context_input",
    )
    expected_choice = st.selectbox(
        "Expected label (optional)",
        options=["unknown"] + LABELS,
        index=0,
        key="single_expected_choice",
    )

    # Run both models once on button click and cache raw outputs in session_state.
    if st.button("Predict", type="primary", key="single_predict_btn"):
        cleaned_inquiry = clean_text(inquiry_input)
        cleaned_context = clean_text(context_input)

        if not cleaned_inquiry.strip():
            st.error("Empty input: please enter a non-empty customer inquiry.")
            st.session_state.pop("single_cache", None)
        else:
            # Compute transformer probabilities once for the inquiry.
            probs, t_labels, t_confs = predict_transformer_batch([cleaned_inquiry], models)
            # Compute linear predictions under all three context conditions for comparison.
            lin_preds = predict_linear_batch(
                [cleaned_inquiry, cleaned_inquiry, cleaned_inquiry],
                [cleaned_context, NEUTRAL_CONTEXT, ""],
                models,
            )
            st.session_state["single_cache"] = {
                "inquiry": cleaned_inquiry,
                "context": cleaned_context,
                "expected": expected_choice,
                "probs": probs[0],
                "t_label": t_labels[0],
                "t_conf": t_confs[0],
                "linear_given": lin_preds[0],
                "linear_neutral": lin_preds[1],
                "linear_none": lin_preds[2],
            }

    # Re-apply routing dynamically from cached raw outputs so sidebar changes update instantly.
    cache = st.session_state.get("single_cache")
    if cache is None:
        return

    routed = apply_routing_row(
        t_label=cache["t_label"],
        t_conf=cache["t_conf"],
        linear_label=cache["linear_given"],
        context=cache["context"],
        context_verified=context_verified,
        low_conf=low_conf,
        guard_thr=guard_thr,
    )

    # Top summary metrics.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Routed label", routed["label"])
    col2.metric("Model used", routed["model_used"])
    review_text = f"Yes ({routed['reason']})" if routed["needs_review"] else "No"
    col3.metric("Needs review", review_text)
    col4.metric("Transformer", f"{cache['t_label']} ({cache['t_conf']:.3f})")

    # Show linear prediction and agreement indicator when non-empty context was supplied.
    if has_non_empty_context(cache["context"]):
        agrees = cache["t_label"] == cache["linear_given"]
        agreement_text = "Models agree" if agrees else "Models disagree"
        st.write(
            f"**Linear+context prediction:** `{cache['linear_given']}` | "
            f"**Agreement:** `{agreement_text}`"
        )
    else:
        st.write("**Linear+context prediction:** Not used (no context supplied)")

    # Show expected-label check when an expected label was selected.
    if cache["expected"] != "unknown":
        if routed["label"] == cache["expected"]:
            st.success(f"Correct: routed prediction `{routed['label']}` matches expected `{cache['expected']}`.")
        else:
            st.error(
                f"Incorrect: routed prediction `{routed['label']}` does not match expected `{cache['expected']}`."
            )

    # Bar chart of the four transformer probabilities.
    fig = render_transformer_probs_chart(cache["probs"])
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # Side-by-side comparison of the same inquiry under three contexts.
    st.markdown("**Context sensitivity (same inquiry under three contexts)**")
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

    conditions = [
        (ctx_col1, "As given", cache["context"], cache["linear_given"]),
        (ctx_col2, "Neutral", NEUTRAL_CONTEXT, cache["linear_neutral"]),
        (ctx_col3, "None (empty)", "", cache["linear_none"]),
    ]
    for target_col, title, ctx_text, lin_pred in conditions:
        cond_routed = apply_routing_row(
            t_label=cache["t_label"],
            t_conf=cache["t_conf"],
            linear_label=lin_pred,
            context=ctx_text,
            context_verified=context_verified,
            low_conf=low_conf,
            guard_thr=guard_thr,
        )
        with target_col:
            st.markdown(f"**{title}**")
            st.caption(ctx_text if ctx_text else "(empty)")
            st.write(f"Routed: `{cond_routed['label']}`")
            st.write(f"Path: `{cond_routed['model_used']}`")
            st.write(f"Linear: `{lin_pred}`")
            flag_str = f"Yes ({cond_routed['reason']})" if cond_routed["needs_review"] else "No"
            st.write(f"Review: `{flag_str}`")


def render_batch_tab(
    models: SentimentModels,
    low_conf: float,
    guard_thr: float,
    context_verified: bool,
) -> None:
    """
    Render Tab 2 for batch CSV/table evaluation, metrics, slices, and error analysis.

    Args:
        models: Loaded SentimentModels container.
        low_conf: Low-confidence threshold.
        guard_thr: Disagreement-guard threshold.
        context_verified: Whether supplied context is treated as verified.
    """
    st.subheader("Batch Evaluation")
    st.caption(
        "Upload a labelled CSV or enter rows manually. Model outputs are cached so threshold changes update routing immediately."
    )

    input_mode = st.radio(
        "Input source",
        options=["Upload CSV", "Manual table (data_editor)"],
        horizontal=True,
    )

    raw_df: pd.DataFrame | None = None

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read CSV file: {exc}")
                return
    else:
        # Provide an editable starter table for quick manual batch testing.
        default_manual = pd.DataFrame(
            [
                {
                    "customer_inquiry": "Asante sana, nimepata pesa yangu baada ya siku tatu!",
                    "conversation_context": "",
                    "sentiment": "positive",
                },
                {
                    "customer_inquiry": "Sawa, nimeelewa vizuri sana.",
                    "conversation_context": "Customer had one earlier chat about a delayed refund with no resolution.",
                    "sentiment": "negative",
                },
            ]
        )
        raw_df = st.data_editor(
            default_manual,
            num_rows="dynamic",
            use_container_width=True,
            key="batch_data_editor",
        )

    if raw_df is None:
        st.info("Upload a CSV file or switch to Manual table to begin.")
        return

    if raw_df.empty or len(raw_df.columns) == 0:
        st.error("Empty input: the provided dataset has no rows or columns.")
        return

    cols = list(raw_df.columns)

    # Auto-select standard column names when present.
    default_inq_idx = cols.index("customer_inquiry") if "customer_inquiry" in cols else 0
    ctx_options = ["(none)"] + cols
    default_ctx_idx = ctx_options.index("conversation_context") if "conversation_context" in cols else 0
    default_lbl_idx = cols.index("sentiment") if "sentiment" in cols else 0

    map_col1, map_col2, map_col3 = st.columns(3)
    inquiry_col = map_col1.selectbox("Inquiry column", options=cols, index=default_inq_idx)
    context_col = map_col2.selectbox("Context column (optional)", options=ctx_options, index=default_ctx_idx)
    label_col = map_col3.selectbox("Expected label column", options=cols, index=default_lbl_idx)

    # Validate that inquiry and expected-label columns are distinct when possible.
    if inquiry_col == label_col:
        st.error("Bad column mapping: Inquiry column and Expected label column must be different.")
        return

    # Allow user to pick optional categorical columns for slice analysis.
    reserved_cols = {inquiry_col, label_col}
    if context_col != "(none)":
        reserved_cols.add(context_col)
    candidate_slice_cols = [c for c in cols if c not in reserved_cols]
    slice_cols = st.multiselect(
        "Slice columns (optional)",
        options=candidate_slice_cols,
        default=[],
    )

    # Button to run transformer + linear inference once and cache results.
    if st.button("Run models", type="primary", key="batch_run_btn"):
        # Check for empty inquiry rows.
        working_df = raw_df.copy()
        working_df["_clean_inquiry"] = working_df[inquiry_col].apply(clean_text)
        if working_df["_clean_inquiry"].str.strip().eq("").all():
            st.error("Empty input: all rows in the selected inquiry column are blank.")
            return

        # Normalize expected labels and separate invalid rows instead of crashing.
        valid_df, invalid_df = normalize_and_filter_labels(working_df, label_col, LABELS)
        if valid_df.empty:
            bad_vals = sorted(invalid_df["expected_normalized"].unique().tolist())
            st.error(
                f"Unknown labels: no rows have a valid label in {LABELS}. "
                f"Found: {bad_vals}"
            )
            return

        inquiries = valid_df[inquiry_col].apply(clean_text).tolist()
        if context_col == "(none)":
            contexts = [""] * len(valid_df)
        else:
            contexts = valid_df[context_col].apply(clean_text).tolist()

        # Run transformer inference with a visible progress bar.
        progress_bar = st.progress(0.0, text="Running transformer inference...")
        t_probs, t_labels, t_confs = predict_transformer_batch(
            inquiries,
            models,
            progress_callback=lambda frac: progress_bar.progress(
                frac, text=f"Running transformer inference ({int(frac * 100)}%)..."
            ),
        )
        progress_bar.empty()

        # Run linear model on supplied context.
        linear_labels = predict_linear_batch(inquiries, contexts, models)

        # Precompute linear robustness variants on rows that have non-empty context
        # so Tab 3 updates without re-running models.
        has_ctx_mask = [has_non_empty_context(c) for c in contexts]
        ctx_indices = [idx for idx, flag in enumerate(has_ctx_mask) if flag]

        robustness_cache: dict[str, Any] | None = None
        if ctx_indices:
            ctx_inquiries = [inquiries[i] for i in ctx_indices]
            ctx_true = [contexts[i] for i in ctx_indices]
            ctx_neutral = [NEUTRAL_CONTEXT] * len(ctx_indices)

            # Shuffle contexts deterministically using fixed seed 42.
            rng = np.random.default_rng(RANDOM_SEED)
            shuffled_order = rng.permutation(len(ctx_true))
            ctx_shuffled = [ctx_true[i] for i in shuffled_order]

            lin_neutral_preds = predict_linear_batch(ctx_inquiries, ctx_neutral, models)
            lin_shuffled_preds = predict_linear_batch(ctx_inquiries, ctx_shuffled, models)

            robustness_cache = {
                "y_true": [valid_df["expected_normalized"].iloc[i] for i in ctx_indices],
                "linear_true": [linear_labels[i] for i in ctx_indices],
                "linear_neutral": lin_neutral_preds,
                "linear_shuffled": lin_shuffled_preds,
                "t_labels": [t_labels[i] for i in ctx_indices],
                "t_confs": [t_confs[i] for i in ctx_indices],
            }

        # Store all raw predictions and metadata in session_state.
        st.session_state["batch_cache"] = {
            "valid_df": valid_df.reset_index(drop=True),
            "invalid_df": invalid_df,
            "inquiries": inquiries,
            "contexts": contexts,
            "has_context": has_ctx_mask,
            "t_probs": t_probs,
            "t_labels": t_labels,
            "t_confs": t_confs,
            "linear_labels": linear_labels,
            "slice_cols": slice_cols,
            "robustness_cache": robustness_cache,
        }

    batch_cache = st.session_state.get("batch_cache")
    if batch_cache is None:
        return

    # Report any excluded rows with unknown expected labels clearly.
    invalid_df = batch_cache["invalid_df"]
    if not invalid_df.empty:
        bad_summary = invalid_df["expected_normalized"].replace("", "(blank)").value_counts().to_dict()
        st.warning(
            f"Excluded {len(invalid_df)} row(s) with unknown expected labels: {bad_summary}"
        )

    # Apply routing dynamically using current sidebar settings without re-running models.
    routed_df = apply_routing_batch(
        t_labels=batch_cache["t_labels"],
        t_confs=batch_cache["t_confs"],
        linear_labels=batch_cache["linear_labels"],
        contexts=batch_cache["contexts"],
        context_verified=context_verified,
        low_conf=low_conf,
        guard_thr=guard_thr,
    )

    # Assemble complete evaluation DataFrame.
    eval_df = batch_cache["valid_df"].copy()
    eval_df["inquiry"] = batch_cache["inquiries"]
    eval_df["context"] = batch_cache["contexts"]
    eval_df["has_context"] = batch_cache["has_context"]
    eval_df["expected"] = eval_df["expected_normalized"]
    eval_df["t_label"] = batch_cache["t_labels"]
    eval_df["t_conf"] = batch_cache["t_confs"]
    eval_df["linear_label"] = batch_cache["linear_labels"]
    eval_df["predicted"] = routed_df["predicted"]
    eval_df["model_used"] = routed_df["model_used"]
    eval_df["needs_review"] = routed_df["needs_review"]
    eval_df["review_reason"] = routed_df["review_reason"]

    # Attach per-class transformer probabilities.
    t_probs = batch_cache["t_probs"]
    for idx, label_name in enumerate(LABELS):
        eval_df[f"prob_{label_name}"] = t_probs[:, idx]

    # Compute routed classification and routing metrics.
    cls_metrics = compute_classification_metrics(
        eval_df["expected"].tolist(),
        eval_df["predicted"].tolist(),
        labels=LABELS,
    )
    rt_metrics = compute_routing_metrics(eval_df)

    # Display headline metrics.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{cls_metrics['accuracy']:.4f}")
    m2.metric("Macro-F1", f"{cls_metrics['macro_f1']:.4f}")
    m3.metric("Weighted-F1", f"{cls_metrics['weighted_f1']:.4f}")
    m4.metric("Review rate", f"{rt_metrics['review_rate'] * 100:.1f}% ({rt_metrics['review_count']}/{rt_metrics['total_rows']})")

    r1, r2, r3, r4 = st.columns(4)
    acc_unflagged = (
        f"{rt_metrics['accuracy_not_flagged']:.4f}"
        if rt_metrics["accuracy_not_flagged"] is not None
        else "N/A"
    )
    acc_flagged = (
        f"{rt_metrics['accuracy_flagged']:.4f}"
        if rt_metrics["accuracy_flagged"] is not None
        else "N/A"
    )
    r1.metric("Accuracy (not flagged)", acc_unflagged)
    r2.metric("Accuracy (flagged)", acc_flagged)
    r3.metric("transformer-inquiry count", rt_metrics["model_counts"].get("transformer-inquiry", 0))
    r4.metric("linear+context count", rt_metrics["model_counts"].get("linear+context", 0))

    # Highlight unflagged errors prominently since they bypass human review.
    unflagged_errs = rt_metrics["unflagged_errors_count"]
    if unflagged_errs > 0:
        st.error(
            f"Unflagged errors (errors NOT caught for review): {unflagged_errs} of {rt_metrics['total_rows']} rows"
        )
    else:
        st.success("Unflagged errors: 0")

    # Per-class metrics table and row-normalized confusion matrix side by side.
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Per-class metrics**")
        per_class_display = cls_metrics["per_class_df"].copy()
        st.dataframe(
            per_class_display.style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right_col:
        cm_fig = render_confusion_matrix(cls_metrics["confusion_matrix_pct"])
        st.pyplot(cm_fig, use_container_width=True)
        plt.close(cm_fig)

    # Side-by-side comparison table.
    st.markdown("**System comparison**")
    comp_df = compute_comparison_table(eval_df, labels=LABELS)
    st.dataframe(
        comp_df.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}, na_rep="N/A"),
        hide_index=True,
        use_container_width=True,
    )

    # Slice analysis for each user-selected categorical column.
    active_slice_cols = [c for c in slice_cols if c in eval_df.columns]
    if active_slice_cols:
        st.markdown("**Slice analysis**")
        st.caption("Groups with fewer than 5 rows are skipped because small-sample metrics are unreliable.")
        for s_col in active_slice_cols:
            slice_df, skipped = compute_slice_analysis(eval_df, s_col, labels=LABELS, min_rows=5)
            st.markdown(f"Slice: `{s_col}`")
            if not slice_df.empty:
                st.dataframe(
                    slice_df.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info(f"No groups in `{s_col}` have at least 5 rows.")
            if skipped:
                skipped_str = ", ".join(f"{item['value']} (n={item['n']})" for item in skipped)
                st.caption(f"Skipped unreliable small groups (<5 rows): {skipped_str}")

    # Misclassified rows table with filter.
    st.markdown("**Misclassified rows**")
    err_filter = st.radio(
        "Error filter",
        options=["All errors", "Unflagged errors only"],
        horizontal=True,
        key="error_table_filter",
    )
    errors_df = eval_df.loc[eval_df["expected"] != eval_df["predicted"]].copy()
    if err_filter == "Unflagged errors only":
        errors_df = errors_df.loc[~errors_df["needs_review"]]

    error_cols = [
        "inquiry",
        "context",
        "expected",
        "predicted",
        "model_used",
        "t_conf",
        "needs_review",
        "review_reason",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
        "prob_mixed",
    ]
    if errors_df.empty:
        st.info("No misclassified rows match the selected filter.")
    else:
        st.dataframe(errors_df[error_cols], use_container_width=True)

    # Export buttons for full results CSV and summary metrics CSV.
    dl_col1, dl_col2 = st.columns(2)
    full_csv = eval_df.to_csv(index=False).encode("utf-8")
    dl_col1.download_button(
        "Download full results CSV",
        data=full_csv,
        file_name="sentiment_batch_results.csv",
        mime="text/csv",
    )

    summary_rows = [
        {"metric": "accuracy", "value": cls_metrics["accuracy"]},
        {"metric": "macro_f1", "value": cls_metrics["macro_f1"]},
        {"metric": "weighted_f1", "value": cls_metrics["weighted_f1"]},
        {"metric": "review_rate", "value": rt_metrics["review_rate"]},
        {"metric": "review_count", "value": rt_metrics["review_count"]},
        {"metric": "accuracy_not_flagged", "value": rt_metrics["accuracy_not_flagged"]},
        {"metric": "accuracy_flagged", "value": rt_metrics["accuracy_flagged"]},
        {"metric": "unflagged_errors_count", "value": rt_metrics["unflagged_errors_count"]},
        {"metric": "total_rows", "value": rt_metrics["total_rows"]},
    ]
    summary_csv = pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8")
    dl_col2.download_button(
        "Download summary metrics CSV",
        data=summary_csv,
        file_name="sentiment_summary_metrics.csv",
        mime="text/csv",
    )


def render_robustness_tab() -> None:
    """Render Tab 3 for context perturbation and disagreement-guard sweep analysis."""
    st.subheader("Context Robustness Test")

    batch_cache = st.session_state.get("batch_cache")
    if batch_cache is None:
        st.info("Run a batch evaluation in Tab 2 first to populate context robustness metrics.")
        return

    rob_cache = batch_cache.get("robustness_cache")
    if rob_cache is None or len(rob_cache["y_true"]) == 0:
        st.info("No rows in the current batch have non-empty conversation context.")
        return

    # Compute robustness across true, neutral, and shuffled contexts.
    rob_df = compute_context_robustness(
        y_true=rob_cache["y_true"],
        preds_true_ctx=rob_cache["linear_true"],
        preds_neutral_ctx=rob_cache["linear_neutral"],
        preds_shuffled_ctx=rob_cache["linear_shuffled"],
        preds_transformer=rob_cache["t_labels"],
        labels=LABELS,
    )

    st.dataframe(
        rob_df.style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}"}),
        hide_index=True,
        use_container_width=True,
    )

    rob_fig = render_robustness_chart(rob_df)
    st.pyplot(rob_fig, use_container_width=False)
    plt.close(rob_fig)

    st.caption(
        "A large fall under shuffled context means the linear model depends on the conversation history being correct."
    )

    # Disagreement guard sweep across [0.0, 0.5, 0.7, 0.9].
    st.markdown("**Disagreement guard sweep**")
    sweep_df = compute_disagreement_guard_sweep(
        y_true=rob_cache["y_true"],
        linear_true_preds=rob_cache["linear_true"],
        linear_shuffled_preds=rob_cache["linear_shuffled"],
        t_labels=rob_cache["t_labels"],
        t_confs=rob_cache["t_confs"],
        thresholds=(0.0, 0.5, 0.7, 0.9),
    )

    sweep_display = sweep_df.copy()
    sweep_display["wrong_caught_fmt"] = sweep_display.apply(
        lambda r: f"{int(r['wrong_caught'])}/{int(r['wrong_total'])} ({r['wrong_caught_share'] * 100:.1f}%)",
        axis=1,
    )
    sweep_display["needless_flags_fmt"] = sweep_display.apply(
        lambda r: f"{int(r['needless_flags'])}/{int(r['correct_total'])} ({r['needless_flags_share'] * 100:.1f}%)",
        axis=1,
    )
    st.dataframe(
        sweep_display[["threshold", "context_condition", "wrong_caught_fmt", "needless_flags_fmt"]].rename(
            columns={
                "threshold": "guard_thr",
                "context_condition": "Context",
                "wrong_caught_fmt": "Wrong linear predictions caught",
                "needless_flags_fmt": "Correct linear predictions flagged needlessly",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


