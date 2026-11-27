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


