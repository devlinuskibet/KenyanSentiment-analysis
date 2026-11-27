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


