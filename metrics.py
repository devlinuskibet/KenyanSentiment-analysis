"""
Metric computation, label validation, slice analysis, and robustness evaluation.

All metric functions run a self-check on predictions and labels before computing
aggregate statistics so mismatched lengths or unknown labels are caught early.
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from predictor import LABELS


def validate_predictions_and_labels(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    valid_labels: Sequence[str] = LABELS,
) -> None:
    """
    Unit-test-style self-check verifying label and prediction alignment and values.

    Args:
        y_true: Ground-truth label sequence.
        y_pred: Predicted label sequence.
        valid_labels: Allowed label set.

    Raises:
        ValueError: If lengths differ, sequences are empty, or unknown labels appear.
    """
    # Verify equal lengths before computing any element-wise metric.
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)} items."
        )

    # Reject empty sequences so division-by-zero or undefined metrics never slip through.
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty sequence.")

    # Check that every label and prediction belongs to the fixed label vocabulary.
    allowed = set(valid_labels)
    invalid_true = sorted({str(val) for val in y_true if val not in allowed})
    if invalid_true:
        raise ValueError(f"Unknown ground-truth labels found: {invalid_true}")

    invalid_pred = sorted({str(val) for val in y_pred if val not in allowed})
    if invalid_pred:
        raise ValueError(f"Unknown predicted labels found: {invalid_pred}")


def normalize_and_filter_labels(
    df: pd.DataFrame,
    label_col: str,
    valid_labels: Sequence[str] = LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize expected labels with strip+lower and separate invalid-label rows.

    Args:
        df: Input dataset DataFrame.
        label_col: Name of the expected-label column.
        valid_labels: Allowed label names.

    Returns:
        Tuple of (valid_df, invalid_df), each preserving original row indices.
    """
    # Work on a copy to avoid mutating the caller's DataFrame unexpectedly.
    working = df.copy()
    allowed = set(valid_labels)

    # Convert NaN/None to "" before stripping and lowercasing.
    normalized = (
        working[label_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    working["expected_normalized"] = normalized

    # Partition rows into valid and invalid label subsets so invalid rows can be reported.
    is_valid = normalized.isin(allowed)
    valid_df = working.loc[is_valid].copy()
    invalid_df = working.loc[~is_valid].copy()
    return valid_df, invalid_df


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """
    Compute accuracy, macro-F1, weighted-F1, per-class metrics, and row-normalized CM.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Dictionary containing scalar metrics, per_class_df, and confusion_matrix_pct.
    """
    # Run self-check before computing any scikit-learn metrics.
    validate_predictions_and_labels(y_true, y_pred, labels)

    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    label_list = list(labels)

    # Compute overall accuracy and F1 aggregates.
    acc = float(accuracy_score(y_true_list, y_pred_list))
    macro_f1 = float(f1_score(y_true_list, y_pred_list, labels=label_list, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_list, y_pred_list, labels=label_list, average="weighted", zero_division=0))

    # Compute per-class precision, recall, F1, and support in fixed label order.
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_list,
        y_pred_list,
        labels=label_list,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(
        {
            "label": label_list,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp.astype(int),
        }
    )

    # Compute raw counts and row-normalize to percentages for the heatmap.
    cm_counts = confusion_matrix(y_true_list, y_pred_list, labels=label_list)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm_counts.astype(float) * 100.0,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums != 0,
    )
    cm_pct_df = pd.DataFrame(cm_pct, index=label_list, columns=label_list)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_df": per_class_df,
        "confusion_matrix_pct": cm_pct_df,
        "confusion_matrix_counts": pd.DataFrame(cm_counts, index=label_list, columns=label_list),
    }


def compute_routing_metrics(df_eval: pd.DataFrame) -> dict[str, Any]:
    """
    Compute routing path counts, review rate, and flagged vs. unflagged accuracy.

    Args:
        df_eval: DataFrame containing expected, predicted, model_used, and needs_review.

    Returns:
        Dictionary of routing and review statistics.
    """
    # Validate alignment of expected and predicted columns before summarizing.
    validate_predictions_and_labels(
        df_eval["expected"].tolist(),
        df_eval["predicted"].tolist(),
        LABELS,
    )

    total_rows = len(df_eval)
    is_correct = df_eval["expected"] == df_eval["predicted"]
    is_flagged = df_eval["needs_review"].astype(bool)

    # Count rows routed through each model path.
    model_counts = df_eval["model_used"].value_counts().to_dict()

    # Compute review counts and rates.
    review_count = int(is_flagged.sum())
    review_rate = float(review_count / total_rows) if total_rows > 0 else 0.0

    # Compute accuracy separately on unflagged and flagged subsets.
    unflagged_mask = ~is_flagged
    accuracy_not_flagged = (
        float(is_correct.loc[unflagged_mask].mean()) if unflagged_mask.any() else None
    )
    accuracy_flagged = (
        float(is_correct.loc[is_flagged].mean()) if is_flagged.any() else None
    )

    # Unflagged errors are silent failures that bypass human review.
    unflagged_errors_count = int((~is_correct & unflagged_mask).sum())

    return {
        "total_rows": total_rows,
        "model_counts": model_counts,
        "review_count": review_count,
        "review_rate": review_rate,
        "accuracy_not_flagged": accuracy_not_flagged,
        "accuracy_flagged": accuracy_flagged,
        "unflagged_errors_count": unflagged_errors_count,
    }


def compute_comparison_table(
    df_eval: pd.DataFrame,
    labels: Sequence[str] = LABELS,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison of transformer alone, linear alone, and routed system.

    Args:
        df_eval: DataFrame with expected, t_label, linear_label, predicted, and has_context.
        labels: Ordered label list.

    Returns:
        DataFrame with columns: system, accuracy, macro_f1, n.
    """
    rows: list[dict[str, Any]] = []

    # 1. Transformer alone across all valid rows.
    t_metrics = compute_classification_metrics(
        df_eval["expected"].tolist(),
        df_eval["t_label"].tolist(),
        labels=labels,
    )
    rows.append(
        {
            "system": "Transformer alone (all rows)",
            "accuracy": t_metrics["accuracy"],
            "macro_f1": t_metrics["macro_f1"],
            "n": len(df_eval),
        }
    )

    # 2. Linear alone evaluated only on rows that supplied non-empty context.
    ctx_df = df_eval.loc[df_eval["has_context"].astype(bool)]
    if len(ctx_df) > 0:
        lin_metrics = compute_classification_metrics(
            ctx_df["expected"].tolist(),
            ctx_df["linear_label"].tolist(),
            labels=labels,
        )
        rows.append(
            {
                "system": "Linear alone (rows with context)",
                "accuracy": lin_metrics["accuracy"],
                "macro_f1": lin_metrics["macro_f1"],
                "n": len(ctx_df),
            }
        )
    else:
        rows.append(
            {
                "system": "Linear alone (rows with context)",
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "n": 0,
            }
        )

    # 3. Routed system across all valid rows.
    r_metrics = compute_classification_metrics(
        df_eval["expected"].tolist(),
        df_eval["predicted"].tolist(),
        labels=labels,
    )
    rows.append(
        {
            "system": "Routed system (all rows)",
            "accuracy": r_metrics["accuracy"],
            "macro_f1": r_metrics["macro_f1"],
            "n": len(df_eval),
        }
    )

    return pd.DataFrame(rows)


def compute_slice_analysis(
    df_eval: pd.DataFrame,
    slice_col: str,
    labels: Sequence[str] = LABELS,
    min_rows: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Compute accuracy, macro-F1, and count per category of a slice column.

    Groups with fewer than min_rows rows are skipped and returned separately
    because small-sample metrics are statistically unreliable.

    Args:
        df_eval: Evaluated DataFrame containing expected and predicted columns.
        slice_col: Categorical column name to slice by.
        labels: Ordered label list.
        min_rows: Minimum group size required to report metrics.

    Returns:
        Tuple of (slice_df, skipped_groups).
    """
    # Normalize missing slice values to a readable label.
    series = df_eval[slice_col].fillna("(missing)").astype(str)
    slice_rows: list[dict[str, Any]] = []
    skipped_groups: list[dict[str, Any]] = []

    # Iterate deterministically over sorted unique slice values.
    for value in sorted(series.unique()):
        subset = df_eval.loc[series == value]
        count = len(subset)
        if count < min_rows:
            skipped_groups.append({"value": value, "n": count})
            continue

        metrics = compute_classification_metrics(
            subset["expected"].tolist(),
            subset["predicted"].tolist(),
            labels=labels,
        )
        slice_rows.append(
            {
                slice_col: value,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "n": count,
            }
        )

    slice_df = (
        pd.DataFrame(slice_rows)
        if slice_rows
        else pd.DataFrame(columns=[slice_col, "accuracy", "macro_f1", "n"])
    )
    return slice_df, skipped_groups


