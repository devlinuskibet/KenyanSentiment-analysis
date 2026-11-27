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


