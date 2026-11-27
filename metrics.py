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


