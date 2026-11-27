"""Unit tests for validation, metrics, slice analysis, and robustness helpers in metrics.py."""

import pandas as pd
import pytest

from metrics import (
    compute_classification_metrics,
    compute_comparison_table,
    compute_context_robustness,
    compute_disagreement_guard_sweep,
    compute_routing_metrics,
    compute_slice_analysis,
    normalize_and_filter_labels,
    validate_predictions_and_labels,
)
from predictor import LABELS


def test_validate_predictions_and_labels_rejects_bad_inputs() -> None:
    """validate_predictions_and_labels catches length mismatches, empties, and unknown labels."""
    validate_predictions_and_labels(["positive", "negative"], ["positive", "neutral"])

    with pytest.raises(ValueError, match="Length mismatch"):
        validate_predictions_and_labels(["positive"], ["positive", "negative"])

    with pytest.raises(ValueError, match="empty"):
        validate_predictions_and_labels([], [])

    with pytest.raises(ValueError, match="Unknown ground-truth"):
        validate_predictions_and_labels(["unknown"], ["positive"])

    with pytest.raises(ValueError, match="Unknown predicted"):
        validate_predictions_and_labels(["positive"], ["bad_label"])


def test_normalize_and_filter_labels() -> None:
    """normalize_and_filter_labels strips/lowercases and separates invalid rows."""
    df = pd.DataFrame(
        {
            "customer_inquiry": ["a", "b", "c", "d"],
            "sentiment": [" Positive ", "MIXED", "invalid", None],
        }
    )
    valid_df, invalid_df = normalize_and_filter_labels(df, "sentiment", LABELS)
    assert valid_df["expected_normalized"].tolist() == ["positive", "mixed"]
    assert len(invalid_df) == 2


def test_compute_classification_metrics() -> None:
    """compute_classification_metrics returns exact accuracy and row-normalized confusion matrix."""
    y_true = ["negative", "neutral", "positive", "mixed"]
    y_pred = ["negative", "neutral", "positive", "positive"]
    res = compute_classification_metrics(y_true, y_pred, LABELS)

    assert res["accuracy"] == pytest.approx(0.75)
    assert list(res["per_class_df"]["label"]) == LABELS
    cm_pct = res["confusion_matrix_pct"]
    assert cm_pct.loc["negative", "negative"] == pytest.approx(100.0)
    assert cm_pct.loc["mixed", "positive"] == pytest.approx(100.0)


def test_compute_routing_metrics_and_comparison_table() -> None:
    """Routing metrics and system comparison table compute counts and unflagged errors."""
    df_eval = pd.DataFrame(
        {
            "expected": ["positive", "negative", "neutral", "mixed"],
            "t_label": ["positive", "positive", "neutral", "positive"],
            "linear_label": ["positive", "negative", "neutral", "mixed"],
            "predicted": ["positive", "negative", "neutral", "positive"],
            "model_used": [
                "transformer-inquiry",
                "linear+context",
                "transformer-inquiry",
                "transformer-inquiry",
            ],
            "needs_review": [False, True, False, False],
            "has_context": [False, True, False, False],
        }
    )
    rt = compute_routing_metrics(df_eval)
    assert rt["total_rows"] == 4
    assert rt["review_count"] == 1
    assert rt["review_rate"] == pytest.approx(0.25)
    assert rt["accuracy_flagged"] == pytest.approx(1.0)
    assert rt["accuracy_not_flagged"] == pytest.approx(2.0 / 3.0)
    assert rt["unflagged_errors_count"] == 1

    comp = compute_comparison_table(df_eval)
    assert comp["n"].tolist() == [4, 1, 4]


def test_compute_slice_analysis_skips_small_groups() -> None:
    """Groups with fewer than min_rows=5 are skipped and reported separately."""
    df_eval = pd.DataFrame(
        {
            "expected": ["positive"] * 5 + ["negative"] * 2,
            "predicted": ["positive"] * 5 + ["negative"] * 2,
            "language_variant": ["English"] * 5 + ["Sheng"] * 2,
        }
    )
    slice_df, skipped = compute_slice_analysis(df_eval, "language_variant", min_rows=5)
    assert slice_df["language_variant"].tolist() == ["English"]
    assert slice_df["accuracy"].iloc[0] == pytest.approx(1.0)
    assert skipped == [{"value": "Sheng", "n": 2}]


def test_compute_context_robustness_and_guard_sweep() -> None:
    """Context robustness and guard sweep produce expected structure and counts."""
    y_true = ["negative", "positive"]
    rob_df = compute_context_robustness(
        y_true=y_true,
        preds_true_ctx=["negative", "positive"],
        preds_neutral_ctx=["neutral", "positive"],
        preds_shuffled_ctx=["positive", "negative"],
        preds_transformer=["positive", "positive"],
    )
    assert len(rob_df) == 4
    assert rob_df.loc[rob_df["condition"] == "Linear: true context", "accuracy"].iloc[0] == pytest.approx(1.0)

    sweep_df = compute_disagreement_guard_sweep(
        y_true=y_true,
        linear_true_preds=["negative", "positive"],
        linear_shuffled_preds=["positive", "negative"],
        t_labels=["positive", "positive"],
        t_confs=[0.95, 0.80],
        thresholds=(0.0, 0.9),
    )
    assert len(sweep_df) == 4
    shuf_09 = sweep_df[
        (sweep_df["threshold"] == 0.9) & (sweep_df["context_condition"] == "Shuffled context")
    ].iloc[0]
    assert shuf_09["wrong_caught"] == 0
    assert shuf_09["wrong_total"] == 2
