"""Real-model integration tests (skipped automatically when sentiment_models/ is absent)."""

from pathlib import Path

import pandas as pd
import pytest

from metrics import compute_classification_metrics, normalize_and_filter_labels
from predictor import (
    LABELS,
    TRANSFORMER_DIR,
    apply_routing_batch,
    apply_routing_row,
    load_models,
    predict_linear_batch,
    predict_transformer_batch,
)


# Skip all tests in this module when local model weights are not present.
pytestmark = pytest.mark.skipif(
    not TRANSFORMER_DIR.exists(),
    reason="sentiment_models/ not present locally; run download_models.py first.",
)

ROOT = Path(__file__).resolve().parent.parent


def test_real_models_single_message_routing() -> None:
    """Verify both required single-message acceptance cases on the real models."""
    models = load_models()

    # Case 1: Kiswahili positive with no context -> transformer-inquiry.
    inq_pos = "Asante sana, nimepata pesa yangu baada ya siku tatu!"
    _, tl1, tc1 = predict_transformer_batch([inq_pos], models)
    r1 = apply_routing_row(tl1[0], tc1[0], None, "", context_verified=True)
    assert r1["label"] == "positive"
    assert r1["model_used"] == "transformer-inquiry"

    # Case 2: Context-dependent negative -> linear+context.
    inq_ctx = "Sawa, nimeelewa vizuri sana."
    ctx = "Customer had one earlier chat about a delayed refund with no resolution."
    _, tl2, tc2 = predict_transformer_batch([inq_ctx], models)
    ll2 = predict_linear_batch([inq_ctx], [ctx], models)
    r2 = apply_routing_row(tl2[0], tc2[0], ll2[0], ctx, context_verified=True)
    assert r2["label"] == "negative"
    assert r2["model_used"] == "linear+context"


def test_real_models_batch_sample_tests() -> None:
    """Verify batch inference and metrics on sample_tests.csv with real models."""
    models = load_models()
    df = pd.read_csv(ROOT / "sample_tests.csv")
    valid_df, invalid_df = normalize_and_filter_labels(df, "sentiment", LABELS)
    assert len(valid_df) == 12
    assert invalid_df.empty

    inquiries = valid_df["customer_inquiry"].fillna("").tolist()
    contexts = valid_df["conversation_context"].fillna("").tolist()

    _, t_labels, t_confs = predict_transformer_batch(inquiries, models)
    lin_labels = predict_linear_batch(inquiries, contexts, models)
    routed = apply_routing_batch(t_labels, t_confs, lin_labels, contexts, context_verified=True)

    cls_metrics = compute_classification_metrics(
        valid_df["expected_normalized"].tolist(),
        routed["predicted"].tolist(),
        labels=LABELS,
    )
    assert cls_metrics["accuracy"] >= 0.80
