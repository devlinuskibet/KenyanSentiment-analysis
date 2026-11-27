"""Validation tests for sample_tests.csv and mocked Streamlit AppTest UI checks."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from predictor import LABELS, SentimentModels


ROOT = Path(__file__).resolve().parent.parent


def _make_fake_models() -> SentimentModels:
    """Create lightweight mock SentimentModels so AppTest runs without model weights."""
    return SentimentModels(
        tokenizer=object(),
        transformer_model=object(),
        linear_model=object(),
        device=torch.device("cpu"),
        versions_info={
            "expected": {
                "scikit-learn": "1.6.1",
                "transformers": "5.20.2",
                "torch": "2.11.0+cu130",
            },
            "installed": {
                "scikit-learn": "1.6.1",
                "transformers": "5.20.2",
                "torch": "2.11.0+cpu",
            },
            "sklearn_mismatch": False,
            "transformers_mismatch": False,
            "torch_mismatch": True,
        },
    )


def _fake_transformer_batch(inquiries: list[str], models: SentimentModels, progress_callback=None):
    """Deterministic stub returning positive for Swahili thanks and high/low confidences."""
    probs = []
    t_labels = []
    t_confs = []
    for text in inquiries:
        if "Asante" in text:
            row = [0.01, 0.01, 0.97, 0.01]
            idx = 2
        else:
            row = [0.15, 0.15, 0.58, 0.12]
            idx = 2
        probs.append(row)
        t_labels.append(LABELS[idx])
        t_confs.append(row[idx])
    if progress_callback is not None:
        progress_callback(1.0)
    return np.array(probs, dtype=np.float32), t_labels, t_confs


def _fake_linear_batch(inquiries: list[str], contexts: list[str], models: SentimentModels) -> list[str]:
    """Deterministic stub returning negative when delayed-refund context is present."""
    preds = []
    for ctx in contexts:
        preds.append("negative" if "delayed refund" in ctx else "positive")
    return preds


def test_sample_tests_csv_schema_and_diversity() -> None:
    """Verify sample_tests.csv has 12 valid rows covering all labels and duplicate-inquiry pair."""
    df = pd.read_csv(ROOT / "sample_tests.csv")
    assert len(df) == 12
    for col in ("customer_inquiry", "conversation_context", "sentiment"):
        assert col in df.columns
    assert set(df["sentiment"].str.strip().str.lower()) == set(LABELS)
    # Same inquiry appears twice with different contexts and different expected labels.
    dup = df[df["customer_inquiry"] == "Sawa, nimeelewa vizuri sana."]
    assert len(dup) == 2
    assert dup["sentiment"].nunique() == 2


def test_app_ui_with_mocked_models() -> None:
    """Run Streamlit AppTest with mocked models to verify routing, errors, and slider updates."""
    fake_models = _make_fake_models()

    with (
        patch("predictor.load_models", return_value=fake_models),
        patch("predictor.predict_transformer_batch", side_effect=_fake_transformer_batch),
        patch("predictor.predict_linear_batch", side_effect=_fake_linear_batch),
    ):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception
        assert any("Loaded" in s.value for s in at.sidebar.success)

        # Single message without context -> transformer-inquiry positive
        at.text_area[0].set_value("Asante sana, nimepata pesa yangu baada ya siku tatu!").run()
        at.text_area[1].set_value("").run()
        at.button[0].click().run()
        assert not at.exception
        m1 = {m.label: m.value for m in at.metric}
        assert m1["Routed label"] == "positive"
        assert m1["Model used"] == "transformer-inquiry"

        # Single message with verified context -> linear+context negative
        at.text_area[0].set_value("Sawa, nimeelewa vizuri sana.").run()
        at.text_area[1].set_value("Customer had one earlier chat about a delayed refund.").run()
        at.button[0].click().run()
        assert not at.exception
        m2 = {m.label: m.value for m in at.metric}
        assert m2["Routed label"] == "negative"
        assert m2["Model used"] == "linear+context"

        # Empty input shows st.error without crashing
        at.text_area[0].set_value("   ").run()
        at.button[0].click().run()
        assert not at.exception
        assert any("Empty input" in e.value for e in at.error)

        # Batch manual table + guard_thr slider update
        at.radio[0].set_value("Manual table (data_editor)").run()
        at.button[1].click().run()
        assert not at.exception
        before = {m.label: m.value for m in at.metric}

        at.sidebar.slider[1].set_value(0.50).run()
        assert not at.exception
        after = {m.label: m.value for m in at.metric}
        assert before["Review rate"] != after["Review rate"]
