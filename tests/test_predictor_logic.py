"""Unit tests for pure inference formatting, validation, and routing logic in predictor.py."""

from pathlib import Path
import math
import pytest

from predictor import (
    LABELS,
    apply_routing_batch,
    apply_routing_row,
    clean_text,
    format_linear_input,
    has_non_empty_context,
    load_models,
)


def test_clean_text_handles_none_nan_and_scalars() -> None:
    """clean_text maps None and NaN to empty string and converts scalars to str."""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(math.nan) == ""
    assert clean_text("hello") == "hello"
    assert clean_text(123) == "123"


def test_has_non_empty_context() -> None:
    """has_non_empty_context ignores whitespace-only, None, and NaN values."""
    assert has_non_empty_context(None) is False
    assert has_non_empty_context(float("nan")) is False
    assert has_non_empty_context("") is False
    assert has_non_empty_context("   \t\n ") is False
    assert has_non_empty_context("Prior refund chat.") is True


def test_format_linear_input_uses_exact_sep() -> None:
    """format_linear_input joins inquiry and context with literal ' [SEP] '."""
    assert format_linear_input("Sawa", "Delayed refund") == "Sawa [SEP] Delayed refund"
    assert format_linear_input("Sawa", None) == "Sawa [SEP] "


def test_apply_routing_row_verified_context_path() -> None:
    """Verified non-empty context routes to linear+context and checks disagreement guard."""
    # Disagreement at or above guard_thr flags models disagree.
    flagged = apply_routing_row(
        t_label="positive",
        t_conf=0.92,
        linear_label="negative",
        context="Earlier unresolved chat.",
        context_verified=True,
        low_conf=0.90,
        guard_thr=0.90,
    )
    assert flagged == {
        "label": "negative",
        "model_used": "linear+context",
        "needs_review": True,
        "reason": "models disagree",
    }

    # Disagreement below guard_thr is not flagged.
    unflagged = apply_routing_row(
        t_label="positive",
        t_conf=0.58,
        linear_label="negative",
        context="Earlier unresolved chat.",
        context_verified=True,
        low_conf=0.90,
        guard_thr=0.90,
    )
    assert unflagged["needs_review"] is False
    assert unflagged["reason"] == ""


def test_apply_routing_row_transformer_fallback_path() -> None:
    """Missing or unverified context routes to transformer-inquiry and checks low_conf."""
    # Unverified context falls back to transformer-inquiry even when context text exists.
    low_conf_row = apply_routing_row(
        t_label="neutral",
        t_conf=0.75,
        linear_label="negative",
        context="Earlier unresolved chat.",
        context_verified=False,
        low_conf=0.90,
        guard_thr=0.90,
    )
    assert low_conf_row == {
        "label": "neutral",
        "model_used": "transformer-inquiry",
        "needs_review": True,
        "reason": "low confidence",
    }

    # High-confidence transformer prediction with empty context is not flagged.
    ok_row = apply_routing_row(
        t_label="positive",
        t_conf=0.95,
        linear_label=None,
        context="",
        context_verified=True,
        low_conf=0.90,
        guard_thr=0.90,
    )
    assert ok_row == {
        "label": "positive",
        "model_used": "transformer-inquiry",
        "needs_review": False,
        "reason": "",
    }


def test_apply_routing_row_requires_linear_label_when_active() -> None:
    """Verified non-empty context without linear_label raises ValueError."""
    with pytest.raises(ValueError):
        apply_routing_row(
            t_label="positive",
            t_conf=0.95,
            linear_label=None,
            context="Some context",
            context_verified=True,
        )


def test_apply_routing_batch_handles_empty_and_mixed_rows() -> None:
    """apply_routing_batch returns expected columns for both empty and populated inputs."""
    empty_df = apply_routing_batch([], [], [], [], context_verified=True)
    assert list(empty_df.columns) == ["predicted", "model_used", "needs_review", "review_reason"]
    assert empty_df.empty

    routed_df = apply_routing_batch(
        t_labels=["positive", "neutral"],
        t_confs=[0.95, 0.70],
        linear_labels=["negative", "neutral"],
        contexts=["Prior chat", ""],
        context_verified=True,
        low_conf=0.90,
        guard_thr=0.90,
    )
    assert routed_df["predicted"].tolist() == ["negative", "neutral"]
    assert routed_df["model_used"].tolist() == ["linear+context", "transformer-inquiry"]
    assert routed_df["needs_review"].tolist() == [True, True]
    assert routed_df["review_reason"].tolist() == ["models disagree", "low confidence"]


def test_load_models_raises_on_missing_directory(tmp_path: Path) -> None:
    """load_models raises FileNotFoundError when model artifacts are absent."""
    with pytest.raises(FileNotFoundError):
        load_models(
            transformer_dir=tmp_path / "missing_transformer",
            linear_model_path=tmp_path / "missing.joblib",
            versions_path=tmp_path / "versions.json",
        )
    assert len(LABELS) == 4
