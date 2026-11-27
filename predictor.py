"""
Model loading, inference, routing, and disagreement-guard logic.

All paths are resolved relative to this file's directory using pathlib so
directory names containing spaces work reliably without hard-coded paths.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Optional
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Fixed label order matching the transformer's output indices (0..3)
# and the linear model's integer class predictions.
LABELS: list[str] = ["negative", "neutral", "positive", "mixed"]

# Neutral context string used for single-message comparison and robustness tests.
NEUTRAL_CONTEXT: str = "No prior interaction."

# Named inference and reproducibility constants.
BATCH_SIZE: int = 64
MAX_LENGTH: int = 96
RANDOM_SEED: int = 42

# Resolve all model paths relative to the project root (location of this file).
BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_DIR: Path = BASE_DIR / "sentiment_models"
TRANSFORMER_DIR: Path = MODEL_DIR / "transformer_inquiry_only"
LINEAR_MODEL_PATH: Path = MODEL_DIR / "linearsvc_context_dropout.joblib"
VERSIONS_PATH: Path = MODEL_DIR / "versions.json"


@dataclass
class SentimentModels:
    """Container for loaded sentiment models, tokenizer, device, and version metadata."""

    tokenizer: Any
    transformer_model: Any
    linear_model: Any
    device: torch.device
    versions_info: dict[str, Any]


def clean_text(value: Any) -> str:
    """
    Convert a text cell value to a clean string, mapping None/NaN to empty string.

    Args:
        value: Raw value from UI input, CSV, or DataFrame cell.

    Returns:
        A string representation, or "" if the input is None or NaN.
    """
    # Treat None and pandas/numpy NaN values as empty strings.
    if value is None or pd.isna(value):
        return ""
    # Convert non-string scalars safely to str.
    return str(value)


def has_non_empty_context(context: Any) -> bool:
    """
    Check whether a context value contains non-whitespace characters.

    Args:
        context: Raw context string or NaN/None value.

    Returns:
        True if context is non-empty after stripping whitespace, else False.
    """
    # Normalize first so NaN/None never raises an AttributeError on .strip().
    return bool(clean_text(context).strip())


def format_linear_input(inquiry: Any, context: Any) -> str:
    """
    Format inquiry and context using the exact separator expected by LinearSVC.

    Args:
        inquiry: Customer inquiry text.
        context: Conversation context text.

    Returns:
        Formatted string "{inquiry} [SEP] {context}".
    """
    # LinearSVC was trained with the literal " [SEP] " delimiter.
    return f"{clean_text(inquiry)} [SEP] {clean_text(context)}"


