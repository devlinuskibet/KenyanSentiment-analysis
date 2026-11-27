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


def load_versions_info(versions_path: Path = VERSIONS_PATH) -> dict[str, Any]:
    """
    Read training library versions from versions.json and compare against runtime.

    Args:
        versions_path: Path to versions.json.

    Returns:
        Dictionary containing expected versions, installed versions, and mismatch flags.

    Raises:
        FileNotFoundError: If versions.json does not exist.
        ValueError: If versions.json is malformed JSON.
    """
    # Ensure the version metadata file exists before reading.
    if not versions_path.exists():
        raise FileNotFoundError(f"Missing versions file: {versions_path}")

    # Parse expected versions recorded during training.
    with versions_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)

    # Collect runtime library versions.
    installed = {
        "scikit-learn": sklearn.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }

    # Check exact match for scikit-learn (critical for joblib compatibility)
    # and informational matches for transformers and torch.
    expected_sklearn = str(expected.get("scikit-learn", ""))
    expected_transformers = str(expected.get("transformers", ""))
    expected_torch = str(expected.get("torch", ""))

    return {
        "expected": expected,
        "installed": installed,
        "sklearn_mismatch": installed["scikit-learn"] != expected_sklearn,
        "transformers_mismatch": installed["transformers"] != expected_transformers,
        "torch_mismatch": installed["torch"] != expected_torch,
    }


def load_models(
    transformer_dir: Path = TRANSFORMER_DIR,
    linear_model_path: Path = LINEAR_MODEL_PATH,
    versions_path: Path = VERSIONS_PATH,
) -> SentimentModels:
    """
    Load the local transformer, tokenizer, and LinearSVC pipeline without network access.

    Args:
        transformer_dir: Directory containing the saved Hugging Face model and tokenizer.
        linear_model_path: Path to the saved scikit-learn joblib pipeline.
        versions_path: Path to versions.json.

    Returns:
        Populated SentimentModels dataclass.

    Raises:
        FileNotFoundError: If any required model directory or file is missing.
    """
    # Verify all required artifacts are present on disk before attempting to load.
    if not transformer_dir.exists():
        raise FileNotFoundError(f"Missing transformer directory: {transformer_dir}")
    if not linear_model_path.exists():
        raise FileNotFoundError(f"Missing linear model file: {linear_model_path}")

    # Load version comparison metadata first.
    versions_info = load_versions_info(versions_path)

    # Select GPU if CUDA is available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and sequence classification model strictly from local files.
    tokenizer = AutoTokenizer.from_pretrained(
        str(transformer_dir),
        local_files_only=True,
    )
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        str(transformer_dir),
        local_files_only=True,
    )

    # Move transformer to target device and switch to evaluation mode.
    transformer_model.to(device)
    transformer_model.eval()

    # Load the scikit-learn TF-IDF + LinearSVC pipeline, capturing any version warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linear_model = joblib.load(linear_model_path)

    return SentimentModels(
        tokenizer=tokenizer,
        transformer_model=transformer_model,
        linear_model=linear_model,
        device=device,
        versions_info=versions_info,
    )


def predict_transformer_batch(
    inquiries: list[str],
    models: SentimentModels,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Run batched transformer inference on inquiry texts only.

    Args:
        inquiries: List of customer inquiry strings.
        models: Loaded SentimentModels container.
        progress_callback: Optional callback receiving progress fraction in [0.0, 1.0].

    Returns:
        Tuple of:
            - probabilities array of shape (N, 4)
            - predicted label strings of length N
            - confidence scores (max probability) of length N
    """
    # Clean all inquiry strings so NaN/None values never break tokenization.
    cleaned = [clean_text(text) for text in inquiries]
    total = len(cleaned)

    # Handle empty input gracefully.
    if total == 0:
        return np.empty((0, len(LABELS)), dtype=np.float32), [], []

    prob_batches: list[np.ndarray] = []

    # Process inputs in fixed-size batches with no gradient tracking.
    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_texts = cleaned[start:end]

            # Tokenize with fixed truncation length and dynamic batch padding.
            encoded = models.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            # Move input tensors to the same device as the transformer model.
            encoded = {key: tensor.to(models.device) for key, tensor in encoded.items()}

            # Forward pass and softmax normalization over class logits.
            outputs = models.transformer_model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            prob_batches.append(probs)

            # Report batch progress to the UI if a callback was provided.
            if progress_callback is not None:
                progress_callback(end / total)

    # Concatenate all batch probability matrices into shape (N, 4).
    all_probs = np.vstack(prob_batches)
    pred_indices = np.argmax(all_probs, axis=1)
    t_labels = [LABELS[int(idx)] for idx in pred_indices]
    t_confs = [float(all_probs[row_idx, int(col_idx)]) for row_idx, col_idx in enumerate(pred_indices)]

    return all_probs, t_labels, t_confs


def predict_linear_batch(
    inquiries: list[str],
    contexts: list[str],
    models: SentimentModels,
) -> list[str]:
    """
    Run LinearSVC inference on formatted "{inquiry} [SEP] {context}" strings.

    Args:
        inquiries: List of customer inquiry strings.
        contexts: List of conversation context strings.
        models: Loaded SentimentModels container.

    Returns:
        List of predicted label strings mapped via LABELS.

    Raises:
        ValueError: If inquiries and contexts have different lengths.
    """
    # Ensure inquiry and context lists align one-to-one.
    if len(inquiries) != len(contexts):
        raise ValueError("Inquiries and contexts must have the same length.")

    if len(inquiries) == 0:
        return []

    # Format every row with the exact "[SEP]" separator used in training.
    combined_inputs = [
        format_linear_input(inquiry, context)
        for inquiry, context in zip(inquiries, contexts)
    ]

    # Predict integer class indices and map to canonical label strings.
    raw_preds = models.linear_model.predict(combined_inputs)
    return [LABELS[int(pred)] for pred in raw_preds]


