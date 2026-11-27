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


