# Sentiment Model Test Environment

Interactive Streamlit test bench for evaluating two trained customer-support sentiment models:

1. **Transformer (inquiry-only)**: local XLM-RoBERTa sequence classifier (`sentiment_models/transformer_inquiry_only`).
2. **LinearSVC + context**: local TF-IDF (word + character n-grams) + LinearSVC pipeline (`sentiment_models/linearsvc_context_dropout.joblib`) using `"{inquiry} [SEP] {context}"`.

All model files are loaded strictly offline (`local_files_only=True`) using `pathlib` relative to `app.py`.

## Setup and Run (Windows PowerShell)

Run these commands from the project root (`C:\Users\linzs\Projects\Sentiment analysis`):

```powershell
# 1. Create the virtual environment (Python 3.11 recommended for scikit-learn 1.6.1 wheels)
py -3.11 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

`scikit-learn` is pinned to `1.6.1` in `requirements.txt` to match `sentiment_models/versions.json`. If the runtime version differs, the sidebar displays a warning because `joblib` pipelines can break across `scikit-learn` versions.

## Input Format

CSV files (such as `sample_tests.csv`) and manual table entries use:

- `customer_inquiry` (required): customer message text.
- `conversation_context` (optional): prior conversation summary; may be blank.
- `sentiment` (required for batch metrics): expected label, one of `negative`, `neutral`, `positive`, `mixed` (normalized with strip + lowercase).
- Optional categorical columns (e.g., `language_variant`, `contains_sarcasm`, `contains_context_dependency`) for slice analysis.

## How to Read the Results

- **Routing**:
  - When **Treat supplied context as verified** is enabled and `conversation_context` is non-empty, the system routes to `linear+context` and flags `models disagree` if the transformer confidence is at or above `guard_thr` and disagrees with the linear prediction.
  - Otherwise, the system routes to `transformer-inquiry` and flags `low confidence` if the transformer confidence is below `low_conf`.
- **Batch Evaluation**:
  - Changing `low_conf`, `guard_thr`, or the context-verified checkbox re-applies routing immediately using cached model outputs without re-running inference.
  - **Unflagged errors** are predictions that were wrong and not flagged for review.
- **Context Robustness Test**:
  - Evaluates the linear model on rows with context under true context, neutral context (`"No prior interaction."`), and shuffled context (seed `42`), alongside a disagreement-guard threshold sweep.
