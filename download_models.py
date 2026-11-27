"""
Download the trained sentiment models from Hugging Face Hub into sentiment_models/.

Run this once after cloning the repository before launching `streamlit run app.py`
so runtime inference remains strictly offline (`local_files_only=True`).
"""

from pathlib import Path
import os

from huggingface_hub import snapshot_download


# Default Hugging Face model repository hosting the trained artifacts.
DEFAULT_REPO_ID: str = "Issme12/kenyan-support-sentiment-models"

# Local target directory relative to the project root.
TARGET_DIR: Path = Path(__file__).resolve().parent / "sentiment_models"


def download_models(
    repo_id: str = DEFAULT_REPO_ID,
    target_dir: Path = TARGET_DIR,
    token: str | None = None,
) -> Path:
    """
    Download model artifacts from Hugging Face Hub into target_dir.

    Args:
        repo_id: Hugging Face Hub repository ID.
        target_dir: Local destination directory for downloaded model files.
        token: Optional Hugging Face access token (needed only for private repos).

    Returns:
        Path to the populated local model directory.
    """
    # Ensure the destination directory exists before downloading.
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download all repository files into sentiment_models/.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        token=token or os.environ.get("HF_TOKEN"),
    )
    return target_dir


if __name__ == "__main__":
    dest = download_models()
    print(f"Downloaded models to: {dest}")
