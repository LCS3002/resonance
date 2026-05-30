"""Pre-download both ML models into MODEL_DIR so they're available offline.

Run once at container build time or before first deploy:
    python -m api.scripts.download_models

Set MODEL_DIR env var to control where models land (default: ./models).
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")


def download_goemotion() -> None:
    logger.info("Downloading GoEmotions (RoBERTa) → %s", MODEL_DIR)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", cache_dir=MODEL_DIR)
    AutoModelForSequenceClassification.from_pretrained(
        "SamLowe/roberta-base-go_emotions", cache_dir=MODEL_DIR
    )
    logger.info("GoEmotions downloaded.")


def download_clip() -> None:
    logger.info("Downloading CLIP (vit-base-patch32) → %s", MODEL_DIR)
    from transformers import CLIPModel, CLIPProcessor
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_DIR)
    logger.info("CLIP downloaded.")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    errors = []
    for fn in (download_goemotion, download_clip):
        try:
            fn()
        except Exception as e:
            logger.error("Failed: %s", e)
            errors.append(e)
    if errors:
        sys.exit(1)
    logger.info("All models ready in %s", MODEL_DIR)
