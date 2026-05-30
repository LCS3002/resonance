"""CLIP-based image emotion scorer.

Uses openai/clip-vit-base-patch32 from HuggingFace.
No API key required. Runs on CPU (~600 MB).

Set MODEL_DIR env var to load from a local folder instead of downloading:
    MODEL_DIR=./models  (default)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# ── Emotion prompts — phrased to activate CLIP's image-text alignment ──────────
_EMOTION_PROMPTS: dict[str, str] = {
    "fomo":       "a photo that evokes scarcity, missing out, and the urgency of a closing window of opportunity",
    "curiosity":  "a photo that evokes intrigue, mystery, and the irresistible pull of wanting to know more",
    "fear":       "a photo that evokes unease, quiet dread, and the risk of something going wrong",
    "excitement": "a photo that evokes energy, anticipation, and the electric thrill of something about to happen",
    "trust":      "a photo that evokes calm, reliability, and the steady confidence of something dependable",
    "pride":      "a photo that evokes ownership, achievement, and the quiet satisfaction of having chosen well",
    "delight":    "a photo that evokes joyful surprise and the warm happiness of an unexpected good moment",
}

EMOTIONS = list(_EMOTION_PROMPTS.keys())


class CLIPScorer:
    """Zero-shot image emotion classifier using CLIP."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._text_features: dict[str, "torch.Tensor"] = {}
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=MODEL_DIR
            )
            self._model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=MODEL_DIR
            )
            self._model.eval()

            # Pre-compute text features for all emotion prompts (done once at load)
            with torch.no_grad():
                for emotion, prompt in _EMOTION_PROMPTS.items():
                    inputs = self._processor(text=[prompt], return_tensors="pt", padding=True)
                    feat = self._model.get_text_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    self._text_features[emotion] = feat

            self._available = True
            logger.info("CLIPScorer loaded from %s", MODEL_DIR)
        except Exception as e:
            logger.warning("CLIPScorer unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def score_image(self, image: "PILImage.Image") -> dict[str, float]:
        """Return cosine similarity scores [0,1] for each emotion given a PIL Image."""
        if not self._available:
            return {e: 0.2 for e in EMOTIONS}

        import torch
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            img_feat = self._model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        scores = {}
        for emotion, text_feat in self._text_features.items():
            sim = float((img_feat * text_feat).sum())
            # CLIP cosine sims sit in ~[-0.3, 0.4] — normalise to [0, 1]
            scores[emotion] = float(max(0.0, min(1.0, (sim + 0.3) / 0.7)))

        return scores

    def score_image_url(self, image_url: str) -> dict[str, float]:
        """Fetch image from URL or data URI and score emotions."""
        if not self._available:
            return {e: 0.2 for e in EMOTIONS}
        try:
            from PIL import Image
            import base64, io, urllib.request

            if image_url.startswith("data:image"):
                _, b64 = image_url.split(",", 1)
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            else:
                with urllib.request.urlopen(image_url, timeout=10) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")

            return self.score_image(img)
        except Exception as e:
            logger.warning("CLIPScorer.score_image_url failed: %s", e)
            return {e: 0.2 for e in EMOTIONS}

    def top_emotion(self, scores: dict[str, float]) -> tuple[str, float]:
        """Return (emotion_name, score) for the highest-scoring emotion."""
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], best[1]
