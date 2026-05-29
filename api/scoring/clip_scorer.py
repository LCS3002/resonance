"""CLIP-based image emotion scorer.

Uses openai/clip-vit-base-patch32 from HuggingFace.
No API key required. Runs on CPU (~600 MB). Fully differentiable.

Emotion scoring: cosine similarity between image embedding and
emotion-primed text embeddings ("a photo that evokes {emotion}").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Emotion prompts — phrased to activate CLIP's image-text alignment
_EMOTION_PROMPTS: dict[str, str] = {
    "aspirational": "a photo that evokes ambition, hope, and reaching for something greater",
    "trustworthy":  "a photo that evokes calm, reliability, and quiet confidence",
    "urgent":       "a photo that evokes immediate action, tension, and urgency",
    "playful":      "a photo that evokes fun, energy, and joyful excitement",
    "premium":      "a photo that evokes luxury, exclusivity, and refined elegance",
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

            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()

            # Pre-compute text features for all emotion prompts (done once)
            import torch
            with torch.no_grad():
                for emotion, prompt in _EMOTION_PROMPTS.items():
                    inputs = self._processor(text=[prompt], return_tensors="pt", padding=True)
                    feat = self._model.get_text_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    self._text_features[emotion] = feat

            self._available = True
            logger.info("CLIPScorer loaded — clip-vit-base-patch32")
        except Exception as e:
            logger.warning(f"CLIPScorer unavailable: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def score_image(self, image: "PILImage.Image") -> dict[str, float]:
        """Return cosine similarity scores for each emotion given a PIL Image.

        Returns dict[emotion_name → float 0-1].
        Falls back to equal weights if model unavailable.
        """
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
            # CLIP cosine sims sit in ~[-0.3, 0.4]; normalise to [0, 1]
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
                # data URI — strip header and decode base64
                _, b64 = image_url.split(",", 1)
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            else:
                with urllib.request.urlopen(image_url, timeout=10) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")

            return self.score_image(img)
        except Exception as e:
            logger.warning(f"CLIPScorer.score_image_url failed: {e}")
            return {e: 0.2 for e in EMOTIONS}

    def top_emotion(self, scores: dict[str, float]) -> tuple[str, float]:
        """Return (emotion_name, score) for the highest scoring emotion."""
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], best[1]
