"""Moondream2 VLM — image → emotion-primed caption.

Load priority:
  1. Local HuggingFace weights (vikhyatk/moondream2, requires ~4 GB VRAM)
  2. Ollama REST endpoint (ollama run moondream — zero-setup, CPU-friendly)
  3. No-op fallback (is_available() returns False)

Caption output is designed to feed directly into GoEmotionScorer.classify().
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_REVISION = "2025-06-21"
_EMOTION_QUERY = (
    "Describe the emotional tone, mood, and psychological effect of this image on a viewer. "
    "What feelings does it evoke? Be specific about tension, curiosity, urgency, or comfort."
)


class MoondreamVLM:
    def __init__(self) -> None:
        self._mode: str | None = None
        self._model = None
        self._tokenizer = None

        if os.getenv("MOONDREAM_SKIP"):
            logger.info("MoondreamVLM disabled via MOONDREAM_SKIP env var")
            return

        # Try local HuggingFace weights first
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision=_REVISION,
                trust_remote_code=True,
                device_map={"": device},
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2", revision=_REVISION
            )
            self._mode = "local"
            logger.info(f"MoondreamVLM loaded locally on {device}")
            return
        except Exception as exc:
            logger.info(f"Moondream local load skipped ({exc}), trying Ollama...")

        # Try Ollama REST endpoint
        try:
            import urllib.request as req
            req.urlopen("http://localhost:11434/api/tags", timeout=2)
            self._mode = "ollama"
            logger.info("MoondreamVLM using Ollama REST (localhost:11434)")
        except Exception:
            logger.info("MoondreamVLM unavailable — no local weights and Ollama not running")

    def is_available(self) -> bool:
        return self._mode is not None

    def analyze_image_emotion(self, image_url: str) -> str:
        """Download image from URL and return emotion-primed caption string."""
        if not self._mode:
            return ""
        try:
            image = _fetch_image(image_url)
            if self._mode == "local":
                return self._query_local(image)
            if self._mode == "ollama":
                return self._query_ollama(image_url)
        except Exception as exc:
            logger.warning(f"Moondream analysis failed: {exc}")
        return ""

    def _query_local(self, image) -> str:
        result = self._model.query(image, _EMOTION_QUERY)
        return result.get("answer", "") if isinstance(result, dict) else str(result)

    def _query_ollama(self, image_url: str) -> str:
        import base64
        import json

        img_bytes = urllib.request.urlopen(image_url, timeout=10).read()
        img_b64 = base64.b64encode(img_bytes).decode()

        payload = json.dumps({
            "model": "moondream",
            "prompt": _EMOTION_QUERY,
            "images": [img_b64],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return body.get("response", "")


def _fetch_image(url: str):
    """Download URL and return a PIL Image."""
    from PIL import Image
    data = urllib.request.urlopen(url, timeout=10).read()
    return Image.open(io.BytesIO(data)).convert("RGB")
