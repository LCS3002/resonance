"""Custom BrainEncoder neural scorer.

Wraps the BrainEncoder from tribe-v2-custom/src/inference/wrapper.py.
Falls back to MockScorer if no checkpoint is available.
"""

import logging
import os
import sys
import time
from pathlib import Path

from .base import BaseScorer, ScoreResult
from .mock_scorer import MockScorer

logger = logging.getLogger(__name__)

_CUSTOM_REPO = Path(__file__).parents[3] / "Tribe V2 and Neural Predictions" / "tribe-v2-custom"


def _region_from_preds(preds, n_verts: int) -> dict[str, float]:
    """Convert (n_verts, n_trs) prediction to per-region engagement scores."""
    import numpy as np

    mean_pred = preds.mean(axis=1) if preds.ndim == 2 else preds
    n_left = n_verts // 2

    def _norm(arr):
        return float(np.clip((arr.mean() + 1.5) / 3.0, 0, 1))

    lang = _norm(mean_pred[:n_left][int(n_left * 0.25):int(n_left * 0.75)])
    vis  = _norm(mean_pred[int(n_verts * 0.6):])
    pre  = _norm(mean_pred[:int(n_left * 0.2)])

    return {"language": lang, "visual": vis, "prefrontal": pre}


class CustomScorer(BaseScorer):
    """Score ad text using the custom BrainDecoder model."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        config_path: str | None = None,
    ):
        self._encoder = None
        self._available = False
        self._mock = MockScorer(model_name="custom-mock", salt="resonance-custom-v1")

        ckpt = Path(ckpt_path) if ckpt_path else None
        if ckpt is None:
            env_path = os.getenv("CUSTOM_CHECKPOINT")
            if env_path:
                ckpt = Path(env_path)

        if ckpt is None or not ckpt.exists():
            logger.info("No custom model checkpoint found — using mock scorer")
            return

        src_path = str(_CUSTOM_REPO / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            import yaml
            from inference.wrapper import BrainEncoder  # noqa: F401

            cfg_path = Path(config_path) if config_path else _CUSTOM_REPO / "configs" / "default.yaml"
            with open(cfg_path) as f:
                config = yaml.safe_load(f)

            self._encoder = BrainEncoder.from_checkpoint(str(ckpt), config)
            self._available = True
            logger.info(f"Custom BrainEncoder loaded from {ckpt}")
        except Exception as e:
            logger.warning(f"Custom model load failed ({e}) — using mock scorer")

    def is_available(self) -> bool:
        return self._available

    def score(self, text: str) -> ScoreResult:
        if not self._available:
            return self._mock.score(text)

        t0 = time.monotonic()
        try:
            import numpy as np

            # Minimal text → feature extraction using a simple sentence embedding
            # (full pipeline requires LLaMA extractor; use SBERT as lightweight proxy)
            try:
                from sentence_transformers import SentenceTransformer
                _st = getattr(self, "_st_model", None)
                if _st is None:
                    self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                emb = self._st_model.encode([text])[0]
                # Pad/truncate to expected feature shape (L=1, D=384, T=1)
                feat = emb[np.newaxis, :, np.newaxis].astype(np.float32)  # (1, D, 1)
                features = {"text": feat}
            except ImportError:
                # If SBERT not available, return mock
                return self._mock.score(text)

            preds = self._encoder.predict_from_features(features)  # (n_verts, n_trs)
            region_scores = _region_from_preds(preds, preds.shape[0])

            return ScoreResult(
                score=float(np.clip(sum(region_scores.values()) / 3, 0, 1)),
                region_scores=region_scores,
                model="custom",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            logger.error(f"Custom scorer inference failed: {e}")
            return self._mock.score(text)
