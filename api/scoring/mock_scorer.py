"""Deterministic mock scorer — used as fallback when model weights are unavailable."""

import hashlib
import time

import numpy as np

from .base import BaseScorer, ScoreResult

# Rough region-sensitivity keywords — gives the mock coherent structure
_LANGUAGE_KEYS = ["buy", "discover", "experience", "feel", "imagine", "join", "read", "story"]
_VISUAL_KEYS   = ["see", "watch", "look", "bright", "bold", "color", "design", "image", "style"]
_PREFRONTAL_KEYS = ["save", "smart", "plan", "decide", "invest", "choose", "future", "performance"]


def _keyword_boost(text: str, keywords: list[str]) -> float:
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    return min(0.15, hits * 0.04)


class MockScorer(BaseScorer):
    """Hash-based deterministic mock — same text always yields same score."""

    def __init__(self, model_name: str = "mock", salt: str = "resonance-v1"):
        self.model_name = model_name
        self.salt = salt

    def score(self, text: str) -> ScoreResult:
        t0 = time.monotonic()

        h = int(hashlib.sha256((text + self.salt).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(h)

        base = 0.52 + rng.random() * 0.32  # [0.52, 0.84]

        lang = float(np.clip(base + rng.uniform(-0.06, 0.06) + _keyword_boost(text, _LANGUAGE_KEYS), 0, 1))
        vis  = float(np.clip(base + rng.uniform(-0.06, 0.06) + _keyword_boost(text, _VISUAL_KEYS),    0, 1))
        pre  = float(np.clip(base + rng.uniform(-0.06, 0.06) + _keyword_boost(text, _PREFRONTAL_KEYS), 0, 1))

        return ScoreResult(
            score=float(np.clip((lang + vis + pre) / 3, 0, 1)),
            region_scores={"language": lang, "visual": vis, "prefrontal": pre},
            model=self.model_name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
