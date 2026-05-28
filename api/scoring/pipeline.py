"""ScorerPipeline: runs TribeV2 + CustomScorer in parallel, combines results."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from .tribe_scorer import TribeScorer
from .custom_scorer import CustomScorer
from .algonauts_scorer import AlgonautsScorer

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class ScorerPipeline:
    def __init__(self):
        self.algonauts = AlgonautsScorer()   # real fMRI-derived scoring
        self.tribe     = TribeScorer()       # Meta TribeV2 (fallback to mock)
        self.custom    = CustomScorer()      # custom BrainDecoder (fallback to mock)
        logger.info(
            f"ScorerPipeline ready — "
            f"Algonauts: {'live' if self.algonauts.is_available() else 'mock'} | "
            f"TribeV2: {'live' if self.tribe.is_available() else 'mock'} | "
            f"Custom: {'live' if self.custom.is_available() else 'mock'}"
        )

    async def score(self, text: str) -> dict:
        """Score ad text with both models in parallel. Returns combined dict."""
        loop = asyncio.get_event_loop()
        algo_r, tribe_r, custom_r = await asyncio.gather(
            loop.run_in_executor(_executor, self.algonauts.score, text),
            loop.run_in_executor(_executor, self.tribe.score,     text),
            loop.run_in_executor(_executor, self.custom.score,    text),
        )

        # Weights: Algonauts (real fMRI) 50%, TribeV2 30%, Custom 20%
        combined = algo_r.score * 0.50 + tribe_r.score * 0.30 + custom_r.score * 0.20
        regions = {
            k: round(
                algo_r.region_scores[k] * 0.50
                + tribe_r.region_scores[k] * 0.30
                + custom_r.region_scores[k] * 0.20,
                4
            )
            for k in ("language", "visual", "prefrontal")
        }

        return {
            "combined_score": round(combined, 4),
            "region_scores": regions,
            "model_scores": {
                "algonauts": round(algo_r.score,   4),
                "tribe":     round(tribe_r.score,  4),
                "custom":    round(custom_r.score, 4),
            },
            "latency_ms": {
                "algonauts": round(algo_r.latency_ms,   1),
                "tribe":     round(tribe_r.latency_ms,  1),
                "custom":    round(custom_r.latency_ms, 1),
            },
            "models_live": {
                "algonauts": self.algonauts.is_available(),
                "tribe":     self.tribe.is_available(),
                "custom":    self.custom.is_available(),
            },
        }

    def score_sync(self, text: str) -> dict:
        """Synchronous wrapper for use in non-async contexts."""
        return asyncio.run(self.score(text))
