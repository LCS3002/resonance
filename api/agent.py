"""Campaign agent: brief → variants → neural scoring → ranked result.

Integrations:
  - Claude (generation)
  - Tavily (brand research)
  - Overmind (tracing + optimization)
  - TribeV2 + Custom (neural scoring)
"""

import asyncio
import logging
import os

from .generation.creative import (
    generate_variants,
    search_brand_context,
    generate_campaign_strategy,
    generate_ad_image,
)
from .scoring.emotion import (
    EMOTIONS,
    predict_emotion,
    emotion_match_score,
    compute_counterfactual_hint,
)
from .scoring import ScorerPipeline

logger = logging.getLogger(__name__)

# ── Overmind tracing ──────────────────────────────────────────────────────────
# Traces every LLM call automatically; feeds Overmind's optimization loop.
# Set OVERMIND_API_KEY in .env to enable. No-op if key is absent.
try:
    import overmind
    if os.getenv("OVERMIND_API_KEY"):
        overmind.init()
        logger.info("Overmind tracing active")
    else:
        logger.info("OVERMIND_API_KEY not set — tracing disabled (set it to enable)")
except ImportError:
    logger.info("overmind package not installed — run: pip install overmind")

# Global scorer instance (loads models once at startup)
_scorer: ScorerPipeline | None = None


def get_scorer() -> ScorerPipeline:
    global _scorer
    if _scorer is None:
        _scorer = ScorerPipeline()
    return _scorer


# ── Human-in-the-loop threshold ───────────────────────────────────────────────
HITL_THRESHOLD = float(os.getenv("HITL_THRESHOLD", "0.45"))


async def run_campaign(
    brief: str,
    brand: str | None = None,
    num_variants: int = 3,
    target_emotion: str | None = None,
) -> dict:
    """Full pipeline: research → generate → score → rank → flag.

    Returns a dict with:
      variants      — ranked list of scored ad variants
      winner        — top-scoring variant
      flagged_count — number of variants below HITL threshold
      brand_context — Tavily research snippet (empty if not found)
      scorer_status — which models are live vs mock
    """

    # Validate target emotion
    if target_emotion and target_emotion not in EMOTIONS:
        target_emotion = None

    # 1. Brand research via Tavily
    context = await search_brand_context(brand or "")

    # 2. Generate variants with Claude (inject target emotion into prompt context)
    emotion_hint = f"Target emotion: {target_emotion}." if target_emotion else ""
    logger.info(f"Generating {num_variants} variants for brief: {brief[:80]}  emotion={target_emotion}")
    variants = await generate_variants(
        brief, context=context, num_variants=num_variants, emotion_hint=emotion_hint
    )

    # 3. Neural scoring — all models in parallel per variant
    scorer = get_scorer()
    scored_variants = []

    for v in variants:
        score_input = f"{v['headline']}. {v['body']}"
        scores = await scorer.score(score_input)

        # Emotion prediction + gap analysis
        predicted_emotion, emotion_confidence = predict_emotion(scores["region_scores"])
        em_score = emotion_match_score(target_emotion or "", scores["region_scores"])
        cf_hint  = compute_counterfactual_hint(target_emotion or "", scores["region_scores"])

        scored_variants.append({
            **v,
            **scores,
            "flagged": scores["combined_score"] < HITL_THRESHOLD,
            "score_label": _score_label(scores["combined_score"]),
            "predicted_emotion":   predicted_emotion,
            "emotion_confidence":  round(emotion_confidence, 3),
            "emotion_match_score": round(em_score, 3) if target_emotion else None,
            "counterfactual_hint": cf_hint,
        })

    # 4. Sort by combined score descending
    scored_variants.sort(key=lambda x: -x["combined_score"])
    winner = scored_variants[0]

    # 5. Campaign strategy + hero image for winner (run in parallel)
    strategy_task = generate_campaign_strategy(
        brief=brief,
        top_variant=winner,
        region_scores=winner["region_scores"],
    )
    image_task = generate_ad_image(winner.get("image_prompt", brief))

    strategy, hero_image_url = await asyncio.gather(strategy_task, image_task)

    return {
        "variants":        scored_variants,
        "winner":          winner,
        "flagged_count":   sum(1 for v in scored_variants if v["flagged"]),
        "brand_context":   context[:200] if context else "",
        "strategy":        strategy,
        "hero_image_url":  hero_image_url,
        "target_emotion":  target_emotion,
        "scorer_status":   scorer.algonauts.is_available() or scorer.tribe.is_available(),
        "hitl_threshold":  HITL_THRESHOLD,
    }


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "High engagement"
    if score >= 0.55:
        return "Moderate engagement"
    if score >= 0.45:
        return "Low engagement"
    return "Below threshold — human review required"
