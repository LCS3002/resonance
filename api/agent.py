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
    classify_text_emotion,
)
from .scoring import ScorerPipeline
from .scoring.goemotion_scorer import GoEmotionScorer, TARGET_LABELS, PROFILE_TO_GOEMOTION
from .scoring.saliency import get_token_saliency, build_feedback_hint
from .scoring.moondream import MoondreamVLM

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

# Global model singletons — loaded once at startup
_scorer: ScorerPipeline | None = None
_goemotion: GoEmotionScorer | None = None
_moondream: MoondreamVLM | None = None


def get_scorer() -> ScorerPipeline:
    global _scorer
    if _scorer is None:
        _scorer = ScorerPipeline()
    return _scorer


def get_goemotion() -> GoEmotionScorer:
    global _goemotion
    if _goemotion is None:
        _goemotion = GoEmotionScorer()
    return _goemotion


def get_moondream() -> MoondreamVLM:
    global _moondream
    if _moondream is None:
        _moondream = MoondreamVLM()
    return _moondream


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
    scorer    = get_scorer()
    goemotion = get_goemotion()
    moondream = get_moondream()
    scored_variants = []

    for v in variants:
        score_input = f"{v['headline']}. {v['body']}"
        scores = await scorer.score(score_input)

        # fMRI region-based emotion prediction + gap analysis
        predicted_emotion, emotion_confidence = predict_emotion(scores["region_scores"])
        em_score = emotion_match_score(target_emotion or "", scores["region_scores"])
        cf_hint  = compute_counterfactual_hint(target_emotion or "", scores["region_scores"])

        # GoEmotions text classification (independent NLP signal)
        goemotion_scores   = goemotion.classify(score_input)
        roberta_emotion, roberta_conf = classify_text_emotion(score_input, goemotion)

        scored_variants.append({
            **v,
            **scores,
            "flagged":             scores["combined_score"] < HITL_THRESHOLD,
            "score_label":         _score_label(scores["combined_score"]),
            "predicted_emotion":   predicted_emotion,
            "emotion_confidence":  round(emotion_confidence, 3),
            "emotion_match_score": round(em_score, 3) if target_emotion else None,
            "counterfactual_hint": cf_hint,
            "roberta_emotion":     roberta_emotion,
            "roberta_confidence":  roberta_conf,
            "goemotion_scores":    goemotion_scores,
        })

    # 4. Sort by combined score descending
    scored_variants.sort(key=lambda x: -x["combined_score"])
    winner = scored_variants[0]

    # Gradient saliency on winner — token-level attribution for target emotion
    if goemotion.is_available() and target_emotion:
        goemotion_target = _profile_to_goemotion_label(target_emotion)
        target_idx = TARGET_LABELS.get(goemotion_target)
        if target_idx is not None:
            model, tok = goemotion.get_model_and_tokenizer()
            winner_text = f"{winner['headline']}. {winner['body']}"
            saliency = get_token_saliency(winner_text, target_idx, model, tok)
            saliency_hint = build_feedback_hint(
                winner_text, goemotion_target, saliency, winner["goemotion_scores"]
            )
            winner["saliency_hint"] = saliency_hint
            # Enrich the counterfactual hint with gradient-level signal
            winner["counterfactual_hint"] = compute_counterfactual_hint(
                target_emotion, winner["region_scores"], saliency_hint=saliency_hint
            )

    # 5. Campaign strategy + hero image for winner (run in parallel)
    strategy_task = generate_campaign_strategy(
        brief=brief,
        top_variant=winner,
        region_scores=winner["region_scores"],
    )
    image_task = generate_ad_image(winner.get("image_prompt", brief))

    strategy, hero_image_url = await asyncio.gather(strategy_task, image_task)

    # Optional: Moondream image emotion analysis on hero image
    if hero_image_url and moondream.is_available():
        caption = moondream.analyze_image_emotion(hero_image_url)
        if caption:
            winner["image_emotion_caption"] = caption
            winner["image_emotion_scores"]  = goemotion.classify(caption)

    return {
        "variants":            scored_variants,
        "winner":              winner,
        "flagged_count":       sum(1 for v in scored_variants if v["flagged"]),
        "brand_context":       context[:200] if context else "",
        "strategy":            strategy,
        "hero_image_url":      hero_image_url,
        "target_emotion":      target_emotion,
        "scorer_status":       scorer.algonauts.is_available() or scorer.tribe.is_available(),
        "goemotion_available": goemotion.is_available(),
        "moondream_available": moondream.is_available(),
        "hitl_threshold":      HITL_THRESHOLD,
    }


def _profile_to_goemotion_label(profile: str) -> str:
    """Return the primary GoEmotions label for a given emotion profile."""
    mapping = PROFILE_TO_GOEMOTION.get(profile, [])
    return mapping[0] if mapping else profile


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "High engagement"
    if score >= 0.55:
        return "Moderate engagement"
    if score >= 0.45:
        return "Low engagement"
    return "Below threshold — human review required"
