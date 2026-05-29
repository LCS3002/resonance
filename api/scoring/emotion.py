"""Emotion feedback utilities for the GAN-like generation loop.

Text path:  GoEmotionScorer → gradient attribution (saliency.py) → text_hint
Image path: GoEmotionScorer on image_prompt → gradient attribution → image_hint
Both hints are combined here, weighted by how far each modality is from the target.
"""

from __future__ import annotations

from .goemotion_scorer import PROFILE_TO_GOEMOTION

# Valid target emotion names — driven by what GoEmotions can classify
EMOTIONS: list[str] = list(PROFILE_TO_GOEMOTION.keys())


def build_combined_feedback(
    target_emotion: str,
    text_hint: str,
    image_hint: str,
    text_score: float,
    image_score: float,
) -> str:
    """Merge text-copy and image-prompt gradient feedback into one agent-ready block.

    Weights each section by (1 - current_score) so the modality furthest from
    the target emotion gets the most attention in the feedback string.
    """
    text_gap  = max(0.0, 1.0 - text_score)
    image_gap = max(0.0, 1.0 - image_score)
    total = text_gap + image_gap + 1e-8

    parts = [f"TARGET EMOTION: {target_emotion}"]

    if text_hint:
        pct = int(text_gap / total * 100)
        parts.append(f"[COPY — {pct}% priority]\n{text_hint}")

    if image_hint:
        pct = int(image_gap / total * 100)
        parts.append(f"[IMAGE PROMPT — {pct}% priority]\n{image_hint}")

    return "\n\n".join(parts)
