"""Token-level gradient saliency + Claude-ready feedback hint builder.

Method A from spec: input embedding gradient norm per token.
Requires PyTorch model with gradients (not ONNX).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SKIP_TOKENS = {"<s>", "</s>", "<pad>", "Ġ", ""}


def get_token_saliency(
    text: str,
    target_emotion_idx: int,
    model,
    tokenizer,
) -> list[tuple[str, float]]:
    """Return [(token, saliency_score), ...] sorted by original token order.

    Saliency = L2 norm of gradient w.r.t. input embeddings, backpropped through
    the target emotion's sigmoid output.
    """
    try:
        model.train()  # enable gradient computation

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Detach and re-attach embeddings so we can get gradients w.r.t. them
        embeds = model.roberta.embeddings.word_embeddings(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        logits = torch.sigmoid(outputs.logits)

        # Maximise the target emotion score
        loss = -logits[0, target_emotion_idx]
        loss.backward()

        saliency = embeds.grad.norm(dim=-1).squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        return list(zip(tokens, saliency if isinstance(saliency, list) else [saliency]))
    except Exception as exc:
        logger.warning(f"Saliency computation failed: {exc}")
        return []
    finally:
        model.eval()


def build_feedback_hint(
    text: str,
    target_emotion: str,
    saliency_pairs: list[tuple[str, float]],
    current_scores: dict[str, float],
) -> str:
    """Build a Claude-ready feedback prefix from saliency attribution.

    Identifies the 3 tokens most/least aligned with the target emotion and
    returns a structured instruction block the caller can prepend to the next
    Claude generation prompt.
    """
    if not saliency_pairs or not current_scores:
        return ""

    predicted_emotion = max(current_scores, key=current_scores.__getitem__)

    # Filter special tokens, sort by saliency ascending
    clean = [(t, s) for t, s in saliency_pairs if t.strip("Ġ") not in _SKIP_TOKENS]
    if not clean:
        return ""

    sorted_pairs = sorted(clean, key=lambda x: x[1])
    hurting  = [_clean_token(t) for t, _ in sorted_pairs[:3]  if _clean_token(t)][:3]
    helping  = [_clean_token(t) for t, _ in sorted_pairs[-3:] if _clean_token(t)][:3]

    target_score   = current_scores.get(target_emotion, 0)
    predicted_score = current_scores.get(predicted_emotion, 0)

    hint = (
        f"EMOTION EVALUATION FEEDBACK\n"
        f"Current copy scored: {predicted_emotion} (score: {predicted_score:.2f})\n"
        f"Target emotion: {target_emotion} (score: {target_score:.2f})\n\n"
        f"Tokens pulling AWAY from {target_emotion}: {hurting}\n"
        f"→ Soften, replace, or remove these words.\n\n"
        f"Tokens helping {target_emotion}: {helping}\n"
        f"→ Amplify or build more phrasing around these.\n\n"
        f"Rewrite the ad copy to increase [{target_emotion}] signal.\n"
        f"Keep the brand message. Change the framing, not the facts."
    )
    return hint


def _clean_token(token: str) -> str:
    """Strip RoBERTa's Ġ prefix and whitespace."""
    return token.lstrip("Ġ").strip()
