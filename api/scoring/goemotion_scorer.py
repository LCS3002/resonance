"""RoBERTa GoEmotions classifier for ad copy emotion scoring.

Model: SamLowe/roberta-base-go_emotions
Output: 28-class sigmoid vector. We expose only the 6 target labels defined in the spec.
Uses PyTorch (not ONNX) so gradients are available for saliency.
"""

import logging

import torch

logger = logging.getLogger(__name__)

TARGET_LABELS: dict[str, int] = {
    "curiosity":   7,
    "desire":      8,   # FOMO proxy
    "excitement":  13,  # arousal proxy
    "nervousness": 19,  # concern proxy
    "surprise":    26,
    "realization": 22,  # pattern-interrupt
}

# Maps our 5 EMOTION_PROFILES to the two GoEmotions labels that best proxy them.
PROFILE_TO_GOEMOTION: dict[str, list[str]] = {
    "aspirational": ["excitement", "desire"],
    "trustworthy":  ["realization", "curiosity"],
    "urgent":       ["nervousness", "surprise"],
    "playful":      ["curiosity", "excitement"],
    "premium":      ["desire", "realization"],
}


class GoEmotionScorer:
    MODEL_ID = "SamLowe/roberta-base-go_emotions"

    def __init__(self) -> None:
        self._available = False
        self._model = None
        self._tokenizer = None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
            self._model.eval()
            self._available = True
            logger.info("GoEmotionScorer loaded — RoBERTa GoEmotions ready")
        except Exception as exc:
            logger.warning(f"GoEmotionScorer unavailable: {exc}")

    def is_available(self) -> bool:
        return self._available

    def classify(self, text: str) -> dict[str, float]:
        """Return sigmoid scores for the 6 target emotion labels.

        Falls back to 0.5 placeholders when model is unavailable.
        """
        if not self._available:
            return {k: 0.5 for k in TARGET_LABELS}
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze()
        return {label: float(probs[idx]) for label, idx in TARGET_LABELS.items()}

    def get_model_and_tokenizer(self):
        """Expose raw model + tokenizer for gradient saliency computation."""
        return self._model, self._tokenizer

    def profile_score(self, goemotion_scores: dict[str, float]) -> dict[str, float]:
        """Aggregate 6 GoEmotions scores into 5 emotion-profile scores (mean of two proxies)."""
        result: dict[str, float] = {}
        for profile, labels in PROFILE_TO_GOEMOTION.items():
            result[profile] = sum(goemotion_scores.get(l, 0.5) for l in labels) / len(labels)
        return result

    def predict(self, text: str) -> tuple[str, float]:
        """Return (best_profile, confidence) based on GoEmotions classification."""
        scores = self.classify(text)
        profile_scores = self.profile_score(scores)
        best = max(profile_scores, key=profile_scores.__getitem__)
        return best, round(profile_scores[best], 3)
