"""RoBERTa GoEmotions classifier for ad copy emotion scoring.

Model: SamLowe/roberta-base-go_emotions
Output: 28-class sigmoid vector. We expose only the 6 raw labels needed to
        proxy our 7 ad-emotion profiles (fomo, curiosity, fear, excitement,
        trust, pride, delight).

Set MODEL_DIR env var to load from a local folder instead of downloading:
    MODEL_DIR=./models  (default)
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# ── Raw GoEmotions label indices (28-class model) ──────────────────────────────
TARGET_LABELS: dict[str, int] = {
    "curiosity":   7,
    "desire":      8,   # proxy: fomo + pride
    "excitement":  13,  # proxy: excitement + delight
    "nervousness": 19,  # proxy: fomo + fear
    "surprise":    26,  # proxy: delight + curiosity
    "realization": 22,  # proxy: trust + pride
}

# ── 7 ad-emotion profiles → two GoEmotions proxies each ───────────────────────
PROFILE_TO_GOEMOTION: dict[str, list[str]] = {
    "fomo":       ["desire", "nervousness"],    # want it + anxiety of missing
    "curiosity":  ["curiosity", "surprise"],    # intrigue + open loop
    "fear":       ["nervousness", "surprise"],  # dread + alarm
    "excitement": ["excitement", "desire"],     # energy + wanting
    "trust":      ["realization", "curiosity"], # recognition + intellectual engagement
    "pride":      ["desire", "realization"],    # ownership + recognition of quality
    "delight":    ["excitement", "surprise"],   # joy + unexpected
}


class GoEmotionScorer:
    MODEL_ID = "SamLowe/roberta-base-go_emotions"

    def __init__(self) -> None:
        self._available = False
        self._model = None
        self._tokenizer = None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, cache_dir=MODEL_DIR
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_ID, cache_dir=MODEL_DIR
            )
            self._model.eval()
            self._available = True
            logger.info("GoEmotionScorer loaded from %s", MODEL_DIR)
        except Exception as exc:
            logger.warning("GoEmotionScorer unavailable: %s", exc)

    def is_available(self) -> bool:
        return self._available

    def classify(self, text: str) -> dict[str, float]:
        """Return sigmoid scores for the 6 raw GoEmotions target labels.

        Falls back to 0.5 placeholders when the model is unavailable.
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
        """Aggregate raw GoEmotions scores into the 7 ad-emotion profile scores."""
        result: dict[str, float] = {}
        for profile, labels in PROFILE_TO_GOEMOTION.items():
            result[profile] = sum(goemotion_scores.get(l, 0.5) for l in labels) / len(labels)
        return result

    def predict(self, text: str) -> tuple[str, float]:
        """Return (best_profile, confidence) across the 7 ad-emotion profiles."""
        scores = self.classify(text)
        profile_scores = self.profile_score(scores)
        best = max(profile_scores, key=profile_scores.__getitem__)
        return best, round(profile_scores[best], 3)
