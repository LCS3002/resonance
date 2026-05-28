"""Emotion prediction from neural region scores.

Maps (language, visual, prefrontal) activation → emotion label.

Based on: valence/arousal research (Russell 1980) + known fMRI correlates.
Post-hack: replace profile vectors with NAPS/IAPS-trained MLP.
"""

import numpy as np

# Each emotion is defined by its expected activation profile across three regions.
# Sources:
#   - Language: Broca/Wernicke, left temporal — storytelling, meaning, narrative
#   - Visual: occipital / ventral stream — imagery, attention, aesthetics
#   - Prefrontal: DLPFC / OFC — decision-making, desire, urgency, reward
EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "aspirational": {"language": 0.82, "visual": 0.65, "prefrontal": 0.88},
    "trustworthy":  {"language": 0.75, "visual": 0.48, "prefrontal": 0.62},
    "urgent":       {"language": 0.52, "visual": 0.72, "prefrontal": 0.94},
    "playful":      {"language": 0.88, "visual": 0.84, "prefrontal": 0.42},
    "premium":      {"language": 0.58, "visual": 0.92, "prefrontal": 0.72},
}

EMOTIONS = list(EMOTION_PROFILES.keys())

# Plain-language copy guidance for each under-activated region
_COPY_GUIDANCE = {
    "language":   "use more narrative storytelling, emotional verbs, and meaning-rich vocabulary",
    "visual":     "add concrete sensory imagery — texture, colour, motion, scale",
    "prefrontal": "inject decision-driving language: social proof, urgency, scarcity, specific outcomes",
}


def predict_emotion(region_scores: dict[str, float]) -> tuple[str, float]:
    """Return (predicted_emotion, confidence 0-1) using relative activation pattern.

    Uses z-scored region activations so small differences (common with fMRI proxies)
    are still meaningfully mapped to emotion profiles.
    """
    vec = np.array([region_scores["language"], region_scores["visual"], region_scores["prefrontal"]])

    # Z-score within the vector to amplify relative differences
    mean, std = vec.mean(), vec.std()
    if std < 1e-6:
        # Flat profile — default to aspirational
        return "aspirational", 0.5
    vec_z = (vec - mean) / std  # shape: (3,) with zero mean

    best_emotion, best_sim = EMOTIONS[0], -np.inf
    for emotion, profile in EMOTION_PROFILES.items():
        p = np.array([profile["language"], profile["visual"], profile["prefrontal"]])
        # Z-score the profile too
        pm, ps = p.mean(), p.std()
        p_z = (p - pm) / (ps + 1e-8)
        # Dot product of z-scored vectors = sensitivity to relative patterns
        sim = float(np.dot(vec_z, p_z))
        if sim > best_sim:
            best_sim, best_emotion = sim, emotion

    # Convert sim to a 0-1 confidence — range of dot products of unit z-vecs is roughly [-3, 3]
    confidence = float(np.clip((best_sim + 3) / 6, 0.1, 0.99))
    return best_emotion, confidence


def emotion_match_score(target: str, region_scores: dict[str, float]) -> float:
    """0-1 score for how well region activations match the target emotion."""
    if target not in EMOTION_PROFILES:
        return 0.5
    profile = EMOTION_PROFILES[target]
    profile_vec = np.array([profile["language"], profile["visual"], profile["prefrontal"]])
    score_vec   = np.array([region_scores["language"], region_scores["visual"], region_scores["prefrontal"]])
    profile_vec /= np.linalg.norm(profile_vec)
    score_vec   /= (np.linalg.norm(score_vec) + 1e-8)
    return float(np.clip((np.dot(profile_vec, score_vec) - 0.3) / 0.7, 0, 1))


def compute_counterfactual_hint(
    target: str,
    region_scores: dict[str, float],
    threshold: float = 0.12,
) -> str:
    """Return a copy-editing hint that closes the emotion gap.

    This is the counterfactual step: find which region is most under-activated
    relative to the target profile, translate to generator guidance.
    """
    if target not in EMOTION_PROFILES:
        return ""

    profile = EMOTION_PROFILES[target]
    gaps = {
        region: profile[region] - region_scores.get(region, 0)
        for region in ("language", "visual", "prefrontal")
    }

    # Only suggest changes for significant gaps
    meaningful = [(r, g) for r, g in gaps.items() if g > threshold]
    if not meaningful:
        return ""

    # Sort by largest gap first
    meaningful.sort(key=lambda x: -x[1])
    hints = [_COPY_GUIDANCE[r] for r, _ in meaningful[:2]]
    return "; ".join(hints)
