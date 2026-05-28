# Flow: Emotion Prediction

Two independent signals — fMRI region-based and RoBERTa GoEmotions NLP-based — run in parallel on every variant.

---

## Signal 1: fMRI Region-Based — `api/scoring/emotion.py`

**Entry:** `predict_emotion(region_scores)` at `api/scoring/emotion.py:34`

```
region_scores = { language: float, visual: float, prefrontal: float }
  │
  └─ z-score the vector (amplifies relative differences)
       Compare z-scored vec against z-scored EMOTION_PROFILES (line 16)
       Dot product → similarity per profile
       Return (best_emotion, confidence 0-1)
```

**EMOTION_PROFILES** at `api/scoring/emotion.py:16`:
```python
{
  "aspirational": {language: 0.82, visual: 0.65, prefrontal: 0.88},
  "trustworthy":  {language: 0.75, visual: 0.48, prefrontal: 0.62},
  "urgent":       {language: 0.52, visual: 0.72, prefrontal: 0.94},
  "playful":      {language: 0.88, visual: 0.84, prefrontal: 0.42},
  "premium":      {language: 0.58, visual: 0.92, prefrontal: 0.72},
}
```

**Emotion match score** (how close to *target* emotion):
`emotion_match_score(target, region_scores)` at `api/scoring/emotion.py:65`
→ Cosine similarity of region vec vs target profile, scaled to [0, 1]

---

## Signal 2: RoBERTa GoEmotions — `api/scoring/goemotion_scorer.py`

**Model:** `SamLowe/roberta-base-go_emotions` — 125M params, 28-class sigmoid

**Entry:** `goemotion.classify(text)` at `api/scoring/goemotion_scorer.py:56`

```
text → RoBERTa tokenizer → model forward pass → sigmoid logits
  └─ extract 6 target label scores → { curiosity, desire, excitement,
                                        nervousness, surprise, realization }
```

**TARGET_LABELS** (index into 28-class output) at `api/scoring/goemotion_scorer.py:17`

**Profile aggregation** — maps 6 GoEmotions labels → 5 EMOTION_PROFILES:
`goemotion.profile_score(goemotion_scores)` at `api/scoring/goemotion_scorer.py:68`

```python
PROFILE_TO_GOEMOTION = {
  "aspirational": ["excitement", "desire"],
  "trustworthy":  ["realization", "curiosity"],
  "urgent":       ["nervousness", "surprise"],
  "playful":      ["curiosity", "excitement"],
  "premium":      ["desire", "realization"],
}
```
→ Each profile score = mean of its two proxy GoEmotions scores.

**Predict profile label:**
`goemotion.predict(text)` at `api/scoring/goemotion_scorer.py:74` → `(profile_label, confidence)`

**Called via wrapper:**
`classify_text_emotion(text, scorer)` at `api/scoring/emotion.py:115`

---

## Where results surface in the response (per variant)
| Field | Source |
|---|---|
| `predicted_emotion` | `predict_emotion(region_scores)` — fMRI signal |
| `emotion_confidence` | confidence from fMRI signal |
| `emotion_match_score` | `emotion_match_score(target, region_scores)` |
| `roberta_emotion` | `classify_text_emotion(text, goemotion)` — NLP signal |
| `roberta_confidence` | GoEmotions confidence |
| `goemotion_scores` | raw 6-label dict from `goemotion.classify()` |

---

## To debug emotion mismatch
- fMRI label wrong → check `region_scores` values and `EMOTION_PROFILES` at `api/scoring/emotion.py:16`
- RoBERTa label wrong → check `PROFILE_TO_GOEMOTION` mapping at `api/scoring/goemotion_scorer.py:26`
- GoEmotions model not loading → check `GoEmotionScorer.__init__` at `api/scoring/goemotion_scorer.py:36`
