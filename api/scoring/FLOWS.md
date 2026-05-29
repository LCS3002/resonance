# Flows: Scoring Subsystem

Files: `goemotion_scorer.py`, `saliency.py`, `clip_scorer.py`, `emotion.py`

---

## Text emotion — GoEmotions

```
copy_text = f"{headline}. {body}"
  └─ GoEmotionScorer.classify(copy_text)        goemotion_scorer.py:56
       RoBERTa → sigmoid logits → 6 target label scores
       { curiosity, desire, excitement, nervousness, surprise, realization }
  └─ GoEmotionScorer.predict(copy_text)          goemotion_scorer.py:74
       → (profile_label, confidence)
       maps 6 labels → 5 profiles via PROFILE_TO_GOEMOTION

PROFILE_TO_GOEMOTION at goemotion_scorer.py:26
TARGET_LABELS (28-class index) at goemotion_scorer.py:17
```

## Text gradient — token attribution

```
target_idx = TARGET_LABELS[ge_label]
model, tok = GoEmotionScorer.get_model_and_tokenizer()   goemotion_scorer.py:64
  └─ get_token_saliency(text, target_idx, model, tok)    saliency.py:14
       word_embeddings → model forward → sigmoid → loss.backward()
       → [(token, saliency_score), ...]
  └─ build_feedback_hint(text, label, pairs, ge_scores)  saliency.py:52
       top 3 helping / bottom 3 hurting tokens
       → Claude-ready feedback block
```

## Image emotion — CLIP zero-shot

```
image_url (SVG data URI or http)
  └─ CLIPScorer.score_image_url(image_url)       clip_scorer.py:score_image_url
       decode base64 or fetch → PIL Image
       CLIP image encoder → cosine sim vs emotion text embeddings
       → { aspirational: float, trustworthy: float, ... }
  └─ CLIPScorer.top_emotion(scores)
       → (emotion_name, score)

Emotion prompts at clip_scorer.py:_EMOTION_PROMPTS
CLIP normalisation range: [-0.3, 0.4] → scaled to [0, 1]
```

## Image gradient — GoEmotions on image_prompt text

```
image_prompt (text)
  └─ get_token_saliency(image_prompt, target_idx, model, tok)
  └─ build_feedback_hint(image_prompt, ge_label, pairs, ge_scores)
       → image_gradient_hint (same format as text hint)

This is the control knob: agent can only change image_prompt, not pixels.
```

## Combine feedback

```
text_gap  = 1 - text_target_score
image_gap = 1 - image_target_score
  └─ build_combined_feedback(target, text_hint, image_hint,  emotion.py
                              text_score, image_score)
       weighted by gap → "TARGET EMOTION: X\n[COPY — N%]...\n[IMAGE — M%]..."
```

---

## Change Index

| Thing to change | Where |
|---|---|
| GoEmotions target labels | `goemotion_scorer.py:TARGET_LABELS` |
| Profile → GoEmotions mapping | `goemotion_scorer.py:PROFILE_TO_GOEMOTION` |
| CLIP emotion text prompts | `clip_scorer.py:_EMOTION_PROMPTS` |
| CLIP model checkpoint | `clip_scorer.py:CLIPScorer._load()` |
| Number of saliency tokens shown | `saliency.py:build_feedback_hint` (top/bottom N) |
| Feedback weighting formula | `emotion.py:build_combined_feedback` |
