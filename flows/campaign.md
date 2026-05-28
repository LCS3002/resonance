# Flow: Campaign Pipeline

**Trigger:** `POST /api/campaign` with `{ brief, brand?, num_variants?, target_emotion? }`

---

## 1. HTTP entry — `api/main.py:74`
```
campaign(req: CampaignRequest)
  └── calls run_campaign(brief, brand, num_variants, target_emotion)
```
Request model: `CampaignRequest` at `api/main.py:46`

---

## 2. Orchestrator — `api/agent.py:58`
```
run_campaign(brief, brand, num_variants, target_emotion)
  │
  ├─ 1. search_brand_context(brand)          api/generation/creative.py:24
  │       Tavily search → brand context string
  │
  ├─ 2. generate_variants(brief, context,    api/generation/creative.py:103
  │       num_variants, emotion_hint)
  │       Claude sonnet-4-6 → list[dict]
  │       Keys: id, headline, body, image_prompt, cta
  │
  ├─ 3. Per-variant scoring loop             api/agent.py:92
  │   ├─ scorer.score(text)                  api/scoring/pipeline.py:28  → see neural_scoring.md
  │   ├─ predict_emotion(region_scores)      api/scoring/emotion.py:34   → see emotion_prediction.md
  │   ├─ emotion_match_score(target, scores) api/scoring/emotion.py:65
  │   ├─ compute_counterfactual_hint(...)    api/scoring/emotion.py:77
  │   ├─ goemotion.classify(text)            api/scoring/goemotion_scorer.py:56 → see emotion_prediction.md
  │   └─ classify_text_emotion(text, scorer) api/scoring/emotion.py:115
  │
  ├─ 4. Sort variants by combined_score desc
  │       winner = scored_variants[0]
  │
  ├─ 5. Gradient saliency on winner          api/agent.py:120  → see gradient_saliency.md
  │       (only if GoEmotions available + target_emotion set)
  │
  ├─ 6. Parallel: strategy + hero image      api/agent.py:148
  │   ├─ generate_campaign_strategy(...)     api/generation/creative.py:46
  │   │   Claude haiku-4-5 → 3-bullet strategy
  │   └─ generate_ad_image(image_prompt)     api/generation/creative.py:80
  │       DALL-E 3 → image URL (optional)
  │
  └─ 7. Optional: Moondream image analysis   api/agent.py:156  → see moondream.md
          (only if moondream.is_available() and hero_image_url)
```

---

## 3. Response shape
```python
{
  "variants":            list[ScoredVariant],  # all, sorted best→worst
  "winner":              ScoredVariant,        # variants[0] + saliency_hint + image_emotion_*
  "flagged_count":       int,
  "brand_context":       str,
  "strategy":            str,                  # 3-bullet Claude narrative
  "hero_image_url":      str | None,
  "target_emotion":      str | None,
  "scorer_status":       bool,
  "goemotion_available": bool,
  "moondream_available": bool,
  "hitl_threshold":      float,
}
```

**ScoredVariant extra fields (each variant):**
- `combined_score`, `region_scores`, `model_scores`, `latency_ms`, `models_live`
- `flagged`, `score_label`
- `predicted_emotion`, `emotion_confidence` — fMRI region-based label
- `emotion_match_score` — cosine match to target profile (if target set)
- `counterfactual_hint` — region gap guidance + optional saliency text
- `roberta_emotion`, `roberta_confidence` — GoEmotions NLP label
- `goemotion_scores` — dict of 6 target emotion sigmoid scores

**Winner-only extra fields:**
- `saliency_hint` — raw Claude-ready gradient feedback block
- `image_emotion_caption` — Moondream caption (if available)
- `image_emotion_scores` — RoBERTa on image caption (if available)

---

## Singletons (loaded once at startup)
| Name | Type | Init function | File |
|---|---|---|---|
| `_scorer` | `ScorerPipeline` | `get_scorer()` | `api/agent.py:47` |
| `_goemotion` | `GoEmotionScorer` | `get_goemotion()` | `api/agent.py:54` |
| `_moondream` | `MoondreamVLM` | `get_moondream()` | `api/agent.py:61` |

---

## To debug
- **Variant scores wrong** → `api/scoring/pipeline.py:28` (`ScorerPipeline.score`)
- **Emotion label wrong** → `api/scoring/emotion.py:34` (`predict_emotion`) or `api/scoring/goemotion_scorer.py:56`
- **Claude not generating** → `api/generation/creative.py:103` (`generate_variants`)
- **Saliency hint empty** → `api/scoring/saliency.py:14` (`get_token_saliency`) + check `goemotion.is_available()`
- **Image analysis missing** → `api/scoring/moondream.py` + check `moondream.is_available()`
