# Flow: Ad Generation (Claude)

All generation lives in `api/generation/creative.py`.

---

## 1. Brand research — `api/generation/creative.py:24`
```
search_brand_context(brand: str) -> str
  └─ TavilyClient.search(query="{brand} brand positioning values marketing 2025 2026")
       max_results=3, search_depth="basic"
       → join first 250 chars of each result with " | "
       → returns "" if TAVILY_API_KEY not set or brand empty
```

---

## 2. Variant generation — `api/generation/creative.py:103`

**Model:** `claude-sonnet-4-6`  
**System prompt:** `api/generation/creative.py:13` — ad copywriter persona (6-10 word headline, 15-25 word body, DALL-E image prompt, 2-4 word CTA)

```
generate_variants(brief, context, num_variants, emotion_hint) -> list[dict]
  │
  ├─ Builds prompt:
  │   "Campaign brief: {brief}
  │    Brand context: {context}       ← optional
  │    Target emotion: {emotion}      ← optional, e.g. "Target emotion: urgent."
  │    Generate {N} distinct ad variants..."
  │
  ├─ Claude returns JSON array (no markdown)
  │   Strips ```json fences if present (api/generation/creative.py:129)
  │
  └─ Returns: [{id, headline, body, image_prompt, cta}, ...]
```

**To add emotion feedback loop (future):** prepend `build_feedback_hint()` output to this prompt before re-running for the winner variant.

---

## 3. Campaign strategy — `api/generation/creative.py:46`

**Model:** `claude-haiku-4-5-20251001` (fast + cheap)

```
generate_campaign_strategy(brief, top_variant, region_scores) -> str
  │
  └─ Prompt includes: brief, winner headline+body, region scores as %
       → Claude returns 3 bullet points (• format):
           1. Target audience + emotional hook
           2. Why copy activates dominant brain region
           3. Optimisation for lowest-scoring region
```

---

## 4. Hero image — `api/generation/creative.py:80`

```
generate_ad_image(image_prompt) -> str | None
  └─ openai.AsyncOpenAI().images.generate(
       model="dall-e-3", size="1024x1024", quality="standard"
     )
     Appends: ". Photorealistic, no text, no overlaid words, clean product shot."
     Returns: URL string or None if openai not installed / key missing
```

---

## Environment variables
| Var | Used by | Effect if missing |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude generation | crash on first request |
| `TAVILY_API_KEY` | Brand research | skips research, returns "" |
| `OPENAI_API_KEY` | DALL-E image | `hero_image_url = None` |

---

## To debug
- Variants are identical angles → check the "each variant must be meaningfully different" instruction at `api/generation/creative.py:117`
- Claude returns invalid JSON → check raw at `api/generation/creative.py:136` (logs first 200 chars)
- Strategy missing → check `generate_campaign_strategy` model/tokens at `api/generation/creative.py:72`
- Image URL null → set `OPENAI_API_KEY` or check `generate_ad_image` exception log
