# Flow: Moondream Image Analysis

**When:** After `generate_ad_image()` returns a non-null URL AND `moondream.is_available()`.
**Entry:** `api/agent.py:156`

---

## Initialization — `api/scoring/moondream.py`

Load priority on startup:

```
MoondreamVLM.__init__()
  │
  ├─ 1. Try local HuggingFace weights              api/scoring/moondream.py:33
  │       AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-06-21")
  │       device: cuda → mps → cpu
  │       Sets self._mode = "local"
  │
  ├─ 2. Try Ollama REST                            api/scoring/moondream.py:53
  │       GET localhost:11434/api/tags (2s timeout)
  │       Sets self._mode = "ollama"
  │       Requires: ollama pull moondream
  │
  └─ 3. No-op fallback
          self._mode = None
          is_available() returns False → silently skipped in agent
```

Skip entirely: set `MOONDREAM_SKIP=1` env var.

---

## Analysis call — `api/scoring/moondream.py:62`

```
analyze_image_emotion(image_url: str) -> str
  │
  ├─ _fetch_image(image_url)                      api/scoring/moondream.py:99
  │   urllib → PIL.Image → RGB
  │
  ├─ [if mode == "local"]
  │   model.query(image, EMOTION_QUERY) → {"answer": caption}
  │
  └─ [if mode == "ollama"]
      Base64-encode image bytes
      POST localhost:11434/api/generate
        { model: "moondream", prompt: EMOTION_QUERY, images: [b64] }
      → response["response"]
```

**EMOTION_QUERY** at `api/scoring/moondream.py:18`:
> "Describe the emotional tone, mood, and psychological effect of this image on a viewer. What feelings does it evoke? Be specific about tension, curiosity, urgency, or comfort."

---

## After caption is returned

Back in `api/agent.py:156`:
```
caption → goemotion.classify(caption)   api/scoring/goemotion_scorer.py:56
  → winner["image_emotion_caption"] = caption
  → winner["image_emotion_scores"]  = { curiosity, desire, excitement,
                                          nervousness, surprise, realization }
```

This gives a parallel emotion reading from the *visual* content of the ad — compare `image_emotion_scores` vs `goemotion_scores` (text) to see if copy and image are emotionally aligned.

---

## To debug
- `moondream.is_available()` returns False → check startup logs for load error
- Local load fails with OOM → needs ~4 GB VRAM; fall back to Ollama
- Ollama mode: `ollama run moondream` must be running, model pulled
- Caption is empty string → check exception log in `analyze_image_emotion`
- `_fetch_image` fails → DALL-E URL may have expired (they expire after ~1h)
