# Flows: Generation Subsystem

Files: `creative.py`

---

## Brand research — `creative.py:search_brand_context()`

```
search_brand_context(brand: str) -> str
  └─ TavilyClient.search(query="{brand} brand positioning values marketing 2025 2026")
       max_results=3, search_depth="basic"
       → join first 250 chars of each snippet with " | "
       → returns "" if TAVILY_API_KEY not set or brand empty
```

Called in: not currently wired into the GAN-loop pipeline.  
To add: call in S0 and pass context into S1 or S2a prompt.

---

## Variant generation — `creative.py:generate_variants()`

```
generate_variants(brief, context, num_variants, emotion_hint) -> list[dict]
  └─ Claude Sonnet: _SYSTEM prompt + user prompt
       Returns JSON array, strips ```json fences
       Each variant: { id, headline, body, image_prompt, cta }
```

Used by: the old `POST /api/campaign` pipeline (removed).  
Currently **not called** by the GAN-loop — the loop generates copy in S2a and image_prompt in S2b separately.  
To reuse: call `generate_variants(brief, num_variants=1)` as an alternative S2 path.

---

## Change Index

| Thing to change | Where |
|---|---|
| System prompt for variants | `creative.py:_SYSTEM` |
| Tavily query | `creative.py:search_brand_context()` |
| Variant JSON keys | `creative.py:generate_variants()` return block |
| TAVILY_API_KEY | `.env` |
