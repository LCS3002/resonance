# Flow: LangGraph Agent Pipeline

**Trigger:** `POST /api/agent/campaign` with `{ brief, mode, run_eval }`

---

## HTTP entry — `api/main.py:agent_campaign()`

```
AgentCampaignRequest:
  brief:    str (5-800 chars)
  mode:     "text" | "text+image"
  run_eval: bool (default true)
```

---

## Phase 1: Agent S0 + S1 (initial draft) — `api/main.py → api/agent_langgraph.py`

```
run_agent(raw_brief, mode, feedback=None)   api/agent_langgraph.py:run_agent()
  │
  ├─ s0_intake (Haiku)                      api/agent_langgraph.py:s0_intake()
  │   Claude parses raw_brief → BriefState:
  │   { brand_name, brand_mission, platform, target_emotion, narrative, mode }
  │
  ├─ s1_generate (Sonnet)                   api/agent_langgraph.py:s1_generate()
  │   Claude generates ONE AdDraft:
  │   { headline, body, cta, image_prompt, image_url }
  │   If mode=text+image: calls image_gen()  → SVG data URI
  │
  └─ Route: feedback=None → skip S2 → s3_format (initial state only)
```

**image_gen tool** — `api/agent_tools.py:image_gen()`
```
Claude Haiku generates 300×250 SVG:
  - <defs> with linearGradient matching emotional tone
  - 3-5 geometric shapes (NO text elements)
  - Returns { image_url: data:image/svg+xml;base64,..., svg_source: str }
PLATFORM_CONSTRAINTS at api/agent_tools.py:14
EMOTION_PALETTES at api/agent_tools.py:18
```

---

## Phase 2: Eval pipeline — `api/main.py:agent_campaign()` (lines ~107-135)

Runs only if `run_eval=True` and initial draft exists.

```
score_text = "{headline}. {body}"
  │
  ├─ scorer.score(score_text)              api/scoring/pipeline.py     → see neural_scoring.md
  │   → neural dict with combined_score, region_scores
  │
  ├─ goemotion.classify(score_text)        api/scoring/goemotion_scorer.py
  │   → 6 target emotion sigmoid scores
  │
  ├─ goemotion.predict(score_text)         api/scoring/goemotion_scorer.py
  │   → (roberta_emotion, roberta_confidence)
  │
  ├─ Gradient saliency (if target_emotion available):
  │   PROFILE_TO_GOEMOTION[target_emotion][0] → GoEmotions label → index
  │   get_token_saliency(text, idx, model, tok)   api/scoring/saliency.py
  │   build_feedback_hint(text, label, pairs, scores) api/scoring/saliency.py
  │   → saliency_hint (Claude-ready feedback block)
  │
  └─ compute_counterfactual_hint(target, regions, saliency_hint=...)
      api/scoring/emotion.py
      → cf_hint = region gap guidance + saliency block combined
```

---

## Phase 3: Agent S2 + S3 (refinement) — `api/agent_langgraph.py`

```
run_agent(raw_brief, mode, feedback=cf_hint, eval_scores=..., saliency_hint=...)
  │
  ├─ s0_intake (Haiku) — re-parses brief (same output)
  ├─ s1_generate (Sonnet) — generates fresh initial draft
  ├─ Route: feedback present → s2_refine
  │
  ├─ s2_refine (Sonnet)                    api/agent_langgraph.py:s2_refine()
  │   Receives: previous draft + feedback string
  │   Only re-calls image_gen if image_prompt changed
  │
  └─ s3_format (Haiku)                     api/agent_langgraph.py:s3_format()
      Checks platform constraints: check_platform_constraints()  api/agent_tools.py
      If violations → Haiku trims headline/body to fit
      Sets status = "done"
```

**Note:** Phase 3 runs S0+S1 again (re-generates the initial draft with the same brief), then S2 refines it with feedback. This means `final_draft` in the response is the feedback-refined version, NOT the S2-refined version of Phase 1's draft.

---

## Graph wiring — `api/agent_langgraph.py:_build_graph()`

```
StateGraph(GraphState)
  s0_intake → s1_generate
  s1_generate →[conditional _route_after_s1]→ s2_refine (if feedback) | s3_format (no feedback)
  s2_refine → s3_format
  s3_format → END
```

Compiled graph: `_graph = _build_graph()` (module-level singleton)

---

## Response shape

```python
{
  "brief_parsed":    BriefState,        # S0 output
  "initial_draft":   AdDraft,           # S1 output (Phase 1)
  "eval": {
    "neural":              dict,        # combined_score, region_scores, model_scores
    "goemotion_scores":    dict,        # 6 target labels
    "roberta_emotion":     str,
    "roberta_confidence":  float,
    "counterfactual_hint": str,
    "saliency_hint":       str,
  },
  "final_draft":     AdDraft,           # S3 output (Phase 3)
  "platform_issues": list[str],         # empty if constraints satisfied
  "iteration_count": int,
  "target_emotion":  str,
  "mode":            str,
}
```

---

## Frontend — Agent Studio

**Overlay:** `#studioOverlay` — `vary-interactive-3d-helmet-sho/index.html`
**Trigger:** Nav link `#openStudio`
**Form:** `#studioBrief` brief field + mode pills + quick-starts
**Progress:** `#pipelineProgress` with steps ps0→ps1→psE→ps2→ps3
**Results:** `#studioResults` with `#initialDraftCard`, `#evalGrid`, `#saliencyBlock`, `#finalDraftCard`
**JS:** `studioRunBtn` click handler, `renderAdCard()`, `renderEval()` functions

---

## To debug
- **S0 returns bad JSON** → check `s0_intake` at `api/agent_langgraph.py:s0_intake()` — strips ` ```json` fences
- **S1 generates no image** → check `mode` is `"text+image"` and `image_gen()` at `api/agent_tools.py`
- **SVG not displaying** → check `image_url` is a valid `data:image/svg+xml;base64,...` URI
- **Eval missing** → check `run_eval=True` in request and `goemotion.is_available()`
- **S2 not running** → verify `feedback` string is non-empty in Phase 3 `run_agent()` call
- **Platform constraints trimming wrong** → check `PLATFORM_CONSTRAINTS` at `api/agent_tools.py:14`
