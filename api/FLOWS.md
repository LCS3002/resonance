# Flows: API Layer

Files: `main.py`, `agent.py`, `agent_langgraph.py`, `agent_tools.py`, `claude_local.py`

---

## Routes — `main.py`

| Method | Path | Handler | Notes |
|---|---|---|---|
| POST | `/api/campaign` | `campaign()` | Full GAN-loop pipeline |
| POST | `/api/generate-visual` | `generate_visual()` | On-demand SVG only |
| GET | `/api/health` | `health()` | GoEmotions + CLIP status |
| GET | `/api/brain-mesh` | `brain_mesh()` | Serves `brain_mesh.json` |
| GET | `/` | `root()` | Frontend index.html |

---

## GAN-loop pipeline — `POST /api/campaign`

```
CampaignRequest { brief, mode }
  └─ run_agent(raw_brief, mode)                 agent_langgraph.py:run_agent()

S0 s0_parse   (Haiku)    raw_brief → BriefState
S1 s1_concept (Sonnet)   BriefState → ConceptState  ← invariant anchor
S2 s2_parallel           two threads:
  ├─ S2a s2a_copy  (Sonnet)  → CopyDraft { headline, body, cta }
  └─ S2b s2b_image (Haiku)   → ImageDraft { image_prompt, image_url }
         if mode=text+image: image_gen()         agent_tools.py:image_gen()
S3 s3_eval               GoEmotions on copy + CLIP on image + gradients
                          → EvalResult + combined_feedback
                          detail: scoring/FLOWS.md
S4 s4_parallel           two threads:
  ├─ S4a s4a_refine_copy  (Sonnet) concept-anchored copy revision
  └─ S4b s4b_refine_image (Haiku)  concept-anchored image revision
S5 s5_format  (Haiku)    platform constraint check → FinalAd
```

Graph wiring: `agent_langgraph.py:_build_graph()`  
Concept is the shared invariant — S4a/S4b cannot change it.

---

## Response shape

```python
{
  "brief":          BriefState,     # brand_name, brand_mission, platform, target_emotion
  "concept":        ConceptState,   # emotional_core, visual_metaphor, tone
  "initial_copy":   CopyDraft,      # S2a — pre-refinement
  "initial_image":  ImageDraft,     # S2b — pre-refinement
  "eval": {
    "text_emotion":          str,
    "text_confidence":       float,
    "text_target_score":     float,
    "text_goemotion_scores": dict,
    "text_gradient_hint":    str,   # Claude-ready token attribution feedback
    "image_emotion":         str,
    "image_target_score":    float,
    "image_clip_scores":     dict,
    "image_gradient_hint":   str,   # gradient on image_prompt text
    "combined_feedback":     str,   # weighted merge → goes to S4
  },
  "final_copy":      CopyDraft,     # S4a output
  "final_image":     ImageDraft,    # S4b output
  "platform_issues": list[str],
  "iteration":       int,
  "mode":            str,
}
```

---

## Singleton model accessors — `agent.py`

```
get_goemotion() → GoEmotionScorer   (lazy, loaded once)
get_clip()      → CLIPScorer        (lazy, loaded once)
```

Called in `main.py:health()`. Also instantiated inside `s3_eval()` directly (LangGraph nodes are isolated — they instantiate their own singletons via the same lazy pattern).

---

## claude_local wrapper — `claude_local.py`

```
local_claude.messages.create(model, max_tokens, messages, system)   — sync
local_claude.messages.acreate(...)                                   — async

Both: claude --model <name> -p <prompt>
Strips ANTHROPIC_API_KEY from env — uses Claude Code session auth.
Timeout: 120s  (_TIMEOUT)
```

To change model for a node: change `model=` in that node function.  
`claude_local.py` itself is model-agnostic — do not add model defaults there.

---

## image_gen — `agent_tools.py`

```
image_gen(image_prompt, target_emotion)        [sync — used in LangGraph nodes]
image_gen_async(image_prompt, target_emotion)  [async — used in FastAPI handlers]
  Claude Haiku → SVG XML (viewBox 300×250, no <text>)
  → "data:image/svg+xml;base64,..." URI
```

---

## To add a new route

1. Add Pydantic request model in `main.py`
2. Add `@app.post/get(...)` handler
3. Add row to routes table above

## To add a new agent state

1. Add `TypedDict` field to `GraphState` in `agent_langgraph.py`
2. Write the node function
3. Add node + edge in `_build_graph()`
4. Add row to state machine table above

---

## Change Index

| Thing to change | Where |
|---|---|
| Route definitions | `main.py` routes section |
| Agent graph topology | `agent_langgraph.py:_build_graph()` |
| S0 parse prompt | `agent_langgraph.py:s0_parse()` |
| S1 concept prompt | `agent_langgraph.py:s1_concept()` |
| S2a/S4a copy prompts | `agent_langgraph.py:s2a_copy()` / `s4a_refine_copy()` |
| S2b/S4b image prompts | `agent_langgraph.py:s2b_image()` / `s4b_refine_image()` |
| Platform char limits | `agent_tools.py:PLATFORM_CONSTRAINTS` |
| SVG colour palettes | `agent_tools.py:_EMOTION_PALETTES` |
| Claude CLI timeout | `claude_local.py:_TIMEOUT` |
| Singleton model init | `agent.py:get_goemotion()` / `get_clip()` |
