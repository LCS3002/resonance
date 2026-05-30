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
CampaignRequest { brief, mode, target_emotion }
  └─ run_agent(raw_brief, mode, target_emotion)   agent_langgraph.py:run_agent()

S0  s0_parse    (Haiku)   raw_brief → BriefState
                           if target_emotion override provided: inject into brief
                           conditional: target_emotion=="infer" → S0b, else → S1

S0b s0b_infer   (Sonnet)  [only if target_emotion=="infer"]
                           brand + audience → infer best of 7 emotions (CoT)
                           → brief.target_emotion, emotion_rationale

S1  s1_concept  (Sonnet)  BriefState → ConceptState  ← invariant anchor
S2  s2_parallel            two threads:
  ├─ S2a s2a_copy  (Sonnet)  → CopyDraft { headline, body, cta }
  └─ S2b s2b_image (Haiku)   → ImageDraft { image_prompt, image_url }
         if mode=text+image: image_gen()           agent_tools.py:image_gen()
S3  s3_eval                GoEmotions on copy + CLIP on image + gradients
                            → EvalResult + combined_feedback
                            detail: scoring/FLOWS.md
S4  s4_parallel            two threads:
  ├─ S4a s4a_refine_copy  (Sonnet) concept-anchored copy revision
  └─ S4b s4b_refine_image (Haiku)  concept-anchored image revision
S5  s5_format   (Haiku)   platform constraint check → FinalAd
```

Graph wiring: `agent_langgraph.py:_build_graph()`  
Concept is the shared invariant — S4a/S4b cannot change it.

## Emotion taxonomy

7 ad-emotion profiles (replaces aspirational|trustworthy|urgent|playful|premium):

| Emotion | Lever | GoEmotions proxies |
|---|---|---|
| fomo | scarcity, "others are in" | desire + nervousness |
| curiosity | intrigue, open loop | curiosity + surprise |
| fear | risk of loss, danger | nervousness + surprise |
| excitement | energy, anticipation | excitement + desire |
| trust | credibility, proof | realization + curiosity |
| pride | ownership, "I chose well" | desire + realization |
| delight | joy, surprise | excitement + surprise |

`target_emotion` flow:
- specific value → injected in S0 after parse
- `"infer"` → S0b CoT inference (Sonnet)
- `""` → S0 extracts from raw brief text (fallback)

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

## Prompt storage — `prompts/`

Each pipeline stage has a versioned prompt file. The loader always picks the highest `vN` for each stage.

```
prompts/
  loader.py              load_prompt(stage) → (system, user_template)
                         render(stage, **kwargs) → (system, user_content)
  s0_parse/v1.txt        few-shot: brief → BriefState JSON
  s1_concept/v1.txt      few-shot: brief → ConceptState JSON  ← most critical
  s2a_copy/v1.txt        few-shot: concept → CopyDraft JSON
  s2b_image/v1.txt       few-shot: concept → image_prompt JSON
  s4a_refine_copy/v1.txt few-shot: copy + feedback → revised CopyDraft
  s4b_refine_image/v1.txt few-shot: image_prompt + feedback → revised image_prompt
  s5_format/v1.txt       few-shot: violations → trimmed CopyDraft
```

File format (each `vN.txt`):
```
## SYSTEM
<system prompt>

## USER
<user template — $variable placeholders, string.Template style>
```

To add a new prompt version: create `vN+1.txt` in the stage directory. Loader picks it up automatically on next restart — no code change needed.

Variables per stage:
| Stage | Template variables |
|---|---|
| s0_parse | `$raw_brief` |
| s1_concept | `$raw_brief`, `$brand_name`, `$target_emotion` |
| s2a_copy | `$emotional_core`, `$tone`, `$brand_name`, `$platform`, `$target_emotion` |
| s2b_image | `$emotional_core`, `$visual_metaphor`, `$target_emotion` |
| s4a_refine_copy | `$emotional_core`, `$tone`, `$brand_name`, `$platform`, `$prev_headline`, `$prev_body`, `$prev_cta`, `$feedback` |
| s4b_refine_image | `$emotional_core`, `$visual_metaphor`, `$target_emotion`, `$prev_image_prompt`, `$feedback` |
| s5_format | `$constraint_str`, `$headline`, `$body`, `$cta` |

---

## Change Index

| Thing to change | Where |
|---|---|
| Route definitions | `main.py` routes section |
| Agent graph topology | `agent_langgraph.py:_build_graph()` |
| Any stage prompt | `prompts/<stage>/v1.txt` — or add `v2.txt` for a new version |
| Emotion taxonomy | `scoring/goemotion_scorer.py:PROFILE_TO_GOEMOTION` + `scoring/clip_scorer.py:_EMOTION_PROMPTS` |
| Emotion inference prompt | `prompts/s0b_infer/v1.txt` |
| Platform char limits | `agent_tools.py:PLATFORM_CONSTRAINTS` |
| SVG colour palettes | `agent_tools.py:_EMOTION_PALETTES` |
| Claude CLI timeout | `claude_local.py:_TIMEOUT` |
| Local model directory | `MODEL_DIR` env var (default `./models`) |
| Pre-download models | `python -m api.scripts.download_models` |
