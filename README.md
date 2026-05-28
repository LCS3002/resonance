<div align="center">

# 🧠 Resonance

### Neural Ad Intelligence — generate ad creative, score it against predicted brain engagement, and deploy only what activates.

*Built for **Cursor × Thrad London 2026***

`FastAPI` · `LangGraph` · `Claude` · `RoBERTa GoEmotions` · `Three.js` · `fMRI brain encoding`

</div>

---

## What it does

Resonance is a closed-loop ad creative engine. You give it a brief; it generates ad copy, **scores that copy against a brain-engagement model trained on real fMRI data**, reads the emotional signal with a transformer, computes which exact words help or hurt the target emotion via **gradient saliency**, and feeds that back to rewrite a stronger ad — all visualised on a live 3D brain that lights up the regions your copy activates.

```
Brief  ─►  Generate  ─►  Neural Eval  ─►  Gradient Feedback  ─►  Refined Ad
            (Claude)      (fMRI + GoEmotions)   (saliency)         (Claude)
                              │
                              ▼
                    3D brain lights up the
                    language / visual / prefrontal
                    regions the copy activates
```

---

## Why it's interesting

| | |
|---|---|
| 🧬 **Grounded in real fMRI** | Engagement scoring uses per-voxel Pearson-r weights from the **Algonauts 2025** brain-encoding challenge — not a black-box heuristic. |
| 🎯 **Dual emotion signal** | fMRI region activation **and** a RoBERTa **GoEmotions** classifier independently cross-validate the emotional read of every ad. |
| 🔬 **Gradient saliency loop** | Token-level input-gradient attribution finds the exact words pulling *away* from the target emotion, then feeds Claude a structured rewrite hint. |
| ⚡ **15ms scoring** | The neural scorer alone runs in ~15ms — fast enough for real-time ad serving, decoupled from generation latency. |
| 🧠 **Live brain viz** | A Three.js `fsaverage5` cortical mesh pulses the language / visual / prefrontal regions your winning ad activates. |
| 🔌 **Zero API key** | Routes Claude calls through the local **Claude Code CLI** (`claude -p`) — runs fully on your own session auth. |

---

## Architecture

```
resonance/
├── api/                          # FastAPI backend
│   ├── main.py                   # Routes incl. SSE streaming pipeline
│   ├── agent.py                  # Campaign orchestration
│   ├── agent_langgraph.py        # LangGraph agent: S0→S1→S2→S3 state machine
│   ├── agent_tools.py            # image_gen (SVG via Claude) + platform constraints
│   ├── claude_local.py           # Claude CLI wrapper — no API key needed
│   ├── scoring/
│   │   ├── pipeline.py           # Weighted ensemble: Algonauts 50 / TribeV2 30 / Custom 20
│   │   ├── algonauts_scorer.py   # Real fMRI weights + SBERT embeddings
│   │   ├── tribe_scorer.py       # Meta TribeV2 brain encoder
│   │   ├── goemotion_scorer.py   # RoBERTa GoEmotions (28-class)
│   │   ├── saliency.py           # Gradient-based counterfactual feedback
│   │   ├── moondream.py          # Optional image→emotion VLM bridge
│   │   └── emotion.py            # Region → emotion mapping
│   └── generation/creative.py    # Claude copy generation + Tavily brand research
├── vary-interactive-3d-helmet-sho/
│   ├── index.html                # 3D brain frontend + streaming demo UI
│   └── brain_mesh.json           # Pre-computed fsaverage5 cortical mesh
├── flows/                        # End-to-end flow traces (file:line maps)
└── start.bat                     # One-command launcher
```

### The agent (LangGraph)

A 4-state machine — the eval pipeline drives termination, the agent only consumes a `feedback: str` and doesn't care where it came from:

| State | Model | Job |
|-------|-------|-----|
| **S0 Intake** | Haiku | Parse brief → `{brand, platform, target_emotion, narrative, mode}` |
| **S1 Generate** | Sonnet | Cold generation → `{headline, body, cta, image_prompt}` |
| **S2 Refine** | Sonnet | Rewrite using gradient-saliency feedback |
| **S3 Format** | Haiku | Enforce platform char limits (FB/IG/Twitter) |

### The scoring ensemble

```
combined_score = 0.50 · Algonauts(fMRI)  +  0.30 · TribeV2  +  0.20 · Custom
```

Each model returns activation across three regions — **language** (Broca/Wernicke), **visual** (occipital), **prefrontal** (DLPFC/OFC) — which map to an emotion profile and drive the brain visualisation. Models gracefully fall back to a deterministic mock scorer when checkpoints are unavailable.

---

## Quickstart

```bash
# 1. Install
pip install -r api/requirements.txt

# 2. Launch (kills stale servers, starts backend, opens browser)
start.bat              # Windows
# or
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** → click **Run Campaign** → watch the timeline fill in as the ad is generated, scored, and refined.

> **No `ANTHROPIC_API_KEY`?** No problem. `claude_local.py` routes every model call through your local Claude Code CLI session — no key required.

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/campaign/stream` | POST | **SSE** — streams initial draft → eval → refined winner as each stage completes |
| `/api/campaign` | POST | Full pipeline: brief → 3 scored variants |
| `/api/agent/campaign` | POST | LangGraph agent: S0→S1→eval→S2→S3 |
| `/api/generate-visual` | POST | Generate a single SVG ad mockup |
| `/api/brain-mesh` | GET | Serve the fsaverage5 cortical mesh |
| `/api/health` · `/api/debug` | GET | Liveness + environment check |

```bash
curl -X POST http://localhost:8000/api/campaign \
  -H "Content-Type: application/json" \
  -d '{"brief":"premium noise-cancelling headphones for remote workers","target_emotion":"premium"}'
```

---

## Measured performance

| Metric | Value |
|--------|-------|
| Brief → 3 scored variants | **~38s** end-to-end |
| Neural scoring (no LLM) | **~15ms** per call |
| Winner emotion-profile match | **0.97** cosine alignment |
| Streaming first-paint (initial draft) | **~20s** (vs 70s monolithic) |

The SSE pipeline (`/api/campaign/stream`) parallelises SVG generation with saliency computation and streams each stage, so the first ad appears in ~20s instead of waiting for the entire chain.

---

## Tech stack

**Backend** FastAPI · LangGraph · Anthropic Claude (Opus/Sonnet/Haiku) · PyTorch · sentence-transformers · Tavily
**Models** Algonauts 2025 fMRI weights · Meta TribeV2 · RoBERTa GoEmotions (`SamLowe/roberta-base-go_emotions`) · Moondream2 (optional)
**Frontend** Three.js · vanilla JS · Server-Sent Events
**Integrations** Model Context Protocol (MCP) · Alpic · Overmind

---

<div align="center">

**Team Resonance** — Lorenz · Kaya · Dmytro
*Cursor × Thrad London 2026*

</div>
