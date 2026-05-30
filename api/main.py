"""Resonance FastAPI server.

Routes:
  POST /api/campaign        — full GAN-loop pipeline: brief → concept → copy+image → eval → refine
  POST /api/generate-visual — on-demand SVG generation for a given image_prompt
  GET  /api/health          — liveness check + model status
  GET  /api/brain-mesh      — serve pre-computed fsaverage5 mesh JSON
  GET  /                    — serve frontend
"""

import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env", override=False)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import get_goemotion, get_clip
from .agent_langgraph import run_agent, stream_agent
from .scoring.emotion import EMOTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resonance",
    description="GAN-loop neural emotion scoring for conversational ad placement",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parents[1] / "vary-interactive-3d-helmet-sho"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Request models ─────────────────────────────────────────────────────────────

class CampaignRequest(BaseModel):
    brief:          str = Field(..., min_length=5, max_length=800)
    mode:           str = Field("text", description="text | text+image")
    target_emotion: str = Field("", description="fomo|curiosity|fear|excitement|trust|pride|delight|infer|empty=let S0 extract")


class VisualRequest(BaseModel):
    image_prompt:   str
    target_emotion: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    goemotion = get_goemotion()
    clip      = get_clip()
    return {
        "status": "ok",
        "claude": "local-cli",
        "models": {
            "goemotion": goemotion.is_available(),
            "clip":      clip.is_available(),
        },
    }


@app.post("/api/campaign")
async def campaign(req: CampaignRequest):
    """Full GAN-loop pipeline.

    S0 parse → S1 concept → S2a copy + S2b image (parallel)
    → S3 eval (GoEmotions + CLIP + gradients)
    → S4a refine copy + S4b refine image (parallel, concept-anchored)
    → S5 format
    """
    try:
        state = await run_agent(raw_brief=req.brief, mode=req.mode, target_emotion=req.target_emotion)
        return {
            "brief":          state.get("brief"),
            "concept":        state.get("concept"),
            "initial_copy":   state.get("copy"),
            "initial_image":  state.get("image"),
            "eval":           state.get("eval"),
            "final_copy":     state.get("refined_copy"),
            "final_image":    state.get("refined_image"),
            "platform_issues": state.get("platform_issues", []),
            "iteration":      state.get("iteration", 1),
            "mode":           req.mode,
        }
    except Exception:
        logger.exception("Campaign pipeline error")
        raise HTTPException(status_code=500, detail="Pipeline error")


@app.post("/api/campaign/stream")
async def campaign_stream(req: CampaignRequest):
    """SSE version of /api/campaign — sends one event per pipeline node as it completes.

    Event shape: { type: "node"|"done", node, label, data }
    Frontend can render each stage the moment it arrives instead of waiting for the full pipeline.
    """
    import json
    from fastapi.responses import StreamingResponse

    async def event_stream():
        try:
            async for event in stream_agent(raw_brief=req.brief, mode=req.mode, target_emotion=req.target_emotion):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Stream pipeline error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:120]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-visual")
async def generate_visual(req: VisualRequest):
    """On-demand SVG generation — called by frontend after initial results load."""
    try:
        from .agent_tools import image_gen_async
        result = await image_gen_async(req.image_prompt, req.target_emotion)
        return {"image_url": result["image_url"]}
    except Exception as e:
        logger.warning(f"Visual generation failed: {e}")
        return {"image_url": None}


@app.get("/api/brain-mesh")
def brain_mesh():
    mesh_path = _FRONTEND / "brain_mesh.json"
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="brain_mesh.json not found")
    return FileResponse(str(mesh_path), media_type="application/json")


@app.get("/")
def root():
    index = _FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Resonance API — see /docs"}


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
