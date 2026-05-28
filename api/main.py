"""Resonance FastAPI server.

Routes:
  POST /api/campaign   — full pipeline: brief → variants → neural scores
  GET  /api/health     — liveness check + model status
  GET  /api/brain-mesh — serve pre-computed fsaverage5 mesh JSON
"""

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import get_scorer, run_campaign

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resonance",
    description="Neural engagement scoring for conversational ad placement",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND = Path(__file__).parents[1] / "vary-interactive-3d-helmet-sho"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class CampaignRequest(BaseModel):
    brief: str = Field(..., min_length=5, max_length=500)
    brand: str | None = Field(None, max_length=100)
    num_variants: int = Field(3, ge=1, le=5)
    target_emotion: str | None = Field(None, description="aspirational|trustworthy|urgent|playful|premium")


class ReviewDecision(BaseModel):
    variant_id: str
    approved: bool
    reviewer_notes: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    scorer = get_scorer()
    return {
        "status": "ok",
        "models": {
            "tribe":  scorer.tribe.is_available(),
            "custom": scorer.custom.is_available(),
        },
    }


@app.post("/api/campaign")
async def campaign(req: CampaignRequest):
    try:
        result = await run_campaign(
            brief=req.brief,
            brand=req.brand,
            num_variants=req.num_variants,
            target_emotion=req.target_emotion,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Campaign pipeline error")
        raise HTTPException(status_code=500, detail="Internal scoring error")


@app.post("/api/review")
async def review(decision: ReviewDecision):
    """Human-in-the-loop review endpoint.

    In production: store to DB, feed to Overmind fine-tuning loop.
    For demo: echo back with confirmation.
    """
    logger.info(
        f"HITL decision — variant={decision.variant_id} "
        f"approved={decision.approved} notes={decision.reviewer_notes!r}"
    )
    return {
        "variant_id": decision.variant_id,
        "approved": decision.approved,
        "message": "Decision recorded. Feeds Overmind optimization loop.",
    }


@app.get("/api/brain-mesh")
def brain_mesh():
    mesh_path = _FRONTEND / "brain_mesh.json"
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="brain_mesh.json not found — run export script")
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
