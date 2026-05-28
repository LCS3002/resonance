"""Resonance FastAPI server.

Routes:
  POST /api/campaign        — full pipeline: brief → variants → neural scores
  POST /api/agent/campaign  — LangGraph agent: S0→S1→eval→S2→S3 combined pipeline
  GET  /api/health          — liveness check + model status
  GET  /api/brain-mesh      — serve pre-computed fsaverage5 mesh JSON
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

from .agent import get_goemotion, get_moondream, get_scorer, run_campaign
from .agent_langgraph import run_agent, refine_existing
from .scoring.emotion import compute_counterfactual_hint
from .scoring.saliency import build_feedback_hint, get_token_saliency
from .scoring.goemotion_scorer import TARGET_LABELS, PROFILE_TO_GOEMOTION


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


class AgentCampaignRequest(BaseModel):
    brief: str = Field(..., min_length=5, max_length=800)
    mode: str = Field("text", description="text | text+image")
    run_eval: bool = Field(True, description="Run neural eval between S1 and S2")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    scorer = get_scorer()
    return {
        "status": "ok",
        "claude": "local-cli",
        "models": {
            "tribe":  scorer.tribe.is_available(),
            "custom": scorer.custom.is_available(),
        },
    }


@app.get("/api/debug")
def debug():
    """Quick env check — shows key status without exposing secret value."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    tavily  = os.getenv("TAVILY_API_KEY", "")
    return {
        "ANTHROPIC_API_KEY": (api_key[:8] + "..." + api_key[-4:]) if len(api_key) > 12 else ("SET" if api_key else "MISSING"),
        "TAVILY_API_KEY":    "SET" if tavily else "MISSING",
        "DEMO_MODE":         DEMO_MODE,
        "note": "If ANTHROPIC_API_KEY shows MISSING or returns 401, run: export ANTHROPIC_API_KEY=sk-ant-... in your terminal",
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


@app.post("/api/agent/campaign")
async def agent_campaign(req: AgentCampaignRequest):
    """Combined LangGraph agent + eval pipeline.

    Flow: S0 intake → S1 generate → [eval: neural + GoEmotions + saliency] → S2 refine → S3 format
    Returns initial draft, eval scores, saliency hint, and final refined ad.
    """
    try:
        # ── Phase 1: S0 + S1 — agent generates initial draft ─────────────────
        initial_state = await run_agent(raw_brief=req.brief, mode=req.mode)
        initial_draft = initial_state["drafts"][-1] if initial_state.get("drafts") else {}
        brief_state   = initial_state.get("brief") or {}
        target_emotion = brief_state.get("target_emotion", "")

        eval_result: dict = {}
        saliency_hint: str = ""

        if req.run_eval and initial_draft:
            # ── Phase 2: eval pipeline on initial draft ───────────────────────
            scorer    = get_scorer()
            goemotion = get_goemotion()

            score_text = f"{initial_draft.get('headline','')}. {initial_draft.get('body','')}"
            neural     = await scorer.score(score_text)
            goemotion_scores = goemotion.classify(score_text)
            roberta_emotion, roberta_conf = goemotion.predict(score_text)

            # Gradient saliency on initial draft for the target emotion
            if goemotion.is_available() and target_emotion:
                # Map profile → primary GoEmotions label → index
                ge_label = (PROFILE_TO_GOEMOTION.get(target_emotion) or [""])[0]
                target_idx = TARGET_LABELS.get(ge_label)
                if target_idx is not None:
                    model, tok = goemotion.get_model_and_tokenizer()
                    saliency_pairs = get_token_saliency(score_text, target_idx, model, tok)
                    saliency_hint  = build_feedback_hint(
                        score_text, ge_label, saliency_pairs, goemotion_scores
                    )

            cf_hint = compute_counterfactual_hint(
                target_emotion, neural.get("region_scores", {}), saliency_hint=saliency_hint
            )

            eval_result = {
                "neural":           neural,
                "goemotion_scores": goemotion_scores,
                "roberta_emotion":  roberta_emotion,
                "roberta_confidence": roberta_conf,
                "counterfactual_hint": cf_hint,
                "saliency_hint":    saliency_hint,
            }

            feedback = cf_hint or saliency_hint or ""
        else:
            feedback = ""

        # ── Phase 3: S2+S3 only — reuse S0/S1 brief, skip re-generation ─────
        if feedback and brief_state:
            final_state = await refine_existing(
                initial_draft=initial_draft,
                brief_state=brief_state,
                raw_brief=req.brief,
                feedback=feedback,
                mode=req.mode,
            )
        else:
            final_state = initial_state

        final_draft = final_state["drafts"][-1] if final_state.get("drafts") else initial_draft

        return {
            "brief_parsed":      final_state.get("brief"),
            "initial_draft":     initial_draft,
            "eval":              eval_result,
            "final_draft":       final_draft,
            "platform_issues":   final_state.get("platform_issues", []),
            "iteration_count":   final_state.get("iteration", 0),
            "target_emotion":    target_emotion,
            "mode":              req.mode,
        }

    except Exception:
        logger.exception("Agent campaign pipeline error")
        raise HTTPException(status_code=500, detail="Agent pipeline error")


class VisualRequest(BaseModel):
    image_prompt: str
    target_emotion: str = ""


@app.post("/api/generate-visual")
async def generate_visual(req: VisualRequest):
    """Generate a single SVG ad mockup. Cheap async call — runs after main results load."""
    try:
        from .agent_tools import image_gen
        result = image_gen(req.image_prompt, req.target_emotion)
        return {"image_url": result["image_url"]}
    except Exception as e:
        logger.warning(f"Visual generation failed: {e}")
        return {"image_url": None}


@app.post("/api/campaign/stream")
async def campaign_stream(req: CampaignRequest):
    """SSE streaming pipeline: sends initial draft, then refined winner, then SVGs.
    Client sees initial copy at ~20s and winner at ~38s instead of waiting for everything.
    """
    import asyncio, json
    from fastapi.responses import StreamingResponse
    from .generation.creative import generate_variants
    from .scoring.emotion import predict_emotion, emotion_match_score, compute_counterfactual_hint, classify_text_emotion
    from .scoring.saliency import get_token_saliency, build_feedback_hint

    async def _run_in_thread(fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def event_stream():
        try:
            scorer    = get_scorer()
            goemotion = get_goemotion()
            target    = req.target_emotion or ""

            # ── Step 1: generate ONE initial variant ─────────────────────────
            yield f"data: {json.dumps({'type':'status','text':'Generating initial draft…'})}\n\n"

            variants = await generate_variants(
                req.brief or "", context="", num_variants=1,
                emotion_hint=f"Target emotion: {target}." if target else "",
            )
            if not variants:
                yield f"data: {json.dumps({'type':'error','text':'Generation failed'})}\n\n"
                return

            draft = variants[0]
            score_text = f"{draft['headline']}. {draft['body']}"

            # ── Step 2: score initial draft (fast) ───────────────────────────
            neural = await scorer.score(score_text)
            goem   = goemotion.classify(score_text)
            em, _  = predict_emotion(neural["region_scores"])
            em_score = emotion_match_score(target, neural["region_scores"])

            yield f"data: {json.dumps({'type':'initial','draft':draft,'neural':neural,'emotion':em,'em_score':round(em_score,3)})}\n\n"

            # ── Step 3: SVG for initial + saliency — parallel ────────────────
            yield f"data: {json.dumps({'type':'status','text':'Generating visual & saliency…'})}\n\n"

            from .agent_tools import image_gen as _image_gen

            def _gen_svg():
                try:   return _image_gen(draft.get("image_prompt",""), target)["image_url"]
                except: return None

            def _saliency():
                hint = ""
                if goemotion.is_available() and target:
                    ge_label = (PROFILE_TO_GOEMOTION.get(target) or [""])[0]
                    idx = TARGET_LABELS.get(ge_label)
                    if idx is not None:
                        m, tok = goemotion.get_model_and_tokenizer()
                        pairs = get_token_saliency(score_text, idx, m, tok)
                        hint  = build_feedback_hint(score_text, ge_label, pairs, goem)
                return hint or compute_counterfactual_hint(target, neural["region_scores"])

            loop = asyncio.get_event_loop()
            svg_task = loop.run_in_executor(None, _gen_svg)
            sal_task = loop.run_in_executor(None, _saliency)
            initial_svg, feedback = await asyncio.gather(svg_task, sal_task)

            if initial_svg:
                yield f"data: {json.dumps({'type':'initial-visual','image_url':initial_svg})}\n\n"

            # ── Step 4: refine with feedback ──────────────────────────────────
            if feedback:
                yield f"data: {json.dumps({'type':'status','text':'Refining with neural feedback…'})}\n\n"

                from .generation.creative import generate_variants as _gen
                from anthropic import Anthropic
                from .claude_local import local_claude as _claude

                resp = _claude.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=600,
                    system="You are an expert ad copywriter. Revise ad copy based on emotion feedback. Keep brand message; change framing.",
                    messages=[{"role":"user","content":(
                        f"Previous headline: {draft['headline']}\n"
                        f"Previous body: {draft['body']}\n"
                        f"Previous CTA: {draft['cta']}\n\n"
                        f"Feedback:\n{feedback[:800]}\n\n"
                        f"Brief: {req.brief}\n\n"
                        "Rewrite. Return ONLY JSON: {\"headline\":\"...\",\"body\":\"...\",\"cta\":\"...\",\"image_prompt\":\"...\"}"
                    )}]
                )
                import json as _json
                try:
                    raw = resp.content[0].text.strip()
                    if raw.startswith("```"): raw = raw.split("```")[1].lstrip("json").strip()
                    refined = _json.loads(raw)
                    refined["id"] = "v_refined"
                    refined.setdefault("image_url", None)
                except Exception:
                    refined = {**draft, "id":"v_refined"}

                # Score refined
                rscore_text = f"{refined.get('headline','')}. {refined.get('body','')}"
                rneural = await scorer.score(rscore_text)
                rem, _  = predict_emotion(rneural["region_scores"])
                rem_score = emotion_match_score(target, rneural["region_scores"])

                yield f"data: {json.dumps({'type':'refined','draft':refined,'neural':rneural,'emotion':rem,'em_score':round(rem_score,3)})}\n\n"

                # SVG for refined
                def _gen_refined_svg():
                    try:   return _image_gen(refined.get("image_prompt",""), target)["image_url"]
                    except: return None

                refined_svg = await loop.run_in_executor(None, _gen_refined_svg)
                if refined_svg:
                    yield f"data: {json.dumps({'type':'refined-visual','image_url':refined_svg})}\n\n"

            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as e:
            logger.exception("Stream pipeline error")
            yield f"data: {json.dumps({'type':'error','text':str(e)[:100]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
