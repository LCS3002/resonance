"""LangGraph ad generation agent — 4-state state machine.

States:
  S0 intake   (Haiku)   — parse brief text → BriefState
  S1 generate (Sonnet)  — cold generation → AdDraft; calls image_gen if mode=text+image
  S2 refine   (Sonnet)  — incorporate feedback string → revised AdDraft
  S3 format   (Haiku)   — validate platform constraints → FinalAd

The agent never decides to stop. Status transitions to "done" externally
(eval pipeline or caller sets it before invoking the graph).
"""

from __future__ import annotations

import json
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agent_tools import check_platform_constraints, image_gen
from .claude_local import local_claude as _claude

logger = logging.getLogger(__name__)


# ── State types ───────────────────────────────────────────────────────────────

class BriefState(TypedDict):
    brand_name: str
    brand_mission: str
    platform: str
    target_emotion: str
    narrative: str
    mode: str  # "text" | "text+image"


class AdDraft(TypedDict):
    headline: str
    body: str
    cta: str
    image_prompt: str | None
    image_url: str | None


class GraphState(TypedDict):
    raw_brief: str          # original user input
    brief: BriefState | None
    mode: str               # "text" | "text+image"
    drafts: list            # list[AdDraft] — full history
    feedback: str | None    # latest feedback string (from eval or human)
    iteration: int
    status: str             # "running" | "done"
    # Eval pipeline fields (injected externally, not computed by agent)
    eval_scores: dict | None
    saliency_hint: str | None
    platform_issues: list   # constraint violations from S3


# ── Node implementations ──────────────────────────────────────────────────────

def s0_intake(state: GraphState) -> dict:
    """Parse raw brief into structured BriefState. Haiku — fast + cheap."""
    raw = state["raw_brief"]
    platform = state.get("mode", "text")  # reuse mode field temporarily; brief has platform

    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"Parse this ad campaign brief into structured JSON:\n\n\"{raw}\"\n\n"
                "Return ONLY valid JSON with these exact keys:\n"
                "{\n"
                '  "brand_name": "brand or product name (infer if not explicit)",\n'
                '  "brand_mission": "one sentence brand purpose",\n'
                '  "platform": "facebook|instagram|twitter|generic",\n'
                '  "target_emotion": "aspirational|trustworthy|urgent|playful|premium",\n'
                '  "narrative": "core story angle in one sentence",\n'
                f'  "mode": "{state.get("mode", "text")}"\n'
                "}"
            ),
        }]
    )
    raw_json = resp.content[0].text.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1].lstrip("json").strip()
    brief = json.loads(raw_json)
    return {"brief": brief, "status": "running"}


def s1_generate(state: GraphState) -> dict:
    """Cold generation from BriefState. Sonnet — quality creative."""
    brief = state["brief"]
    mode  = state.get("mode", "text")

    emotion_context = (
        f"Target emotion: {brief['target_emotion']}. "
        f"Write copy that triggers {brief['target_emotion']} response in the reader.\n"
        if brief.get("target_emotion") else ""
    )

    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=(
            "You are an expert ad copywriter. Write concise, benefit-led ad copy "
            "for conversational AI placements. No exclamation marks. No clichés."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Brief: {state['raw_brief']}\n"
                f"Brand: {brief.get('brand_name', '')}\n"
                f"Narrative: {brief.get('narrative', '')}\n"
                f"Platform: {brief.get('platform', 'generic')}\n"
                f"{emotion_context}\n"
                "Generate ONE ad variant. Return ONLY JSON:\n"
                "{\n"
                '  "headline": "6-10 words, active verb",\n'
                '  "body": "15-25 words, one specific benefit, conversational",\n'
                '  "cta": "2-4 words, imperative",\n'
                '  "image_prompt": "vivid DALL-E style description of the ad visual (required even for text mode)"\n'
                "}"
            ),
        }]
    )

    draft = _parse_draft(resp.content[0].text)
    draft["image_url"] = None

    if mode == "text+image" and draft.get("image_prompt"):
        result = image_gen(draft["image_prompt"], brief.get("target_emotion", ""))
        draft["image_url"] = result["image_url"]

    drafts = list(state.get("drafts", []))
    drafts.append(draft)
    return {"drafts": drafts, "iteration": state.get("iteration", 0) + 1}


def s2_refine(state: GraphState) -> dict:
    """Incorporate feedback into a new AdDraft. Sonnet."""
    brief    = state["brief"]
    feedback = state.get("feedback", "")
    mode     = state.get("mode", "text")
    prev     = state["drafts"][-1] if state.get("drafts") else {}

    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=(
            "You are an expert ad copywriter. Revise ad copy based on "
            "emotion feedback. Keep the brand message; change the framing."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Previous draft:\n"
                f"  Headline: {prev.get('headline','')}\n"
                f"  Body: {prev.get('body','')}\n"
                f"  CTA: {prev.get('cta','')}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Brief context: {state['raw_brief']}\n"
                f"Brand: {brief.get('brand_name','')}\n"
                f"Platform: {brief.get('platform','generic')}\n\n"
                "Rewrite the ad copy applying the feedback. Return ONLY JSON:\n"
                "{\n"
                '  "headline": "...",\n'
                '  "body": "...",\n'
                '  "cta": "...",\n'
                '  "image_prompt": "revised visual description if framing changed, else same as before"\n'
                "}"
            ),
        }]
    )

    draft = _parse_draft(resp.content[0].text)
    draft["image_url"] = None

    # Only re-generate image if image_prompt meaningfully changed
    prev_prompt = prev.get("image_prompt", "")
    new_prompt  = draft.get("image_prompt", "")
    image_changed = new_prompt and new_prompt != prev_prompt

    if mode == "text+image":
        if image_changed:
            result = image_gen(new_prompt, brief.get("target_emotion", ""))
            draft["image_url"] = result["image_url"]
        else:
            draft["image_url"] = prev.get("image_url")
            draft["image_prompt"] = prev_prompt

    drafts = list(state.get("drafts", []))
    drafts.append(draft)
    return {"drafts": drafts, "iteration": state.get("iteration", 0) + 1}


def s3_format(state: GraphState) -> dict:
    """Validate platform constraints and assemble FinalAd. Haiku."""
    brief    = state["brief"] or {}
    platform = brief.get("platform", "generic")
    final    = state["drafts"][-1] if state.get("drafts") else {}

    issues = check_platform_constraints(
        final.get("headline", ""),
        final.get("body", ""),
        platform,
    )

    if issues:
        # Ask Haiku to trim to fit constraints
        constraints_str = "; ".join(issues)
        resp = _claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"Trim this ad copy to fix these constraint violations: {constraints_str}\n\n"
                    f"Headline: {final.get('headline','')}\n"
                    f"Body: {final.get('body','')}\n"
                    f"CTA: {final.get('cta','')}\n\n"
                    "Return ONLY JSON: {\"headline\": \"...\", \"body\": \"...\", \"cta\": \"...\"}"
                ),
            }]
        )
        trimmed = _parse_draft(resp.content[0].text)
        final = {**final, **trimmed}
        issues = []

    return {
        "drafts": state["drafts"][:-1] + [final],
        "platform_issues": issues,
        "status": "done",
    }


# ── Graph wiring ──────────────────────────────────────────────────────────────

def _route_after_s1(state: GraphState) -> str:
    """Go to S2 if feedback is waiting, else S3."""
    return "s2_refine" if state.get("feedback") else "s3_format"


def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("s0_intake",   s0_intake)
    g.add_node("s1_generate", s1_generate)
    g.add_node("s2_refine",   s2_refine)
    g.add_node("s3_format",   s3_format)

    g.set_entry_point("s0_intake")
    g.add_edge("s0_intake", "s1_generate")
    g.add_conditional_edges("s1_generate", _route_after_s1, {
        "s2_refine": "s2_refine",
        "s3_format": "s3_format",
    })
    g.add_edge("s2_refine", "s3_format")
    g.add_edge("s3_format", END)
    return g.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

async def run_agent(
    raw_brief: str,
    mode: str = "text",
    feedback: str | None = None,
    eval_scores: dict | None = None,
    saliency_hint: str | None = None,
) -> GraphState:
    """Run the full S0→S1→(S2?)→S3 graph and return final state.

    If feedback is provided (e.g. from eval pipeline), S2 runs automatically.
    """
    initial: GraphState = {
        "raw_brief":      raw_brief,
        "brief":          None,
        "mode":           mode,
        "drafts":         [],
        "feedback":       feedback,
        "iteration":      0,
        "status":         "running",
        "eval_scores":    eval_scores,
        "saliency_hint":  saliency_hint,
        "platform_issues": [],
    }
    result = await _graph.ainvoke(initial)
    return result


async def refine_existing(
    initial_draft: AdDraft,
    brief_state: BriefState,
    raw_brief: str,
    feedback: str,
    mode: str = "text",
) -> dict:
    """Run S2→S3 only on an existing draft — skips S0+S1 to avoid double generation.

    Cuts claude subprocess calls roughly in half vs a full second run_agent call.
    Returns { drafts, platform_issues, iteration }.
    """
    state: GraphState = {
        "raw_brief":       raw_brief,
        "brief":           brief_state,
        "mode":            mode,
        "drafts":          [initial_draft],
        "feedback":        feedback,
        "iteration":       1,
        "status":          "running",
        "eval_scores":     None,
        "saliency_hint":   None,
        "platform_issues": [],
    }
    state = s2_refine(state)
    state = s3_format(state)
    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_draft(text: str) -> AdDraft:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Agent returned non-JSON draft: {raw[:120]}")
        d = {"headline": raw[:80], "body": "", "cta": "Learn more", "image_prompt": None}
    return {
        "headline":     str(d.get("headline", "")),
        "body":         str(d.get("body", "")),
        "cta":          str(d.get("cta", "Learn more")),
        "image_prompt": d.get("image_prompt"),
        "image_url":    d.get("image_url"),
    }
