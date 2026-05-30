"""LangGraph ad generation agent — GAN-like loop.

States:
  S0  Parse     (Haiku)   — raw brief → BriefState
  S1  Concept   (Sonnet)  — creative anchor: emotional_core + visual_metaphor + tone
  S2a Copy      (Sonnet)  — generate headline/body/cta from concept         ┐ parallel
  S2b Image     (Haiku)   — generate image_prompt + SVG from concept        ┘
  S3  Eval               — GoEmotions on copy + CLIP on image → combined gradient feedback
  S4a RefineCopy (Sonnet) — revise copy anchored to concept + gradient hint ┐ parallel
  S4b RefineImg  (Haiku)  — revise image_prompt + SVG anchored to concept   ┘
  S5  Format    (Haiku)   — validate platform constraints, assemble FinalAd

The concept (S1) is the invariant both copy and image are always constrained by.
Drift is impossible: S4a/S4b refine framing only, never the concept.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agent_tools import check_platform_constraints, image_gen
from .claude_local import local_claude as _claude
from .prompts.loader import render as _render

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

# ── State types ────────────────────────────────────────────────────────────────

class BriefState(TypedDict):
    brand_name:     str
    brand_mission:  str
    platform:       str
    target_emotion: str   # aspirational|trustworthy|urgent|playful|premium
    raw_brief:      str


class ConceptState(TypedDict):
    emotional_core:   str   # one-sentence emotional truth ("freedom of sorting your finances")
    visual_metaphor:  str   # visual anchor ("open road, warm morning light")
    tone:             str   # e.g. "aspirational but grounded"


class CopyDraft(TypedDict):
    headline: str
    body:     str
    cta:      str


class ImageDraft(TypedDict):
    image_prompt: str
    image_url:    str | None   # base64 SVG data URI


class EvalResult(TypedDict):
    # GoEmotions on copy text
    text_emotion:       str
    text_confidence:    float
    text_target_score:  float   # score of target emotion label specifically
    text_goemotion_scores: dict[str, float]
    text_gradient_hint: str     # token attribution → agent-ready feedback

    # CLIP on SVG image
    image_emotion:       str
    image_target_score:  float
    image_clip_scores:   dict[str, float]

    # GoEmotions gradient on image_prompt text
    image_gradient_hint: str

    # Combined weighted feedback block
    combined_feedback:   str


class GraphState(TypedDict):
    raw_brief:               str
    mode:                    str             # "text" | "text+image"
    target_emotion_override: str             # "" | "infer" | one of 7 valid emotions
    brief:                   BriefState | None
    concept:                 ConceptState | None
    copy:                    CopyDraft | None
    image:                   ImageDraft | None
    eval:                    EvalResult | None
    refined_copy:            CopyDraft | None
    refined_image:           ImageDraft | None
    emotion_rationale:       str | None      # set by s0b_infer when target_emotion_override="infer"
    platform_issues:         list[str]
    iteration:               int
    status:                  str             # "running" | "done"


# ── Node implementations ───────────────────────────────────────────────────────

def s0_parse(state: GraphState) -> dict:
    """Parse raw brief → BriefState. Haiku — fast."""
    logger.info("S0 parse — brief: %s", state["raw_brief"][:80])
    system, user = _render("s0_parse", raw_brief=state["raw_brief"])
    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    brief = _parse_json(resp.content[0].text)
    brief.setdefault("raw_brief", state["raw_brief"])

    # If the frontend sent a specific emotion (not "infer", not empty), honour it
    override = state.get("target_emotion_override", "")
    if override and override != "infer":
        brief["target_emotion"] = override

    logger.info("S0 done — brand=%s platform=%s emotion=%s",
                brief.get("brand_name"), brief.get("platform"), brief.get("target_emotion"))
    return {"brief": brief, "status": "running"}


def s0b_infer(state: GraphState) -> dict:
    """Infer the best target emotion from the brand + audience profile. Sonnet."""
    logger.info("S0b infer — deducing target emotion for brand=%s", (state["brief"] or {}).get("brand_name"))
    brief = state["brief"] or {}
    system, user = _render(
        "s0b_infer",
        brand_name=brief.get("brand_name", ""),
        brand_mission=brief.get("brand_mission", ""),
        raw_brief=state["raw_brief"],
    )
    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    result = _parse_json(resp.content[0].text)
    emotion   = result.get("emotion", "")
    rationale = result.get("rationale", "")
    if emotion:
        brief = {**brief, "target_emotion": emotion}
    logger.info("S0b done — inferred_emotion=%s", emotion)
    return {"brief": brief, "emotion_rationale": rationale}


def _route_after_s0(state: GraphState) -> str:
    """Route to s0b_infer if emotion inference was requested, else straight to s1_concept."""
    return "s0b_infer" if state.get("target_emotion_override") == "infer" else "s1_concept"


def s1_concept(state: GraphState) -> dict:
    """Generate the creative concept anchor. Sonnet — quality matters here."""
    logger.info("S1 concept — target_emotion=%s", (state["brief"] or {}).get("target_emotion"))
    brief = state["brief"] or {}
    system, user = _render(
        "s1_concept",
        raw_brief=state["raw_brief"],
        brand_name=brief.get("brand_name", ""),
        target_emotion=brief.get("target_emotion", ""),
    )
    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    concept = _parse_json(resp.content[0].text)
    logger.info("S1 done — core=%s", concept.get("emotional_core", "")[:60])
    return {"concept": concept}


def s2a_copy(state: GraphState) -> dict:
    """Generate ad copy anchored to the concept. Sonnet."""
    logger.info("S2a copy — generating headline/body/cta")
    brief   = state["brief"] or {}
    concept = state["concept"] or {}
    system, user = _render(
        "s2a_copy",
        emotional_core=concept.get("emotional_core", ""),
        tone=concept.get("tone", ""),
        brand_name=brief.get("brand_name", ""),
        platform=brief.get("platform", "generic"),
        target_emotion=brief.get("target_emotion", ""),
    )
    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    copy = _parse_json(resp.content[0].text)
    logger.info("S2a done — headline=%s", copy.get("headline", "")[:50])
    return {"copy": copy}


def s2b_image(state: GraphState) -> dict:
    """Generate image_prompt and SVG anchored to the concept. Haiku."""
    logger.info("S2b image — generating image_prompt%s", " + SVG" if state.get("mode") == "text+image" else "")
    brief   = state["brief"] or {}
    concept = state["concept"] or {}
    mode    = state.get("mode", "text")

    system, user = _render(
        "s2b_image",
        emotional_core=concept.get("emotional_core", ""),
        visual_metaphor=concept.get("visual_metaphor", ""),
        target_emotion=brief.get("target_emotion", ""),
    )
    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    image_prompt = _parse_json(resp.content[0].text).get("image_prompt", "")

    image_url = None
    if mode == "text+image" and image_prompt:
        result    = image_gen(image_prompt, brief.get("target_emotion", ""))
        image_url = result["image_url"]

    logger.info("S2b done — image_url=%s", "SVG ready" if image_url else "text-only")
    return {"image": {"image_prompt": image_prompt, "image_url": image_url}}


def s3_eval(state: GraphState) -> dict:
    """Evaluate copy (GoEmotions) and image (CLIP) — produce combined gradient feedback."""
    logger.info("S3 eval — scoring copy + image for target=%s", (state["brief"] or {}).get("target_emotion"))
    from .scoring.goemotion_scorer import GoEmotionScorer, TARGET_LABELS, PROFILE_TO_GOEMOTION
    from .scoring.clip_scorer import CLIPScorer
    from .scoring.saliency import get_token_saliency, build_feedback_hint
    from .scoring.emotion import build_combined_feedback

    brief   = state["brief"] or {}
    copy    = state["copy"] or {}
    image   = state["image"] or {}
    target  = brief.get("target_emotion", "")

    goemotion = GoEmotionScorer()
    clip      = CLIPScorer()

    # ── Text path ────────────────────────────────────────────────────────────
    copy_text = f"{copy.get('headline', '')}. {copy.get('body', '')}"
    ge_scores = goemotion.classify(copy_text)
    text_emotion, text_conf = goemotion.predict(copy_text)

    # Score of the specific target label
    ge_label       = (PROFILE_TO_GOEMOTION.get(target) or [""])[0]
    text_target_sc = ge_scores.get(ge_label, 0.0)

    # Gradient attribution on copy text for target emotion
    text_gradient_hint = ""
    target_idx = TARGET_LABELS.get(ge_label)
    if goemotion.is_available() and target_idx is not None:
        model, tok  = goemotion.get_model_and_tokenizer()
        pairs       = get_token_saliency(copy_text, target_idx, model, tok)
        text_gradient_hint = build_feedback_hint(copy_text, ge_label, pairs, ge_scores)

    # ── Image path ───────────────────────────────────────────────────────────
    image_clip_scores  = {}
    image_emotion      = target or "aspirational"
    image_target_score = 0.2
    image_gradient_hint = ""

    image_url = image.get("image_url")
    if clip.is_available() and image_url:
        image_clip_scores  = clip.score_image_url(image_url)
        image_emotion, _   = clip.top_emotion(image_clip_scores)
        image_target_score = image_clip_scores.get(target, 0.2)

    # Gradient on image_prompt text (what the agent can actually control)
    image_prompt = image.get("image_prompt", "")
    if goemotion.is_available() and image_prompt and target_idx is not None:
        model, tok  = goemotion.get_model_and_tokenizer()
        pairs       = get_token_saliency(image_prompt, target_idx, model, tok)
        image_gradient_hint = build_feedback_hint(image_prompt, ge_label, pairs, ge_scores)

    # ── Combine ──────────────────────────────────────────────────────────────
    combined = build_combined_feedback(
        target_emotion=target,
        text_hint=text_gradient_hint,
        image_hint=image_gradient_hint,
        text_score=text_target_sc,
        image_score=image_target_score,
    )

    logger.info("S3 done — text_emotion=%s (%.2f) image_emotion=%s (%.2f)",
                text_emotion, text_target_sc, image_emotion, image_target_score)
    return {"eval": {
        "text_emotion":          text_emotion,
        "text_confidence":       round(text_conf, 3),
        "text_target_score":     round(text_target_sc, 3),
        "text_goemotion_scores": ge_scores,
        "text_gradient_hint":    text_gradient_hint,
        "image_emotion":         image_emotion,
        "image_target_score":    round(image_target_score, 3),
        "image_clip_scores":     image_clip_scores,
        "image_gradient_hint":   image_gradient_hint,
        "combined_feedback":     combined,
    }}


def s4a_refine_copy(state: GraphState) -> dict:
    """Revise copy with gradient feedback. Concept is the hard constraint. Sonnet."""
    logger.info("S4a refine copy — applying gradient feedback")
    brief    = state["brief"] or {}
    concept  = state["concept"] or {}
    copy     = state["copy"] or {}
    feedback = (state["eval"] or {}).get("combined_feedback", "")

    system, user = _render(
        "s4a_refine_copy",
        emotional_core=concept.get("emotional_core", ""),
        tone=concept.get("tone", ""),
        brand_name=brief.get("brand_name", ""),
        platform=brief.get("platform", "generic"),
        prev_headline=copy.get("headline", ""),
        prev_body=copy.get("body", ""),
        prev_cta=copy.get("cta", ""),
        feedback=feedback[:800],
    )
    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    refined = _parse_json(resp.content[0].text)
    logger.info("S4a done — headline=%s", refined.get("headline", "")[:50])
    return {"refined_copy": refined}


def s4b_refine_image(state: GraphState) -> dict:
    """Revise image_prompt with gradient feedback. Concept is the hard constraint. Haiku."""
    logger.info("S4b refine image — revising image_prompt")
    brief    = state["brief"] or {}
    concept  = state["concept"] or {}
    image    = state["image"] or {}
    feedback = (state["eval"] or {}).get("combined_feedback", "")
    mode     = state.get("mode", "text")

    system, user = _render(
        "s4b_refine_image",
        emotional_core=concept.get("emotional_core", ""),
        visual_metaphor=concept.get("visual_metaphor", ""),
        target_emotion=brief.get("target_emotion", ""),
        prev_image_prompt=image.get("image_prompt", ""),
        feedback=feedback[:400],
    )
    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system or None,
        messages=[{"role": "user", "content": user}],
    )
    new_prompt = _parse_json(resp.content[0].text).get("image_prompt", image.get("image_prompt", ""))

    image_url = image.get("image_url")  # keep old SVG unless prompt changed meaningfully
    if mode == "text+image" and new_prompt and new_prompt != image.get("image_prompt", ""):
        result    = image_gen(new_prompt, brief.get("target_emotion", ""))
        image_url = result["image_url"]

    logger.info("S4b done — image_url=%s", "new SVG" if image_url != (image or {}).get("image_url") else "reused")
    return {"refined_image": {"image_prompt": new_prompt, "image_url": image_url}}


def s5_format(state: GraphState) -> dict:
    """Validate platform constraints. Trim if needed. Assemble FinalAd. Haiku."""
    logger.info("S5 format — checking platform=%s constraints", (state["brief"] or {}).get("platform"))
    brief   = state["brief"] or {}
    platform = brief.get("platform", "generic")

    # Use refined drafts if available, else fall back to initial
    final_copy  = state.get("refined_copy")  or state.get("copy")  or {}
    final_image = state.get("refined_image") or state.get("image") or {}

    issues = check_platform_constraints(
        final_copy.get("headline", ""),
        final_copy.get("body", ""),
        platform,
    )

    if issues:
        constraint_str = "; ".join(issues)
        system, user = _render(
            "s5_format",
            constraint_str=constraint_str,
            headline=final_copy.get("headline", ""),
            body=final_copy.get("body", ""),
            cta=final_copy.get("cta", ""),
        )
        resp = _claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system or None,
            messages=[{"role": "user", "content": user}],
        )
        trimmed   = _parse_json(resp.content[0].text)
        final_copy = {**final_copy, **trimmed}
        issues     = []

    logger.info("S5 done — issues=%s status=done", issues or "none")
    return {
        "refined_copy":   final_copy,
        "refined_image":  final_image,
        "platform_issues": issues,
        "iteration":      state.get("iteration", 0) + 1,
        "status":         "done",
    }


# ── Parallel S2 and S4 wrappers ───────────────────────────────────────────────

def s2_parallel(state: GraphState) -> dict:
    """Run S2a (copy) and S2b (image) in parallel threads, merge results."""
    fut_copy  = _executor.submit(s2a_copy, state)
    fut_image = _executor.submit(s2b_image, state)
    copy_out  = fut_copy.result()
    image_out = fut_image.result()
    return {**copy_out, **image_out}


def s4_parallel(state: GraphState) -> dict:
    """Run S4a (refine copy) and S4b (refine image) in parallel threads."""
    fut_copy  = _executor.submit(s4a_refine_copy, state)
    fut_image = _executor.submit(s4b_refine_image, state)
    copy_out  = fut_copy.result()
    image_out = fut_image.result()
    return {**copy_out, **image_out}


# ── Graph wiring ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("s0_parse",    s0_parse)
    g.add_node("s0b_infer",   s0b_infer)
    g.add_node("s1_concept",  s1_concept)
    g.add_node("s2_parallel", s2_parallel)
    g.add_node("s3_eval",     s3_eval)
    g.add_node("s4_parallel", s4_parallel)
    g.add_node("s5_format",   s5_format)

    g.set_entry_point("s0_parse")
    g.add_conditional_edges("s0_parse", _route_after_s0, {
        "s0b_infer":  "s0b_infer",
        "s1_concept": "s1_concept",
    })
    g.add_edge("s0b_infer",   "s1_concept")
    g.add_edge("s1_concept",  "s2_parallel")
    g.add_edge("s2_parallel", "s3_eval")
    g.add_edge("s3_eval",     "s4_parallel")
    g.add_edge("s4_parallel", "s5_format")
    g.add_edge("s5_format",   END)
    return g.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def _initial_state(raw_brief: str, mode: str, target_emotion: str = "") -> GraphState:
    return {
        "raw_brief":               raw_brief,
        "mode":                    mode,
        "target_emotion_override": target_emotion,
        "brief":                   None,
        "concept":                 None,
        "copy":                    None,
        "image":                   None,
        "eval":                    None,
        "refined_copy":            None,
        "refined_image":           None,
        "emotion_rationale":       None,
        "platform_issues":         [],
        "iteration":               0,
        "status":                  "running",
    }


# Human-readable labels for each node — sent to the frontend
_NODE_LABELS: dict[str, str] = {
    "s0_parse":    "Parsing brief",
    "s0b_infer":   "Inferring target emotion",
    "s1_concept":  "Building creative concept",
    "s2_parallel": "Generating initial copy + image",
    "s3_eval":     "Evaluating emotions + gradients",
    "s4_parallel": "Refining with feedback",
    "s5_format":   "Formatting final ad",
}

# Keys from each node's output that are worth sending to the frontend
_NODE_FIELDS: dict[str, list[str]] = {
    "s0_parse":    ["brief"],
    "s0b_infer":   ["brief", "emotion_rationale"],
    "s1_concept":  ["concept"],
    "s2_parallel": ["copy", "image"],
    "s3_eval":     ["eval"],
    "s4_parallel": ["refined_copy", "refined_image"],
    "s5_format":   ["refined_copy", "refined_image", "platform_issues", "iteration"],
}


async def stream_agent(raw_brief: str, mode: str = "text", target_emotion: str = ""):
    """Async generator — yields one SSE-ready dict per node that completes.

    Each dict: { type, node, label, data }
    Final dict: { type: "done", data: <full final state> }
    """
    initial = _initial_state(raw_brief, mode, target_emotion)
    final_state = initial.copy()

    async for chunk in _graph.astream(initial, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            final_state.update(node_output)
            fields = _NODE_FIELDS.get(node_name, [])
            payload = {k: node_output.get(k) for k in fields if k in node_output}
            yield {
                "type":  "node",
                "node":  node_name,
                "label": _NODE_LABELS.get(node_name, node_name),
                "data":  payload,
            }

    yield {"type": "done", "data": final_state}


async def run_agent(raw_brief: str, mode: str = "text", target_emotion: str = "") -> GraphState:
    """Run the full S0→[S0b]→S1→S2→S3→S4→S5 pipeline. Returns final GraphState."""
    return await _graph.ainvoke(_initial_state(raw_brief, mode, target_emotion))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    raw = text.strip()
    # Extract from ```json fences first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    else:
        # Find the outermost JSON object — handles preamble / chain-of-thought before the JSON
        obj = re.search(r"\{[\s\S]*\}", raw)
        if obj:
            raw = obj.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON response (first 200): %s", raw[:200])
        return {}
