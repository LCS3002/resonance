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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agent_tools import check_platform_constraints, image_gen
from .claude_local import local_claude as _claude

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
    raw_brief:      str
    mode:           str             # "text" | "text+image"
    brief:          BriefState | None
    concept:        ConceptState | None
    copy:           CopyDraft | None
    image:          ImageDraft | None
    eval:           EvalResult | None
    refined_copy:   CopyDraft | None
    refined_image:  ImageDraft | None
    platform_issues: list[str]
    iteration:      int
    status:         str             # "running" | "done"


# ── Node implementations ───────────────────────────────────────────────────────

def s0_parse(state: GraphState) -> dict:
    """Parse raw brief → BriefState. Haiku — fast."""
    logger.info("S0 parse — brief: %s", state["raw_brief"][:80])
    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": (
            f"Parse this ad campaign brief into structured JSON:\n\n\"{state['raw_brief']}\"\n\n"
            "Return ONLY valid JSON with these exact keys:\n"
            "{\n"
            '  "brand_name": "brand or product name (infer if not explicit)",\n'
            '  "brand_mission": "one sentence brand purpose",\n'
            '  "platform": "facebook|instagram|twitter|generic",\n'
            '  "target_emotion": "aspirational|trustworthy|urgent|playful|premium",\n'
            f'  "raw_brief": "{state["raw_brief"]}"\n'
            "}"
        )}],
    )
    brief = _parse_json(resp.content[0].text)
    brief.setdefault("raw_brief", state["raw_brief"])
    logger.info("S0 done — brand=%s platform=%s emotion=%s",
                brief.get("brand_name"), brief.get("platform"), brief.get("target_emotion"))
    return {"brief": brief, "status": "running"}


def s1_concept(state: GraphState) -> dict:
    """Generate the creative concept anchor. Sonnet — quality matters here."""
    logger.info("S1 concept — target_emotion=%s", (state["brief"] or {}).get("target_emotion"))
    brief = state["brief"] or {}
    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "You are a creative director. Distil a campaign brief into a tight creative concept "
            "that will govern both ad copy and visuals. The concept is the invariant — "
            "copy and images are both expressions of it."
        ),
        messages=[{"role": "user", "content": (
            f"Brief: {state['raw_brief']}\n"
            f"Brand: {brief.get('brand_name', '')}\n"
            f"Target emotion: {brief.get('target_emotion', '')}\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "emotional_core": "one sentence — the feeling this ad should leave the viewer with",\n'
            '  "visual_metaphor": "concrete visual anchor, e.g. open road at golden hour",\n'
            '  "tone": "2-4 words describing the voice and register"\n'
            "}"
        )}],
    )
    concept = _parse_json(resp.content[0].text)
    logger.info("S1 done — core=%s", concept.get("emotional_core", "")[:60])
    return {"concept": concept}


def s2a_copy(state: GraphState) -> dict:
    """Generate ad copy anchored to the concept. Sonnet."""
    logger.info("S2a copy — generating headline/body/cta")
    brief   = state["brief"] or {}
    concept = state["concept"] or {}

    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system="You are an expert ad copywriter. No exclamation marks. No clichés.",
        messages=[{"role": "user", "content": (
            f"Creative concept: {concept.get('emotional_core', '')}\n"
            f"Tone: {concept.get('tone', '')}\n"
            f"Brand: {brief.get('brand_name', '')}\n"
            f"Platform: {brief.get('platform', 'generic')}\n"
            f"Target emotion: {brief.get('target_emotion', '')}\n\n"
            "Write ONE ad variant. Return ONLY JSON:\n"
            '{"headline": "6-10 words, active verb", '
            '"body": "15-25 words, one specific benefit, conversational", '
            '"cta": "2-4 words, imperative"}'
        )}],
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

    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": (
            f"Creative concept: {concept.get('emotional_core', '')}\n"
            f"Visual metaphor: {concept.get('visual_metaphor', '')}\n"
            f"Target emotion: {brief.get('target_emotion', '')}\n\n"
            "Write a vivid image_prompt (1-2 sentences, no text in image). "
            "Return ONLY JSON:\n"
            '{"image_prompt": "..."}'
        )}],
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

    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=(
            "You are an expert ad copywriter. Revise copy based on emotion feedback. "
            "The creative concept is the hard constraint — keep it. Change only framing."
        ),
        messages=[{"role": "user", "content": (
            f"Creative concept (DO NOT change this): {concept.get('emotional_core', '')}\n"
            f"Tone: {concept.get('tone', '')}\n"
            f"Brand: {brief.get('brand_name', '')}\n"
            f"Platform: {brief.get('platform', 'generic')}\n\n"
            f"Previous copy:\n"
            f"  Headline: {copy.get('headline', '')}\n"
            f"  Body: {copy.get('body', '')}\n"
            f"  CTA: {copy.get('cta', '')}\n\n"
            f"Emotion feedback:\n{feedback[:800]}\n\n"
            "Rewrite. Return ONLY JSON:\n"
            '{"headline": "...", "body": "...", "cta": "..."}'
        )}],
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

    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": (
            f"Creative concept (DO NOT change this): {concept.get('emotional_core', '')}\n"
            f"Visual metaphor: {concept.get('visual_metaphor', '')}\n"
            f"Target emotion: {brief.get('target_emotion', '')}\n\n"
            f"Previous image_prompt: {image.get('image_prompt', '')}\n\n"
            f"Emotion feedback:\n{feedback[:400]}\n\n"
            "Revise the image_prompt to better hit the target emotion. "
            "Keep the visual metaphor. Return ONLY JSON:\n"
            '{"image_prompt": "..."}'
        )}],
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
        resp = _claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": (
                f"Trim this ad copy to fix: {constraint_str}\n\n"
                f"Headline: {final_copy.get('headline', '')}\n"
                f"Body: {final_copy.get('body', '')}\n"
                f"CTA: {final_copy.get('cta', '')}\n\n"
                'Return ONLY JSON: {"headline": "...", "body": "...", "cta": "..."}'
            )}],
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
    g.add_node("s1_concept",  s1_concept)
    g.add_node("s2_parallel", s2_parallel)   # copy + image in parallel
    g.add_node("s3_eval",     s3_eval)
    g.add_node("s4_parallel", s4_parallel)   # refine copy + image in parallel
    g.add_node("s5_format",   s5_format)

    g.set_entry_point("s0_parse")
    g.add_edge("s0_parse",    "s1_concept")
    g.add_edge("s1_concept",  "s2_parallel")
    g.add_edge("s2_parallel", "s3_eval")
    g.add_edge("s3_eval",     "s4_parallel")
    g.add_edge("s4_parallel", "s5_format")
    g.add_edge("s5_format",   END)
    return g.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def _initial_state(raw_brief: str, mode: str) -> GraphState:
    return {
        "raw_brief":      raw_brief,
        "mode":           mode,
        "brief":          None,
        "concept":        None,
        "copy":           None,
        "image":          None,
        "eval":           None,
        "refined_copy":   None,
        "refined_image":  None,
        "platform_issues": [],
        "iteration":      0,
        "status":         "running",
    }


# Human-readable labels for each node — sent to the frontend
_NODE_LABELS: dict[str, str] = {
    "s0_parse":    "Parsing brief",
    "s1_concept":  "Building creative concept",
    "s2_parallel": "Generating initial copy + image",
    "s3_eval":     "Evaluating emotions + gradients",
    "s4_parallel": "Refining with feedback",
    "s5_format":   "Formatting final ad",
}

# Keys from each node's output that are worth sending to the frontend
_NODE_FIELDS: dict[str, list[str]] = {
    "s0_parse":    ["brief"],
    "s1_concept":  ["concept"],
    "s2_parallel": ["copy", "image"],
    "s3_eval":     ["eval"],
    "s4_parallel": ["refined_copy", "refined_image"],
    "s5_format":   ["refined_copy", "refined_image", "platform_issues", "iteration"],
}


async def stream_agent(raw_brief: str, mode: str = "text"):
    """Async generator — yields one SSE-ready dict per node that completes.

    Each dict: { type, node, label, data }
    Final dict: { type: "done", data: <full final state> }
    """
    initial = _initial_state(raw_brief, mode)
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


async def run_agent(raw_brief: str, mode: str = "text") -> GraphState:
    """Run the full S0→S1→S2→S3→S4→S5 pipeline. Returns final GraphState."""
    return await _graph.ainvoke(_initial_state(raw_brief, mode))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Non-JSON response (first 120): {raw[:120]}")
        return {}
