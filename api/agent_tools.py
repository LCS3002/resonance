"""Tools for the LangGraph ad generation agent.

image_gen: Claude Haiku generates an SVG ad mockup — no external image API required.
Hackathon-safe: works fully offline, zero cost beyond the Claude token call.
"""

import base64
import re

from .claude_local import local_claude as _claude

PLATFORM_CONSTRAINTS: dict[str, dict[str, int]] = {
    "facebook":  {"headline_max": 40,  "body_max": 125},
    "instagram": {"headline_max": 125, "body_max": 300},
    "twitter":   {"headline_max": 70,  "body_max": 280},
    "generic":   {"headline_max": 80,  "body_max": 200},
}

_EMOTION_PALETTES = {
    "aspirational": "blues and warm golds, upward diagonal gradients",
    "trustworthy":  "deep navy and clean whites, steady horizontal gradients",
    "urgent":       "high-contrast red-orange and black, sharp diagonal cuts",
    "playful":      "bright saturated colours, curved shapes and circles",
    "premium":      "near-black with gold accents, subtle texture gradients",
}


def _svg_prompt(image_prompt: str, target_emotion: str) -> str:
    palette = _EMOTION_PALETTES.get(target_emotion, "clean neutrals with brand accent colour")
    return (
        f"Create a minimal SVG ad visual for: {image_prompt}\n\n"
        f"Colour palette: {palette}\n\n"
        "Rules:\n"
        "- Exactly viewBox=\"0 0 300 250\"\n"
        "- Use <defs> with at least one <linearGradient>\n"
        "- 3-5 geometric shapes (rect, circle, ellipse, path) to suggest the scene\n"
        "- NO <text> elements\n"
        "- Modern, clean, ad-agency aesthetic\n\n"
        "Return ONLY the SVG XML starting with <svg, no markdown fences, no commentary."
    )


def _parse_svg(text: str) -> dict:
    raw = text.strip()
    match = re.search(r"<svg[\s\S]*?</svg>", raw, re.IGNORECASE)
    svg = match.group(0) if match else raw
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return {"image_url": data_uri, "svg_source": svg}


def image_gen(image_prompt: str, target_emotion: str = "") -> dict:
    """Sync SVG generation — used by LangGraph nodes (run in thread executor)."""
    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1800,
        messages=[{"role": "user", "content": _svg_prompt(image_prompt, target_emotion)}],
    )
    return _parse_svg(resp.content[0].text)


async def image_gen_async(image_prompt: str, target_emotion: str = "") -> dict:
    """Async SVG generation — use in FastAPI handlers and SSE generators."""
    resp = await _claude.messages.acreate(
        model="claude-haiku-4-5-20251001",
        max_tokens=1800,
        messages=[{"role": "user", "content": _svg_prompt(image_prompt, target_emotion)}],
    )
    return _parse_svg(resp.content[0].text)


def check_platform_constraints(
    headline: str,
    body: str,
    platform: str,
) -> list[str]:
    """Return list of constraint violations for the given platform."""
    limits = PLATFORM_CONSTRAINTS.get(platform, PLATFORM_CONSTRAINTS["generic"])
    issues = []
    if len(headline) > limits["headline_max"]:
        issues.append(f"Headline {len(headline)} chars (max {limits['headline_max']} for {platform})")
    if len(body) > limits["body_max"]:
        issues.append(f"Body {len(body)} chars (max {limits['body_max']} for {platform})")
    return issues
