"""Ad creative generation — brand research + variant copy via Claude CLI."""

import json
import logging
import os

from ..claude_local import local_claude as _claude

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert ad copywriter specialising in conversational AI placements
(ChatGPT-style in-chat sponsored answers). You write variants that are crisp, benefit-led,
and natural inside a chat context — not banner-ad language.

Rules:
- headline: 6-10 words, active verb, no exclamation marks
- body: 15-25 words, one specific benefit, conversational tone
- image_prompt: vivid visual description of the ad scene, no text in image
- cta: 2-4 words, imperative"""


async def search_brand_context(brand: str) -> str:
    """Tavily search for brand positioning context. Returns '' if key missing."""
    key = os.getenv("TAVILY_API_KEY", "")
    if not key or not brand:
        return ""
    try:
        from tavily import TavilyClient
        tc = TavilyClient(api_key=key)
        res = tc.search(
            query=f"{brand} brand positioning values marketing 2025 2026",
            max_results=3,
            search_depth="basic",
        )
        snippets = [r["content"][:250] for r in res.get("results", [])]
        return " | ".join(snippets)
    except Exception as e:
        logger.warning(f"Tavily search failed ({e}) — proceeding without brand context")
        return ""


async def generate_variants(
    brief: str,
    context: str = "",
    num_variants: int = 3,
    emotion_hint: str = "",
) -> list[dict]:
    """Generate N distinct ad variants as structured JSON via Claude."""
    brand_block   = f"\nBrand context:\n{context}" if context else ""
    emotion_block = f"\n{emotion_hint}" if emotion_hint else ""

    prompt = (
        f"Campaign brief: {brief}{brand_block}{emotion_block}\n\n"
        f"Generate {num_variants} distinct ad variants for a conversational AI placement.\n"
        f"Each variant must be meaningfully different in angle, not just wording.\n"
        f"Return ONLY a JSON array of {num_variants} objects with keys: "
        f"headline, body, image_prompt, cta. No markdown, no commentary."
    )

    resp = await _claude.messages.acreate(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Claude returned invalid JSON: {raw[:200]}")
        raise ValueError("Claude generation returned invalid JSON")

    return [
        {
            "id":           f"v{i + 1}",
            "headline":     str(v.get("headline", "")),
            "body":         str(v.get("body", "")),
            "image_prompt": str(v.get("image_prompt", "")),
            "cta":          str(v.get("cta", "Learn more")),
        }
        for i, v in enumerate(variants[:num_variants])
    ]
