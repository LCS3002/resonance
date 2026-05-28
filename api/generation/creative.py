"""Ad creative generation using Claude + Tavily brand grounding."""

import json
import logging
import os

logger = logging.getLogger(__name__)

from ..claude_local import local_claude as _claude

_SYSTEM = """You are an expert ad copywriter specialising in conversational AI placements
(ChatGPT-style in-chat sponsored answers). You write variants that are crisp, benefit-led,
and natural inside a chat context — not banner-ad language.

Rules:
- headline: 6-10 words, active verb, no exclamation marks
- body: 15-25 words, one specific benefit, conversational tone
- image_prompt: vivid DALL-E 3 prompt, photorealistic, no text in image
- cta: 2-4 words, imperative"""


async def search_brand_context(brand: str) -> str:
    """Tavily search for brand positioning context."""
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
        ctx = " | ".join(snippets)
        logger.info(f"Tavily returned {len(snippets)} snippets for brand '{brand}'")
        return ctx
    except Exception as e:
        logger.warning(f"Tavily search failed ({e}) — proceeding without brand context")
        return ""


async def generate_campaign_strategy(
    brief: str,
    top_variant: dict,
    region_scores: dict[str, float],
) -> str:
    """Use Claude to write a campaign strategy narrative grounded in neural scores."""

    regions_text = (
        f"Language cortex: {region_scores.get('language', 0):.0%}  |  "
        f"Visual cortex: {region_scores.get('visual', 0):.0%}  |  "
        f"Prefrontal (attention): {region_scores.get('prefrontal', 0):.0%}"
    )

    prompt = (
        f"Campaign brief: {brief}\n\n"
        f"Top-performing ad variant:\n"
        f"  Headline: {top_variant['headline']}\n"
        f"  Body: {top_variant['body']}\n\n"
        f"Neural engagement profile (from fMRI-trained model):\n{regions_text}\n\n"
        f"Write a concise campaign strategy in 3 bullet points (max 20 words each):\n"
        f"1. Target audience and emotional hook\n"
        f"2. Why this copy activates the dominant brain region\n"
        f"3. One specific optimisation to increase the lowest-scoring region\n\n"
        f"Format: plain text, three bullet points starting with •"
    )

    resp = _claude.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for strategy
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


async def generate_ad_image(image_prompt: str) -> str | None:
    """Generate a hero image via DALL-E 3. Returns URL or None if unavailable."""
    try:
        import openai
        client = openai.AsyncOpenAI()
        resp = await client.images.generate(
            model="dall-e-3",
            prompt=image_prompt + ". Photorealistic, no text, no overlaid words, clean product shot.",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = resp.data[0].url
        logger.info(f"DALL-E 3 image generated: {url[:60]}...")
        return url
    except ImportError:
        logger.info("openai package not installed — skipping image generation")
        return None
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        return None


async def generate_variants(
    brief: str,
    context: str = "",
    num_variants: int = 3,
    emotion_hint: str = "",
) -> list[dict]:
    """Call Claude to produce N distinct ad variants as structured JSON."""

    brand_block = f"\nBrand context (from live research):\n{context}" if context else ""

    emotion_block = f"\n{emotion_hint}" if emotion_hint else ""
    prompt = (
        f"Campaign brief: {brief}{brand_block}{emotion_block}\n\n"
        f"Generate {num_variants} distinct ad variants for a conversational AI placement.\n"
        f"Each variant must be meaningfully different in angle, not just wording.\n"
        f"Return ONLY a JSON array of {num_variants} objects with keys: "
        f"headline, body, image_prompt, cta. No markdown, no commentary."
    )

    resp = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
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
            "id": f"v{i + 1}",
            "headline":     str(v.get("headline", "")),
            "body":         str(v.get("body", "")),
            "image_prompt": str(v.get("image_prompt", "")),
            "cta":          str(v.get("cta", "Learn more")),
        }
        for i, v in enumerate(variants[:num_variants])
    ]
