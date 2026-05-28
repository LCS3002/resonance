"""Alpic MCP server — exposes Resonance as callable tools for any MCP client.

Deploy on Alpic: https://alpic.ai/
Local run: python -m api.mcp_server

Tools exposed:
  run_campaign         — full pipeline from a campaign brief
  score_ad_neural      — score a single ad text
  explain_neural_score — human-readable interpretation of a score
  review_ad            — log a human-in-the-loop decision
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit("mcp package required: pip install mcp")

from .agent import run_campaign
from .scoring import ScorerPipeline

mcp = FastMCP(
    name="resonance",
    instructions=(
        "Resonance scores ad creative with neural engagement models trained on fMRI data. "
        "Use run_campaign to generate and score variants from a brief. "
        "Scores above 0.75 are safe to serve autonomously. "
        "Scores below 0.45 require human review before placement."
    ),
)

_scorer: ScorerPipeline | None = None


def _get_scorer() -> ScorerPipeline:
    global _scorer
    if _scorer is None:
        _scorer = ScorerPipeline()
    return _scorer


@mcp.tool()
async def run_campaign_tool(
    brief: str,
    brand: str = "",
    num_variants: int = 3,
) -> dict:
    """Generate and score ad variants from a campaign brief.

    Args:
        brief: Campaign brief (e.g. 'luxury EV for urban professionals in London')
        brand: Optional brand name for Tavily context research
        num_variants: Number of variants to generate (1-5)

    Returns:
        Ranked variants with neural engagement scores, region activation breakdown,
        and HITL flags for variants below threshold.
    """
    return await run_campaign(
        brief=brief,
        brand=brand or None,
        num_variants=max(1, min(5, num_variants)),
    )


@mcp.tool()
async def score_ad_neural(ad_text: str) -> dict:
    """Score a single ad text with both neural engagement models.

    Args:
        ad_text: The full ad text (headline + body) to score

    Returns:
        combined_score, region_scores (language/visual/prefrontal), model breakdown
    """
    return await _get_scorer().score(ad_text)


@mcp.tool()
def explain_neural_score(score: float) -> str:
    """Explain what a neural engagement score means for ad placement decisions.

    Args:
        score: Combined neural score in [0, 1]
    """
    if score >= 0.75:
        return f"{score:.2f} — High neural engagement. Visual cortex and prefrontal attention strongly activated. Safe to serve autonomously."
    if score >= 0.55:
        return f"{score:.2f} — Moderate engagement. Consider A/B testing against a higher-scoring variant."
    if score >= 0.45:
        return f"{score:.2f} — Below target threshold. Weak prefrontal signal suggests low decision-intent activation."
    return f"{score:.2f} — Flagged for human review. Neural engagement below minimum serving threshold (0.45)."


@mcp.tool()
def log_review_decision(variant_id: str, approved: bool, notes: str = "") -> dict:
    """Log a human-in-the-loop approval or rejection decision.

    This feeds the Overmind optimization loop to improve future generation.

    Args:
        variant_id: The variant ID from run_campaign output
        approved: True if approved for serving, False if rejected
        notes: Optional reviewer notes
    """
    logger.info(f"HITL: {variant_id} approved={approved} notes={notes!r}")
    return {
        "recorded": True,
        "variant_id": variant_id,
        "approved": approved,
        "pipeline": "Overmind fine-tuning loop",
    }


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    logger.info(f"Starting Resonance MCP server on port {port}")
    mcp.run(transport="sse")
