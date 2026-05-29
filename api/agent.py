"""Singleton accessors for shared model instances.

Loaded once at startup; reused across requests.
"""

import logging
import os

from .scoring.goemotion_scorer import GoEmotionScorer
from .scoring.clip_scorer import CLIPScorer

logger = logging.getLogger(__name__)

# Overmind tracing — no-op if key absent
try:
    import overmind
    if os.getenv("OVERMIND_API_KEY"):
        overmind.init()
        logger.info("Overmind tracing active")
except ImportError:
    pass

_goemotion: GoEmotionScorer | None = None
_clip: CLIPScorer | None = None


def get_goemotion() -> GoEmotionScorer:
    global _goemotion
    if _goemotion is None:
        _goemotion = GoEmotionScorer()
    return _goemotion


def get_clip() -> CLIPScorer:
    global _clip
    if _clip is None:
        _clip = CLIPScorer()
    return _clip
