"""TribeV2 (Meta) neural engagement scorer.

Wraps the TribeModel from the tribe-v2-reference repo.
Falls back to MockScorer if weights are unavailable.

Reference: facebook/tribev2 on HuggingFace
Paper: Decoding Brain Activity with Meta AI
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

from .base import BaseScorer, ScoreResult
from .mock_scorer import MockScorer

logger = logging.getLogger(__name__)

# Path to the reference repo (relative to project root)
_TRIBE_REPO = Path(__file__).parents[3] / "Tribe V2 and Neural Predictions" / "tribe-v2-reference"

# fsaverage5 region vertex ranges (rough splits used to extract region scores)
# These correspond to the normalised y-coordinate thresholds used in the frontend
_REGION_SPLITS = {
    "visual":     {"y_max": -0.30},   # posterior
    "language":   {"y_min": -0.10, "y_max": 0.50, "left_only": True},
    "prefrontal": {"y_min": 0.55},    # anterior-superior
}


def _mean_activation(preds, region: str, n_left: int) -> float:
    """Extract mean activation for a named region from (n_vertices,) array."""
    import numpy as np

    n = len(preds)
    indices = np.arange(n)
    if region == "visual":
        # Keep roughly posterior quarter of vertices — crude but sufficient for scoring
        mask = indices > int(n * 0.6)
    elif region == "language":
        mask = (indices < n_left) & (indices > int(n_left * 0.25)) & (indices < int(n_left * 0.75))
    elif region == "prefrontal":
        mask = (indices < n_left) & (indices < int(n_left * 0.2))
    else:
        mask = np.ones(n, dtype=bool)

    vals = preds[mask]
    if len(vals) == 0:
        return 0.0
    # Normalise to [0,1]: assume preds are in roughly [-3, 3] z-score range
    return float(np.clip((vals.mean() + 1.5) / 3.0, 0, 1))


class TribeScorer(BaseScorer):
    """Score ad text using TribeV2 brain encoding model."""

    def __init__(self, checkpoint_dir: str = "facebook/tribev2", cache_folder: str = ".cache/tribe"):
        self._model = None
        self._available = False
        self._mock = MockScorer(model_name="tribe-mock")
        self._cache = cache_folder

        if str(_TRIBE_REPO) not in sys.path:
            sys.path.insert(0, str(_TRIBE_REPO))

        try:
            from tribev2.demo_utils import TribeModel  # noqa: F401
            self._TribeModel = TribeModel
            logger.info("TribeV2 module imported successfully")
        except ImportError as e:
            logger.warning(f"TribeV2 import failed ({e}) — using mock scorer")
            return

        try:
            logger.info(f"Loading TribeV2 from {checkpoint_dir} ...")
            self._model = TribeModel.from_pretrained(
                checkpoint_dir,
                cache_folder=cache_folder,
                device="cpu",
            )
            self._available = True
            logger.info("TribeV2 loaded successfully")
        except Exception as e:
            logger.warning(f"TribeV2 checkpoint load failed ({e}) — using mock scorer")

    def is_available(self) -> bool:
        return self._available

    def score(self, text: str) -> ScoreResult:
        if not self._available:
            return self._mock.score(text)

        t0 = time.monotonic()
        try:
            import numpy as np

            with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
                f.write(text)
                txt_path = f.name

            events = self._model.get_events_dataframe(text_path=txt_path)
            preds, _ = self._model.predict(events, verbose=False)  # (n_segs, n_verts)

            mean_pred = preds.mean(axis=0)  # (n_verts,)
            n_verts = len(mean_pred)
            n_left  = n_verts // 2

            lang = _mean_activation(mean_pred, "language", n_left)
            vis  = _mean_activation(mean_pred, "visual",   n_left)
            pre  = _mean_activation(mean_pred, "prefrontal", n_left)

            return ScoreResult(
                score=float(np.clip((lang + vis + pre) / 3, 0, 1)),
                region_scores={"language": lang, "visual": vis, "prefrontal": pre},
                model="tribev2",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            logger.error(f"TribeV2 inference failed: {e}")
            return self._mock.score(text)
