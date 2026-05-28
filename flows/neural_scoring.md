# Flow: Neural Scoring

**Called from:** `api/agent.py:96` — `await scorer.score(score_input)`

Input: `score_input = f"{v['headline']}. {v['body']}"`

---

## ScorerPipeline — `api/scoring/pipeline.py:28`
```
ScorerPipeline.score(text)
  │
  ├─ [parallel via ThreadPoolExecutor]
  │   ├─ AlgonautsScorer.score(text)    api/scoring/algonauts_scorer.py   weight: 50%
  │   ├─ TribeScorer.score(text)        api/scoring/tribe_scorer.py       weight: 30%
  │   └─ CustomScorer.score(text)       api/scoring/custom_scorer.py      weight: 20%
  │
  └─ Weighted combination → combined_score + region_scores
```

Each scorer returns a `ScoreResult` (defined at `api/scoring/base.py`):
```python
@dataclass
class ScoreResult:
    score: float                    # 0-1 overall
    region_scores: dict[str, float] # language, visual, prefrontal
    model: str
    latency_ms: float
```

---

## AlgonautsScorer — `api/scoring/algonauts_scorer.py`
- Uses `sentence-transformers/all-MiniLM-L6-v2` for 384-dim embedding
- Loads `per_voxel_r.npy` — real Pearson r from Algonauts 2025 fMRI challenge
- Extracts semantic features: richness, visual keywords, decision-language
- Maps to regions via voxel masks → `{language, visual, prefrontal}` scores

## TribeScorer — `api/scoring/tribe_scorer.py`
- Loads TribeV2 checkpoint from `TRIBE_V2_CHECKPOINT` env var
- Converts text to events, runs inference on fsaverage5 brain surface
- Falls back to `MockScorer` if checkpoint missing

## CustomScorer — `api/scoring/custom_scorer.py`
- Loads `BrainEncoder` from `CUSTOM_CHECKPOINT` env var
- Falls back to `MockScorer` if checkpoint missing

## MockScorer — `api/scoring/mock_scorer.py`
- Hash-based deterministic fallback
- Adds keyword-based score boosts (identical text → identical score)
- Used whenever a real model is unavailable

---

## Output dict from `ScorerPipeline.score()`
```python
{
  "combined_score": float,           # weighted average
  "region_scores": {
    "language":   float,
    "visual":     float,
    "prefrontal": float,
  },
  "model_scores": {
    "algonauts": float,
    "tribe":     float,
    "custom":    float,
  },
  "latency_ms":  { "algonauts": float, "tribe": float, "custom": float },
  "models_live": { "algonauts": bool,  "tribe": bool,  "custom": bool  },
}
```

---

## To add a new scorer
1. Create `api/scoring/my_scorer.py` extending `BaseScorer` (`api/scoring/base.py`)
2. Instantiate in `ScorerPipeline.__init__()` (`api/scoring/pipeline.py:16`)
3. Add to `asyncio.gather()` at `api/scoring/pipeline.py:31`
4. Add weight in the combination block at `api/scoring/pipeline.py:38`
