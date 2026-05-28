# Flow: Server & Routing

**Entry point:** `api/main.py`

---

## FastAPI app — `api/main.py:25`

CORS: `allow_origins=["*"]` — open for hackathon (tighten for prod).

Static files: `vary-interactive-3d-helmet-sho/` mounted at `/static` if directory exists.

---

## Routes

| Method | Path | Handler | File:line |
|---|---|---|---|
| POST | `/api/campaign` | `campaign()` | `api/main.py:73` |
| GET | `/api/health` | `health()` | `api/main.py:61` |
| POST | `/api/review` | `review()` | `api/main.py:90` |
| GET | `/api/brain-mesh` | `brain_mesh()` | `api/main.py:108` |
| GET | `/` | `root()` | `api/main.py:116` |

---

## Request models — `api/main.py:46`

```python
class CampaignRequest(BaseModel):
    brief: str           # 5-500 chars, required
    brand: str | None    # optional brand name for Tavily research
    num_variants: int    # 1-5, default 3
    target_emotion: str  # aspirational|trustworthy|urgent|playful|premium
```

```python
class ReviewDecision(BaseModel):
    variant_id: str
    approved: bool
    reviewer_notes: str  # default ""
```

---

## Health check — `api/main.py:61`

```
GET /api/health
  └─ get_scorer()   api/agent.py:47
       → { status: "ok", models: { tribe: bool, custom: bool } }
```

Note: does not initialize GoEmotionScorer or MoondreamVLM on health check — those load lazily on first `/api/campaign` call.

---

## Run locally

```bash
cd api
uvicorn main:app --reload --port 8000
```

Or via `__main__`:
```bash
python -m api.main
```

Uses `PORT` env var (default 8000).

---

## Frontend

Served from `vary-interactive-3d-helmet-sho/index.html` at `GET /`.
3D brain visualization built with Three.js.
Brain mesh data: `GET /api/brain-mesh` → serves `brain_mesh.json`.

---

## To add a new route
1. Define Pydantic model in `api/main.py` (after line 54)
2. Add `@app.post(...)` or `@app.get(...)` handler after line 58
3. Add route to `flows/server.md` routes table above
