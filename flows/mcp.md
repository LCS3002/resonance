# Flow: MCP Server (Claude-Callable Tools)

**File:** `api/mcp_server.py`  
**Port:** `MCP_PORT` env var (default 8001)  
**Protocol:** Model Context Protocol — exposes Resonance as tools callable by any MCP client (e.g. Claude via Alpic)

---

## Tools

### `run_campaign_tool`
```
Input: brief, brand, num_variants
  └─ calls run_campaign() (same as POST /api/campaign)
     → returns full campaign result dict
```

### `score_ad_neural`
```
Input: ad_text: str
  └─ get_scorer().score(ad_text)
     → { combined_score, region_scores, model_scores, ... }
```

### `explain_neural_score`
```
Input: score: float (0-1)
  └─ Returns plain-text explanation of what the score means
     for ad placement decisions
```

### `log_review_decision`
```
Input: variant_id, approved, notes
  └─ Logs HITL decision (same as POST /api/review)
     → { variant_id, approved, message }
```

---

## To add a new MCP tool
1. Add `@mcp.tool()` decorated async function to `api/mcp_server.py`
2. Document inputs/outputs clearly (MCP clients use the docstring)
3. Add entry to this file
