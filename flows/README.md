# Flows

Each file traces one end-to-end flow through the codebase — entry point → every function touched → exit. Read the relevant flow file instead of grepping the repo.

| Flow | File | When to read |
|---|---|---|
| Campaign pipeline (full) | [campaign.md](campaign.md) | Anything touching POST /api/campaign |
| Neural scoring | [neural_scoring.md](neural_scoring.md) | fMRI models, region scores, combined score |
| Emotion prediction | [emotion_prediction.md](emotion_prediction.md) | Region-based + RoBERTa GoEmotions emotion labels |
| Gradient saliency | [gradient_saliency.md](gradient_saliency.md) | Token attribution, counterfactual hints, feedback loop |
| Ad generation (Claude) | [ad_generation.md](ad_generation.md) | Variant generation, brand research, strategy, image |
| Moondream image analysis | [moondream.md](moondream.md) | VLM image→caption→emotion scoring |
| Server & routing | [server.md](server.md) | FastAPI routes, request models, health check |
| MCP tools | [mcp.md](mcp.md) | Claude-callable tools via Model Context Protocol |
| Agent pipeline (LangGraph) | [agent_pipeline.md](agent_pipeline.md) | POST /api/agent/campaign — S0→S1→eval→S2→S3 combined flow |
