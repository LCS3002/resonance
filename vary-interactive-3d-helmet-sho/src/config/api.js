// All backend endpoint URLs live here.
// Import ENDPOINTS in any component — never hardcode paths.

const API_BASE = '';

export const ENDPOINTS = {
  campaign:       `${API_BASE}/api/campaign`,        // POST — full pipeline, single JSON response
  campaignStream: `${API_BASE}/api/campaign/stream`, // POST — SSE, one event per node
  generateVisual: `${API_BASE}/api/generate-visual`, // POST — on-demand SVG
  brainMesh:      `${API_BASE}/api/brain-mesh`,      // GET  — 3D mesh JSON
  health:         `${API_BASE}/api/health`,           // GET  — model status
};
