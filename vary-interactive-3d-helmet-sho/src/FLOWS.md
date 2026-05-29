# Flows: Frontend

Files: `main.js`, `organisms/BrainCanvas.js`, `organisms/DemoOverlay.js`, `organisms/StudioOverlay.js`, `config/api.js`

---

## Entry point — `main.js`

```
main.js
  ├─ initBrainCanvas(#brainCanvas, #modeToggle)   BrainCanvas.js
  ├─ initDemoOverlay()                             DemoOverlay.js
  └─ initStudioOverlay()                           StudioOverlay.js
```

---

## DemoOverlay — pipeline UI

**Trigger:** `#openDemo` click → `#demoOverlay` opens  
**API:** `ENDPOINTS.campaign` → `POST /api/campaign`

```
analyseBtn click
  └─ fetch(ENDPOINTS.campaign, { brief, mode })
       → data = JSON response
       fillTimelineSlot('slot-initial', data.initial_copy, ...)
       inject data.initial_image.image_url into #visual-initial
       fillTimelineSlot('slot-refined',  data.final_copy, ...)
       inject data.final_image.image_url  into #visual-refined
       window.setApiActivation(ge.desire, ge.excitement, ge.nervousness)
```

Phase bar steps: `ph0` → `ph1` → `ph2` → `ph3` → `ph4`  
Brain activation mapped from `data.eval.text_goemotion_scores`.

---

## StudioOverlay — agent pipeline UI

**Trigger:** `#openStudio` click → `#studioOverlay` opens  
**API:** `ENDPOINTS.campaign` → `POST /api/campaign`

```
studioRunBtn click
  └─ fetch(ENDPOINTS.campaign, { brief, mode })
       → data = JSON response
       renderAdCard(#initialDraftCard, data.initial_copy)
       renderEval(data.eval)
       renderAdCard(#finalDraftCard,   data.final_copy)
```

Pipeline step indicators: `ps0` → `ps1` → `psE` → `ps2` → `ps3`

---

## API config — `config/api.js`

Single source of truth for all backend URLs. Import `ENDPOINTS` — never hardcode paths.

```javascript
export const ENDPOINTS = {
  campaign:       '/api/campaign',        // POST — full pipeline, single JSON response
  campaignStream: '/api/campaign/stream', // POST — SSE, one event per node (preferred)
  generateVisual: '/api/generate-visual', // POST — on-demand SVG
  brainMesh:      '/api/brain-mesh',      // GET  — 3D mesh JSON
  health:         '/api/health',          // GET  — model status
}
```

Both overlays use `campaignStream`. Each SSE event: `{ type: "node"|"done"|"error", node, label, data }`.  
Node sequence: `s0_parse → s1_concept → s2_parallel → s3_eval → s4_parallel → s5_format`

---

## Change Index

| Thing to change | Where |
|---|---|
| Add / rename a backend URL | `config/api.js:ENDPOINTS` |
| DemoOverlay request body | `DemoOverlay.js:analyseBtn` click handler |
| DemoOverlay response rendering | `DemoOverlay.js:fillTimelineSlot()` calls |
| StudioOverlay request body | `StudioOverlay.js:studioRunBtn` click handler |
| StudioOverlay response rendering | `StudioOverlay.js:renderAdCard()` / `renderEval()` calls |
| Brain mesh activation mapping | `DemoOverlay.js` — `window.setApiActivation` args |
| Phase bar timing | `DemoOverlay.js:startPhaseAnimation()` |
