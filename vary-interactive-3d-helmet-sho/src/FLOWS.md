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
**API:** `ENDPOINTS.campaignStream` → `POST /api/campaign/stream` (SSE)

Form: `#briefInput` (free text) + `#demoEmotionChips` → `selectedEmotion` (default `"infer"`)

```
analyseBtn click
  └─ fetch(ENDPOINTS.campaignStream, { brief, mode: 'text', target_emotion: selectedEmotion })
       SSE events:
       s2_parallel → fillTimelineSlot('slot-initial', copy) + inject image_url
       s3_eval     → window.setApiActivation(ge scores for brain mesh)
       s4_parallel → fillTimelineSlot('slot-refined', refined_copy) + inject image_url
```

Phase bar: `ph0` → `ph1` → `ph2` → `ph3` → `ph4` (time-based animation, not SSE-driven)  
Emotion chips: `#demoEmotionChips` — 7 emotions + "Let AI choose" (infer)

---

## StudioOverlay — agent pipeline UI

**Trigger:** `#openStudio` click → `#studioOverlay` opens  
**API:** `ENDPOINTS.campaignStream` → `POST /api/campaign/stream` (SSE)

Form fields: `#studioCompany` + `#studioBriefLine` → combined into `brief` string  
Platform chips: `#platform-chips` → `studioPlatform` → appended to brief  
Emotion chips: `#studioEmotionChips` → `studioEmotion` → sent as `target_emotion`  
Mode pills: `#mode-pill` → `studioMode` → sent as `mode`

```
studioRunBtn click
  brief = "{studioCompany}. {studioBriefLine}. Platform: {studioPlatform}."
  └─ fetch(ENDPOINTS.campaignStream, { brief, mode, target_emotion })
       SSE: s0_parse → [s0b_infer if infer] → s1_concept → s2_parallel → s3_eval → s4_parallel → s5_format
       s0b_infer event → show #emotionRationaleSection with chosen emotion + rationale
       s2_parallel event → renderAdCard(#initialDraftCard, copy)
       s3_eval event → renderEval(eval)
       s4_parallel event → renderAdCard(#finalDraftCard, refined_copy)
```

Pipeline step indicators: `ps0` → `ps0b` (shown only when emotion=infer) → `ps1` → `psE` → `ps2` → `ps3`

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
| Demo emotion chip list | `index.html #demoEmotionChips` + `DemoOverlay.js` default |
| Studio emotion chip list | `index.html #studioEmotionChips` + `StudioOverlay.js` default |
| Studio platform chips | `index.html .platform-chips` |
| DemoOverlay request body | `DemoOverlay.js:analyseBtn` click handler |
| DemoOverlay response rendering | `DemoOverlay.js:fillTimelineSlot()` calls |
| StudioOverlay form → brief construction | `StudioOverlay.js:studioRunBtn` (brief = company + description + platform) |
| StudioOverlay response rendering | `StudioOverlay.js:renderAdCard()` / `renderEval()` |
| Brain mesh activation mapping | `DemoOverlay.js` — `window.setApiActivation` args |
| Phase bar timing | `DemoOverlay.js:startPhaseAnimation()` |
