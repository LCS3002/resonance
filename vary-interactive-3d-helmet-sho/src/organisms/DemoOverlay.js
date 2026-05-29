import { ENDPOINTS } from '../config/api.js';

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scorePillClass(s) {
  if (s >= 0.65) return 'good';
  if (s >= 0.50) return 'mid';
  return 'low';
}

function regionBarsHtml(rs) {
  return ['language','visual','prefrontal'].map(r =>
    `<div class="region-bar-row">
      <span class="region-bar-label">${r}</span>
      <div class="region-bar-track">
        <div class="region-bar-fill fill-${r}" style="width:${Math.round((rs[r]||0)*100)}%"></div>
      </div>
    </div>`
  ).join('');
}

function fillTimelineSlot(slotId, draft, neural, emotion, isWinner) {
  const slot = document.getElementById(slotId);
  if (!slot) return;
  const score = (neural || {}).combined_score || 0;
  const rs    = (neural || {}).region_scores  || {};
  const pc    = scorePillClass(score);
  const label = isWinner
    ? '<div class="tl-label refined">Winner <span class="tl-deploy-badge">Deploy →</span></div>'
    : '<div class="tl-label">Draft</div>';

  slot.innerHTML = label + `
    <div class="ad-preview ${isWinner ? 'winner' : ''}">
      <div class="ad-visual" id="visual-${isWinner ? 'refined' : 'initial'}">
        <span class="ad-visual-placeholder">Generating…</span>
      </div>
      <div class="ad-copy">
        <div class="ad-copy-headline">${escHtml(draft.headline || '')}</div>
        <div class="ad-copy-body">${escHtml(draft.body || '')}</div>
        <div class="ad-copy-footer">
          <span class="ad-copy-cta">${escHtml(draft.cta || 'Learn more')}</span>
          <span class="ad-score-pill ${pc}">${Math.round(score * 100)}</span>
        </div>
      </div>
      <div class="ad-meta">
        <div class="ad-meta-emotion">
          ${emotion ? `<span class="emotion-tag">${escHtml(emotion)}</span>` : ''}
        </div>
        <div class="region-bars">${regionBarsHtml(rs)}</div>
      </div>
    </div>`;
}

function renderTimelineSkeleton(brandCtxBox, resultsInner) {
  brandCtxBox.innerHTML = '';
  resultsInner.innerHTML = `
    <div class="tl-wrap">
      <div class="tl-slot" id="slot-initial">
        <div class="tl-label">Draft</div>
        <div class="ad-preview">
          <div class="ad-visual" id="visual-initial"><span class="ad-visual-placeholder">—</span></div>
          <div class="ad-copy">
            <div class="tl-skeleton"></div>
            <div class="tl-skeleton tl-sk-short"></div>
          </div>
        </div>
      </div>
      <div class="tl-arrow">→</div>
      <div class="tl-slot" id="slot-refined">
        <div class="tl-label">Refined</div>
        <div class="ad-preview">
          <div class="ad-visual" id="visual-refined"><span class="ad-visual-placeholder">—</span></div>
          <div class="ad-copy">
            <div class="tl-skeleton"></div>
            <div class="tl-skeleton tl-sk-short"></div>
          </div>
        </div>
      </div>
    </div>`;
}

export function initDemoOverlay() {
  const overlay      = document.getElementById('demoOverlay');
  const briefInput   = document.getElementById('briefInput');
  const analyseBtn   = document.getElementById('analyseBtn');
  const demoStatus   = document.getElementById('demoStatus');
  const overlayRes   = document.getElementById('overlayResults');
  const resultsInner = document.getElementById('resultsInner');
  const brandCtxBox  = document.getElementById('brandContextBox');
  const phaseBar     = document.getElementById('phaseBar');
  const phaseSteps   = ['ph0','ph1','ph2','ph3','ph4'];

  document.getElementById('openDemo').addEventListener('click', () => {
    overlay.classList.add('open');
    setTimeout(() => briefInput.focus(), 300);
  });
  document.getElementById('closeDemo').addEventListener('click', () => {
    overlay.classList.remove('open');
    window.setApiActivation?.(0, 0, 0);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && overlay.classList.contains('open')) overlay.classList.remove('open');
  });

  let selectedEmotion = '';
  document.querySelectorAll('.emotion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.emotion-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      selectedEmotion = chip.dataset.emotion;
    });
  });

  document.querySelectorAll('.demo-suggest').forEach(s => {
    s.addEventListener('click', () => { briefInput.value = s.dataset.brief; analyseBtn.click(); });
  });
  briefInput.addEventListener('keydown', e => { if (e.key === 'Enter') analyseBtn.click(); });

  function setPhase(idx) {
    phaseSteps.forEach((id, i) => {
      const el = document.getElementById(id);
      el.classList.remove('active', 'done');
      if (i < idx)  el.classList.add('done');
      if (i === idx) el.classList.add('active');
    });
  }

  function startPhaseAnimation() {
    phaseBar.classList.add('show');
    [0, 4000, 18000, 38000, 58000].forEach((d, i) => setTimeout(() => setPhase(i), d));
  }

  function stopPhaseAnimation() {
    phaseSteps.forEach(id => {
      const el = document.getElementById(id);
      el.classList.remove('active'); el.classList.add('done');
    });
    setTimeout(() => phaseBar.classList.remove('show'), 1200);
  }

  function handleStreamEvent(evt) {
    if (evt.type === 'status') {
      demoStatus.textContent = evt.text;
    } else if (evt.type === 'initial') {
      fillTimelineSlot('slot-initial', evt.draft, evt.neural, evt.emotion, false);
      const rs = (evt.neural || {}).region_scores || {};
      window.setApiActivation?.(rs.language||0, rs.visual||0, rs.prefrontal||0);
    } else if (evt.type === 'initial-visual') {
      const v = document.getElementById('visual-initial');
      if (v) v.innerHTML = `<img src="${evt.image_url}" style="width:100%;height:100%;object-fit:cover;display:block;" />`;
    } else if (evt.type === 'refined') {
      fillTimelineSlot('slot-refined', evt.draft, evt.neural, evt.emotion, true);
      const rs = (evt.neural || {}).region_scores || {};
      window.setApiActivation?.(rs.language||0, rs.visual||0, rs.prefrontal||0);
      demoStatus.textContent = '';
    } else if (evt.type === 'refined-visual') {
      const v = document.getElementById('visual-refined');
      if (v) v.innerHTML = `<img src="${evt.image_url}" style="width:100%;height:100%;object-fit:cover;display:block;" />`;
    } else if (evt.type === 'done') {
      stopPhaseAnimation(); demoStatus.textContent = '';
    } else if (evt.type === 'error') {
      demoStatus.textContent = 'Error: ' + evt.text; stopPhaseAnimation();
    }
  }

  analyseBtn.addEventListener('click', async () => {
    const brief = briefInput.value.trim();
    if (!brief) { briefInput.focus(); return; }

    analyseBtn.disabled = true;
    analyseBtn.textContent = 'Generating…';
    demoStatus.textContent = '';
    overlayRes.classList.remove('show');
    startPhaseAnimation();
    renderTimelineSkeleton(brandCtxBox, resultsInner);
    overlayRes.classList.add('show');
    setTimeout(() => overlayRes.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);

    try {
      const res = await fetch(ENDPOINTS.campaignStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief, target_emotion: selectedEmotion || null }),
      });
      if (!res.ok) throw new Error('API ' + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try { handleStreamEvent(JSON.parse(line.slice(6))); } catch {}
        }
      }
      stopPhaseAnimation();
    } catch (e) {
      stopPhaseAnimation();
      demoStatus.textContent = e.message.includes('fetch')
        ? 'Cannot reach API — run start.bat first'
        : 'Error: ' + e.message;
    }

    analyseBtn.disabled = false;
    analyseBtn.textContent = 'Analyse →';
  });
}
