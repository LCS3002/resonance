import { ENDPOINTS } from '../config/api.js';

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function setStepState(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (state) el.classList.add(state);
}

function renderAdCard(container, draft) {
  if (!draft) { container.innerHTML = '<p style="color:var(--text-secondary);font-size:13px;">No draft available</p>'; return; }
  container.innerHTML = `
    <div class="ad-card-headline">${escHtml(draft.headline || '')}</div>
    <div class="ad-card-body">${escHtml(draft.body || '')}</div>
    <div class="ad-card-cta">${escHtml(draft.cta || '')}</div>
    ${draft.image_url ? `<div class="ad-card-image"><img src="${draft.image_url}" alt="Ad visual mockup" /></div>` : ''}
  `;
}

function renderEval(evalData) {
  const evalSection   = document.getElementById('evalSection');
  const evalGrid      = document.getElementById('evalGrid');
  const saliencyBlock = document.getElementById('saliencyBlock');
  const saliencyText  = document.getElementById('saliencyText');

  if (!evalData?.neural) { evalSection.style.display = 'none'; return; }
  evalSection.style.display = 'block';

  const n           = evalData.neural;
  const combinedPct = Math.round((n.combined_score || 0) * 100);
  const emotion     = evalData.roberta_emotion || '—';
  const conf        = evalData.roberta_confidence ? Math.round(evalData.roberta_confidence * 100) : '—';
  const ge          = evalData.goemotion_scores || {};
  const topGE       = Object.entries(ge).sort((a,b)=>b[1]-a[1]).slice(0,2).map(([k,v])=>`${k} ${Math.round(v*100)}%`).join(' · ');

  evalGrid.innerHTML = `
    <div class="eval-cell">
      <div class="eval-cell-label">Neural Score</div>
      <div class="eval-cell-value">${combinedPct}</div>
      <div class="eval-cell-sub">/ 100 engagement</div>
    </div>
    <div class="eval-cell">
      <div class="eval-cell-label">RoBERTa Emotion</div>
      <div class="eval-cell-value" style="font-size:16px;text-transform:capitalize;">${escHtml(emotion)}</div>
      <div class="eval-cell-sub">${conf}% confidence</div>
    </div>
    <div class="eval-cell" style="grid-column:1/-1">
      <div class="eval-cell-label">GoEmotions top signals</div>
      <div class="eval-cell-sub" style="font-size:12px;margin-top:4px;">${escHtml(topGE) || '—'}</div>
    </div>
  `;

  const hint = evalData.saliency_hint || evalData.counterfactual_hint || '';
  saliencyBlock.style.display = hint ? 'block' : 'none';
  if (hint) saliencyText.textContent = hint;
}

export function initStudioOverlay() {
  const studioOverlay    = document.getElementById('studioOverlay');
  const studioRunBtn     = document.getElementById('studioRunBtn');
  const studioBrief      = document.getElementById('studioBrief');
  const pipelineProgress = document.getElementById('pipelineProgress');
  const pipelineStatus   = document.getElementById('pipelineStatus');
  const studioResults    = document.getElementById('studioResults');

  document.getElementById('openStudio').addEventListener('click', e => {
    e.preventDefault(); studioOverlay.classList.add('open');
  });
  document.getElementById('closeStudio').addEventListener('click', () => studioOverlay.classList.remove('open'));

  let studioMode = 'text';
  document.querySelectorAll('.mode-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.mode-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      studioMode = pill.dataset.mode;
    });
  });

  studioOverlay.querySelectorAll('.demo-suggest').forEach(btn => {
    btn.addEventListener('click', () => { studioBrief.value = btn.dataset.brief; });
  });

  studioRunBtn.addEventListener('click', async () => {
    const brief = studioBrief.value.trim();
    if (!brief) { studioBrief.focus(); return; }

    studioRunBtn.disabled = true;
    studioResults.classList.remove('show');
    pipelineProgress.classList.add('show');
    ['ps0','ps1','psE','ps2','ps3'].forEach(id => setStepState(id, ''));

    try {
      setStepState('ps0', 'active'); pipelineStatus.textContent = 'S0 — Parsing brief…';
      await delay(300);
      setStepState('ps0', 'done'); setStepState('ps1', 'active');
      pipelineStatus.textContent = 'S1 — Generating initial draft…';

      const resp = await fetch(ENDPOINTS.agentCampaign, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief, mode: studioMode, run_eval: true }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      setStepState('ps1', 'done'); setStepState('psE', 'active');
      pipelineStatus.textContent = 'Eval — Neural scoring + saliency…';
      await delay(200);

      setStepState('psE', 'done'); setStepState('ps2', 'active');
      pipelineStatus.textContent = 'S2 — Refining with gradient feedback…';
      await delay(200);

      setStepState('ps2', 'done'); setStepState('ps3', 'active');
      pipelineStatus.textContent = 'S3 — Formatting final ad…';
      await delay(200);

      setStepState('ps3', 'done');
      pipelineStatus.textContent = `Done · ${data.iteration_count || 2} iterations · target: ${data.target_emotion || 'auto'}`;

      renderAdCard(document.getElementById('initialDraftCard'), data.initial_draft);
      renderEval(data.eval);
      renderAdCard(document.getElementById('finalDraftCard'), data.final_draft);
      studioResults.classList.add('show');

    } catch(e) {
      pipelineStatus.textContent = `Error: ${e.message}`;
      ['ps0','ps1','psE','ps2','ps3'].forEach(id => setStepState(id, ''));
    } finally {
      studioRunBtn.disabled = false;
    }
  });
}
