import { ENDPOINTS } from '../config/api.js';

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}


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

    // Map pipeline nodes → step indicator IDs
    const NODE_TO_STEP = {
      s0_parse:    'ps0',
      s1_concept:  'ps1',
      s2_parallel: 'ps1',  // copy+image = initial draft phase
      s3_eval:     'psE',
      s4_parallel: 'ps2',
      s5_format:   'ps3',
    };
    let finalData = {};

    try {
      setStepState('ps0', 'active');
      pipelineStatus.textContent = 'Parsing brief…';

      const resp = await fetch(ENDPOINTS.campaignStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief, mode: studioMode }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader  = resp.body.getReader();
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
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }

          if (evt.type === 'node') {
            const stepId = NODE_TO_STEP[evt.node];
            if (stepId) {
              // Mark previous steps done, current active
              const steps = ['ps0','ps1','psE','ps2','ps3'];
              const idx = steps.indexOf(stepId);
              steps.forEach((id, i) => {
                if (i < idx)  setStepState(id, 'done');
                if (i === idx) setStepState(id, 'active');
              });
            }
            pipelineStatus.textContent = (evt.label || evt.node) + '…';

            // Incrementally render as data arrives
            const d = evt.data || {};
            if (evt.node === 's2_parallel' && d.copy) {
              renderAdCard(document.getElementById('initialDraftCard'), d.copy);
            }
            if (evt.node === 's3_eval' && d.eval) {
              renderEval(d.eval);
            }
            if (evt.node === 's4_parallel' && d.refined_copy) {
              renderAdCard(document.getElementById('finalDraftCard'), d.refined_copy);
            }
          }

          if (evt.type === 'done') {
            finalData = evt.data || {};
            ['ps0','ps1','psE','ps2','ps3'].forEach(id => setStepState(id, 'done'));
            pipelineStatus.textContent = `Done · ${finalData.iteration || 1} iteration(s) · target: ${finalData.brief?.target_emotion || 'auto'}`;
            studioResults.classList.add('show');
          }

          if (evt.type === 'error') {
            throw new Error(evt.message);
          }
        }
      }

    } catch(e) {
      pipelineStatus.textContent = `Error: ${e.message}`;
      ['ps0','ps1','psE','ps2','ps3'].forEach(id => setStepState(id, ''));
    } finally {
      studioRunBtn.disabled = false;
    }
  });
}
