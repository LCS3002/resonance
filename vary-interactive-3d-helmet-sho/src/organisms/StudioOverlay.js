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
  if (!draft) {
    container.innerHTML = '<p style="color:var(--text-secondary);font-size:13px;">No draft available</p>';
    return;
  }
  container.innerHTML = `
    <div class="ad-card-headline">${escHtml(draft.headline || '')}</div>
    <div class="ad-card-body">${escHtml(draft.body || '')}</div>
    <div class="ad-card-cta">${escHtml(draft.cta || '')}</div>
    ${draft.image_url ? `<div class="ad-card-image"><img src="${draft.image_url}" alt="Ad visual" /></div>` : ''}
  `;
}

function renderEval(evalData) {
  const evalSection   = document.getElementById('evalSection');
  const evalGrid      = document.getElementById('evalGrid');
  const saliencyBlock = document.getElementById('saliencyBlock');
  const saliencyText  = document.getElementById('saliencyText');

  if (!evalData) { evalSection.style.display = 'none'; return; }
  evalSection.style.display = 'block';

  const emotion  = evalData.text_emotion || '—';
  const conf     = evalData.text_confidence ? Math.round(evalData.text_confidence * 100) : '—';
  const score    = evalData.text_target_score ? Math.round(evalData.text_target_score * 100) : '—';
  const ge       = evalData.text_goemotion_scores || {};
  const topGE    = Object.entries(ge).sort((a,b)=>b[1]-a[1]).slice(0,3)
                     .map(([k,v])=>`${k} ${Math.round(v*100)}%`).join(' · ');

  evalGrid.innerHTML = `
    <div class="eval-cell">
      <div class="eval-cell-label">Target Score</div>
      <div class="eval-cell-value">${score}</div>
      <div class="eval-cell-sub">/ 100</div>
    </div>
    <div class="eval-cell">
      <div class="eval-cell-label">Detected Emotion</div>
      <div class="eval-cell-value" style="font-size:16px;text-transform:capitalize;">${escHtml(emotion)}</div>
      <div class="eval-cell-sub">${conf}% confidence</div>
    </div>
    <div class="eval-cell" style="grid-column:1/-1">
      <div class="eval-cell-label">GoEmotions signals</div>
      <div class="eval-cell-sub" style="font-size:12px;margin-top:4px;">${escHtml(topGE) || '—'}</div>
    </div>
  `;

  const hint = evalData.text_gradient_hint || '';
  saliencyBlock.style.display = hint ? 'block' : 'none';
  if (hint) saliencyText.textContent = hint;
}

export function initStudioOverlay() {
  const studioOverlay    = document.getElementById('studioOverlay');
  const studioRunBtn     = document.getElementById('studioRunBtn');
  const studioCompany    = document.getElementById('studioCompany');
  const studioBriefLine  = document.getElementById('studioBriefLine');
  const pipelineProgress = document.getElementById('pipelineProgress');
  const pipelineStatus   = document.getElementById('pipelineStatus');
  const studioResults    = document.getElementById('studioResults');

  document.getElementById('openStudio').addEventListener('click', e => {
    e.preventDefault(); studioOverlay.classList.add('open');
  });
  document.getElementById('closeStudio').addEventListener('click', () => studioOverlay.classList.remove('open'));

  // Mode toggle
  let studioMode = 'text';
  document.querySelectorAll('.mode-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.mode-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      studioMode = pill.dataset.mode;
    });
  });

  // Platform chips
  let studioPlatform = 'instagram';
  document.querySelectorAll('.platform-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.platform-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      studioPlatform = chip.dataset.platform;
    });
  });

  // Emotion chips
  let studioEmotion = 'infer';
  document.querySelectorAll('#studioEmotionChips .emotion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#studioEmotionChips .emotion-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      studioEmotion = chip.dataset.emotion;
    });
  });

  // Map pipeline nodes → step indicator IDs (includes optional s0b step)
  const NODE_TO_STEP = {
    s0_parse:    'ps0',
    s0b_infer:   'ps0b',
    s1_concept:  'ps1',
    s2_parallel: 'ps1',
    s3_eval:     'psE',
    s4_parallel: 'ps2',
    s5_format:   'ps3',
  };
  const ALL_STEPS = ['ps0', 'ps0b', 'ps1', 'psE', 'ps2', 'ps3'];

  studioRunBtn.addEventListener('click', async () => {
    const company   = studioCompany.value.trim();
    const briefLine = studioBriefLine.value.trim();
    if (!company || !briefLine) {
      (company ? studioBriefLine : studioCompany).focus();
      return;
    }

    // Construct brief string from structured fields
    const brief = `${company}. ${briefLine}. Platform: ${studioPlatform}.`;

    studioRunBtn.disabled = true;
    studioResults.classList.remove('show');
    pipelineProgress.classList.add('show');
    ALL_STEPS.forEach(id => setStepState(id, ''));

    // Hide s0b step initially — only show it when the node fires
    const s0bEl = document.getElementById('ps0b');
    if (s0bEl) s0bEl.style.display = studioEmotion === 'infer' ? '' : 'none';

    // Reset result panels
    document.getElementById('emotionRationaleSection').style.display = 'none';
    document.getElementById('evalSection').style.display = 'none';

    try {
      setStepState('ps0', 'active');
      pipelineStatus.textContent = 'Parsing brief…';

      const resp = await fetch(ENDPOINTS.campaignStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief, mode: studioMode, target_emotion: studioEmotion }),
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
              const idx = ALL_STEPS.indexOf(stepId);
              ALL_STEPS.forEach((id, i) => {
                if (i < idx)  setStepState(id, 'done');
                if (i === idx) setStepState(id, 'active');
              });
            }
            pipelineStatus.textContent = (evt.label || evt.node) + '…';

            const d = evt.data || {};

            if (evt.node === 's0b_infer' && d.emotion_rationale) {
              const sec = document.getElementById('emotionRationaleSection');
              const box = document.getElementById('emotionRationaleBox');
              const emotion = d.brief?.target_emotion || '';
              box.innerHTML = `
                <span class="emotion-tag" style="margin-bottom:8px;display:inline-block;">${escHtml(emotion)}</span>
                <p style="font-size:13px;color:var(--text-secondary);margin:0;">${escHtml(d.emotion_rationale)}</p>
              `;
              sec.style.display = 'block';
            }

            if (evt.node === 's2_parallel' && d.copy) {
              renderAdCard(document.getElementById('initialDraftCard'), {
                ...d.copy,
                image_url: d.image?.image_url,
              });
            }
            if (evt.node === 's3_eval' && d.eval) {
              renderEval(d.eval);
            }
            if (evt.node === 's4_parallel' && d.refined_copy) {
              renderAdCard(document.getElementById('finalDraftCard'), {
                ...d.refined_copy,
                image_url: d.refined_image?.image_url,
              });
            }
          }

          if (evt.type === 'done') {
            const finalData = evt.data || {};
            ALL_STEPS.forEach(id => setStepState(id, 'done'));
            const emotion  = finalData.brief?.target_emotion || 'auto';
            const inferred = finalData.emotion_rationale ? ' (AI inferred)' : '';
            pipelineStatus.textContent = `Done · emotion: ${emotion}${inferred}`;
            studioResults.classList.add('show');
          }

          if (evt.type === 'error') {
            throw new Error(evt.message);
          }
        }
      }

    } catch(e) {
      pipelineStatus.textContent = `Error: ${e.message}`;
      ALL_STEPS.forEach(id => setStepState(id, ''));
    } finally {
      studioRunBtn.disabled = false;
    }
  });
}
