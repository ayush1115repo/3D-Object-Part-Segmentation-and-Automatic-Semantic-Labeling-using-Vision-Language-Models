// ── PartField Studio — App Controller ────────────────────────────────────────
(function () {
  'use strict';

  // Palette MUST match backend modal_app.py PALETTE exactly (same order, same hex)
  // MUST match backend PALETTE exactly — same order, same hex
  const PALETTE = [
    "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
    "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
    "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
    "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
    "#6432C8","#149650","#C85014","#3264C8",
  ];

  let currentFile  = null;
  let currentJobId = null;
  let pollTimer    = null;
  let resultData   = null;
  let allLabels    = [];
  let previewBlob  = null;   // for AI recommend

  // ── File handling ──────────────────────────────────────────────────────────
  const dropZone  = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag');
    if (e.dataTransfer.files[0]) acceptFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) acceptFile(fileInput.files[0]);
  });

  function getExt(name) {
    return name.includes('.') ? name.split('.').pop().toLowerCase() : '';
  }

  function acceptFile(f) {
    const e = getExt(f.name);
    if (!['obj','glb','gltf','ply','stl'].includes(e)) {
      showToast(`Unsupported: .${e}`, 'err'); return;
    }
    clearInterval(pollTimer);
    currentFile = f; currentJobId = null; resultData = null;
    allLabels = []; window._renderMap = {};

    ['upload','partfield','cluster','render','gemini','done'].forEach(id => setPip(id,'idle','—'));
    document.getElementById('pipeline').style.display    = 'none';
    document.getElementById('resultsArea').style.display = 'none';
    document.getElementById('resultsIdle').style.display = '';
    showOverlayLoad(false);

    document.getElementById('filePill').style.display = 'flex';
    document.getElementById('pillName').textContent    = f.name;
    document.getElementById('pillSize').textContent    = fmtBytes(f.size);

    // Preview locally
    const blobUrl = URL.createObjectURL(f);
    loadModel(blobUrl, e, null);

    document.getElementById('overlayIdle').style.display = 'none';
    document.getElementById('toolbar').style.display     = '';
    document.getElementById('runBtn').disabled           = false;

    // Auto-recommend parts if category is filled
    const cat = document.getElementById('categoryInput').value.trim();
    if (cat) _requestRecommend(f, cat);
  }

  window.clearFile = function () {
    currentFile = null; fileInput.value = '';
    document.getElementById('filePill').style.display = 'none';
    document.getElementById('runBtn').disabled         = true;
    clearScene();
    document.getElementById('overlayIdle').style.display = '';
  };

  // Category input → trigger recommend when user finishes typing
  const catInput = document.getElementById('categoryInput');
  let catTimer;
  catInput.addEventListener('input', () => {
    clearTimeout(catTimer);
    catTimer = setTimeout(() => {
      const cat = catInput.value.trim();
      if (cat && currentFile) _requestRecommend(currentFile, cat);
    }, 800);
  });

  // ── AI Recommend parts ─────────────────────────────────────────────────────
  async function _requestRecommend(file, category) {
    const btn = document.getElementById('recommendBtn');
    if (btn) { btn.textContent = '⟳ Thinking…'; btn.disabled = true; }

    try {
      const fd = new FormData();
      fd.append('category', category);

      // Try canvas screenshot first (works after model is loaded)
      const canvas = document.getElementById('threeCanvas');
      let previewSent = false;
      if (canvas) {
        try {
          // Wait a tick for any pending renders
          await new Promise(r => setTimeout(r, 200));
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          // Only use if canvas is not blank (not all black)
          if (dataUrl && dataUrl.length > 5000) {
            const blob = await (await fetch(dataUrl)).blob();
            fd.append('preview', blob, 'preview.jpg');
            previewSent = true;
          }
        } catch (e) { console.warn('Canvas capture failed:', e); }
      }

      // Fallback: use original uploaded file as preview
      if (!previewSent && file) {
        fd.append('preview', file, file.name);
      }

      const r = await fetch('/api/recommend', { method: 'POST', body: fd });
      const d = await r.json();

      const slider = document.getElementById('partsRange');
      const hint   = document.getElementById('partsHint');
      if (slider && d.parts) {
        slider.value = d.parts;
        if (hint) hint.textContent = d.parts;
        updateSlider();
        showToast(`✦ AI recommends ${d.parts} parts — ${d.reason}`, 'ok');
      }
    } catch (e) {
      console.warn('Recommend failed:', e);
      showToast('AI Recommend unavailable — using current value', 'err');
    } finally {
      if (btn) { btn.textContent = '✦ AI Recommend'; btn.disabled = false; }
    }
  }

  window.requestRecommend = function () {
    const cat = document.getElementById('categoryInput').value.trim();
    if (!cat)  { showToast('Enter object category first', 'err'); return; }
    if (!currentFile) { showToast('Upload a model first', 'err'); return; }
    _requestRecommend(currentFile, cat);
  };

  // ── Start pipeline ─────────────────────────────────────────────────────────
  window.startPipeline = async function () {
    if (!currentFile) return;
    const category = document.getElementById('categoryInput').value.trim() || 'object';
    const nParts   = parseInt(document.getElementById('partsRange').value, 10);

    clearInterval(pollTimer);
    resultData = null; allLabels = []; window._renderMap = {};
    clearScene();

    document.getElementById('runBtn').disabled          = true;
    document.getElementById('pipeline').style.display   = '';
    document.getElementById('resultsArea').style.display = 'none';
    document.getElementById('resultsIdle').style.display = '';
    showOverlayLoad(true, 'Uploading model…', 2);

    ['upload','partfield','cluster','render','gemini','done'].forEach(id => setPip(id,'idle','—'));
    setPip('upload', 'active', 'Uploading…');

    const fd = new FormData();
    fd.append('model', currentFile);
    fd.append('category', category);
    fd.append('n_parts', nParts);

    let jid;
    for (let attempt = 1; attempt <= 6; attempt++) {
      try {
        showOverlayLoad(true,
          attempt === 1 ? 'Uploading model…' : `Server warming up… retry ${attempt}/6`,
          Math.min(attempt * 8, 40));
        const r    = await fetch('/api/upload', { method: 'POST', body: fd });
        const text = await r.text();
        let d;
        try { d = JSON.parse(text); }
        catch (_) {
          if (attempt < 6) { await sleep(3000); continue; }
          throw new Error('Server still warming up. Wait 10s and retry.');
        }
        if (d.error) throw new Error(d.error);
        jid = d.job_id; currentJobId = jid;
        break;
      } catch (err) {
        if (attempt === 6) { showPipelineError('Upload failed: ' + err.message); return; }
        await sleep(3000);
      }
    }
    if (!jid) { showPipelineError('Upload failed.'); return; }

    setPip('upload', 'done', 'Done ✓');
    setPip('partfield', 'active', 'Submitting to PartField GPU (A10G)…');
    startPolling(jid);
  };

  // ── Polling ────────────────────────────────────────────────────────────────
  function startPolling(jid) {
    clearInterval(pollTimer);
    let pollErrors = 0;
    pollTimer = setInterval(async () => {
      try {
        const r    = await fetch(`/api/status/${jid}`);
        const text = await r.text();
        let d;
        try { d = JSON.parse(text); }
        catch (_) {
          pollErrors++;
          if (pollErrors > 20) { clearInterval(pollTimer); showPipelineError('Server stopped responding.'); }
          return;
        }
        pollErrors = 0;
        onJobUpdate(d);
        if (d.status === 'done' || d.status === 'error') clearInterval(pollTimer);
      } catch (e) {
        pollErrors++;
        if (pollErrors > 20) { clearInterval(pollTimer); showPipelineError('Connection lost.'); }
      }
    }, 900);
  }

  function onJobUpdate(job) {
    const pct   = job.progress || 0;
    const stage = job.stage   || '';
    updateOverlayLoad(pct, stage);

    if (pct >= 5)  setPip('partfield', pct>=55?'done':'active', pct>=55?'Done ✓':'Running PartField GPU inference…');
    if (pct >= 55) setPip('cluster',   pct>=60?'done':'active', pct>=60?'Done ✓':'Agglomerative clustering…');
    if (pct >= 60) setPip('render',    pct>=72?'done':'active', pct>=72?'Done ✓':'Rendering cluster previews…');
    if (pct >= 72) setPip('gemini',    pct>=92?'done':'active', pct>=92?'Done ✓':'Gemma 4 open-vocabulary labelling…');

    if (job.status === 'done' && job.result) { setPip('done','done','Complete ✓'); onDone(job.result); }
    if (job.status === 'error') showPipelineError(job.error || 'Unknown error');
  }

  async function onDone(result) {
    resultData = result;

    // Build render map
    window._renderMap = {};
    (result.renders || []).forEach(r => {
      if (r.cluster_id != null && r.cluster_id >= 0) window._renderMap[r.cluster_id] = r.url;
    });

    // CRITICAL: colors MUST match backend PLY vertex colors
    // Backend: vert_colors[vi] = PALETTE[best_cid % len(PALETTE)]
    // Frontend: lbl.color = PALETTE[cluster_id % PALETTE.length]
    // Both use cluster_id directly — NOT sort index
    const rawLabels = result.labels || [];
    rawLabels.sort((a, b) => a.cluster_id - b.cluster_id);
    rawLabels.forEach((lbl) => {
      lbl.color = PALETTE[lbl.cluster_id % PALETTE.length];
    });
    allLabels = rawLabels;

    showOverlayLoad(false);
    document.getElementById('runBtn').disabled = false;
    renderResultsPanel(result);

    // Fetch FL from dedicated endpoint (too large for status poll)
    let FL = [];
    if (result.has_fl && result.job_id) {
      try {
        const r = await fetch(`/api/fl/${result.job_id}`);
        const d = await r.json();
        FL = d.fl || [];
        console.log(`FL: ${FL.length} labels, ${new Set(FL).size} unique clusters`);
      } catch (e) { console.warn('FL fetch failed:', e); }
    }

    // Load model — try colored PLY first, fallback to original
    const hasColoredPly = result.job_id && result.colored_model_ext;
    const origUrl  = '/model/'   + result.job_id;
    const origExt  = result.model_ext || 'glb';

    if (hasColoredPly) {
      // Verify the colored PLY exists before trying to load it
      fetch('/colored/' + result.job_id, { method: 'HEAD' })
        .then(r => {
          if (r.ok) {
            loadModelWithFallback(
              '/colored/' + result.job_id, result.colored_model_ext,
              origUrl, origExt, allLabels, FL
            );
          } else {
            console.warn('Colored PLY not found, loading original model');
            loadModel(origUrl, origExt, allLabels);
          }
        })
        .catch(() => loadModel(origUrl, origExt, allLabels));
    } else if (result.job_id) {
      loadModel(origUrl, origExt, allLabels);
    }

    showToast(`Done — ${allLabels.length} parts labelled!`, 'ok');
  }

  // ── Results panel ──────────────────────────────────────────────────────────
  function renderResultsPanel(result) {
    document.getElementById('resultsIdle').style.display  = 'none';
    document.getElementById('resultsArea').style.display  = 'flex';

    const descEl = document.getElementById('objDesc');
    if (descEl) {
      descEl.textContent = result.object_description || '';
      descEl.style.display = result.object_description ? '' : 'none';
    }
    const metaEl = document.getElementById('resultsMeta');
    if (metaEl) metaEl.innerHTML =
      `Category: <b>${result.category}</b> &nbsp;·&nbsp; Segments: <b>${(result.clusters||[]).length}</b> &nbsp;·&nbsp; Labels: <b>${allLabels.length}</b>`;

    buildPartsList(allLabels, result.clusters || []);

    const jaPre = document.getElementById('jaPre');
    if (jaPre) jaPre.textContent = JSON.stringify({
      category: result.category,
      object_description: result.object_description || '',
      parts: allLabels,
    }, null, 2);
  }

  function buildPartsList(labels, clusters) {
    const list = document.getElementById('partsList');
    list.innerHTML = '';
    const clusterMap = {};
    clusters.forEach(c => { clusterMap[c.id] = c; });

    labels.forEach(lbl => {
      const c    = clusterMap[lbl.cluster_id] || {};
      const conf = lbl.confidence != null ? Math.round(lbl.confidence * 100) : null;
      const card = document.createElement('div');
      card.className     = 'part-card';
      card.dataset.label = (lbl.label || '').toLowerCase();
      card.innerHTML = `
        <div class="pc-swatch" style="background:${lbl.color};box-shadow:0 0 10px ${lbl.color}55"></div>
        <div class="pc-info">
          <div class="pc-label">${lbl.label.replace(/_/g,' ')}</div>
          <div class="pc-desc">${lbl.description || ''}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
          <div class="pc-badge">#${lbl.cluster_id}</div>
          <div class="pc-actions">
            <div class="pc-act-btn" onclick="event.stopPropagation();isolateCluster(${lbl.cluster_id})">Isolate</div>
          </div>
        </div>
        ${conf !== null ? `<div class="pc-conf">${conf}%</div>` : ''}`;
      card.addEventListener('mouseenter', () => { highlightCluster(lbl.cluster_id); card.classList.add('active'); });
      card.addEventListener('mouseleave', () => { highlightCluster(null); card.classList.remove('active'); });
      list.appendChild(card);
    });
  }

  window.filterParts = q => {
    q = q.toLowerCase();
    document.querySelectorAll('.part-card').forEach(c => {
      c.style.display = c.dataset.label.includes(q) ? '' : 'none';
    });
  };

  window.toggleJSON = function () {
    const pre   = document.getElementById('jaPre');
    const arrow = document.getElementById('jaArrow');
    if (!pre) return;
    const open = pre.style.display === 'none';
    pre.style.display = open ? '' : 'none';
    if (arrow) arrow.textContent = open ? '▼' : '▶';
  };

  window.exportResult = function () {
    if (!resultData) { showToast('No result yet', 'err'); return; }
    const blob = new Blob([JSON.stringify({
      category: resultData.category,
      object_description: resultData.object_description || '',
      parts: allLabels, clusters: resultData.clusters,
    }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `partfield_${currentJobId || 'result'}.json`;
    a.click();
  };

  // ── UI helpers ─────────────────────────────────────────────────────────────
  function showOverlayLoad(on, stage, pct) {
    const el = document.getElementById('overlayLoad');
    el.style.display = on ? '' : 'none';
    if (on) {
      document.getElementById('loadStage').textContent = stage || 'Processing…';
      document.getElementById('loadFill').style.width  = (pct||0) + '%';
      document.getElementById('loadPct').textContent   = (pct||0) + '%';
    }
  }
  function updateOverlayLoad(pct, stage) {
    if (pct >= 100) { document.getElementById('overlayLoad').style.display = 'none'; return; }
    document.getElementById('overlayLoad').style.display = '';
    document.getElementById('loadStage').textContent = stage || 'Processing…';
    document.getElementById('loadFill').style.width  = pct + '%';
    document.getElementById('loadPct').textContent   = pct + '%';
  }
  function showPipelineError(msg) {
    showOverlayLoad(false);
    document.getElementById('runBtn').disabled = false;
    setPip('done', 'err', 'Error: ' + msg);
    showToast('Error: ' + msg, 'err');
  }
  function setPip(id, state, msg) {
    const el  = document.getElementById('pip-' + id);
    const mel = document.getElementById('pip-' + id + '-msg');
    if (!el || !mel) return;
    el.className    = 'pip-step ' + state;
    mel.textContent = msg;
  }
  function fmtBytes(b) {
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Toast ──────────────────────────────────────────────────────────────────
  function showToast(msg, type) {
    let t = document.getElementById('toast');
    if (!t) {
      t = document.createElement('div'); t.id = 'toast';
      t.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#0f1824;border:1px solid;border-radius:8px;padding:10px 20px;font-family:monospace;font-size:11px;z-index:9999;transition:opacity .3s;white-space:nowrap;max-width:90vw;overflow:hidden;text-overflow:ellipsis';
      document.body.appendChild(t);
    }
    const color = type === 'err' ? '#ff3b3b' : '#22d97b';
    t.style.borderColor = color; t.style.color = color;
    t.textContent = msg; t.style.opacity = '1';
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.style.opacity = '0'; }, 5000);
  }

  // ── Range slider ───────────────────────────────────────────────────────────
  const slider = document.getElementById('partsRange');
  function updateSlider() {
    const v = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(90deg,var(--ac) ${v}%,var(--bd2) ${v}%)`;
    const vl = document.getElementById('partsVal');
    if (vl) vl.textContent = slider.value;
  }
  slider.addEventListener('input', updateSlider);
  updateSlider();

})();
