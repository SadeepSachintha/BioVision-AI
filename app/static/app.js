/**
 * BioVision AI — Interactive Web Demo Logic
 * Handles image ingest, parallel ONNX / Grad-CAM inference queries,
 * and thermodynamic heatmap alpha blending.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const presetButtons = document.querySelectorAll('.preset-card');
  const systemStatusText = document.getElementById('system-status-text');

  // Result views
  const emptyState = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultsContent = document.getElementById('results-content');

  // Visualizer elements
  const rawLeafImg = document.getElementById('raw-leaf-img');
  const heatmapImg = document.getElementById('gradcam-heatmap-img');
  const opacitySlider = document.getElementById('opacity-slider');
  const opacityLabel = document.getElementById('opacity-label');
  const pillButtons = document.querySelectorAll('.pill-btn');
  const latencyBadge = document.getElementById('latency-badge');

  // Diagnosis details
  const diagCrop = document.getElementById('diag-crop');
  const diagCondition = document.getElementById('diag-condition');
  const diagScientific = document.getElementById('diag-scientific');
  const diagConfidence = document.getElementById('diag-confidence');
  const severityPill = document.getElementById('severity-pill');
  const yieldRiskText = document.getElementById('yield-risk-text');
  const candidatesList = document.getElementById('candidates-list');

  // Recommendations
  const recImmediateList = document.getElementById('rec-immediate-list');
  const recChemical = document.getElementById('rec-chemical');
  const recOrganic = document.getElementById('rec-organic');
  const recPreventative = document.getElementById('rec-preventative');

  // 1. Check System Health
  async function checkSystemHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        systemStatusText.textContent = `Online: ${data.inference_provider}`;
      } else {
        systemStatusText.textContent = 'Service Degraded';
      }
    } catch {
      systemStatusText.textContent = 'Offline / Connecting...';
    }
  }
  checkSystemHealth();

  // Drift Telemetry Check
  const driftStatusBadge = document.getElementById('drift-status-badge');
  const driftPulseDot = document.getElementById('drift-pulse-dot');
  const driftStatusText = document.getElementById('drift-status-text');

  async function checkDriftStatus() {
    try {
      const res = await fetch('/drift');
      if (res.ok) {
        const data = await res.json();
        if (driftStatusText) {
          if (data.drift_detected) {
            driftStatusText.textContent = `Drift: ${data.status.replace(/_/g, ' ')}`;
            if (driftPulseDot) driftPulseDot.style.backgroundColor = '#ef4444';
            if (driftStatusBadge) driftStatusBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          } else {
            driftStatusText.textContent = `Drift: ${data.status} (${data.sample_count} samples)`;
            if (driftPulseDot) driftPulseDot.style.backgroundColor = 'var(--emerald-400)';
            if (driftStatusBadge) driftStatusBadge.style.borderColor = 'var(--border-subtle)';
          }
        }
      }
    } catch {
      // ignore
    }
  }
  checkDriftStatus();

  // 2. Drag & Drop Event Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageUpload(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageUpload(e.target.files[0]);
    }
  });

  // 3. Preset Cards Click
  presetButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const sampleUrl = btn.getAttribute('data-sample');
      try {
        setLoading(true);
        const response = await fetch(sampleUrl);
        const blob = await response.blob();
        const file = new File([blob], 'preset_specimen.jpg', { type: 'image/jpeg' });
        await executeDiagnosis(file, sampleUrl);
      } catch (err) {
        alert('Failed to load specimen preset: ' + err.message);
        setLoading(false);
      }
    });
  });

  // 4. File Upload Handler
  function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, or WEBP).');
      return;
    }
    presetButtons.forEach(b => b.classList.remove('active'));
    const previewUrl = URL.createObjectURL(file);
    executeDiagnosis(file, previewUrl);
  }

  // 5. Execute Diagnosis (Parallel Predict + Explain)
  async function executeDiagnosis(file, previewUrl) {
    setLoading(true);

    const formDataPredict = new FormData();
    formDataPredict.append('file', file);

    const formDataExplain = new FormData();
    formDataExplain.append('file', file);

    try {
      // Execute both prediction and Grad-CAM in parallel
      const [predictRes, explainRes] = await Promise.all([
        fetch('/predict?top_k=3', { method: 'POST', body: formDataPredict }),
        fetch('/explain', { method: 'POST', body: formDataExplain })
      ]);

      if (!predictRes.ok) {
        const err = await predictRes.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      if (!explainRes.ok) {
        const err = await explainRes.json();
        throw new Error(err.detail || 'Grad-CAM explainability failed');
      }

      const predictData = await predictRes.json();
      const explainData = await explainRes.json();

      renderResults(predictData, explainData, previewUrl);
      checkDriftStatus();
    } catch (err) {
      alert('Inference error: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  // 6. Render Results into UI
  function renderResults(predictData, explainData, rawUrl) {
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Visualizer images
    rawLeafImg.src = rawUrl;
    heatmapImg.src = `data:image/jpeg;base64,${explainData.heatmap_image_base64}`;
    
    // Latency
    latencyBadge.textContent = `⚡ ${predictData.inference_latency_ms}ms`;

    // Reset slider to balanced 50%
    setOpacity(50);

    // Primary diagnosis
    diagCrop.textContent = predictData.crop;
    diagCondition.textContent = predictData.disease;
    diagScientific.textContent = predictData.scientific_name || 'Associated agricultural pathogen';
    diagConfidence.textContent = `${(predictData.confidence * 100).toFixed(1)}%`;

    // Severity pill styling
    const sev = (predictData.severity || 'MODERATE').toLowerCase();
    severityPill.textContent = predictData.severity || 'MODERATE';
    severityPill.className = `severity-pill ${sev}`;

    // Yield risk
    yieldRiskText.textContent = predictData.yield_loss_risk_pct;

    // Candidate progress bars
    candidatesList.innerHTML = '';
    predictData.top_k_predictions.forEach(cand => {
      const pct = (cand.confidence * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'candidate-row';
      row.innerHTML = `
        <div class="cand-meta">
          <span class="cand-name">${cand.crop} — ${cand.condition}</span>
          <span class="cand-score">${pct}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${pct}%;"></div>
        </div>
      `;
      candidatesList.appendChild(row);
    });

    // Recommendations
    recImmediateList.innerHTML = '';
    const immediates = predictData.agronomic_recommendations.immediate_actions || [];
    immediates.forEach(action => {
      const li = document.createElement('li');
      li.textContent = action;
      recImmediateList.appendChild(li);
    });

    recChemical.textContent = predictData.agronomic_recommendations.chemical_control || 'Consult local agronomic regulations.';
    recOrganic.textContent = predictData.agronomic_recommendations.organic_bio_control || 'Standard bio-control treatments.';
    recPreventative.textContent = predictData.agronomic_recommendations.preventative_practices || 'Crop rotation and aeration.';
  }

  // 7. Slider & Opacity Logic
  function setOpacity(val) {
    opacitySlider.value = val;
    opacityLabel.textContent = `Heatmap Opacity: ${val}%`;
    heatmapImg.style.opacity = val / 100;

    pillButtons.forEach(btn => {
      if (parseInt(btn.getAttribute('data-val')) === val) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  opacitySlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    opacityLabel.textContent = `Heatmap Opacity: ${val}%`;
    heatmapImg.style.opacity = val / 100;
    pillButtons.forEach(btn => btn.classList.remove('active'));
  });

  pillButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const val = parseInt(btn.getAttribute('data-val'));
      setOpacity(val);
    });
  });

  function setLoading(isLoading) {
    if (isLoading) {
      emptyState.classList.add('hidden');
      resultsContent.classList.add('hidden');
      loadingState.classList.remove('hidden');
    } else {
      loadingState.classList.add('hidden');
    }
  }
});
