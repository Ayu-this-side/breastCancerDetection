/* ==========================================================================
   ONCOVISION AI - FRONTEND INTERACTION & CHART JS LOGIC
   ========================================================================== */

// 10 Key Wisconsin Breast Cancer Dataset Features Metadata & Ranges
const FEATURES_CONFIG = [
  { id: 'mean_radius', label: 'Mean Radius', unit: 'mm', min: 6.0, max: 30.0, step: 0.1, default: 14.12, desc: 'Average distance from center to perimeter points' },
  { id: 'mean_texture', label: 'Mean Texture', unit: 'sd', min: 9.0, max: 40.0, step: 0.1, default: 19.29, desc: 'Standard deviation of gray-scale values' },
  { id: 'mean_perimeter', label: 'Mean Perimeter', unit: 'mm', min: 40.0, max: 190.0, step: 0.5, default: 91.96, desc: 'Nuclear cell boundary length' },
  { id: 'mean_area', label: 'Mean Area', unit: 'mm²', min: 140.0, max: 2500.0, step: 1.0, default: 654.8, desc: 'Total surface area of cell nucleus' },
  { id: 'mean_smoothness', label: 'Mean Smoothness', unit: 'var', min: 0.05, max: 0.17, step: 0.005, default: 0.096, desc: 'Local variation in radius lengths' },
  { id: 'mean_compactness', label: 'Mean Compactness', unit: 'ratio', min: 0.02, max: 0.35, step: 0.005, default: 0.104, desc: 'Perimeter² / area - 1.0' },
  { id: 'mean_concavity', label: 'Mean Concavity', unit: 'ratio', min: 0.0, max: 0.45, step: 0.005, default: 0.088, desc: 'Severity of concave portions of the contour' },
  { id: 'mean_concave_points', label: 'Mean Concave Points', unit: 'pts', min: 0.0, max: 0.20, step: 0.005, default: 0.048, desc: 'Number of concave portions of contour' },
  { id: 'mean_symmetry', label: 'Mean Symmetry', unit: 'ratio', min: 0.10, max: 0.35, step: 0.005, default: 0.181, desc: 'Cellular symmetry alignment index' },
  { id: 'mean_fractal_dimension', label: 'Mean Fractal Dimension', unit: 'fd', min: 0.04, max: 0.10, step: 0.001, default: 0.062, desc: 'Coastline approximation boundary roughness' }
];

// Presets based on actual Wisconsin Clinical Samples
const PRESETS = {
  malignant: {
    mean_radius: 17.99,
    mean_texture: 10.38,
    mean_perimeter: 122.8,
    mean_area: 1001.0,
    mean_smoothness: 0.1184,
    mean_compactness: 0.2776,
    mean_concavity: 0.3001,
    mean_concave_points: 0.1471,
    mean_symmetry: 0.2419,
    mean_fractal_dimension: 0.0787
  },
  benign: {
    mean_radius: 13.54,
    mean_texture: 14.36,
    mean_perimeter: 87.46,
    mean_area: 566.3,
    mean_smoothness: 0.09779,
    mean_compactness: 0.08129,
    mean_concavity: 0.06664,
    mean_concave_points: 0.04781,
    mean_symmetry: 0.1885,
    mean_fractal_dimension: 0.05766
  },
  baseline: {
    mean_radius: 14.12,
    mean_texture: 19.29,
    mean_perimeter: 91.96,
    mean_area: 654.8,
    mean_smoothness: 0.096,
    mean_compactness: 0.104,
    mean_concavity: 0.088,
    mean_concave_points: 0.048,
    mean_symmetry: 0.181,
    mean_fractal_dimension: 0.062
  }
};

let populationStats = null;
let radarChartInstance = null;
let barChartInstance = null;

// Initialize Web App on Load
document.addEventListener('DOMContentLoaded', () => {
  renderFeatureInputGrid();
  fetchModelMetaData();
  
  // Trigger initial baseline prediction after small render delay
  setTimeout(() => {
    runPrediction();
  }, 400);
});

// Render the 10 inputs dynamically into HTML
function renderFeatureInputGrid() {
  const container = document.getElementById('features-container');
  if (!container) return;

  container.innerHTML = FEATURES_CONFIG.map(feat => `
    <div class="input-group">
      <div class="input-label-row">
        <label for="${feat.id}" class="input-label">${feat.label}</label>
        <span class="input-val-badge" id="badge-${feat.id}">${feat.default} ${feat.unit}</span>
      </div>
      <div class="slider-container">
        <input 
          type="range" 
          id="slider-${feat.id}" 
          min="${feat.min}" 
          max="${feat.max}" 
          step="${feat.step}" 
          value="${feat.default}"
          oninput="syncFeatureInput('${feat.id}', this.value, 'slider')"
        >
        <input 
          type="number" 
          id="num-${feat.id}" 
          class="glass-num-input" 
          min="${feat.min}" 
          max="${feat.max}" 
          step="${feat.step}" 
          value="${feat.default}"
          onchange="syncFeatureInput('${feat.id}', this.value, 'num')"
        >
      </div>
    </div>
  `).join('');
}

// Synchronize Slider and Numeric Inputs
function syncFeatureInput(id, val, source) {
  const numVal = parseFloat(val) || 0;
  const config = FEATURES_CONFIG.find(f => f.id === id);
  const badge = document.getElementById(`badge-${id}`);
  const slider = document.getElementById(`slider-${id}`);
  const num = document.getElementById(`num-${id}`);

  if (source === 'slider' && num) num.value = numVal;
  if (source === 'num' && slider) slider.value = numVal;
  if (badge && config) badge.textContent = `${numVal} ${config.unit}`;
}

// Load Presets (Malignant, Benign, Baseline)
function loadPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  Object.keys(preset).forEach(featId => {
    const val = preset[featId];
    syncFeatureInput(featId, val, 'slider');
    syncFeatureInput(featId, val, 'num');
  });

  // Run prediction automatically on preset switch
  runPrediction();
}

// Fetch Model Info & Baseline Population Stats from Flask API
async function fetchModelMetaData() {
  try {
    const res = await fetch('/api/model-info');
    const data = await res.json();
    if (data.status === 'success') {
      const metrics = data.metrics;
      populationStats = data.population_stats;

      // Update counters in Hero section
      if (document.getElementById('val-accuracy')) {
        document.getElementById('val-accuracy').textContent = `${metrics.accuracy}%`;
      }
      if (document.getElementById('val-auc')) {
        document.getElementById('val-auc').textContent = `${metrics.roc_auc}%`;
      }

      // Render Bar Chart for Feature Importances
      initBarChart(metrics.feature_importances);
    }
  } catch (err) {
    console.warn('Could not fetch model metadata:', err);
  }
}

// Execute Prediction Request to Flask Server
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span>Computing Cell Metrics...</span>`;
  }

  // Collect input data
  const payload = {};
  FEATURES_CONFIG.forEach(feat => {
    const input = document.getElementById(`num-${feat.id}`);
    payload[feat.id] = input ? parseFloat(input.value) : feat.default;
  });

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (result.status === 'success') {
      updateResultDisplay(result.prediction, result.anomalies);
      updateRadarChart(payload);
    } else {
      alert(`Prediction Error: ${result.message}`);
    }

  } catch (err) {
    console.error('API Error:', err);
    alert('Failed to connect to Flask backend server.');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span>Execute Diagnostic Analysis</span> &rarr;`;
    }
  }
}

// Update Gauge Meter & Diagnosis Badges
function updateResultDisplay(pred, anomalies) {
  const probVal = pred.malignant_probability;
  const probEl = document.getElementById('result-probability');
  const badgeEl = document.getElementById('result-badge');
  const confTagEl = document.getElementById('confidence-tag');
  const gaugeArc = document.getElementById('gauge-arc');
  const gaugeWrapper = document.querySelector('.gauge-wrapper');

  if (gaugeWrapper) {
    gaugeWrapper.style.transform = 'scale(0.95)';
    setTimeout(() => { gaugeWrapper.style.transform = 'scale(1)'; }, 200);
  }

  if (probEl) probEl.textContent = `${probVal}%`;
  
  if (badgeEl) {
    badgeEl.textContent = pred.diagnosis;
    badgeEl.style.color = pred.risk_color;
    badgeEl.style.borderColor = pred.risk_color;
    badgeEl.style.background = `${pred.risk_color}18`; // Translucent glow
  }

  if (confTagEl) {
    confTagEl.textContent = `${pred.risk_level} • ${pred.confidence_tag}`;
  }

  // Update Arc Dash Offset (Full Circumference = 2 * PI * 75 ≈ 471)
  if (gaugeArc) {
    const maxOffset = 471;
    const offset = maxOffset - (maxOffset * (probVal / 100));
    gaugeArc.style.strokeDashoffset = offset;
    gaugeArc.style.stroke = pred.risk_color;
  }

  // Populate Anomalies list
  const anomalyBox = document.getElementById('anomaly-box');
  const anomalyList = document.getElementById('anomaly-list');
  if (anomalyBox && anomalyList) {
    if (anomalies && anomalies.length > 0) {
      anomalyBox.style.display = 'block';
      anomalyList.innerHTML = anomalies.map(anom => `
        <div class="anomaly-item">
          <span>${anom.label}</span>
          <span class="anomaly-tag">+${anom.elevation_pct}% vs Benign</span>
        </div>
      `).join('');
    } else {
      anomalyBox.style.display = 'none';
    }
  }
}

// Render & Update Chart.js Radar Chart
function updateRadarChart(userInput) {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;

  const labels = FEATURES_CONFIG.map(f => f.label.replace('Mean ', ''));

  // Normalize values on scale [0, 100] for visual comparison
  const normalize = (val, featId) => {
    const config = FEATURES_CONFIG.find(f => f.id === featId);
    if (!config) return 50;
    return Math.min(100, Math.max(0, ((val - config.min) / (config.max - config.min)) * 100));
  };

  const patientNormalized = FEATURES_CONFIG.map(f => normalize(userInput[f.id], f.id));
  
  const benignMeans = populationStats ? populationStats.benign_means : {};
  const malignantMeans = populationStats ? populationStats.malignant_means : {};

  const benignNormalized = FEATURES_CONFIG.map(f => normalize(benignMeans[f.id] || PRESETS.benign[f.id], f.id));
  const malignantNormalized = FEATURES_CONFIG.map(f => normalize(malignantMeans[f.id] || PRESETS.malignant[f.id], f.id));

  if (radarChartInstance) {
    radarChartInstance.data.datasets[0].data = patientNormalized;
    radarChartInstance.update();
    return;
  }

  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Patient Input Profile',
          data: patientNormalized,
          borderColor: '#FF2E93',
          backgroundColor: 'rgba(255, 46, 147, 0.25)',
          pointBackgroundColor: '#FF65B2',
          pointBorderColor: '#FFF',
          borderWidth: 2.5
        },
        {
          label: 'Benign Average Baseline',
          data: benignNormalized,
          borderColor: '#00E699',
          backgroundColor: 'rgba(0, 230, 153, 0.1)',
          pointBackgroundColor: '#00E699',
          borderWidth: 1.5,
          borderDash: [4, 4]
        },
        {
          label: 'Malignant Population Average',
          data: malignantNormalized,
          borderColor: '#FFB800',
          backgroundColor: 'rgba(255, 184, 0, 0.1)',
          pointBackgroundColor: '#FFB800',
          borderWidth: 1.5,
          borderDash: [4, 4]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { color: '#94A3B8', font: { size: 10, family: 'Plus Jakarta Sans' } },
          ticks: { display: false, max: 100, min: 0 }
        }
      },
      plugins: {
        legend: { labels: { color: '#F0F4FC', font: { family: 'Plus Jakarta Sans', size: 11 } } }
      }
    }
  });
}

// Render Feature Importance Bar Chart
function initBarChart(importances) {
  const ctx = document.getElementById('barChart');
  if (!ctx || !importances) return;

  const sortedPairs = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  const labels = sortedPairs.map(p => p[0].replace('mean_', '').replace('_', ' ').toUpperCase());
  const dataVals = sortedPairs.map(p => p[1] * 100);

  if (barChartInstance) barChartInstance.destroy();

  barChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Influence Weight (%)',
        data: dataVals,
        backgroundColor: 'rgba(255, 46, 147, 0.65)',
        borderColor: '#FF2E93',
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94A3B8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#F0F4FC', font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}
