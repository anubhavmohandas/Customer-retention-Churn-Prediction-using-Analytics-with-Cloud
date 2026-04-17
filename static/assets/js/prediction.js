// --- Global Variables ---
let currentUploadedFile = null;

/**
 * UI State Management: Tab Switching
 */
function switchTab(tabName) {
    document.getElementById('tab-single').classList.toggle('active', tabName === 'single');
    document.getElementById('tab-bulk').classList.toggle('active', tabName === 'bulk');
    document.getElementById('content-single').classList.toggle('hidden', tabName !== 'single');
    document.getElementById('content-bulk').classList.toggle('hidden', tabName !== 'bulk');

    if (tabName === 'single') {
        document.getElementById('single-outcome-view').classList.remove('hidden');
        document.getElementById('bulk-outcome-view').classList.add('hidden');
    } else {
        document.getElementById('single-outcome-view').classList.add('hidden');
        if (currentUploadedFile) document.getElementById('bulk-outcome-view').classList.remove('hidden');
    }
}

/**
 * Slider UI Sync
 */
function updateSlider(sliderId, valId, suffix, isPrefix = false) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(valId).textContent = isPrefix ? suffix + val : val + suffix;

    // Auto-update preview (BUT DO NOT SAVE TO HISTORY)
    runSingleSimulation(false);
}

/**
 * API Utils
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

/**
 * 🚀 SINGLE SUBSCRIBER SIMULATION
 */
async function runSingleSimulation(shouldSave = false) {
    const btn = document.getElementById('run-single-btn');
    const spinner = document.getElementById('single-spinner');
    const btnText = document.getElementById('single-btn-text');

    // Only show loading state if we are performing an "Official Run"
    if (shouldSave) {
        btn.disabled = true;
        spinner.classList.remove('hidden');
        btnText.textContent = "Processing...";
    }

    const payload = {
        tenure: parseFloat(document.getElementById('slider-tenure').value),
        monthly_charges: parseFloat(document.getElementById('slider-charges').value),
        contract: document.getElementById('select-contract').value,
        model: document.getElementById('model-select-single').value,
        save_to_history: shouldSave === true  // Ensure boolean type
    };

    try {
        const response = await fetch('/api/predict-single/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (response.ok) {
            // UPDATED: Pass data.confidence to the update function
            updateSingleUI(data.probability, payload.monthly_charges, payload.tenure, payload.contract, data.confidence);
            if (shouldSave) {
                console.log("Snapshot archived to History");
            }
        }
    } catch (err) {
        console.error("Simulation failed", err);
    } finally {
        if (shouldSave) {
            btn.disabled = false;
            spinner.classList.add('hidden');
            btnText.textContent = "Run ML Prediction";
        }
    }
}

// UPDATED: Added confidence to the function arguments
function updateSingleUI(prob, charges, tenure, contract, confidence) {
    const outcomeVal = document.getElementById('outcome-value');
    const badge = document.getElementById('risk-badge');
    const mrrImpact = document.getElementById('mrr-impact');
    const recommendation = document.getElementById('ai-recommendation');
    const confidenceEl = document.getElementById('ai-confidence'); // Target new ID
    const risk = (prob * 100).toFixed(1);

    outcomeVal.textContent = risk + '%';
    mrrImpact.textContent = '$' + (charges * prob).toFixed(2);

    // UPDATED: Set the dynamic confidence score if provided by backend
    if (confidence) {
        confidenceEl.textContent = confidence + '%';
    }

    // Update Factor Bars
    document.getElementById('factor-contract-bar').style.width = (contract === 'Month-to-month' ? '88%' : '15%');
    document.getElementById('factor-tenure-bar').style.width = Math.max(10, 100 - (tenure * 1.4)) + '%';

    if (risk > 70) {
        outcomeVal.className = "text-8xl font-black mb-4 text-red-500 animate-pulse";
        badge.textContent = "CRITICAL RISK";
        badge.className = "inline-block px-5 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest bg-red-100 text-red-600";
        recommendation.textContent = "Immediate intervention required. High probability of churn due to pricing/tenure ratio.";
    } else if (risk > 30) {
        outcomeVal.className = "text-8xl font-black mb-4 text-yellow-500";
        badge.textContent = "MODERATE RISK";
        badge.className = "inline-block px-5 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest bg-yellow-100 text-yellow-700";
        recommendation.textContent = "Monitor account closely. Recommend proactive loyalty outreach.";
    } else {
        outcomeVal.className = "text-8xl font-black mb-4 text-emerald-500";
        badge.textContent = "STABLE ACCOUNT";
        badge.className = "inline-block px-5 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest bg-emerald-100 text-emerald-600";
        recommendation.textContent = "Subscriber is healthy. High potential for cross-selling value-added services.";
    }
}

/**
 * 🚀 BULK BATCH ANALYSIS
 */
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('csv-upload');

if (dropZone && fileInput) {
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = handleFiles;
    dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('bg-gray-100'); };
    dropZone.ondragleave = () => { dropZone.classList.remove('bg-gray-100'); };
    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('bg-gray-100');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFiles();
        }
    };
}

function handleFiles() {
    const file = fileInput.files[0];
    if (file && file.name.endsWith('.csv')) {
        currentUploadedFile = file;
        document.getElementById('file-name-text').textContent = file.name;
        document.getElementById('upload-zone-container').classList.add('hidden');
        document.getElementById('data-preview-container').classList.remove('hidden');

        const btn = document.getElementById('run-bulk-btn');
        btn.disabled = false;
        btn.className = "w-full bg-black dark:bg-white text-white dark:text-black py-5 curved font-extrabold text-sm tracking-widest uppercase transition-all";
    }
}

function resetUpload() {
    currentUploadedFile = null;
    fileInput.value = '';
    document.getElementById('upload-zone-container').classList.remove('hidden');
    document.getElementById('data-preview-container').classList.add('hidden');
    document.getElementById('run-bulk-btn').disabled = true;
}

async function runBulkAnalysis() {
    if (!currentUploadedFile) return;

    const btn = document.getElementById('run-bulk-btn');
    const spinner = document.getElementById('bulk-spinner');
    const btnText = document.getElementById('bulk-btn-text');
    const model = document.getElementById('model-select-single').value;

    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append('file', currentUploadedFile);
    formData.append('model', model);

    try {
        const response = await fetch('/api/predict-bulk/', {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': getCookie('csrftoken') }
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('bulk-avg-risk').textContent = data.avg_risk + '%';
            document.getElementById('bulk-high-risk').textContent = data.high_risk_count;

            const total = data.total_processed;
            updateDistribution('low', data.low_risk_count, total);
            updateDistribution('med', data.med_risk_count, total);
            updateDistribution('high', data.high_risk_count, total);

            document.getElementById('bulk-outcome-view').classList.remove('hidden');
            document.getElementById('single-outcome-view').classList.add('hidden');
        }
    } catch (err) { alert("Server error"); }
    finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.textContent = "Process Batch";
    }
}

function updateDistribution(id, count, total) {
    const pct = total > 0 ? ((count / total) * 100).toFixed(0) : 0;
    document.getElementById(`dist-${id}-num`).textContent = count;
    document.getElementById(`dist-${id}-bar`).style.width = pct + '%';
}