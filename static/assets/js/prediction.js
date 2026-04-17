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

    const singleOutcome = document.getElementById('single-outcome-view');
    const bulkOutcome = document.getElementById('bulk-outcome-view');

    if (tabName === 'single') {
        singleOutcome.classList.remove('hidden');
        bulkOutcome.classList.add('hidden');
    } else {
        singleOutcome.classList.add('hidden');
        if (currentUploadedFile) {
            bulkOutcome.classList.remove('hidden');
        }
    }
}

/**
 * Slider UI Sync with Live Simulation
 */
function updateSlider(sliderId, valId, suffix, isPrefix = false) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(valId).textContent = isPrefix ? suffix + val : val + suffix;
    // Real-time feedback as the user moves the slider
    runSingleSimulation();
}

/**
 * File Handling Logic for Bulk CSV
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
    document.getElementById('bulk-outcome-view').classList.add('hidden');
    document.getElementById('run-bulk-btn').disabled = true;
}

/**
 * 🚀 API INTEGRATION
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

// 1. Single Subscriber Simulation (With Intelligence Features)
async function runSingleSimulation() {
    const outcomeVal = document.getElementById('outcome-value');
    const badge = document.getElementById('risk-badge');
    const mrrImpact = document.getElementById('mrr-impact');
    const recommendation = document.getElementById('ai-recommendation');

    const monthlyCharges = document.getElementById('slider-charges').value;
    const payload = {
        tenure: document.getElementById('slider-tenure').value,
        monthly_charges: monthlyCharges,
        contract: document.getElementById('select-contract').value,
        model: document.getElementById('model-select-single').value
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
            const prob = data.probability;
            const risk = (prob * 100).toFixed(1);

            // A. Update Probability
            outcomeVal.textContent = risk + '%';

            // B. Update Financial Impact (Revenue at Risk)
            const impact = (monthlyCharges * prob).toFixed(2);
            mrrImpact.textContent = '$' + impact;

            // C. UI Color & AI Recommendation Logic
            if (risk > 70) {
                outcomeVal.className = "text-7xl font-extrabold mb-2 text-red-500 animate-pulse";
                badge.textContent = "CRITICAL RISK";
                badge.className = "inline-block px-4 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-red-100 text-red-600";
                recommendation.textContent = "High priority! Offer a 'Contract Migration' credit or a hardware upgrade to lock in the subscriber.";
            } else if (risk > 30) {
                outcomeVal.className = "text-7xl font-extrabold mb-2 text-yellow-500";
                badge.textContent = "MODERATE RISK";
                badge.className = "inline-block px-4 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-yellow-100 text-yellow-700";
                recommendation.textContent = "Send a 'Loyalty Appreciation' discount and check for recent network service tickets.";
            } else {
                outcomeVal.className = "text-7xl font-extrabold mb-2 text-emerald-500";
                badge.textContent = "STABLE ACCOUNT";
                badge.className = "inline-block px-4 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-emerald-100 text-emerald-600";
                recommendation.textContent = "Customer is healthy. This is an ideal candidate for upselling high-tier data plans.";
            }
        }
    } catch (err) {
        console.error("Simulation failed", err);
    }
}

// 2. Bulk Batch Analysis
async function runBulkAnalysis() {
    if (!currentUploadedFile) return;

    const btn = document.getElementById('run-bulk-btn');
    const spinner = document.getElementById('bulk-spinner');
    const btnText = document.getElementById('bulk-btn-text');
    const selectedModel = document.getElementById('model-select-single').value;

    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = "Analyzing Batch...";

    const formData = new FormData();
    formData.append('file', currentUploadedFile);
    formData.append('model', selectedModel);

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
            updateDistribution('low', data.low_risk_count || 0, total);
            updateDistribution('med', data.med_risk_count || 0, total);
            updateDistribution('high', data.high_risk_count || 0, total);

            document.getElementById('bulk-outcome-view').classList.remove('hidden');
            document.getElementById('single-outcome-view').classList.add('hidden');
        } else {
            alert("Error: " + data.error);
        }
    } catch (err) {
        alert("Server error during batch analysis.");
    } finally {
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