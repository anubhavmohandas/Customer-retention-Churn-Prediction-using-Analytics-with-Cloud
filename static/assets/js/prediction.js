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

const MAX_CSV_BYTES = 10 * 1024 * 1024; // 10 MB

function handleFiles() {
    const file = fileInput.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
        alert('Only CSV files are accepted.');
        fileInput.value = '';
        return;
    }

    if (file.size > MAX_CSV_BYTES) {
        alert(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum allowed size is 10 MB.`);
        fileInput.value = '';
        return;
    }

    currentUploadedFile = file;
    document.getElementById('file-name-text').textContent = file.name;
    document.getElementById('upload-zone-container').classList.add('hidden');
    document.getElementById('data-preview-container').classList.remove('hidden');

    const btn = document.getElementById('run-bulk-btn');
    btn.disabled = false;
    btn.className = "w-full bg-black dark:bg-white text-white dark:text-black py-5 curved font-extrabold text-sm tracking-widest uppercase transition-all";

    // Enable the Auto-Train button when a file is loaded
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) trainBtn.disabled = false;
}

function resetUpload() {
    currentUploadedFile = null;
    fileInput.value = '';
    document.getElementById('upload-zone-container').classList.remove('hidden');
    document.getElementById('data-preview-container').classList.add('hidden');
    document.getElementById('bulk-outcome-view').classList.add('hidden');
    document.getElementById('run-bulk-btn').disabled = true;
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) trainBtn.disabled = true;
}

/**
 * 🚀 API INTEGRATION
 */

function getCsrfToken() {
    // Read from cookie (CSRF_COOKIE_HTTPONLY must be False for this to work)
    const name = 'csrftoken';
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith(name + '='));
    return cookie ? decodeURIComponent(cookie.trim().substring(name.length + 1)) : '';
}

// 1. Single Subscriber Simulation
// saveToHistory=true  → "Run ML Prediction" button — saves result to ReportHistory
// saveToHistory=false → slider/select changes — live preview only
async function runSingleSimulation(saveToHistory = false) {
    const outcomeVal = document.getElementById('outcome-value');
    const badge = document.getElementById('risk-badge');
    const mrrImpact = document.getElementById('mrr-impact');
    const recommendation = document.getElementById('ai-recommendation');

    const monthlyCharges = parseFloat(document.getElementById('slider-charges').value);
    const tenure = parseFloat(document.getElementById('slider-tenure').value);
    const contract = document.getElementById('select-contract').value;

    const payload = {
        tenure,
        monthly_charges: monthlyCharges,
        contract,
        model: document.getElementById('model-select-single').value,
        save_to_history: saveToHistory,
    };

    // Show loading state on the explicit "Run" button
    const runBtn = document.getElementById('run-single-btn');
    const spinner = document.getElementById('single-spinner');
    const btnText = document.getElementById('single-btn-text');
    if (saveToHistory && runBtn) {
        runBtn.disabled = true;
        spinner.classList.remove('hidden');
        btnText.textContent = 'Running…';
    }

    try {
        const response = await fetch('/api/predict-single/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (response.ok) {
            const prob = data.probability;
            const risk = (prob * 100).toFixed(1);

            // A. Probability display
            outcomeVal.textContent = risk + '%';

            // B. AI Confidence
            const confidenceEl = document.getElementById('ai-confidence');
            if (confidenceEl) {
                const confidence = data.confidence
                    ? data.confidence.toFixed(1)
                    : (Math.max(prob, 1 - prob) * 100).toFixed(1);
                confidenceEl.textContent = confidence + '%';
            }

            // C. Revenue at Risk
            const impact = (monthlyCharges * prob).toFixed(2);
            mrrImpact.textContent = '$' + impact;

            // D. Factor bars — contract weight and tenure weight
            const contractBar = document.getElementById('factor-contract-bar');
            const tenureBar   = document.getElementById('factor-tenure-bar');
            if (contractBar) {
                // Month-to-month is the highest risk contract; weight reflects that
                const contractWeight = contract === 'Month-to-month' ? Math.min(95, risk * 1.1)
                                     : contract === 'One year'       ? Math.min(60, risk * 0.7)
                                     :                                  Math.min(30, risk * 0.4);
                contractBar.style.width = contractWeight.toFixed(0) + '%';
            }
            if (tenureBar) {
                // Shorter tenure = higher risk; invert the 1–72 range
                const tenureWeight = Math.max(5, 100 - (tenure / 72) * 100);
                tenureBar.style.width = Math.min(tenureWeight, risk * 1.05).toFixed(0) + '%';
            }

            // E. Risk badge & recommendation
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

            // F. Notify user if saved
            if (saveToHistory && data.saved) {
                btnText.textContent = '✓ Saved to History';
                setTimeout(() => { btnText.textContent = 'Run ML Prediction'; }, 2000);
            }
        } else {
            console.error("Prediction error:", data.error);
        }
    } catch (err) {
        console.error("Simulation failed", err);
    } finally {
        if (saveToHistory && runBtn) {
            runBtn.disabled = false;
            spinner.classList.add('hidden');
            if (btnText.textContent === 'Running…') btnText.textContent = 'Run ML Prediction';
        }
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
            headers: { 'X-CSRFToken': getCsrfToken() }
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
async function trainNewModel() {
    if (!currentUploadedFile) {
        alert("Please upload a CSV dataset first!");
        return;
    }

    const btn = document.getElementById('train-btn');
    const btnText = document.getElementById('train-btn-text');

    btn.disabled = true;
    btnText.textContent = "Training... (takes 10–30 seconds)";
    btn.classList.add("animate-pulse");

    const formData = new FormData();
    formData.append('file', currentUploadedFile);

    try {
        const response = await fetch('/api/train-models/', {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': getCsrfToken() }
        });
        const data = await response.json();

        if (response.ok) {
            alert("✅ Training complete! Models retrained on the new dataset. Visit AI Registry to see updated metrics.");
        } else {
            alert("❌ Training failed: " + data.error);
        }
    } catch (err) {
        alert("❌ Server connection failed.");
        console.error("Auto-train error", err);
    } finally {
        btnText.textContent = "🧠 Auto-Train New AI Brain";
        btn.disabled = false;
        btn.classList.remove("animate-pulse");
    }
}
