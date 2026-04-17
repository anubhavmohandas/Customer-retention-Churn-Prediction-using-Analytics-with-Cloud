// --- Global Variables ---
let currentUploadedFile = null;

// --- Slider UI Sync ---
function updateSlider(sliderId, valId, suffix, isPrefix = false) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(valId).textContent = isPrefix ? suffix + val : val + suffix;
    // For single simulation, we can auto-run on slider change for a "live" feel
    runSingleSimulation();
}

// --- Tab Switching Logic ---
function switchTab(tabName) {
    // Button States
    document.getElementById('tab-single').classList.toggle('active', tabName === 'single');
    document.getElementById('tab-bulk').classList.toggle('active', tabName === 'bulk');

    // Content Visibility
    document.getElementById('content-single').classList.toggle('hidden', tabName !== 'single');
    document.getElementById('content-bulk').classList.toggle('hidden', tabName !== 'bulk');

    // Outcome Area Logic
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

// --- File Handling Logic ---
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

        // Enable Button
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

// --- 🚀 API INTEGRATION ---

// Helper for CSRF (Django Security)
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

// 1. Single Subscriber Simulation
async function runSingleSimulation() {
    const payload = {
        tenure: document.getElementById('slider-tenure').value,
        charges: document.getElementById('slider-charges').value,
        // Match the text in your CSV exactly
        contract: document.getElementById('select-contract').options[document.getElementById('select-contract').selectedIndex].text,
        model: document.getElementById('model-select-single').value
    };
    // ... rest of fetch call ...
}

// 2. Bulk Batch Analysis
async function runBulkAnalysis() {
    if (!currentUploadedFile) return;

    const btn = document.getElementById('run-bulk-btn');
    const spinner = document.getElementById('bulk-spinner');
    const btnText = document.getElementById('bulk-btn-text');

    // UI Loading
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = "Analyzing Batch...";

    const formData = new FormData();
    formData.append('file', currentUploadedFile);
    formData.append('model', 'ensemble'); // You can add a selector for this too

    try {
        const response = await fetch('/api/predict-bulk/', {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': getCookie('csrftoken') }
        });

        const data = await response.json();

        if (response.ok) {
            // Fill Cards
            document.getElementById('bulk-avg-risk').textContent = data.summary.average_risk + '%';
            document.getElementById('bulk-high-risk').textContent = data.summary.high_risk;

            // Fill Distribution Bars
            const total = data.summary.total_customers;
            updateDistribution('low', data.summary.low_risk, total);
            updateDistribution('med', data.summary.medium_risk, total);
            updateDistribution('high', data.summary.high_risk, total);

            // Switch UI
            document.getElementById('bulk-outcome-view').classList.remove('hidden');
            document.getElementById('single-outcome-view').classList.add('hidden');
        }
    } catch (err) {
        alert("Server error during batch analysis.");
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.textContent = "Run Batch Prediction";
    }
}

function updateDistribution(id, count, total) {
    const pct = ((count / total) * 100).toFixed(0);
    document.getElementById(`dist-${id}-num`).textContent = count;
    document.getElementById(`dist-${id}-bar`).style.width = pct + '%';
}