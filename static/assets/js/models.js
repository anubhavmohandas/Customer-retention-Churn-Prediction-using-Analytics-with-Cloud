/**
 * RetainIQ - Model Comparison Engine
 * Handles dynamic UI updates for Model Registry
 */

let rocChart;
let comparisonData;

document.addEventListener('DOMContentLoaded', () => {
    const dataElement = document.getElementById('model-comparison-data');
    if (!dataElement) return;

    try {
        // Parse the JSON data injected by Django
        comparisonData = JSON.parse(dataElement.textContent);
    } catch (e) {
        console.error("Failed to parse model data", e);
        return;
    }

    // Check if we have any valid data to display (accuracy > 0)
    // This allows the UI to stay in "initial" state if data is empty
    const defaultModel = 'random_forest';

    // Initialize Chart first
    initChart();

    // Switch to default model to populate UI
    switchModel(defaultModel);
});

/**
 * Updates all text metrics, the confusion matrix, and the ROC chart
 * @param {string} modelKey - 'decision_tree', 'random_forest', or 'xgboost'
 */
function switchModel(modelKey) {
    const model = comparisonData[modelKey];
    if (!model) return;

    // 1. Update Numeric Metrics with 1-decimal precision
    // Using parseFloat to ensure we handle numbers correctly even if they come as strings
    document.getElementById('accuracy-val').innerText = parseFloat(model.accuracy).toFixed(1) + '%';
    document.getElementById('precision-val').innerText = parseFloat(model.precision).toFixed(1) + '%';
    document.getElementById('recall-val').innerText = parseFloat(model.recall).toFixed(1) + '%';
    document.getElementById('f1-val').innerText = parseFloat(model.f1).toFixed(1) + '%';

    // 2. Update Confusion Matrix values
    document.getElementById('tn-val').innerText = model.tn;
    document.getElementById('fp-val').innerText = model.fp;
    document.getElementById('fn-val').innerText = model.fn;
    document.getElementById('tp-val').innerText = model.tp;

    // 3. Update Chart Data
    // Expects model.roc to be an array of coordinates: [[0,0], [x,y], [1,1]]
    if (rocChart && model.roc) {
        rocChart.data.datasets[0].data = model.roc.map(point => ({
            x: point[0],
            y: point[1]
        }));
        rocChart.update();
    }

    // 4. Update Tab Styling
    updateTabStyling(modelKey);
}

/**
 * Handles the visual active/inactive states of the model selection tabs
 */
function updateTabStyling(activeKey) {
    const tabMap = {
        'logistic_regression': 'tab-lr',
        'random_forest':       'tab-rf',
        'xgboost':             'tab-xgb',
    };

    // Remove active styles from all tabs
    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.classList.remove('bg-white', 'dark:bg-zinc-700', 'text-black', 'dark:text-white', 'shadow-sm');
        tab.classList.add('text-gray-500');
    });

    // Add active styles to the selected tab
    const activeTab = document.getElementById(tabMap[activeKey]);
    if (activeTab) {
        activeTab.classList.remove('text-gray-500');
        activeTab.classList.add('bg-white', 'dark:bg-zinc-700', 'text-black', 'dark:text-white', 'shadow-sm');
    }
}

/**
 * Creates the initial Chart.js instance with responsive configuration
 */
function initChart() {
    const canvas = document.getElementById('rocCurveChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const isDark = document.documentElement.classList.contains('dark');

    // Theme-aware colors
    const textColor = isDark ? '#9ca3af' : '#6b7280';
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
    const primaryColor = '#10b981'; // Emerald-500

    rocChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Model Performance',
                    data: [], // Populated in switchModel
                    borderColor: primaryColor,
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    backgroundColor: isDark ? 'rgba(16, 185, 129, 0.05)' : 'rgba(16, 185, 129, 0.02)',
                    pointRadius: 2,
                    pointBackgroundColor: primaryColor
                },
                {
                    label: 'Random Chance (Baseline)',
                    data: [{x: 0, y: 0}, {x: 1, y: 1}],
                    borderColor: isDark ? '#3f3f46' : '#d4d4d8',
                    borderDash: [6, 6],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: isDark ? '#18181b' : '#ffffff',
                    titleColor: isDark ? '#ffffff' : '#000000',
                    bodyColor: isDark ? '#a1a1aa' : '#71717a',
                    borderColor: isDark ? '#27272a' : '#e4e4e7',
                    borderWidth: 1,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'False Positive Rate', color: textColor, font: { size: 10, weight: 'bold' } },
                    min: 0,
                    max: 1,
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { size: 10 } }
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'True Positive Rate', color: textColor, font: { size: 10, weight: 'bold' } },
                    min: 0,
                    max: 1,
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { size: 10 } }
                }
            }
        }
    });
}