// Global variables
let rocChart = null;
let modelData = {};

document.addEventListener('DOMContentLoaded', () => {
    const dataScript = document.getElementById('model-comparison-data');
    if (dataScript) {
        // Parse the JSON data sent from Django
        modelData = JSON.parse(dataScript.textContent);

        // Load Random Forest by default on page load
        if (Object.keys(modelData).length > 0 && modelData['random_forest']) {
            switchModel('random_forest');
        }
    }
});

function switchModel(modelName) {
    // 1. Tab UI Mapping
    const tabs = {
        'logistic_regression': 'tab-lr', // Updated from Decision Tree
        'random_forest': 'tab-rf',
        'xgboost': 'tab-xgb'
    };

    // 2. Reset all tabs to inactive styling
    Object.values(tabs).forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('bg-white', 'dark:bg-zinc-700', 'text-black', 'dark:text-white', 'shadow-sm', 'active');
            el.classList.add('text-gray-500');
        }
    });

    // 3. Highlight the active tab
    const activeTab = document.getElementById(tabs[modelName]);
    if (activeTab) {
        activeTab.classList.remove('text-gray-500');
        activeTab.classList.add('bg-white', 'dark:bg-zinc-700', 'text-black', 'dark:text-white', 'shadow-sm', 'active');
    }

    // 4. Get the specific data for the selected model
    const data = modelData[modelName];
    if (!data) return;

    // 5. Update Metrics Cards
    document.getElementById('accuracy-val').textContent = data.accuracy.toFixed(1) + '%';
    document.getElementById('precision-val').textContent = data.precision.toFixed(1) + '%';
    document.getElementById('recall-val').textContent = data.recall.toFixed(1) + '%';
    document.getElementById('f1-val').textContent = data.f1.toFixed(1) + '%';

    // 6. Update Confusion Matrix Matrix
    document.getElementById('tn-val').textContent = data.tn || 0;
    document.getElementById('fp-val').textContent = data.fp || 0;
    document.getElementById('fn-val').textContent = data.fn || 0;
    document.getElementById('tp-val').textContent = data.tp || 0;

    // 7. Render/Update the ROC Chart
    updateRocChart(data.roc);
}

function updateRocChart(rocData) {
    const ctx = document.getElementById('rocCurveChart').getContext('2d');

    // Convert array format [[0,0], [0.5, 0.8]] to Chart.js format [{x:0, y:0}]
    const formattedData = rocData ? rocData.map(pt => ({ x: pt[0], y: pt[1] })) : [{ x: 0, y: 0 }, { x: 1, y: 1 }];

    if (rocChart) {
        // Update existing chart
        rocChart.data.datasets[0].data = formattedData;
        rocChart.update();
    } else {
        // Create new chart
        rocChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Model ROC',
                        data: formattedData,
                        borderColor: '#10b981', // Emerald green
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        showLine: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointBackgroundColor: '#10b981'
                    },
                    {
                        label: 'Random Guess',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: '#9ca3af',
                        borderDash: [5, 5],
                        showLine: true,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        min: 0, max: 1,
                        title: { display: true, text: 'False Positive Rate' },
                        grid: { color: 'rgba(156, 163, 175, 0.1)' }
                    },
                    y: {
                        min: 0, max: 1,
                        title: { display: true, text: 'True Positive Rate' },
                        grid: { color: 'rgba(156, 163, 175, 0.1)' }
                    }
                }
            }
        });
    }
}