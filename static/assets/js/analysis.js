/**
 * Retain.Ai - Risk Analysis Engine
 * Handles Dynamic Chart.js Visualizations from Django Data
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Data Extraction from the HTML Bridge
    const dataElement = document.getElementById('analysis-data');
    if (!dataElement) {
        console.warn("Analysis data bridge not found. Using placeholder logic.");
        return;
    }

    const rawData = JSON.parse(dataElement.textContent);

    // 2. Theme Detection
    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? '#9ca3af' : '#6b7280';
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';

    // --- Chart 1: Contract Risk Breakdown (Bar) ---
    const contractCtx = document.getElementById('contractRiskChart');
    if (contractCtx) {
        new Chart(contractCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: rawData.contractLabels,
                datasets: [{
                    label: 'Avg Churn Risk %',
                    data: rawData.contractValues,
                    backgroundColor: rawData.contractValues.map(v =>
                        v > 50 ? '#ef4444' : (v > 25 ? '#eab308' : '#10b981')
                    ),
                    borderRadius: 12,
                    barThickness: 40
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: gridColor },
                        ticks: { color: textColor, callback: v => v + '%' }
                    },
                    x: { grid: { display: false }, ticks: { color: textColor } }
                }
            }
        });
    }

    // --- Chart 2: Correlation Heatmap (Bubble) ---
    const correlationCtx = document.getElementById('correlationChart');
    if (correlationCtx) {
        new Chart(correlationCtx.getContext('2d'), {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'Risk Clusters',
                    data: rawData.correlationData,
                    backgroundColor: 'rgba(239, 68, 68, 0.5)',
                    hoverBackgroundColor: '#ef4444',
                    borderWidth: 1,
                    borderColor: '#ef4444'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (c) => `Tenure: ${c.raw.x}mo | Charge: $${c.raw.y} | Risk Size: ${c.raw.r}`
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Monthly Charge ($)', color: textColor, font: { weight: 'bold' } },
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    },
                    x: {
                        title: { display: true, text: 'Tenure (Months)', color: textColor, font: { weight: 'bold' } },
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    }

    // --- Chart 3: Subscriber Survival Curve (Line) ---
    const survivalCtx = document.getElementById('survivalChart');
    if (survivalCtx) {
        new Chart(survivalCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['0m', '6m', '12m', '18m', '24m', '36m', '48m', '72m'],
                datasets: [{
                    label: 'Survival Rate %',
                    data: [100, 82, 68, 55, 48, 42, 38, 35], // Predictive trend
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#10b981'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: textColor, callback: v => v + '%' }
                    },
                    x: { grid: { display: false }, ticks: { color: textColor } }
                }
            }
        });
    }
});