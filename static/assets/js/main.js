/* Mobile Sidebar */
function openSidebar() {
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('sidebar-overlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebar-overlay').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSidebar();
});

/* Dark Mode Toggle */
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

function applyTheme(isDark) {
  html.classList.toggle('dark', isDark);
  const icon = document.getElementById('themeIcon');
  const text = document.getElementById('themeText');
  if (icon) icon.textContent = isDark ? '☀️' : '🌙';
  if (text) text.textContent = isDark ? 'Light Mode' : 'Dark Mode';
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

if (themeToggle) {
  themeToggle.addEventListener('click', () => applyTheme(!html.classList.contains('dark')));
}

const saved = localStorage.getItem('theme');
if (saved) {
  applyTheme(saved === 'dark');
} else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
  applyTheme(true);
}

// --- Inside runBulkAnalysis in main.js ---
async function runBulkAnalysis() {
    // ... (previous setup code) ...
    try {
        const response = await fetch('/api/predict-bulk/', {
            method: 'POST',
            body: formData,
            headers: { 'X-CSRFToken': getCookie('csrftoken') }
        });

        const data = await response.json();

        if (response.ok) {
            const riskDisplay = document.getElementById('bulk-avg-risk');
            const statusLabel = document.getElementById('bulk-status-label'); // Ensure this ID exists in HTML

            riskDisplay.textContent = data.avg_risk + '%';

            // --- HIGH ALARM UI UPDATE ---
            if (data.status === 'danger' || data.high_risk_count > 0) {
                // Trigger Red Alarm
                riskDisplay.classList.add('text-red-500', 'animate-pulse');
                riskDisplay.classList.remove('text-emerald-500', 'dark:text-white');

                if(statusLabel) {
                    statusLabel.textContent = `CRITICAL: ${data.high_risk_count} Subscribers at High Risk`;
                    statusLabel.className = "text-[10px] font-black text-red-500 uppercase tracking-widest animate-bounce";
                }
            } else {
                // Stable UI
                riskDisplay.classList.add('text-emerald-500');
                riskDisplay.classList.remove('text-red-500', 'animate-pulse');
                if(statusLabel) statusLabel.textContent = "Batch Analysis Stable";
            }

            // Update Distribution Bars
            updateDistribution('low', data.low_risk_count, data.total_processed);
            updateDistribution('med', data.med_risk_count, data.total_processed);
            updateDistribution('high', data.high_risk_count, data.total_processed);

            document.getElementById('bulk-outcome-view').classList.remove('hidden');
        }
    } catch (err) {
        console.error("Bulk Analysis Error:", err);
    }
}