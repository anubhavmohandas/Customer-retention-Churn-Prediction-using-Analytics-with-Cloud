/* Retain.Ai UI Utilities */

// --- Mobile Sidebar Logic ---
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebar-overlay');

function openSidebar() {
    if (sidebar && overlay) {
        sidebar.classList.add('translate-x-0'); // Using Tailwind transforms is smoother
        overlay.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function closeSidebar() {
    if (sidebar && overlay) {
        sidebar.classList.remove('translate-x-0');
        overlay.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// Close on ESC key
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeSidebar();
});

// --- Dark Mode Engine ---
function applyTheme(isDark) {
    const html = document.documentElement;
    const icon = document.getElementById('themeIcon');
    const text = document.getElementById('themeText');

    html.classList.toggle('dark', isDark);

    if (icon) icon.textContent = isDark ? '☀️' : '🌙';
    if (text) text.textContent = isDark ? 'Light Mode' : 'Dark Mode';

    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Initial Sync
const savedTheme = localStorage.getItem('theme');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
applyTheme(savedTheme === 'dark' || (!savedTheme && prefersDark));

// Toggle Event
document.getElementById('themeToggle')?.addEventListener('click', () => {
    applyTheme(!document.documentElement.classList.contains('dark'));
});