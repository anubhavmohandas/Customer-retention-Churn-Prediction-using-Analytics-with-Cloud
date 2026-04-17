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