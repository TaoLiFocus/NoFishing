/**
 * NoFishing Popup Script - Redesigned with Tabbed Interface
 */

// Initialize popup
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Initialize toast
    if (typeof toast !== 'undefined' && toast.init) {
        toast.init();
    }

    // Initialize API client
    if (typeof apiClient !== 'undefined') {
        await apiClient.init();
    }

    // Setup tabs
    setupTabs();

    // Check authentication
    await checkAuth();

    // Health check
    await checkHealth();

    // Initialize home tab (default)
    if (typeof initHomeTab === 'function') {
        await initHomeTab();
    }
}

/**
 * Setup tabs
 */
function setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', async () => {
            const tabName = tab.dataset.tab;

            // Remove active class from all tabs and panels
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            // Add active class to clicked tab
            tab.classList.add('active');

            // Show corresponding panel
            const panel = document.getElementById(`${tabName}-panel`);
            if (panel) {
                panel.classList.add('active');
            }

            // Initialize tab specific content
            if (tabName === 'home' && typeof initHomeTab === 'function') {
                await initHomeTab();
            } else if (tabName === 'history' && typeof initHistoryTab === 'function') {
                await initHistoryTab();
            } else if (tabName === 'settings' && typeof initSettingsTab === 'function') {
                await initSettingsTab();
            }
        });
    });
}

/**
 * Check authentication status
 */
async function checkAuth() {
    const result = await chrome.storage.local.get(['apiToken', 'apiKey']);

    const tokenStatus = document.getElementById('tokenStatus');
    if (tokenStatus) {
        const dot = tokenStatus.querySelector('.status-dot');
        const text = tokenStatus.querySelector('.status-text');

        if (result.apiToken || result.apiKey) {
            if (dot) dot.classList.add('online');
            if (text) text.textContent = '已登录';
        } else {
            if (dot) dot.classList.add('offline');
            if (text) text.textContent = '未登录';

            // Show login modal after a short delay
            setTimeout(() => {
                showLoginModal();
            }, 500);
        }
    }
}

/**
 * Check API health
 */
async function checkHealth() {
    const apiStatus = document.getElementById('apiStatus');
    if (!apiStatus) return;

    const dot = apiStatus.querySelector('.status-dot');
    const text = apiStatus.querySelector('.status-text');

    // Set checking state
    if (dot) dot.style.backgroundColor = 'rgba(255, 255, 255, 0.4)';
    if (text) text.textContent = '检查中...';

    try {
        let data;
        if (typeof apiClient !== 'undefined') {
            data = await apiClient.healthCheck();
        } else {
            // Fallback to direct fetch
            const response = await fetch('http://localhost:8080/api/v1/health');
            data = await response.json();
        }

        if (dot) {
            dot.classList.remove('online', 'offline');
            if (data.mlService === 'UP' || data.status === 'healthy') {
                dot.classList.add('online');
                text.textContent = 'API在线';
            } else {
                dot.classList.add('offline');
                text.textContent = 'API离线';
            }
        }
    } catch (error) {
        console.error('Health check failed:', error);
        if (dot) {
            dot.classList.remove('online', 'offline');
            dot.classList.add('offline');
        }
        if (text) text.textContent = 'API离线';
    }
}

/**
 * Show login modal
 */
function showLoginModal() {
    const modal = document.getElementById('loginModal');
    if (modal) {
        modal.classList.remove('hidden');

        // Setup login modal events
        const closeBtn = document.getElementById('closeLoginModal');
        const cancelBtn = document.getElementById('cancelLoginBtn');
        const loginBtn = document.getElementById('loginBtn');

        if (closeBtn && !closeBtn.hasAttribute('data-bound')) {
            closeBtn.setAttribute('data-bound', 'true');
            closeBtn.addEventListener('click', hideLoginModal);
        }

        if (cancelBtn && !cancelBtn.hasAttribute('data-bound')) {
            cancelBtn.setAttribute('data-bound', 'true');
            cancelBtn.addEventListener('click', hideLoginModal);
        }

        if (loginBtn && !loginBtn.hasAttribute('data-bound')) {
            loginBtn.setAttribute('data-bound', 'true');
            loginBtn.addEventListener('click', handleLogin);
        }

        // Setup tabs
        const tabs = modal.querySelectorAll('.login-tab');
        tabs.forEach(tab => {
            if (!tab.hasAttribute('data-bound')) {
                tab.setAttribute('data-bound', 'true');
                tab.addEventListener('click', (e) => {
                    tabs.forEach(t => t.classList.remove('active'));
                    e.target.classList.add('active');
                    switchLoginTab(e.target.dataset.tab);
                });
            }
        });

        // Setup overlay click
        const overlay = modal.querySelector('.modal-overlay');
        if (overlay && !overlay.hasAttribute('data-bound')) {
            overlay.setAttribute('data-bound', 'true');
            overlay.addEventListener('click', hideLoginModal);
        }
    }
}

/**
 * Hide login modal
 */
function hideLoginModal() {
    const modal = document.getElementById('loginModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

/**
 * Switch login tab
 */
function switchLoginTab(tabName) {
    const panels = document.querySelectorAll('.login-panel');
    panels.forEach(p => p.classList.remove('active'));

    const targetPanel = document.getElementById(tabName + 'Login');
    if (targetPanel) {
        targetPanel.classList.add('active');
    }
}

/**
 * Handle login
 */
async function handleLogin() {
    const loginBtn = document.getElementById('loginBtn');
    const activeTab = document.querySelector('.login-tab.active');

    if (!activeTab) return;

    const tabType = activeTab.dataset.tab;
    loginBtn.disabled = true;
    loginBtn.textContent = '登录中...';

    try {
        if (tabType === 'token') {
            const tokenInput = document.getElementById('apiTokenInput');
            const token = tokenInput ? tokenInput.value.trim() : '';

            if (!token) {
                showToast('请输入 API Token', 'error');
                return;
            }

            await apiClient.setApiKey(token);
        } else {
            const usernameInput = document.getElementById('usernameInput');
            const passwordInput = document.getElementById('passwordInput');
            const username = usernameInput ? usernameInput.value.trim() : '';
            const password = passwordInput ? passwordInput.value.trim() : '';

            if (!username || !password) {
                showToast('请输入用户名和密码', 'error');
                return;
            }

            const result = await apiClient.login(username, password);
        }

        showToast('登录成功', 'success');
        hideLoginModal();
        await checkAuth();
        await loadApiConfig();
    } catch (error) {
        console.error('Login failed:', error);
        showToast('登录失败: ' + error.message, 'error');
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = '登录';
    }
}

/**
 * Load API configuration (for settings tab)
 */
async function loadApiConfig() {
    const result = await chrome.storage.local.get(['apiToken', 'apiKey']);

    const tokenDisplay = document.getElementById('token-display');
    const tokenStatus = document.getElementById('token-status');

    if (tokenDisplay && tokenStatus) {
        const credential = result.apiToken || result.apiKey;
        if (credential) {
            const masked = credential.length > 12
                ? credential.substring(0, 6) + '...' + credential.substring(credential.length - 4)
                : '***';
            tokenDisplay.textContent = masked;
            tokenStatus.textContent = '✅ 已登录';
            tokenStatus.className = 'token-status logged-in';
        } else {
            tokenDisplay.textContent = '未设置';
            tokenStatus.textContent = '⚠️ 未登录';
            tokenStatus.className = 'token-status not-logged-in';
        }
    }
}

/**
 * Close details modal
 */

/**
 * Close details modal
 */
function closeDetailsModal() {
    const modal = document.getElementById('details-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

/**
 * Truncate URL for display
 */
function truncateUrl(url, maxLength) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    if (typeof toast !== 'undefined' && toast.show) {
        return toast.show(message, type, duration);
    }

    // Fallback if toast is not available
    console.log(`[${type}] ${message}`);
    return null;
}

// Make functions globally available for inline event handlers
window.showLoginModal = showLoginModal;
window.closeDetailsModal = closeDetailsModal;
window.showToast = showToast;
