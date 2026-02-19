/**
 * Settings Tab - Quick toggles + API configuration
 */

/**
 * Initialize settings tab
 */
async function initSettingsTab() {
    await loadSettings();
    await loadApiConfig();
    setupSettingsEventListeners();
}

/**
 * Load settings values
 */
async function loadSettings() {
    const result = await chrome.storage.local.get([
        'autoBlock',
        'showNotifications',
        'autoScan',
        'sensitivity'
    ]);

    // Auto-block toggle
    const autoBlockToggle = document.getElementById('auto-block-toggle');
    if (autoBlockToggle) {
        autoBlockToggle.checked = result.autoBlock || false;
    }

    // Notifications toggle
    const notificationsToggle = document.getElementById('notifications-toggle');
    if (notificationsToggle) {
        notificationsToggle.checked = result.showNotifications !== false;
    }

    // Auto-scan toggle
    const autoScanToggle = document.getElementById('auto-scan-toggle');
    if (autoScanToggle) {
        autoScanToggle.checked = result.autoScan !== false;
    }

    // Sensitivity dropdown
    const sensitivitySelect = document.getElementById('sensitivity-select');
    if (sensitivitySelect) {
        sensitivitySelect.value = result.sensitivity || 'medium';
    }
}

/**
 * Load API configuration
 */
async function loadApiConfig() {
    const result = await chrome.storage.local.get(['apiToken', 'apiKey', 'apiUrl']);

    const tokenStatus = document.getElementById('token-status');
    const tokenDisplay = document.getElementById('token-display');

    if (result.apiToken || result.apiKey) {
        // Mask the token for display
        const credential = result.apiToken || result.apiKey;
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

/**
 * Setup settings tab event listeners
 */
function setupSettingsEventListeners() {
    // Auto-block toggle
    const autoBlockToggle = document.getElementById('auto-block-toggle');
    if (autoBlockToggle) {
        autoBlockToggle.addEventListener('change', (e) => {
            updateSetting('autoBlock', e.target.checked);
        });
    }

    // Notifications toggle
    const notificationsToggle = document.getElementById('notifications-toggle');
    if (notificationsToggle) {
        notificationsToggle.addEventListener('change', (e) => {
            updateSetting('showNotifications', e.target.checked);
        });
    }

    // Auto-scan toggle
    const autoScanToggle = document.getElementById('auto-scan-toggle');
    if (autoScanToggle) {
        autoScanToggle.addEventListener('change', (e) => {
            updateSetting('autoScan', e.target.checked);
        });
    }

    // Sensitivity dropdown
    const sensitivitySelect = document.getElementById('sensitivity-select');
    if (sensitivitySelect) {
        sensitivitySelect.addEventListener('change', (e) => {
            updateSetting('sensitivity', e.target.value);
        });
    }

    // Update token button
    const updateTokenBtn = document.getElementById('update-token-btn');
    if (updateTokenBtn) {
        updateTokenBtn.addEventListener('click', showLoginModal);
    }

    // Logout button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // Clear cache button
    const clearCacheBtn = document.getElementById('clear-cache-btn');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', handleClearCache);
    }

    // Clear history button
    const clearHistoryBtn = document.getElementById('clear-history-settings-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', handleClearHistory);
    }
}

/**
 * Update a setting
 */
async function updateSetting(key, value) {
    try {
        await chrome.storage.local.set({ [key]: value });

        // Notify background script of setting change
        chrome.runtime.sendMessage({
            action: 'settingsChanged',
            key: key,
            value: value
        });

        showToast('设置已保存', 'success');
    } catch (error) {
        console.error('Failed to update setting:', error);
        showToast('保存失败: ' + error.message, 'error');
    }
}

/**
 * Handle logout
 */
async function handleLogout() {
    if (!confirm('确定要退出登录吗？')) {
        return;
    }

    try {
        const { apiClient } = await import('../utils/api.js');
        await apiClient.logout();

        showToast('已退出登录', 'success');
        await loadApiConfig();
    } catch (error) {
        console.error('Logout failed:', error);
        showToast('退出失败: ' + error.message, 'error');
    }
}

/**
 * Handle clear cache
 */
async function handleClearCache() {
    if (!confirm('确定要清除检测缓存吗？')) {
        return;
    }

    try {
        const { clearCache } = await import('../utils/storage.js');
        await clearCache();

        showToast('缓存已清除', 'success');
    } catch (error) {
        console.error('Failed to clear cache:', error);
        showToast('清除失败: ' + error.message, 'error');
    }
}

/**
 * Handle clear history
 */
async function handleClearHistory() {
    if (!confirm('确定要清空检测历史吗？')) {
        return;
    }

    try {
        const { clearHistory } = await import('../utils/storage.js');
        await clearHistory();

        showToast('检测历史已清空', 'success');
    } catch (error) {
        console.error('Failed to clear history:', error);
        showToast('清空失败: ' + error.message, 'error');
    }
}

// Export for use in popup.js
window.initSettingsTab = initSettingsTab;
