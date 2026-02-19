/**
 * NoFishing Settings Script
 */

// API Configuration
const DEFAULT_API_URL = 'http://localhost:8080/api/v1';

// Elements
const backBtn = document.getElementById('backBtn');
const apiUrl = document.getElementById('apiUrl');
const apiKey = document.getElementById('apiKey');
const testApiBtn = document.getElementById('testApiBtn');
const apiTestResult = document.getElementById('apiTestResult');

// Detection settings
const autoScan = document.getElementById('autoScan');
const showNotifications = document.getElementById('showNotifications');
const blockPhishing = document.getElementById('blockPhishing');
const sensitivity = document.getElementById('sensitivity');

// Whitelist/Blacklist
const whitelistForm = document.getElementById('whitelistForm');
const whitelistUrl = document.getElementById('whitelistUrl');
const whitelistItems = document.getElementById('whitelistItems');
const blacklistForm = document.getElementById('blacklistForm');
const blacklistUrl = document.getElementById('blacklistUrl');
const blacklistItems = document.getElementById('blacklistItems');

// Data management
const exportDataBtn = document.getElementById('exportDataBtn');
const importDataBtn = document.getElementById('importDataBtn');
const importFile = document.getElementById('importFile');
const resetDataBtn = document.getElementById('resetDataBtn');

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Load settings
    await loadSettings();

    // Load whitelist and blacklist
    await loadWhitelist();
    await loadBlacklist();

    // Setup event listeners
    setupEventListeners();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    backBtn.addEventListener('click', () => {
        window.location.href = 'popup.html';
    });

    // API settings
    apiUrl.addEventListener('change', saveSettings);
    apiKey.addEventListener('change', saveSettings);
    testApiBtn.addEventListener('click', testApiConnection);

    // Detection settings
    autoScan.addEventListener('change', saveSettings);
    showNotifications.addEventListener('change', saveSettings);
    blockPhishing.addEventListener('change', saveSettings);
    sensitivity.addEventListener('change', saveSettings);

    // Whitelist
    whitelistForm.addEventListener('submit', handleAddWhitelist);

    // Blacklist
    blacklistForm.addEventListener('submit', handleAddBlacklist);

    // Data management
    exportDataBtn.addEventListener('click', exportData);
    importDataBtn.addEventListener('click', () => importFile.click());
    importFile.addEventListener('change', importData);
    resetDataBtn.addEventListener('click', resetData);
}

/**
 * Load settings from storage
 */
async function loadSettings() {
    return new Promise((resolve) => {
        chrome.storage.local.get([
            'apiUrl',
            'apiKey',
            'autoScan',
            'showNotifications',
            'blockPhishing',
            'sensitivity'
        ], (result) => {
            apiUrl.value = result.apiUrl || DEFAULT_API_URL;
            apiKey.value = result.apiKey || '';
            autoScan.checked = result.autoScan !== false;
            showNotifications.checked = result.showNotifications !== false;
            blockPhishing.checked = result.blockPhishing !== false;
            sensitivity.value = result.sensitivity || 'medium';
            resolve();
        });
    });
}

/**
 * Save settings to storage
 */
async function saveSettings() {
    return new Promise((resolve) => {
        chrome.storage.local.set({
            apiUrl: apiUrl.value,
            apiKey: apiKey.value,
            autoScan: autoScan.checked,
            showNotifications: showNotifications.checked,
            blockPhishing: blockPhishing.checked,
            sensitivity: sensitivity.value
        }, () => {
            // Notify popup of settings change
            chrome.runtime.sendMessage({ action: 'settingsChanged' });
            resolve();
        });
    });
}

/**
 * Test API connection
 */
async function testApiConnection() {
    testApiBtn.disabled = true;
    testApiBtn.textContent = '测试中...';
    apiTestResult.classList.add('hidden');

    try {
        const url = apiUrl.value || DEFAULT_API_URL;
        const response = await fetch(`${url}/health`);

        if (response.ok) {
            const data = await response.json();
            if (data.mlService === 'UP') {
                showTestResult(true, 'API 连接成功！ML 服务在线。');
            } else {
                showTestResult(false, 'API 连接成功，但 ML 服务离线。');
            }
        } else {
            showTestResult(false, `API 连接失败: ${response.status}`);
        }
    } catch (error) {
        showTestResult(false, `连接错误: ${error.message}`);
    } finally {
        testApiBtn.disabled = false;
        testApiBtn.textContent = '测试连接';
    }
}

/**
 * Show test result
 */
function showTestResult(success, message) {
    apiTestResult.classList.remove('hidden', 'success', 'error');
    apiTestResult.classList.add(success ? 'success' : 'error');
    apiTestResult.textContent = message;
}

/**
 * Load whitelist
 */
async function loadWhitelist() {
    return new Promise((resolve) => {
        chrome.storage.local.get(['whitelist'], (result) => {
            const whitelist = result.whitelist || [];
            renderList(whitelistItems, whitelist, 'whitelist');
            resolve();
        });
    });
}

/**
 * Load blacklist
 */
async function loadBlacklist() {
    return new Promise((resolve) => {
        chrome.storage.local.get(['blacklist'], (result) => {
            const blacklist = result.blacklist || [];
            renderList(blacklistItems, blacklist, 'blacklist');
            resolve();
        });
    });
}

/**
 * Render list items
 */
function renderList(container, items, type) {
    if (items.length === 0) {
        container.innerHTML = '<div class="empty-list">暂无项目</div>';
        return;
    }

    container.innerHTML = items.map((item, index) => `
        <div class="list-item">
            <span class="list-item-url">${escapeHtml(item)}</span>
            <button class="list-item-remove" data-index="${index}" data-type="${type}" title="删除">×</button>
        </div>
    `).join('');

    // Add remove listeners
    container.querySelectorAll('.list-item-remove').forEach(btn => {
        btn.addEventListener('click', handleRemoveItem);
    });
}

/**
 * Handle add whitelist item
 */
async function handleAddWhitelist(e) {
    e.preventDefault();
    const url = whitelistUrl.value.trim();
    if (!url) return;

    const result = await chrome.storage.local.get(['whitelist']);
    const whitelist = result.whitelist || [];

    if (!whitelist.includes(url)) {
        whitelist.push(url);
        await chrome.storage.local.set({ whitelist });
        renderList(whitelistItems, whitelist, 'whitelist');
        whitelistUrl.value = '';
    }
}

/**
 * Handle add blacklist item
 */
async function handleAddBlacklist(e) {
    e.preventDefault();
    const url = blacklistUrl.value.trim();
    if (!url) return;

    const result = await chrome.storage.local.get(['blacklist']);
    const blacklist = result.blacklist || [];

    if (!blacklist.includes(url)) {
        blacklist.push(url);
        await chrome.storage.local.set({ blacklist });
        renderList(blacklistItems, blacklist, 'blacklist');
        blacklistUrl.value = '';
    }
}

/**
 * Handle remove item
 */
async function handleRemoveItem(e) {
    const index = parseInt(e.target.dataset.index);
    const type = e.target.dataset.type;

    const result = await chrome.storage.local.get([type]);
    const list = result[type] || [];

    list.splice(index, 1);
    await chrome.storage.local.set({ [type]: list });

    if (type === 'whitelist') {
        renderList(whitelistItems, list, type);
    } else {
        renderList(blacklistItems, list, type);
    }
}

/**
 * Export data
 */
async function exportData() {
    const data = await chrome.storage.local.get(null);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `nofishing-backup-${new Date().toISOString().split('T')[0]}.json`;
    a.click();

    URL.revokeObjectURL(url);
}

/**
 * Import data
 */
async function importData(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        const text = await file.text();
        const data = JSON.parse(text);

        if (confirm('这将覆盖当前所有设置，确定要导入吗？')) {
            await chrome.storage.local.clear();
            await chrome.storage.local.set(data);

            // Reload settings
            await loadSettings();
            await loadWhitelist();
            await loadBlacklist();

            alert('数据导入成功！');
        }
    } catch (error) {
        alert('导入失败：无效的文件格式');
    }

    e.target.value = '';
}

/**
 * Reset data
 */
async function resetData() {
    if (confirm('确定要重置所有数据吗？此操作不可撤销。')) {
        await chrome.storage.local.clear();
        await loadSettings();
        await loadWhitelist();
        await loadBlacklist();
        alert('数据已重置');
    }
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
