/**
 * NoFishing Popup Script
 */

// API Configuration
const DEFAULT_API_URL = 'http://localhost:8080/api/v1';
let API_BASE_URL = DEFAULT_API_URL;

// Elements
const apiStatus = document.getElementById('apiStatus');
const currentUrl = document.getElementById('currentUrl');
const currentStatus = document.getElementById('currentStatus');
const currentSite = document.getElementById('currentSite');
const scanBtn = document.getElementById('scanBtn');
const checkForm = document.getElementById('checkForm');
const checkUrlInput = document.getElementById('checkUrl');
const checkResult = document.getElementById('checkResult');
const settingsBtn = document.getElementById('settingsBtn');
const clearCacheBtn = document.getElementById('clearCacheBtn');
const scannedCount = document.getElementById('scannedCount');
const blockedCount = document.getElementById('blockedCount');

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Load settings first
    await loadSettings();

    // Handle logo image error
    const logoImg = document.getElementById('logoImg');
    if (logoImg) {
        logoImg.onerror = function() {
            this.style.display = 'none';
        };
    }

    // Load stats
    loadStats();

    // Health check
    await checkHealth();

    // Get current tab info
    await getCurrentTabInfo();

    // Setup event listeners
    setupEventListeners();
}

/**
 * Load settings from storage
 */
async function loadSettings() {
    return new Promise((resolve) => {
        chrome.storage.local.get(['apiUrl'], (result) => {
            API_BASE_URL = result.apiUrl || DEFAULT_API_URL;
            resolve();
        });
    });
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    scanBtn.addEventListener('click', scanCurrentSite);
    checkForm.addEventListener('submit', handleQuickCheck);
    settingsBtn.addEventListener('click', openSettings);
    clearCacheBtn.addEventListener('click', clearCache);
}

/**
 * Check API health
 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.mlService === 'UP') {
            updateApiStatus('online');
        } else {
            updateApiStatus('offline');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateApiStatus('offline');
    }
}

/**
 * Update API status indicator
 */
function updateApiStatus(status) {
    const dot = apiStatus.querySelector('.status-dot');
    const text = apiStatus.querySelector('.status-text');

    dot.classList.remove('online', 'offline');
    dot.classList.add(status);
    text.textContent = status === 'online' ? 'API在线' : 'API离线';
}

/**
 * Get current tab info
 */
async function getCurrentTabInfo() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (tab && tab.url) {
            currentUrl.textContent = truncateUrl(tab.url, 40);
            currentStatus.textContent = '准备扫描';
            currentStatus.className = 'site-status';
        }
    } catch (error) {
        console.error('Failed to get current tab:', error);
        currentUrl.textContent = '无法获取URL';
    }
}

/**
 * Scan current site
 */
async function scanCurrentSite() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            alert('无法获取页面URL');
            return;
        }

        setScanButtonLoading(true);

        const result = await detectUrl(tab.url);

        if (result) {
            displayResult(result);

            // Update stats
            incrementStats('scanned');
            if (result.isPhishing) {
                incrementStats('blocked');
            }
        }

    } catch (error) {
        console.error('Scan failed:', error);
        alert('扫描失败: ' + error.message);
    } finally {
        setScanButtonLoading(false);
    }
}

/**
 * Handle quick check form
 */
async function handleQuickCheck(e) {
    e.preventDefault();

    const url = checkUrlInput.value.trim();
    if (!url) return;

    const submitBtn = checkForm.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="spinner"></span>';
    submitBtn.disabled = true;

    try {
        const result = await detectUrl(url);
        displayCheckResult(result);
    } catch (error) {
        console.error('Quick check failed:', error);
        checkResult.innerHTML = '<p style="color: var(--danger-color)">检测失败</p>';
        checkResult.classList.remove('hidden');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

/**
 * Detect URL via API
 */
async function detectUrl(url) {
    const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            url: url,
            fetchContent: false
        })
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
}

/**
 * Display scan result in current site section
 */
function displayResult(result) {
    const statusClass = result.isPhishing ? 'danger' : 'safe';
    const statusText = result.isPhishing ? '检测到钓鱼网站' : '安全';

    currentStatus.textContent = statusText;
    currentStatus.className = `site-status ${statusClass}`;

    // Add confidence info
    const confidence = Math.round(result.confidence * 100);
    currentStatus.textContent += ` (${confidence}% 置信度, ${result.riskLevel})`;
}

/**
 * Display quick check result
 */
function displayCheckResult(result) {
    checkResult.classList.remove('hidden', 'safe', 'danger');

    if (result.isPhishing) {
        checkResult.classList.add('danger');
        checkResult.innerHTML = `
            <strong>⚠️ 钓鱼网站警告</strong><br>
            该网站是钓鱼网站<br>
            风险等级: ${result.riskLevel}<br>
            置信度: ${Math.round(result.confidence * 100)}%
        `;
    } else {
        checkResult.classList.add('safe');
        checkResult.innerHTML = `
            <strong>✓ 安全网站</strong><br>
            该网站是安全的<br>
            风险等级: ${result.riskLevel}<br>
            置信度: ${Math.round(result.confidence * 100)}%
        `;
    }
}

/**
 * Set scan button loading state
 */
function setScanButtonLoading(loading) {
    if (loading) {
        scanBtn.innerHTML = '<span class="spinner"></span> 扫描中...';
        scanBtn.disabled = true;
    } else {
        scanBtn.innerHTML = '<span class="btn-icon">🔍</span> 扫描当前网站';
        scanBtn.disabled = false;
    }
}

/**
 * Load statistics from storage
 */
function loadStats() {
    chrome.storage.local.get(['scannedCount', 'blockedCount'], (result) => {
        scannedCount.textContent = result.scannedCount || 0;
        blockedCount.textContent = result.blockedCount || 0;
    });
}

/**
 * Increment statistics
 */
function incrementStats(type) {
    chrome.storage.local.get([type + 'Count'], (result) => {
        const current = result[type + 'Count'] || 0;
        chrome.storage.local.set({ [type + 'Count']: current + 1 });

        if (type === 'scanned') {
            scannedCount.textContent = current + 1;
        } else if (type === 'blocked') {
            blockedCount.textContent = current + 1;
        }
    });
}

/**
 * Open settings
 */
function openSettings() {
    chrome.tabs.create({ url: chrome.runtime.getURL('src/settings/settings.html') });
}

/**
 * Clear detection cache
 */
async function clearCache() {
    try {
        await chrome.runtime.sendMessage({ action: 'clearCache' });
        alert('缓存已清除');
    } catch (error) {
        console.error('Failed to clear cache:', error);
        alert('清除缓存失败');
    }
}

/**
 * Truncate URL for display
 */
function truncateUrl(url, maxLength) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
}
