/**
 * Home Tab - Current site info + quick actions
 */

/**
 * Initialize home tab
 */
async function initHomeTab() {
    await loadCurrentSiteInfo();
    await loadTodayStats();
    setupHomeEventListeners();
}

/**
 * Load current site information
 */
async function loadCurrentSiteInfo() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            updateSiteInfo(null);
            return;
        }

        // Get cached result if available
        const { getCachedResult } = await import('../utils/storage.js');
        const cached = await getCachedResult(tab.url);

        if (cached) {
            updateSiteInfo(tab.url, cached);
        } else {
            updateSiteInfo(tab.url, null);
        }
    } catch (error) {
        console.error('Failed to load current site:', error);
        updateSiteInfo(null);
    }
}

/**
 * Update site info display
 */
function updateSiteInfo(url, result) {
    const urlElement = document.getElementById('current-url');
    const statusElement = document.getElementById('current-status');
    const confidenceElement = document.getElementById('current-confidence');

    if (!url) {
        urlElement.textContent = '无法获取页面URL';
        statusElement.textContent = '-';
        statusElement.className = 'site-status';
        confidenceElement.textContent = '';
        return;
    }

    // Truncate URL for display
    urlElement.textContent = truncateUrl(url, 40);

    if (!result) {
        statusElement.textContent = '未扫描';
        statusElement.className = 'site-status';
        confidenceElement.textContent = '';
        return;
    }

    // Update status
    if (result.isPhishing) {
        statusElement.textContent = '检测到钓鱼网站';
        statusElement.className = 'site-status danger';
    } else {
        statusElement.textContent = '安全';
        statusElement.className = 'site-status safe';
    }

    // Update confidence
    const confidence = Math.round(result.confidence * 100);
    confidenceElement.textContent = `置信度: ${confidence}%`;
}

/**
 * Load today's statistics
 */
async function loadTodayStats() {
    const result = await chrome.storage.local.get(['scannedCount', 'blockedCount']);

    const scannedElement = document.getElementById('today-scanned');
    const blockedElement = document.getElementById('today-blocked');

    if (scannedElement) {
        scannedElement.textContent = result.scannedCount || 0;
    }
    if (blockedElement) {
        blockedElement.textContent = result.blockedCount || 0;
    }
}

/**
 * Setup home tab event listeners
 */
function setupHomeEventListeners() {
    // Scan current site button
    const scanBtn = document.getElementById('scan-current-btn');
    if (scanBtn) {
        scanBtn.addEventListener('click', scanCurrentSite);
    }

    // Whitelist button
    const whitelistBtn = document.getElementById('whitelist-btn');
    if (whitelistBtn) {
        whitelistBtn.addEventListener('click', addToWhitelist);
    }

    // Blacklist button
    const blacklistBtn = document.getElementById('blacklist-btn');
    if (blacklistBtn) {
        blacklistBtn.addEventListener('click', addToBlacklist);
    }

    // Quick check form
    const quickCheckForm = document.getElementById('quick-check-form');
    if (quickCheckForm) {
        quickCheckForm.addEventListener('submit', handleQuickCheck);
    }
}

/**
 * Scan current site
 */
async function scanCurrentSite() {
    const scanBtn = document.getElementById('scan-current-btn');

    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            showToast('无法获取页面URL', 'error');
            return;
        }

        // Show loading state
        scanBtn.disabled = true;
        scanBtn.innerHTML = '<span class="spinner"></span> 扫描中...';

        // Call background to detect
        const response = await chrome.runtime.sendMessage({
            action: 'detectUrl',
            url: tab.url
        });

        if (response && response.error) {
            showToast(response.error, 'error');
        } else if (response) {
            updateSiteInfo(tab.url, response);
            showToast('扫描完成', 'success');
        }
    } catch (error) {
        console.error('Scan failed:', error);
        showToast('扫描失败: ' + error.message, 'error');
    } finally {
        scanBtn.disabled = false;
        scanBtn.innerHTML = '<span class="icon">🔍</span> 重新扫描';
    }
}

/**
 * Add current site to whitelist
 */
async function addToWhitelist() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            showToast('无法获取页面URL', 'error');
            return;
        }

        const { apiClient } = await import('../utils/api.js');

        // Show loading
        showToast('正在添加到白名单...', 'info');

        await apiClient.addToWhitelist(tab.url, '通过扩展添加');

        showToast('已添加到白名单', 'success');
    } catch (error) {
        console.error('Failed to add to whitelist:', error);
        if (error.message === 'UNAUTHORIZED') {
            showToast('请先登录', 'error');
            showLoginModal();
        } else {
            showToast('添加失败: ' + error.message, 'error');
        }
    }
}

/**
 * Add current site to blacklist
 */
async function addToBlacklist() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            showToast('无法获取页面URL', 'error');
            return;
        }

        const { apiClient } = await import('../utils/api.js');

        // Show loading
        showToast('正在添加到黑名单...', 'info');

        await apiClient.addToBlacklist(tab.url, '通过扩展添加');

        showToast('已添加到黑名单', 'success');
    } catch (error) {
        console.error('Failed to add to blacklist:', error);
        if (error.message === 'UNAUTHORIZED') {
            showToast('请先登录', 'error');
            showLoginModal();
        } else {
            showToast('添加失败: ' + error.message, 'error');
        }
    }
}

/**
 * Handle quick check form
 */
async function handleQuickCheck(e) {
    e.preventDefault();

    const input = document.getElementById('quick-check-input');
    const url = input.value.trim();

    if (!url) {
        showToast('请输入URL', 'warning');
        return;
    }

    const submitBtn = e.target.querySelector('button');
    const originalHTML = submitBtn.innerHTML;

    try {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span>';

        const { apiClient } = await import('../utils/api.js');
        const result = await apiClient.detectUrl(url);

        displayQuickCheckResult(result);

        // Add to history
        const { addHistoryEntry } = await import('../utils/storage.js');
        await addHistoryEntry({
            url: url,
            isPhishing: result.isPhishing,
            confidence: result.confidence,
            riskLevel: result.riskLevel
        });

        input.value = '';
    } catch (error) {
        console.error('Quick check failed:', error);
        if (error.message === 'UNAUTHORIZED') {
            showToast('请先登录', 'error');
            showLoginModal();
        } else {
            showToast('检测失败: ' + error.message, 'error');
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalHTML;
    }
}

/**
 * Display quick check result
 */
function displayQuickCheckResult(result) {
    const resultContainer = document.getElementById('quick-check-result');

    resultContainer.classList.remove('hidden', 'safe', 'danger', 'warning');

    if (result.isPhishing) {
        resultContainer.classList.add('danger');
        resultContainer.innerHTML = `
            <div class="result-icon">⛔</div>
            <div class="result-content">
                <div class="result-title">检测到钓鱼网站</div>
                <div class="result-details">
                    风险等级: ${result.riskLevel}<br>
                    置信度: ${Math.round(result.confidence * 100)}%
                </div>
            </div>
        `;
    } else {
        resultContainer.classList.add('safe');
        resultContainer.innerHTML = `
            <div class="result-icon">✅</div>
            <div class="result-content">
                <div class="result-title">安全网站</div>
                <div class="result-details">
                    风险等级: ${result.riskLevel}<br>
                    置信度: ${Math.round(result.confidence * 100)}%
                </div>
            </div>
        `;
    }

    resultContainer.classList.remove('hidden');
}

/**
 * Truncate URL for display
 */
function truncateUrl(url, maxLength) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
}

// Export for use in popup.js
window.initHomeTab = initHomeTab;
