/**
 * NoFishing - Background Service Worker
 * Handles URL interception and phishing detection
 */

// API Configuration - 使用Spring Boot后端API
const BACKEND_API_URL = 'http://localhost:8080/api/v1';
const DETECT_ENDPOINT = `${BACKEND_API_URL}/detect`;
const HEALTH_ENDPOINT = `${BACKEND_API_URL}/health`;

// Cache settings
const CACHE_TTL = 3600000; // 1 hour in milliseconds
const MAX_CACHE_SIZE = 1000;

// Detection cache
const detectionCache = new Map();

/**
 * Initialize the extension
 */
chrome.runtime.onInstalled.addListener(() => {
    console.log('NoFishing extension installed');

    // Set default settings
    chrome.storage.local.set({
        enabled: true,
        showNotifications: true,
        autoBlock: false,
        apiEndpoint: BACKEND_API_URL
    });
});

/**
 * Handle navigation events
 */
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    // Only handle main frame navigation
    if (details.frameId !== 0) return;

    const url = details.url;

    // Skip chrome:// and edge:// URLs
    if (url.startsWith('chrome://') || url.startsWith('edge://')) return;

    // Check if extension is enabled
    const settings = await getSettings();
    if (!settings.enabled) return;

    // Check if URL is in cache
    const cachedResult = getCachedResult(url);
    if (cachedResult) {
        console.log('[NoFishing] Cache HIT:', url, cachedResult);
        if (cachedResult.isPhishing) {
            await handlePhishingUrl(url, cachedResult);
        }
        return;
    }

    // Perform detection
    console.log('[NoFishing] Detecting URL:', url);
    const result = await detectUrl(url);

    if (result && result.isPhishing) {
        // Cache the result
        cacheResult(url, result);

        // Handle phishing URL
        await handlePhishingUrl(url, result);
    }
});

/**
 * Detect if a URL is phishing
 */
async function detectUrl(url) {
    try {
        const response = await fetch(DETECT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url
            })
        });

        if (!response.ok) {
            console.error('[NoFishing] API error:', response.status);
            return null;
        }

        const mlResult = await response.json();
        console.log('[NoFishing] ML API result:', mlResult);

        // 转换Flask响应格式为扩展期望的格式
        const result = {
            url: mlResult.url,
            isPhishing: mlResult.is_phishing,
            confidence: mlResult.probability,
            riskLevel: mlResult.risk_level,
            processingTimeMs: mlResult.processing_time_ms
        };

        return result;

    } catch (error) {
        console.error('[NoFishing] Detection failed:', error);
        return null;
    }
}

/**
 * Handle a phishing URL
 */
async function handlePhishingUrl(url, result) {
    const settings = await getSettings();

    // Show notification
    if (settings.showNotifications) {
        showNotification(result);
    }

    // Auto-block if enabled
    if (settings.autoBlock) {
        // Get current tab and redirect to warning page
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.id) {
            chrome.tabs.update(tab.id, {
                url: chrome.runtime.getURL('public/warning.html') +
                      '?url=' + encodeURIComponent(url) +
                      '&risk=' + encodeURIComponent(result.riskLevel) +
                      '&confidence=' + result.confidence
            });
        }
    }

    // Update badge
    updateBadge(result.riskLevel);
}

/**
 * Show desktop notification
 */
function showNotification(result) {
    const riskLevel = result.riskLevel;
    const confidence = Math.round(result.confidence * 100);

    chrome.notifications.create({
        type: 'basic',
        title: '⚠️ 钓鱼网站警告',
        message: `检测到钓鱼网站！\n风险等级: ${riskLevel}\n置信度: ${confidence}%`,
        priority: 2,
        requireInteraction: true
    });
}

/**
 * Update extension badge
 */
function updateBadge(riskLevel) {
    const colors = {
        'LOW': '#4CAF50',
        'MEDIUM': '#FF9800',
        'HIGH': '#FF5722',
        'CRITICAL': '#D32F2F'
    };

    const badgeColors = {
        'LOW': '#4CAF50',
        'MEDIUM': '#FF9800',
        'HIGH': '#FF5722',
        'CRITICAL': '#D32F2F'
    };

    const badgeText = {
        'LOW': '✓',
        'MEDIUM': '!',
        'HIGH': '⚠',
        'CRITICAL': '⛔'
    };

    chrome.action.setBadgeText({ text: badgeText[riskLevel] || '?' });
    chrome.action.setBadgeTextColor({ color: '#FFFFFF' });
    chrome.action.setBadgeBackgroundColor({ color: badgeColors[riskLevel] || '#999999' });
}

/**
 * Cache a detection result
 */
function cacheResult(url, result) {
    // Enforce cache size limit
    if (detectionCache.size >= MAX_CACHE_SIZE) {
        const firstKey = detectionCache.keys().next().value;
        detectionCache.delete(firstKey);
    }

    detectionCache.set(url, {
        ...result,
        timestamp: Date.now()
    });
}

/**
 * Get cached result
 */
function getCachedResult(url) {
    const cached = detectionCache.get(url);
    if (!cached) return null;

    // Check if cache is expired
    if (Date.now() - cached.timestamp > CACHE_TTL) {
        detectionCache.delete(url);
        return null;
    }

    return cached;
}

/**
 * Get extension settings
 */
async function getSettings() {
    return new Promise((resolve) => {
        chrome.storage.local.get(
            ['autoScan', 'showNotifications', 'blockPhishing', 'sensitivity', 'apiUrl'],
            (result) => {
                resolve({
                    enabled: result.autoScan !== false,
                    showNotifications: result.showNotifications !== false,
                    autoBlock: result.blockPhishing || false,
                    sensitivity: result.sensitivity || 'medium',
                    apiEndpoint: result.apiUrl || BACKEND_API_URL
                });
            }
        );
    });
}

/**
 * Health check for API
 */
async function healthCheck() {
    try {
        const settings = await getSettings();
        const response = await fetch(settings.apiEndpoint + '/health');

        if (response.ok) {
            const data = await response.json();
            console.log('[NoFishing] API Health:', data);
            return data.status === 'healthy';
        }
        return false;
    } catch (error) {
        console.error('[NoFishing] Health check failed:', error);
        return false;
    }
}

// Periodic health check
setInterval(healthCheck, 60000); // Every minute

/**
 * Handle messages from popup
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'detectUrl') {
        detectUrl(request.url).then(sendResponse);
        return true;
    }

    if (request.action === 'healthCheck') {
        healthCheck().then(sendResponse);
        return true;
    }

    if (request.action === 'clearCache') {
        detectionCache.clear();
        sendResponse({ success: true });
        return true;
    }

    if (request.action === 'settingsChanged') {
        // Reload settings cache
        console.log('[NoFishing] Settings changed, reloading...');
        sendResponse({ success: true });
        return true;
    }
});

// Log startup
console.log('[NoFishing] Background service worker initialized');
