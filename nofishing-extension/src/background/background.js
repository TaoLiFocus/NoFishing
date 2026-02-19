/**
 * NoFishing - Background Service Worker
 * Handles URL interception and phishing detection with token auth
 */

// API Configuration
const BACKEND_API_URL = 'http://localhost:8080/api/v1';
const DETECT_ENDPOINT = `${BACKEND_API_URL}/detect`;
const HEALTH_ENDPOINT = `${BACKEND_API_URL}/health`;

// Cache settings
const CACHE_TTL = 3600000;
const MAX_CACHE_SIZE = 1000;

// Detection cache
const detectionCache = new Map();

/**
 * Initialize the extension
 */
chrome.runtime.onInstalled.addListener(() => {
    console.log('[NoFishing] Extension installed');

    // Set default settings
    chrome.storage.local.set({
        settings: {
            autoBlock: false,
            showNotifications: true,
            autoScan: true,
            sensitivity: 'medium'
        },
        scannedCount: 0,
        blockedCount: 0
    });
});

/**
 * Get authorization headers
 */
async function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };

    // Try JWT token first
    const result = await chrome.storage.local.get(['apiToken', 'apiKey']);

    if (result.apiToken) {
        headers['Authorization'] = `Bearer ${result.apiToken}`;
    } else if (result.apiKey) {
        headers['X-API-Key'] = result.apiKey;
    }

    return headers;
}

/**
 * Handle navigation events
 */
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    if (details.frameId !== 0) return;

    const url = details.url;

    // Skip chrome:// and edge:// URLs
    if (url.startsWith('chrome://') || url.startsWith('edge://')) return;

    // Check settings
    const settings = await getSettings();
    if (!settings.autoScan) return;

    // Check cache first
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
        cacheResult(url, result);
        await handlePhishingUrl(url, result);
    }
});

/**
 * Detect if a URL is phishing
 */
async function detectUrl(url) {
    try {
        const headers = await getAuthHeaders();

        const response = await fetch(DETECT_ENDPOINT, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            console.error('[NoFishing] API error:', response.status);

            // Handle 401 Unauthorized - clear credentials
            if (response.status === 401) {
                console.log('[NoFishing] Unauthorized, clearing credentials...');
                await chrome.storage.local.set({ apiToken: null, apiKey: null });
            }

            return null;
        }

        const mlResult = await response.json();
        console.log('[NoFishing] ML API result:', mlResult);

        const result = {
            url: mlResult.url,
            isPhishing: mlResult.is_phishing,
            confidence: mlResult.probability,
            riskLevel: mlResult.risk_level,
            processingTimeMs: mlResult.processing_time_ms
        };

        // Add to history
        await addToHistory(result);

        return result;

    } catch (error) {
        console.error('[NoFishing] Detection failed:', error);
        return null;
    }
}

/**
 * Add detection to history
 */
async function addToHistory(result) {
    chrome.storage.local.get(['detectionHistory', 'scannedCount', 'blockedCount'], (data) => {
        const history = data.detectionHistory || [];
        let scannedCount = data.scannedCount || 0;
        let blockedCount = data.blockedCount || 0;

        const entry = {
            id: Date.now(),
            url: result.url,
            isPhishing: result.isPhishing,
            confidence: result.confidence,
            riskLevel: result.riskLevel,
            timestamp: Date.now(),
            processingTimeMs: result.processingTimeMs
        };

        history.unshift(entry);

        // Trim to 200 entries
        if (history.length > 200) {
            history.splice(200);
        }

        // Remove entries older than 30 days
        const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
        const filtered = history.filter(e => e.timestamp > thirtyDaysAgo);

        // Increment stats
        scannedCount += 1;
        if (result.isPhishing) {
            blockedCount += 1;
        }

        chrome.storage.local.set({
            detectionHistory: filtered,
            scannedCount: scannedCount,
            blockedCount: blockedCount
        });
    });
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
        chrome.storage.local.get(['settings'], (result) => {
            const defaults = {
                autoBlock: false,
                showNotifications: true,
                autoScan: true,
                sensitivity: 'medium'
            };
            resolve({ ...defaults, ...result.settings });
        });
    });
}

/**
 * Health check for API
 */
async function healthCheck() {
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(HEALTH_ENDPOINT, { headers });

        if (response.ok) {
            const data = await response.json();
            console.log('[NoFishing] API Health:', data);
            return data.mlService === 'UP';
        }
        return false;
    } catch (error) {
        console.error('[NoFishing] Health check failed:', error);
        return false;
    }
}

// Periodic health check
setInterval(healthCheck, 60000);

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

    if (request.action === 'getCachedResult') {
        const cached = getCachedResult(request.url);
        sendResponse({ result: cached });
        return true;
    }

    if (request.action === 'settingsChanged') {
        console.log('[NoFishing] Settings changed');
        sendResponse({ success: true });
        return true;
    }
});

// Log startup
console.log('[NoFishing] Background service worker initialized');
