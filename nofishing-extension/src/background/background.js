/**
 * NoFishing - Background Service Worker
 * Handles URL interception and phishing detection with token auth
 */

// API Configuration
const BACKEND_API_URL = 'http://localhost:8080/api/v1';
const DETECT_ENDPOINT = `${BACKEND_API_URL}/detect`;
const HEALTH_ENDPOINT = `${BACKEND_API_URL}/health`;
const WHITELIST_CHECK_ENDPOINT = `${BACKEND_API_URL}/whitelist/check`;
const BLACKLIST_CHECK_ENDPOINT = `${BACKEND_API_URL}/blacklist/check`;

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
            autoBlock: true,  // Default to blocking phishing sites
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
 * Extract domain from URL
 */
function extractDomain(url) {
    try {
        const urlObj = new URL(url);
        return urlObj.hostname;
    } catch (e) {
        return null;
    }
}

/**
 * Check if URL is in whitelist
 */
async function checkWhitelist(url) {
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${WHITELIST_CHECK_ENDPOINT}?url=${encodeURIComponent(url)}`, {
            method: 'GET',
            headers: headers
        });

        if (response.ok) {
            const data = await response.json();
            return data.whitelisted || false;
        }
        return false;
    } catch (error) {
        console.error('[NoFishing] Whitelist check failed:', error);
        return false;
    }
}

/**
 * Check if URL is in blacklist
 */
async function checkBlacklist(url) {
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${BLACKLIST_CHECK_ENDPOINT}?url=${encodeURIComponent(url)}`, {
            method: 'GET',
            headers: headers
        });

        if (response.ok) {
            const data = await response.json();
            return data.blacklisted || false;
        }
        return false;
    } catch (error) {
        console.error('[NoFishing] Blacklist check failed:', error);
        return false;
    }
}

/**
 * Handle navigation events
 */
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    if (details.frameId !== 0) return;

    const url = details.url;

    // Skip chrome:// and edge:// URLs
    if (url.startsWith('chrome://') || url.startsWith('edge://')) return;

    // Extract domain for list checks and caching
    const domain = extractDomain(url);
    if (!domain) return;

    // Check if URL is in bypass list (user clicked "proceed anyway")
    const result = await chrome.storage.local.get(['bypassList']);
    const bypassList = result.bypassList || {};
    const bypassTime = bypassList[url];

    // Clear expired bypass entries (older than 5 minutes)
    const now = Date.now();
    const fiveMinutesAgo = now - (5 * 60 * 1000);
    let shouldBypass = false;

    if (bypassTime && bypassTime > fiveMinutesAgo) {
        console.log('[NoFishing] URL in bypass list, allowing:', url);
        shouldBypass = true;
        // Remove from bypass list after allowing (one-time bypass)
        delete bypassList[url];
        await chrome.storage.local.set({ bypassList });
    }

    // Clean up old bypass entries
    for (const key in bypassList) {
        if (bypassList[key] < fiveMinutesAgo) {
            delete bypassList[key];
        }
    }
    if (Object.keys(bypassList).length === 0) {
        await chrome.storage.local.set({ bypassList: {} });
    } else {
        await chrome.storage.local.set({ bypassList });
    }

    if (shouldBypass) {
        return; // Skip checking for this URL
    }

    // Check settings
    const settings = await getSettings();
    if (!settings.autoScan) return;

    // Check cache first (use domain as cache key)
    const cachedResult = getCachedResult(domain);
    if (cachedResult) {
        console.log('[NoFishing] Cache HIT:', domain, cachedResult);
        if (cachedResult.isPhishing) {
            await handlePhishingUrl(url, cachedResult);
        }
        return;
    }

    // PRIORITY 1: Check whitelist first (use domain for check)
    const isWhitelisted = await checkWhitelist(domain);
    if (isWhitelisted) {
        console.log('[NoFishing] Domain is in whitelist:', domain);
        const whitelistResult = {
            url: url,
            isPhishing: false,
            confidence: 0,
            riskLevel: 'SAFE_WHITELIST',
            processingTimeMs: 0,
            inWhitelist: true
        };
        cacheResult(domain, whitelistResult);
        await addToHistory(whitelistResult);
        return;
    }

    // PRIORITY 2: Check blacklist (use domain for check)
    const isBlacklisted = await checkBlacklist(domain);
    if (isBlacklisted) {
        console.log('[NoFishing] Domain is in blacklist:', domain);
        const blacklistResult = {
            url: url,
            isPhishing: true,
            confidence: 1.0,
            riskLevel: 'CRITICAL_BLACKLIST',
            processingTimeMs: 0,
            inBlacklist: true
        };
        cacheResult(domain, blacklistResult);
        await addToHistory(blacklistResult);
        await handlePhishingUrl(url, blacklistResult);
        return;
    }

    // PRIORITY 3: Perform ML detection
    console.log('[NoFishing] Detecting URL:', url);
    const detectResult = await detectUrl(url);

    if (detectResult) {
        cacheResult(domain, detectResult);
        if (detectResult.isPhishing) {
            await handlePhishingUrl(url, detectResult);
        }
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
            // Add defensive checks for confidence and riskLevel
            const confidence = (typeof result.confidence === 'number' && !isNaN(result.confidence))
                ? result.confidence
                : 0;
            const riskLevel = result.riskLevel || 'CRITICAL';

            chrome.tabs.update(tab.id, {
                url: chrome.runtime.getURL('public/warning.html') +
                      '?url=' + encodeURIComponent(url) +
                      '&risk=' + encodeURIComponent(riskLevel) +
                      '&confidence=' + confidence
            });
        }
    }

    // Update badge
    updateBadge(result.riskLevel || 'CRITICAL');
}

/**
 * Show desktop notification
 */
function showNotification(result) {
    const riskLevel = result.riskLevel;
    // Add defensive check for confidence
    const confidence = (typeof result.confidence === 'number' && !isNaN(result.confidence))
        ? Math.round(result.confidence * 100)
        : 0;

    chrome.notifications.create({
        type: 'basic',
        iconUrl: chrome.runtime.getURL('icons/icon48.png'),
        title: '⚠️ 钓鱼网站警告',
        message: `检测到钓鱼网站！\n风险等级: ${riskLevel}\n置信度: ${confidence}%`,
        priority: 2,
        requireInteraction: true
    }, (notificationId) => {
        if (chrome.runtime.lastError) {
            console.error('[NoFishing] Notification error:', chrome.runtime.lastError);
        } else {
            console.log('[NoFishing] Notification created:', notificationId);
        }
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
                autoBlock: true,
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

// Periodic cleanup of bypass list (every 5 minutes)
setInterval(async () => {
    const result = await chrome.storage.local.get(['bypassList']);
    const bypassList = result.bypassList || {};
    const now = Date.now();
    const fiveMinutesAgo = now - (5 * 60 * 1000);
    let cleaned = false;

    for (const key in bypassList) {
        if (bypassList[key] < fiveMinutesAgo) {
            delete bypassList[key];
            cleaned = true;
        }
    }

    if (cleaned) {
        await chrome.storage.local.set({ bypassList });
        console.log('[NoFishing] Cleaned up expired bypass entries');
    }
}, 5 * 60 * 1000);

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
