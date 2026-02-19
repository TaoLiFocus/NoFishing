/**
 * NoFishing Storage Helpers
 * Utilities for managing chrome.storage.local
 */

// Storage keys
const STORAGE_KEYS = {
    API_TOKEN: 'apiToken',
    API_KEY: 'apiKey',
    API_URL: 'apiUrl',
    DETECTION_HISTORY: 'detectionHistory',
    SETTINGS: 'settings',
    SCANNED_COUNT: 'scannedCount',
    BLOCKED_COUNT: 'blockedCount'
};

// Default settings
const DEFAULT_SETTINGS = {
    autoBlock: false,
    showNotifications: true,
    autoScan: true,
    sensitivity: 'medium' // 'low', 'medium', 'high'
};

/**
 * Get data from chrome.storage.local
 */
function getStorage(keys) {
    return new Promise((resolve) => {
        chrome.storage.local.get(keys, (result) => resolve(result || {}));
    });
}

/**
 * Set data in chrome.storage.local
 */
function setStorage(data) {
    return new Promise((resolve) => {
        chrome.storage.local.set(data, () => resolve());
    });
}

/**
 * Remove data from chrome.storage.local
 */
function removeStorage(keys) {
    return new Promise((resolve) => {
        chrome.storage.local.remove(keys, () => resolve());
    });
}

/**
 * Get API token
 */
async function getApiToken() {
    const data = await getStorage([STORAGE_KEYS.API_TOKEN]);
    return data[STORAGE_KEYS.API_TOKEN] || null;
}

/**
 * Set API token
 */
async function setApiToken(token) {
    return setStorage({ [STORAGE_KEYS.API_TOKEN]: token });
}

/**
 * Get API key
 */
async function getApiKey() {
    const data = await getStorage([STORAGE_KEYS.API_KEY]);
    return data[STORAGE_KEYS.API_KEY] || null;
}

/**
 * Set API key
 */
async function setApiKey(apiKey) {
    return setStorage({ [STORAGE_KEYS.API_KEY]: apiKey });
}

/**
 * Get API URL
 */
async function getApiUrl() {
    const data = await getStorage([STORAGE_KEYS.API_URL]);
    return data[STORAGE_KEYS.API_URL] || 'http://localhost:8080/api/v1';
}

/**
 * Set API URL
 */
async function setApiUrl(url) {
    return setStorage({ [STORAGE_KEYS.API_URL]: url });
}

/**
 * Get settings
 */
async function getSettings() {
    const data = await getStorage([STORAGE_KEYS.SETTINGS]);
    return { ...DEFAULT_SETTINGS, ...data[STORAGE_KEYS.SETTINGS] };
}

/**
 * Set settings
 */
async function setSettings(settings) {
    const current = await getSettings();
    return setStorage({ [STORAGE_KEYS.SETTINGS]: { ...current, ...settings } });
}

/**
 * Get detection history
 */
async function getDetectionHistory() {
    const data = await getStorage([STORAGE_KEYS.DETECTION_HISTORY]);
    return data[STORAGE_KEYS.DETECTION_HISTORY] || [];
}

/**
 * Add detection to history
 */
async function addDetectionHistory(entry) {
    const history = await getDetectionHistory();

    // Create history entry
    const historyEntry = {
        id: Date.now(),
        url: entry.url,
        isPhishing: entry.isPhishing,
        confidence: entry.confidence,
        riskLevel: entry.riskLevel,
        timestamp: entry.timestamp || Date.now(),
        processingTimeMs: entry.processingTimeMs
    };

    // Add to beginning of array
    history.unshift(historyEntry);

    // Trim to max 200 entries
    if (history.length > 200) {
        history.splice(200);
    }

    // Remove entries older than 30 days
    const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    const filtered = history.filter(entry => entry.timestamp > thirtyDaysAgo);

    return setStorage({ [STORAGE_KEYS.DETECTION_HISTORY]: filtered });
}

/**
 * Clear detection history
 */
async function clearDetectionHistory() {
    return setStorage({ [STORAGE_KEYS.DETECTION_HISTORY]: [] });
}

/**
 * Filter detection history
 */
async function filterDetectionHistory(filter = 'all') {
    const history = await getDetectionHistory();

    if (filter === 'all') {
        return history;
    }

    return history.filter(entry => {
        if (filter === 'safe') {
            return !entry.isPhishing;
        } else if (filter === 'suspicious') {
            return entry.riskLevel === 'MEDIUM';
        } else if (filter === 'phishing') {
            return entry.isPhishing;
        }
        return true;
    });
}

/**
 * Get stats
 */
async function getStats() {
    const data = await getStorage([STORAGE_KEYS.SCANNED_COUNT, STORAGE_KEYS.BLOCKED_COUNT]);
    return {
        scanned: data[STORAGE_KEYS.SCANNED_COUNT] || 0,
        blocked: data[STORAGE_KEYS.BLOCKED_COUNT] || 0
    };
}

/**
 * Increment stat
 */
async function incrementStat(type) {
    const key = type === 'scanned' ? STORAGE_KEYS.SCANNED_COUNT : STORAGE_KEYS.BLOCKED_COUNT;
    const data = await getStorage([key]);
    const current = data[key] || 0;
    return setStorage({ [key]: current + 1 });
}

/**
 * Clear all credentials
 */
async function clearCredentials() {
    return removeStorage([STORAGE_KEYS.API_TOKEN, STORAGE_KEYS.API_KEY]);
}

// Export to global scope for Chrome extension
window.getStorage = getStorage;
window.setStorage = setStorage;
window.removeStorage = removeStorage;
window.getApiToken = getApiToken;
window.setApiToken = setApiToken;
window.getApiKey = getApiKey;
window.setApiKey = setApiKey;
window.getApiUrl = getApiUrl;
window.setApiUrl = setApiUrl;
window.getSettings = getSettings;
window.setSettings = setSettings;
window.getDetectionHistory = getDetectionHistory;
window.addDetectionHistory = addDetectionHistory;
window.clearDetectionHistory = clearDetectionHistory;
window.filterDetectionHistory = filterDetectionHistory;
window.getStats = getStats;
window.incrementStat = incrementStat;
window.clearCredentials = clearCredentials;

// Aliases for backward compatibility
window.addHistoryEntry = addDetectionHistory;
window.clearHistory = clearDetectionHistory;
window.getHistory = getDetectionHistory;
