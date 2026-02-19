/**
 * NoFishing Storage Utilities
 * Handles chrome.storage.local operations
 */

/**
 * Storage keys
 */
const STORAGE_KEYS = {
    ENABLED: 'enabled',
    SHOW_NOTIFICATIONS: 'showNotifications',
    AUTO_BLOCK: 'autoBlock',
    API_ENDPOINT: 'apiEndpoint',
    SCANNED_COUNT: 'scannedCount',
    BLOCKED_COUNT: 'blockedCount',
    CACHE: 'detectionCache'
};

/**
 * Get settings from storage
 */
async function getSettings() {
    return new Promise((resolve) => {
        chrome.storage.local.get(
            Object.values(STORAGE_KEYS),
            (result) => {
                resolve({
                    enabled: result[STORAGE_KEYS.ENABLED] !== false,
                    showNotifications: result[STORAGE_KEYS.SHOW_NOTIFICATIONS] !== false,
                    autoBlock: result[STORAGE_KEYS.AUTO_BLOCK] || false,
                    apiEndpoint: result[STORAGE_KEYS.API_ENDPOINT] || 'http://localhost:8080/api/v1',
                    scannedCount: result[STORAGE_KEYS.SCANNED_COUNT] || 0,
                    blockedCount: result[STORAGE_KEYS.BLOCKED_COUNT] || 0
                });
            }
        );
    });
}

/**
 * Save settings to storage
 */
async function saveSettings(settings) {
    return new Promise((resolve) => {
        chrome.storage.local.set(settings, () => {
            resolve(true);
        });
    });
}

/**
 * Get cached detection result for URL
 */
async function getCachedResult(url) {
    return new Promise((resolve) => {
        chrome.storage.local.get([STORAGE_KEYS.CACHE], (result) => {
            const cache = result[STORAGE_KEYS.CACHE] || {};
            const cached = cache[url];

            if (!cached) {
                resolve(null);
                return;
            }

            // Check if cache is expired (1 hour)
            const CACHE_TTL = 3600000;
            if (Date.now() - cached.timestamp > CACHE_TTL) {
                delete cache[url];
                chrome.storage.local.set({ [STORAGE_KEYS.CACHE]: cache });
                resolve(null);
                return;
            }

            resolve(cached);
        });
    });
}

/**
 * Cache a detection result
 */
async function cacheResult(url, result) {
    return new Promise((resolve) => {
        chrome.storage.local.get([STORAGE_KEYS.CACHE], (result_data) => {
            const cache = result_data[STORAGE_KEYS.CACHE] || {};

            // Enforce cache size limit
            const MAX_CACHE_SIZE = 500;
            const cacheKeys = Object.keys(cache);
            if (cacheKeys.length >= MAX_CACHE_SIZE) {
                const oldestKey = cacheKeys[0];
                delete cache[oldestKey];
            }

            cache[url] = {
                ...result,
                timestamp: Date.now()
            };

            chrome.storage.local.set({ [STORAGE_KEYS.CACHE]: cache }, () => {
                resolve(true);
            });
        });
    });
}

/**
 * Clear detection cache
 */
async function clearCache() {
    return new Promise((resolve) => {
        chrome.storage.local.set({ [STORAGE_KEYS.CACHE]: {} }, () => {
            resolve(true);
        });
    });
}

/**
 * Increment counter
 */
async function incrementCounter(type) {
    return new Promise((resolve) => {
        const key = type === 'scanned' ? STORAGE_KEYS.SCANNED_COUNT : STORAGE_KEYS.BLOCKED_COUNT;

        chrome.storage.local.get([key], (result) => {
            const current = result[key] || 0;
            chrome.storage.local.set({ [key]: current + 1 }, () => {
                resolve(current + 1);
            });
        });
    });
}

/**
 * Reset all statistics
 */
async function resetStats() {
    return new Promise((resolve) => {
        chrome.storage.local.set({
            [STORAGE_KEYS.SCANNED_COUNT]: 0,
            [STORAGE_KEYS.BLOCKED_COUNT]: 0
        }, () => {
            resolve(true);
        });
    });
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        STORAGE_KEYS,
        getSettings,
        saveSettings,
        getCachedResult,
        cacheResult,
        clearCache,
        incrementCounter,
        resetStats
    };
}
