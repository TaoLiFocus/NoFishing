/**
 * NoFishing API Client
 * Handles all API calls with token authentication
 */

// API Configuration
const DEFAULT_API_BASE = 'http://localhost:8080/api/v1';

class ApiClient {
    constructor() {
        this.apiBase = DEFAULT_API_BASE;
        this.token = null;
        this.apiKey = null;
    }

    /**
     * Initialize API client with stored credentials
     */
    async init() {
        const result = await this.getStorageData(['apiToken', 'apiKey', 'apiUrl']);
        this.token = result.apiToken || null;
        this.apiKey = result.apiKey || null;
        this.apiBase = result.apiUrl || DEFAULT_API_BASE;
    }

    /**
     * Get data from chrome.storage.local
     */
    getStorageData(keys) {
        return new Promise((resolve) => {
            chrome.storage.local.get(keys, resolve);
        });
    }

    /**
     * Set data in chrome.storage.local
     */
    setStorageData(data) {
        return new Promise((resolve) => {
            chrome.storage.local.set(data, resolve);
        });
    }

    /**
     * Get authorization headers
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        } else if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        return headers;
    }

    /**
     * Make API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        const config = {
            ...options,
            headers: {
                ...this.getHeaders(),
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                if (response.status === 401) {
                    await this.clearCredentials();
                    throw new Error('UNAUTHORIZED');
                }

                // Try to parse error message from response
                let errorMessage = `API Error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.message || errorMessage;
                } catch (e) {
                    // Response body is empty or not JSON
                }
                throw new Error(errorMessage);
            }

            // Handle empty responses (e.g., 204 No Content)
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                return null;
            }

            const text = await response.text();
            if (!text) {
                return null;
            }

            return JSON.parse(text);
        } catch (error) {
            if (error.message === 'UNAUTHORIZED') {
                throw error;
            }
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * GET request
     */
    async get(endpoint, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'GET'
        });
    }

    /**
     * POST request
     */
    async post(endpoint, data, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT request
     */
    async put(endpoint, data, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    async delete(endpoint, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'DELETE'
        });
    }

    /**
     * Login with username/password
     */
    async login(username, password) {
        const data = await this.post('/auth/login', { username, password }, {
            headers: { 'Content-Type': 'application/json' }
        });

        // IMPORTANT: Clear existing apiKey to avoid header conflicts
        this.apiKey = null;
        this.token = data.token;
        await this.setStorageData({ apiToken: this.token, apiKey: null });

        return data;
    }

    /**
     * Verify token by getting current user
     */
    async verifyToken() {
        if (!this.token) {
            return null;
        }

        try {
            const user = await this.get('/auth/me');
            return user;
        } catch (error) {
            await this.clearCredentials();
            return null;
        }
    }

    /**
     * Set and verify API key
     * Uses a detection request to verify the token is valid
     */
    async setApiKey(apiKey) {
        // IMPORTANT: Clear existing token to avoid header conflicts
        this.token = null;

        // Temporarily set the key for verification
        this.apiKey = apiKey;

        // Verify the key by making an actual API call that requires authentication
        // We'll use a simple test URL detection to verify the token works
        try {
            const testUrl = 'https://example.com';
            const response = await this.post('/detect', { url: testUrl });

            // If we get here without 401 error, the token is valid
            console.log('[NoFishing] API Token verified successfully');
            // Also clear apiToken from storage to prevent conflicts
            await this.setStorageData({ apiKey, apiToken: null });
            return true;
        } catch (error) {
            // Clear the invalid key
            this.apiKey = null;
            console.error('[NoFishing] API Token verification failed:', error);

            if (error.message === 'UNAUTHORIZED') {
                throw new Error('API Token无效或已过期');
            } else if (error.message && error.message.includes('fetch')) {
                throw new Error('无法连接到后端服务，请确保服务已启动');
            } else {
                throw new Error('API Token验证失败: ' + (error.message || '未知错误'));
            }
        }
    }

    /**
     * Clear all credentials
     */
    async clearCredentials() {
        this.token = null;
        this.apiKey = null;
        await this.setStorageData({ apiToken: null, apiKey: null });
    }

    /**
     * Logout
     */
    async logout() {
        try {
            await this.post('/auth/logout', {});
        } catch (error) {
            // Ignore logout errors
        }
        await this.clearCredentials();
        // Re-initialize to ensure credentials are cleared from memory
        await this.init();
    }

    // ==================== Detection API ====================

    async detectUrl(url, options = {}) {
        try {
            const mlResult = await this.post('/detect', { url, ...options });

            console.log('[NoFishing] Raw API response:', mlResult);

            // Handle null/undefined response
            if (!mlResult) {
                console.error('[NoFishing] API returned null response');
                return {
                    url: url,
                    isPhishing: false,
                    confidence: 0,
                    riskLevel: 'LOW'
                };
            }

            // Transform backend response to frontend format
            // Backend: is_phishing, probability, risk_level
            // Frontend: isPhishing, confidence, riskLevel
            const result = {
                url: mlResult.url || url,
                isPhishing: Boolean(mlResult.is_phishing),
                confidence: this.extractConfidence(mlResult),
                riskLevel: mlResult.risk_level || mlResult.riskLevel || 'LOW',
                processingTimeMs: mlResult.processing_time_ms || mlResult.processingTimeMs || 0
            };

            console.log('[NoFishing] Transformed result:', result);
            return result;
        } catch (error) {
            console.error('[NoFishing] Detection error:', error);
            throw error;
        }
    }

    /**
     * Extract confidence from various possible field names
     */
    extractConfidence(mlResult) {
        // Try different possible field names with type conversion
        const fields = ['probability', 'confidence', 'score'];

        for (const field of fields) {
            const value = mlResult[field];

            // Handle number
            if (typeof value === 'number') {
                return value;
            }

            // Handle string (convert to number)
            if (typeof value === 'string') {
                const num = parseFloat(value);
                if (!isNaN(num)) {
                    return num;
                }
            }
        }

        // Log warning if no valid confidence found
        console.warn('[NoFishing] No valid confidence field in response:', mlResult);
        return 0;  // Return 0 as fallback
    }

    async checkUrl(url) {
        return this.get(`/check?url=${encodeURIComponent(url)}`);
    }

    // ==================== Whitelist API ====================

    async getWhitelist(page = 0, size = 10) {
        return this.get(`/whitelist?page=${page}&size=${size}`);
    }

    async addToWhitelist(pattern, comment = '') {
        return this.post('/whitelist', {
            pattern,
            comment,
            addedBy: 'extension'
        });
    }

    async removeFromWhitelist(id) {
        return this.delete(`/whitelist/${id}`);
    }

    async checkWhitelist(url) {
        return this.get(`/whitelist/check?url=${encodeURIComponent(url)}`);
    }

    // ==================== Blacklist API ====================

    async getBlacklist(page = 0, size = 10) {
        return this.get(`/blacklist?page=${page}&size=${size}`);
    }

    async addToBlacklist(pattern, comment = '') {
        return this.post('/blacklist', {
            pattern,
            comment,
            addedBy: 'extension'
        });
    }

    async removeFromBlacklist(id) {
        return this.delete(`/blacklist/${id}`);
    }

    async checkBlacklist(url) {
        return this.get(`/blacklist/check?url=${encodeURIComponent(url)}`);
    }

    // ==================== Health Check ====================

    async healthCheck() {
        try {
            const data = await this.get('/health');
            return data;
        } catch (error) {
            return { status: 'down', mlService: 'DOWN' };
        }
    }
}

// Create singleton instance
const apiClient = new ApiClient();

// Auto-initialize on load
apiClient.init().catch(console.error);
