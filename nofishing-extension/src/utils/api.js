/**
 * NoFishing API Client
 * Handles communication with backend API
 */

// API Configuration
const API_CONFIG = {
    base_url: 'http://localhost:8080/api/v1',
    endpoints: {
        detect: '/detect',
        health: '/health',
        check: '/check'
    },
    timeout: 5000
};

/**
 * API Client Class
 */
class NoFishingAPI {
    constructor(config = API_CONFIG) {
        this.config = config;
        this.baseUrl = config.base_url;
    }

    /**
     * Detect if a URL is phishing
     */
    async detectUrl(url, options = {}) {
        const {
            fetchContent = false,
            timeout = this.config.timeout
        } = options;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(`${this.baseUrl}${this.config.endpoints.detect}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: url,
                    fetchContent: fetchContent
                }),
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    /**
     * Quick check a URL
     */
    async quickCheck(url) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        try {
            const response = await fetch(`${this.baseUrl}${this.config.endpoints.check}?url=${encodeURIComponent(url)}`, {
                method: 'GET',
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            return await response.json();

        } finally {
            clearTimeout(timeoutId);
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}${this.config.endpoints.health}`);

            if (!response.ok) {
                return { status: 'DOWN', mlService: 'DOWN' };
            }

            return await response.json();

        } catch (error) {
            return { status: 'DOWN', mlService: 'DOWN', error: error.message };
        }
    }

    /**
     * Update base URL
     */
    setBaseUrl(url) {
        this.baseUrl = url;
    }
}

// Export singleton instance
const api = new NoFishingAPI();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NoFishingAPI, api };
}
