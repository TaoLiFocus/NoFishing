/**
 * NoFishing Content Script
 * Runs on page load to monitor for suspicious behavior
 */

(function() {
    'use strict';

    // Configuration
    const CHECK_INTERVAL = 2000; // Check every 2 seconds
    const MAX_CHECKS = 3; // Maximum number of checks

    let checkCount = 0;

    /**
     * Initialize content script
     */
    function init() {
        // Only run on http/https pages
        if (!window.location.protocol.startsWith('http')) {
            return;
        }

        console.log('[NoFishing] Content script initialized on', window.location.href);

        // Run initial checks
        performChecks();

        // Set up mutation observer for DOM changes
        observePageChanges();
    }

    /**
     * Perform page security checks
     */
    function performChecks() {
        if (checkCount >= MAX_CHECKS) return;

        checkCount++;

        // Check for password inputs on non-HTTPS
        checkPasswordInputs();

        // Check for suspicious forms
        checkSuspiciousForms();

        // Check for iframe embedding
        checkIframeEmbedding();
    }

    /**
     * Check for password inputs on non-HTTPS pages
     */
    function checkPasswordInputs() {
        if (window.location.protocol === 'https:') return;

        const passwordInputs = document.querySelectorAll('input[type="password"]');

        if (passwordInputs.length > 0) {
            console.warn('[NoFishing] Password inputs found on HTTP page!');
            chrome.runtime.sendMessage({
                action: 'securityWarning',
                type: 'passwordOnHttp',
                url: window.location.href
            });
        }
    }

    /**
     * Check for suspicious form actions
     */
    function checkSuspiciousForms() {
        const forms = document.querySelectorAll('form');

        forms.forEach(form => {
            const action = form.getAttribute('action') || '';
            if (!action) return; // Skip forms without action

            // Convert relative URLs to absolute URLs for checking
            let fullActionUrl;
            try {
                fullActionUrl = new URL(action, window.location.href).href;
            } catch (e) {
                return; // Invalid URL, skip
            }

            const suspiciousPatterns = [
                /login/i,
                /signin/i,
                /verify/i,
                /account/i,
                /password/i
            ];

            // Check for external form action (different origin)
            const actionOrigin = new URL(fullActionUrl).origin;
            const currentOrigin = window.location.origin;

            if (actionOrigin !== currentOrigin) {
                console.warn('[NoFishing] Suspicious form with external action:', {
                    action,
                    fullUrl: fullActionUrl,
                    currentOrigin
                });
            }

            // Check for suspicious keywords in path
            const actionPath = new URL(fullActionUrl).pathname.toLowerCase();
            if (suspiciousPatterns.some(pattern => pattern.test(actionPath))) {
                // Only warn if it's an external form or on HTTP
                if (actionOrigin !== currentOrigin || window.location.protocol === 'http:') {
                    console.warn('[NoFishing] Form with suspicious action:', {
                        action,
                        fullUrl: fullActionUrl
                    });
                }
            }
        });
    }

    /**
     * Check if page is embedded in iframe
     */
    function checkIframeEmbedding() {
        if (window.self !== window.top) {
            console.warn('[NoFishing] Page is embedded in iframe');
            chrome.runtime.sendMessage({
                action: 'securityWarning',
                type: 'iframeEmbed',
                url: window.location.href,
                parentUrl: document.referrer
            });
        }
    }

    /**
     * Observe page changes for dynamic content
     */
    function observePageChanges() {
        const observer = new MutationObserver((mutations) => {
            let hasFormChanges = false;

            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) {
                        if (node.tagName === 'FORM' ||
                            node.tagName === 'INPUT' ||
                            node.querySelector && node.querySelector('form, input[type="password"]')) {
                            hasFormChanges = true;
                        }
                    }
                });
            });

            if (hasFormChanges) {
                performChecks();
            }
        });

        observer.observe(document.body || document.documentElement, {
            childList: true,
            subtree: true
        });

        // Disconnect after some time to avoid performance impact
        setTimeout(() => observer.disconnect(), 30000);
    }

    /**
     * Get page information for analysis
     */
    function getPageInfo() {
        return {
            url: window.location.href,
            title: document.title,
            hasPasswordInputs: document.querySelectorAll('input[type="password"]').length,
            formCount: document.querySelectorAll('form').length,
            inputCount: document.querySelectorAll('input').length,
            linkCount: document.querySelectorAll('a[href]').length,
            scriptCount: document.querySelectorAll('script').length,
            isIframe: window.self !== window.top
        };
    }

    // Listen for messages from background
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === 'getPageInfo') {
            sendResponse(getPageInfo());
            return true;
        }
    });

    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
