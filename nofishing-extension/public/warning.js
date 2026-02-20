/**
 * Warning page script - handles user interactions
 */

// Apply translations
function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const message = chrome.i18n.getMessage(key);
        if (message) {
            el.textContent = message;
        }
    });
    document.title = chrome.i18n.getMessage('warningPageTitle');
}

// Initialize page
function initWarningPage() {
    console.log('[NoFishing] Initializing warning page');

    // Parse URL parameters
    const params = new URLSearchParams(window.location.search);
    const url = params.get('url');
    const risk = params.get('risk');
    const confidence = params.get('confidence');

    console.log('[NoFishing] Warning page params:', { url, risk, confidence });

    // Apply translations
    applyTranslations();

    // Update URL display
    const urlDisplay = document.getElementById('urlDisplay');
    if (urlDisplay) {
        urlDisplay.textContent = url || chrome.i18n.getMessage('unknownUrl') || '未知URL';
    }

    // Update confidence display
    if (confidence) {
        const confidencePercent = Math.round(parseFloat(confidence) * 100);
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceFill = document.getElementById('confidenceFill');

        if (confidenceValue) {
            confidenceValue.textContent = confidencePercent + '%';
        }
        if (confidenceFill) {
            confidenceFill.style.width = confidencePercent + '%';
        }
    }

    // Update alert level
    const alertLevel = document.getElementById('alertLevel');
    if (alertLevel) {
        const riskLevel = risk ? risk.toUpperCase() : 'CRITICAL';
        const levelKey = 'risk_' + (risk ? risk.toLowerCase() : 'critical');
        alertLevel.textContent = chrome.i18n.getMessage(levelKey) || riskLevel + ' 风险';
        alertLevel.className = 'alert-level ' + (risk ? risk.toLowerCase() : 'critical');
    }

    // Setup button handlers
    setupButtonHandlers(url);
}

// Setup button event handlers
function setupButtonHandlers(url) {
    const goBackBtn = document.getElementById('goBackBtn');
    const closeTabBtn = document.getElementById('closeTabBtn');
    const proceedBtn = document.getElementById('proceedBtn');

    console.log('[NoFishing] Setting up button handlers');

    if (goBackBtn) {
        goBackBtn.addEventListener('click', () => {
            console.log('[NoFishing] Go back button clicked');
            if (window.history.length > 1) {
                window.history.back();
            } else {
                window.close();
            }
        });
    }

    if (closeTabBtn) {
        closeTabBtn.addEventListener('click', () => {
            console.log('[NoFishing] Close tab button clicked');
            window.close();
        });
    }

    if (proceedBtn) {
        proceedBtn.addEventListener('click', () => {
            console.log('[NoFishing] Proceed button clicked, URL:', url);
            const confirmMsg = chrome.i18n.getMessage('areYouSureToProceed') || '您确定要继续吗？这不推荐。';
            if (confirm(confirmMsg)) {
                if (url) {
                    window.location.href = url;
                }
            }
        });
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWarningPage);
} else {
    initWarningPage();
}
