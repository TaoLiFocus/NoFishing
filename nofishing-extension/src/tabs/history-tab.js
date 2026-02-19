/**
 * History Tab Logic
 */

class HistoryTab {
    constructor() {
        this.currentFilter = 'all';
    }

    async init() {
        this.bindEvents();
        await this.loadHistory();
    }

    bindEvents() {
        const filterSelect = document.getElementById('history-filter');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.currentFilter = e.target.value;
                this.loadHistory();
            });
        }

        const clearBtn = document.getElementById('clear-history-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearHistory());
        }
    }

    async loadHistory() {
        const history = await filterDetectionHistory(this.currentFilter);
        this.renderHistory(history);
    }

    renderHistory(history) {
        const container = document.getElementById('history-list');
        if (!container) return;

        if (history.length === 0) {
            container.innerHTML = '<div class="history-empty"><div class="empty-icon">📋</div><div class="empty-title">暂无检测记录</div><div class="empty-text">检测历史将显示在这里</div></div>';
            return;
        }

        container.innerHTML = history.map(entry => this.createHistoryEntry(entry)).join('');
    }

    createHistoryEntry(entry) {
        const icon = entry.isPhishing ? '⛔' : '✅';
        const statusClass = entry.isPhishing ? 'danger' : 'safe';
        const timeAgo = this.formatTimeAgo(entry.timestamp);
        const confidence = Math.round(entry.confidence * 100);

        return `
            <div class="history-item ${statusClass}" data-id="${entry.id}">
                <div class="history-icon">${icon}</div>
                <div class="history-content">
                    <div class="history-url">${this.truncateUrl(entry.url, 45)}</div>
                    <div class="history-meta">
                        <span class="history-status ${statusClass}">${entry.riskLevel}</span>
                        <span class="history-confidence">${confidence}%</span>
                        <span class="history-time">${timeAgo}</span>
                    </div>
                </div>
                <button class="history-details-btn" onclick="showHistoryDetails(${entry.id})">详情</button>
            </div>
        `;
    }

    truncateUrl(url, maxLength) {
        if (url.length <= maxLength) return url;
        return url.substring(0, maxLength) + '...';
    }

    formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        if (seconds < 60) return '刚刚';
        if (seconds < 3600) return Math.floor(seconds / 60) + '分钟前';
        if (seconds < 86400) return Math.floor(seconds / 3600) + '小时前';
        if (seconds < 2592000) return Math.floor(seconds / 86400) + '天前';
        const date = new Date(timestamp);
        return date.toLocaleDateString('zh-CN');
    }

    async clearHistory() {
        if (!confirm('确定要清除所有检测历史吗？')) return;
        try {
            await clearHistory();
            await this.loadHistory();
            showToast('历史记录已清除', 'success');
        } catch (error) {
            console.error('Failed to clear history:', error);
            showToast('清除失败: ' + error.message, 'error');
        }
    }
}

/**
 * Show history entry details
 */
async function showHistoryDetails(entryId) {
    const history = await getHistory();
    const entry = history.find(e => e.id === entryId);

    if (!entry) {
        showToast('未找到检测记录', 'error');
        return;
    }

    const modal = document.getElementById('details-modal');
    const content = document.getElementById('details-content');

    if (modal && content) {
        const statusClass = entry.isPhishing ? 'danger' : 'safe';
        const statusText = entry.isPhishing ? '钓鱼网站' : '安全';

        content.innerHTML = `
            <div class="detail-row">
                <div class="detail-label">URL</div>
                <div class="detail-value">${entry.url}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">检测结果</div>
                <div class="detail-value ${statusClass}">${statusText}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">风险等级</div>
                <div class="detail-value ${statusClass}">${entry.riskLevel}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">置信度</div>
                <div class="detail-value">${Math.round(entry.confidence * 100)}%</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">检测时间</div>
                <div class="detail-value">${new Date(entry.timestamp).toLocaleString('zh-CN')}</div>
            </div>
            ${entry.processingTimeMs ? `
            <div class="detail-row">
                <div class="detail-label">处理耗时</div>
                <div class="detail-value">${entry.processingTimeMs} ms</div>
            </div>
            ` : ''}
        `;

        modal.classList.remove('hidden');
    }
}

// Export to global scope
window.showHistoryDetails = showHistoryDetails;

let historyTab = null;

/**
 * Initialize history tab (called from popup.js)
 */
async function initHistoryTab() {
    if (!historyTab) {
        historyTab = new HistoryTab();
    }
    await historyTab.init();
}

// Export for use in popup.js
window.initHistoryTab = initHistoryTab;
