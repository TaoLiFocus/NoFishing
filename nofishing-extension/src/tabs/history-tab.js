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
        const filterSelect = document.getElementById('historyFilter');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.currentFilter = e.target.value;
                this.loadHistory();
            });
        }

        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearHistory());
        }
    }

    async loadHistory() {
        const history = await filterDetectionHistory(this.currentFilter);
        this.renderHistory(history);
    }

    renderHistory(history) {
        const container = document.getElementById('historyList');
        if (!container) return;

        if (history.length === 0) {
            container.innerHTML = '<p class="empty-message">暂无检测记录</p>';
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
            <div class="history-entry ${statusClass}" data-id="${entry.id}">
                <div class="history-icon">${icon}</div>
                <div class="history-content">
                    <div class="history-url">${this.truncateUrl(entry.url, 45)}</div>
                    <div class="history-meta">
                        ${entry.riskLevel} • ${confidence}% • ${timeAgo}
                    </div>
                </div>
                <button class="history-details-btn" data-id="${entry.id}">查看详情</button>
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
            await clearDetectionHistory();
            await this.loadHistory();
            showToast('历史记录已清除', 'success');
        } catch (error) {
            console.error('Failed to clear history:', error);
            showToast('清除失败: ' + error.message, 'error');
        }
    }
}

let historyTab = null;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        historyTab = new HistoryTab();
    });
} else {
    historyTab = new HistoryTab();
}
