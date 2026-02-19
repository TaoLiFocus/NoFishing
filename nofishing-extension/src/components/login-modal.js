/**
 * Login Modal Component
 */

class LoginModal {
    constructor() {
        this.modal = null;
        this.isVisible = false;
    }

    init() {
        this.createModal();
    }

    createModal() {
        const modalHTML = `
            <div id="loginModal" class="modal hidden">
                <div class="modal-overlay"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>🔐 API Token 登录</h2>
                        <button class="modal-close" id="closeLoginModal">&times;</button>
                    </div>
                    <div class="modal-body">
                        <p class="modal-description">请输入您的 API Token 或用户名/密码登录</p>

                        <div class="login-tabs">
                            <button class="login-tab active" data-tab="token">Token 登录</button>
                            <button class="login-tab" data-tab="password">密码登录</button>
                        </div>

                        <div id="tokenLogin" class="login-panel active">
                            <input type="text" id="apiTokenInput" placeholder="输入 API Token..." class="input-field">
                        </div>

                        <div id="passwordLogin" class="login-panel">
                            <input type="text" id="usernameInput" placeholder="用户名" class="input-field">
                            <input type="password" id="passwordInput" placeholder="密码" class="input-field">
                        </div>

                        <p class="modal-hint">💡 从 Dashboard 获取 Token</p>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" id="cancelLoginBtn">取消</button>
                        <button class="btn btn-primary" id="loginBtn">登录</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.modal = document.getElementById('loginModal');
        this.bindEvents();
    }

    bindEvents() {
        const closeBtn = document.getElementById('closeLoginModal');
        const cancelBtn = document.getElementById('cancelLoginBtn');
        const loginBtn = document.getElementById('loginBtn');

        if (closeBtn) closeBtn.addEventListener('click', () => this.hide());
        if (cancelBtn) cancelBtn.addEventListener('click', () => this.hide());
        if (loginBtn) loginBtn.addEventListener('click', () => this.handleLogin());

        const tabs = this.modal.querySelectorAll('.login-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                tabs.forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                this.switchTab(e.target.dataset.tab);
            });
        });

        const overlay = this.modal.querySelector('.modal-overlay');
        if (overlay) overlay.addEventListener('click', () => this.hide());
    }

    switchTab(tabName) {
        this.modal.querySelectorAll('.login-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(tabName + 'Login').classList.add('active');
    }

    async handleLogin() {
        const activeTab = this.modal.querySelector('.login-tab.active').dataset.tab;
        const loginBtn = document.getElementById('loginBtn');
        loginBtn.disabled = true;
        loginBtn.textContent = '登录中...';

        try {
            if (activeTab === 'token') {
                const token = document.getElementById('apiTokenInput').value.trim();
                if (!token) { showToast('请输入 API Token', 'error'); return; }
                await apiClient.setApiKey(token);
            } else {
                const username = document.getElementById('usernameInput').value.trim();
                const password = document.getElementById('passwordInput').value.trim();
                if (!username || !password) { showToast('请输入用户名和密码', 'error'); return; }
                await apiClient.login(username, password);
            }
            showToast('登录成功', 'success');
            this.hide();
            window.location.reload();
        } catch (error) {
            console.error('Login failed:', error);
            showToast('登录失败: ' + error.message, 'error');
        } finally {
            loginBtn.disabled = false;
            loginBtn.textContent = '登录';
        }
    }

    show() {
        if (!this.modal) this.createModal();
        this.modal.classList.remove('hidden');
        this.isVisible = true;
    }

    hide() {
        if (this.modal) this.modal.classList.add('hidden');
        this.isVisible = false;
    }
}

const loginModal = new LoginModal();

window.addEventListener('load', async () => {
    const token = await getApiToken();
    const apiKey = await getApiKey();
    if (!token && !apiKey) {
        loginModal.init();
        setTimeout(() => loginModal.show(), 500);
    }
});