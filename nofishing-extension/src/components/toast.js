/**
 * Toast Notification Component
 */

class Toast {
    constructor() {
        this.container = null;
        this.toasts = [];
    }

    init() {
        this.createContainer();
    }

    createContainer() {
        const containerHTML = '<div id="toastContainer" class="toast-container"></div>';
        document.body.insertAdjacentHTML('beforeend', containerHTML);
        this.container = document.getElementById('toastContainer');
    }

    show(message, type = 'info', duration = 3000) {
        if (!this.container) this.createContainer();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-message">${message}</span>
            <button class="toast-close">&times;</button>
        `;

        toast.querySelector('.toast-close').addEventListener('click', () => this.remove(toast));
        this.container.appendChild(toast);
        this.toasts.push(toast);

        requestAnimationFrame(() => toast.classList.add('show'));

        if (duration > 0) setTimeout(() => this.remove(toast), duration);
        return toast;
    }

    remove(toast) {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
            this.toasts = this.toasts.filter(t => t !== toast);
        }, 300);
    }

    success(message, duration) { return this.show(message, 'success', duration); }
    error(message, duration) { return this.show(message, 'error', duration); }
    warning(message, duration) { return this.show(message, 'warning', duration); }
    info(message, duration) { return this.show(message, 'info', duration); }
}

const toast = new Toast();

function showToast(message, type = 'info', duration = 3000) {
    return toast.show(message, type, duration);
}