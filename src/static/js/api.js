/**
 * API client — wraps all fetch calls to the backend.
 */

const API = {
    // ── Chat endpoints ──────────────────────────────────────────
    async listChats() {
        const res = await fetch('/api/chats');
        return res.json();
    },

    async createChat() {
        const res = await fetch('/api/chats', { method: 'POST' });
        return res.json();
    },

    async deleteChat(chatId) {
        const res = await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
        return res.json();
    },

    async getMessages(chatId) {
        const res = await fetch(`/api/chats/${chatId}/messages`);
        return res.json();
    },

    /**
     * Send a message. Returns an EventSource-like reader for SSE.
     * We use fetch + ReadableStream since POST can't use EventSource.
     */
    sendMessage(chatId, content) {
        return fetch(`/api/chats/${chatId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });
    },

    editMessage(chatId, messageId, content) {
        return fetch(`/api/chats/${chatId}/messages/${messageId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });
    },

    // ── Settings endpoints ──────────────────────────────────────
    async getSettings() {
        const res = await fetch('/api/settings');
        return res.json();
    },

    async updateSettings(data) {
        const res = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return res.json();
    },

    async resetSystemPrompt() {
        const res = await fetch('/api/settings/reset-system-prompt', { method: 'POST' });
        return res.json();
    },

    async listModels() {
        const res = await fetch('/api/settings/models');
        return res.json();
    },

    // ── Data directory endpoints ────────────────────────────────
    async listDirectories() {
        const res = await fetch('/api/data-directories');
        return res.json();
    },

    async addDirectory(path, dirType) {
        const res = await fetch('/api/data-directories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, dir_type: dirType }),
        });
        return res.json();
    },

    async deleteDirectory(dirId) {
        const res = await fetch(`/api/data-directories/${dirId}`, { method: 'DELETE' });
        return res.json();
    },

    // ── Ingestion endpoints ─────────────────────────────────────
    async triggerIngestion() {
        const res = await fetch('/api/ingestion/trigger', { method: 'POST' });
        return res.json();
    },

    async refreshOutput() {
        const res = await fetch('/api/ingestion/refresh-output', { method: 'POST' });
        return res.json();
    },

    // ── Stats ───────────────────────────────────────────────────
    async getStats() {
        const res = await fetch('/api/stats');
        return res.json();
    },
};
