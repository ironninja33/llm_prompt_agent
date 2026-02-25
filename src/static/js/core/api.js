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

    /**
     * Send a message with file attachments via multipart/form-data.
     * Do NOT set Content-Type — the browser sets it with the boundary.
     */
    sendMessageWithAttachments(chatId, formData) {
        return fetch(`/api/chats/${chatId}/messages`, {
            method: 'POST',
            body: formData,
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

    // ── ComfyUI endpoints ──────────────────────────────────────
    async checkComfyUIHealth() {
        const res = await fetch('/api/comfyui/health');
        return res.json();
    },

    async getComfyUIModels(modelType) {
        const res = await fetch(`/api/comfyui/models/${modelType}`);
        return res.json();
    },

    async getComfyUIOutputFolders() {
        const res = await fetch('/api/comfyui/output-folders');
        return res.json();
    },

    async validateWorkflow(path) {
        const res = await fetch('/api/comfyui/validate-workflow', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        return res.json();
    },

    async getWorkflowInfo() {
        const res = await fetch('/api/comfyui/workflow');
        return res.json();
    },

    async uploadWorkflowAPI(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/comfyui/workflow/api', {
            method: 'POST',
            body: formData,
        });
        return res.json();
    },

    async uploadWorkflowUI(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/comfyui/workflow/ui', {
            method: 'POST',
            body: formData,
        });
        return res.json();
    },

    async deleteWorkflow() {
        const res = await fetch('/api/comfyui/workflow', { method: 'DELETE' });
        return res.json();
    },

    async submitGeneration(chatId, messageId, settings) {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: chatId, message_id: messageId, settings }),
        });
        return res.json();
    },

    async getChatGenerations(chatId) {
        const res = await fetch(`/api/generate/chat/${chatId}`);
        return res.json();
    },

    async deleteGeneratedImage(jobId, imageId) {
        const res = await fetch(`/api/generate/image/${jobId}/${imageId}`, {
            method: 'DELETE',
        });
        return res.json();
    },

    async deleteGenerationJob(jobId) {
        const res = await fetch(`/api/generate/job/${jobId}`, {
            method: 'DELETE',
        });
        return res.json();
    },
};
