/**
 * LLM Prompt Agent — Sidebar / Chat list management.
 *
 * Depends on: api.js, app.js ($, $$, escapeHtml, currentChatId, chats)
 *             chatPane.js (loadMessages, renderEmptyState)
 */

// ── Chat List ───────────────────────────────────────────────────────────

async function loadChats() {
    chats = await API.listChats();
    renderChatList();

    // Auto-select first chat if available
    if (chats.length > 0 && !currentChatId) {
        selectChat(chats[0].id);
    }
}

function renderChatList() {
    const list = $('#chat-list');
    if (!list) return;

    list.innerHTML = chats.map(chat => {
        const isActive = chat.id === currentChatId;
        const isStreamActive = typeof StreamRegistry !== 'undefined' && StreamRegistry.isActive(chat.id);
        const classes = ['chat-item'];
        if (isActive) classes.push('active');
        if (isStreamActive) classes.push('chat-streaming');

        return `
            <div class="${classes.join(' ')}"
                 onclick="selectChat('${chat.id}')" data-chat-id="${chat.id}">
                <span class="chat-title">${escapeHtml(chat.title || 'New Chat')}</span>
                ${isStreamActive ? '<span class="chat-streaming-indicator" title="Agent is responding…"></span>' : ''}
                <button class="btn-delete-chat" onclick="event.stopPropagation(); deleteChat('${chat.id}')" title="Delete">&times;</button>
            </div>
        `;
    }).join('');
}

async function selectChat(chatId) {
    cleanupGenerationPollers();  // Clean up active polling from previous chat
    currentChatId = chatId;
    renderChatList();
    await loadMessages(chatId);

    // After loading messages, check if this chat has an active agent stream.
    // On page refresh/reopen, StreamRegistry is empty so isActive() returns false,
    // and we fall through to setStreaming(false) — ensuring input is unlocked.
    if (typeof StreamRegistry !== 'undefined' && StreamRegistry.isActive(chatId)) {
        // Restore the in-progress streaming UI from the buffer
        const buffer = StreamRegistry.getBuffer(chatId);
        addStreamingMessage();
        if (buffer) {
            const el = $('#streaming-message');
            if (el) {
                el.dataset.rawText = buffer;
                el.innerHTML = renderMarkdown(buffer);
            }
        }
        setStreaming(true);
        scrollToBottom();
    } else {
        // No active stream for this chat — ensure input is unlocked.
        // This handles fresh page load, page refresh, browser reopen.
        setStreaming(false);
    }
}

async function createNewChat() {
    // Don't create another if there's already an empty "New Chat"
    const existingEmpty = chats.find(c => c.title === 'New Chat');
    if (existingEmpty) {
        await selectChat(existingEmpty.id);
        return;
    }

    const chat = await API.createChat();
    chats.unshift(chat);
    await selectChat(chat.id);
    renderChatList();
}

async function deleteChat(chatId) {
    await API.deleteChat(chatId);
    chats = chats.filter(c => c.id !== chatId);

    if (currentChatId === chatId) {
        currentChatId = null;
        if (chats.length > 0) {
            selectChat(chats[0].id);
        } else {
            renderEmptyState();
        }
    }

    renderChatList();
}

// ── Sidebar Toggle ──────────────────────────────────────────────────────

function toggleSidebar() {
    const sidebar = $('#sidebar');
    if (sidebar) sidebar.classList.toggle('collapsed');
}
