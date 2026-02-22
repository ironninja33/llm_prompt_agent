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

    list.innerHTML = chats.map(chat => `
        <div class="chat-item ${chat.id === currentChatId ? 'active' : ''}"
             onclick="selectChat('${chat.id}')" data-chat-id="${chat.id}">
            <span class="chat-title">${escapeHtml(chat.title || 'New Chat')}</span>
            <button class="btn-delete-chat" onclick="event.stopPropagation(); deleteChat('${chat.id}')" title="Delete">&times;</button>
        </div>
    `).join('');
}

async function selectChat(chatId) {
    cleanupGenerationPollers();  // Clean up active polling from previous chat
    currentChatId = chatId;
    renderChatList();
    await loadMessages(chatId);
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
