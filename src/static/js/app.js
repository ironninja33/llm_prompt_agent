/**
 * LLM Prompt Agent — Main application logic.
 *
 * Depends on: api.js, sse.js, markdown.js (loaded before this file)
 */

// ── State ───────────────────────────────────────────────────────────────
let currentChatId = null;
let isStreaming = false;
let chats = [];

// ── DOM References ──────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Initialization ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    await loadChats();
    loadStats();
    monitorIngestion();
});

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

// ── Messages ────────────────────────────────────────────────────────────

async function loadMessages(chatId) {
    const messages = await API.getMessages(chatId);
    renderMessages(messages);
}

function renderMessages(messages) {
    const container = $('#messages');
    if (!container) return;

    if (messages.length === 0) {
        renderEmptyState();
        return;
    }

    $('#empty-state')?.classList.add('hidden');

    container.innerHTML = '';
    messages.forEach((msg, idx) => {
        const el = createMessageElement(msg, idx === messages.length - 1 && msg.role === 'user');
        container.appendChild(el);
    });

    scrollToBottom();
}

function renderEmptyState() {
    const container = $('#messages');
    if (!container) return;

    container.innerHTML = `
        <div id="empty-state" class="empty-state">
            <h2>Welcome to LLM Prompt Agent</h2>
            <p>Start a conversation to generate creative image prompts.</p>
            <p class="hint">The agent can search your training data and generated outputs to create new, tailored prompts.</p>
        </div>
    `;
}

function createMessageElement(msg, isLastUserMsg = false) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    div.dataset.messageId = msg.id;

    if (msg.role === 'assistant') {
        div.innerHTML = renderMarkdown(msg.content);
    } else {
        div.textContent = msg.content;
    }

    // Edit button on the last user message
    if (isLastUserMsg && msg.role === 'user') {
        const editBtn = document.createElement('button');
        editBtn.className = 'edit-btn';
        editBtn.textContent = '✎';
        editBtn.title = 'Edit and resubmit';
        editBtn.onclick = () => startEditMessage(msg.id, msg.content);
        div.appendChild(editBtn);
    }

    return div;
}

function addStreamingMessage() {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'message assistant streaming';
    div.id = 'streaming-message';
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function appendToStreamingMessage(text) {
    const el = $('#streaming-message');
    if (!el) return;

    // Accumulate raw text in a data attribute
    const rawText = (el.dataset.rawText || '') + text;
    el.dataset.rawText = rawText;

    // Render as markdown
    el.innerHTML = renderMarkdown(rawText);
    scrollToBottom();
}

function finalizeStreamingMessage(messageId, toolCalls = null) {
    const el = $('#streaming-message');
    if (!el) {
        return;
    }

    el.classList.remove('streaming');
    el.id = '';
    if (messageId) {
        el.dataset.messageId = messageId;
    }

    // Insert tool calls introspection section if present
    if (toolCalls && toolCalls.length > 0) {
        const section = buildToolCallsSection(toolCalls);
        el.insertBefore(section, el.firstChild);
    }
}

function buildToolCallsSection(toolCalls) {
    const details = document.createElement('details');
    details.className = 'tool-calls-section';

    const summary = document.createElement('summary');
    summary.className = 'tool-calls-toggle';
    summary.innerHTML =
        `<span class="tool-calls-icon">🔧</span>` +
        `<span>${toolCalls.length} tool call${toolCalls.length !== 1 ? 's' : ''}</span>` +
        `<span class="tool-calls-chevron">▶</span>`;
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'tool-calls-list';

    toolCalls.forEach(call => {
        const item = document.createElement('div');
        item.className = 'tool-call-item';

        const header = document.createElement('div');
        header.className = 'tool-call-header';
        header.innerHTML = `<span class="tool-call-name">${escapeHtml(call.tool)}</span>`;
        item.appendChild(header);

        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'tool-call-details';

        // Args
        const argsDiv = document.createElement('div');
        argsDiv.className = 'tool-call-args';
        const argsLabel = document.createElement('span');
        argsLabel.className = 'tool-call-label';
        argsLabel.textContent = 'Args:';
        const argsPre = document.createElement('pre');
        argsPre.textContent = JSON.stringify(call.args, null, 2);
        argsDiv.appendChild(argsLabel);
        argsDiv.appendChild(argsPre);
        detailsDiv.appendChild(argsDiv);

        // Result
        const resultDiv = document.createElement('div');
        resultDiv.className = 'tool-call-result';
        const resultLabel = document.createElement('span');
        resultLabel.className = 'tool-call-label';
        resultLabel.textContent = 'Result:';
        const resultPre = document.createElement('pre');
        resultPre.textContent = JSON.stringify(call.result, null, 2);
        resultDiv.appendChild(resultLabel);
        resultDiv.appendChild(resultPre);
        detailsDiv.appendChild(resultDiv);

        item.appendChild(detailsDiv);
        list.appendChild(item);
    });

    details.appendChild(list);
    return details;
}

function addStatusMessage(text) {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'tool-status';
    div.innerHTML = `<div class="spinner"></div><span>${escapeHtml(text)}</span>`;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function removeStatusMessages() {
    $$('.tool-status').forEach(el => el.remove());
}

function addErrorMessage(text) {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'message error';
    div.textContent = text;
    container.appendChild(div);
    scrollToBottom();
}

// ── Send Message ────────────────────────────────────────────────────────

async function sendMessage() {
    const input = $('#message-input');
    const content = input.value.trim();
    if (!content || isStreaming) return;
    let pendingToolCalls = null;

    // Create chat if none selected
    if (!currentChatId) {
        await createNewChat();
    }

    // Add user message to UI
    const container = $('#messages');
    $('#empty-state')?.classList.add('hidden');

    const userDiv = document.createElement('div');
    userDiv.className = 'message user';
    userDiv.textContent = content;
    container.appendChild(userDiv);

    input.value = '';
    autoResizeInput(input);
    setStreaming(true);
    scrollToBottom();

    try {
        const response = await API.sendMessage(currentChatId, content);
        if (!response.ok) {
            const err = await response.json();
            addErrorMessage(err.error || 'Failed to send message');
            setStreaming(false);
            return;
        }

        const streamingEl = addStreamingMessage();

        await readSSEStream(response, {
            token(data) {
                appendToStreamingMessage(data.text || '');
            },
            status(data) {
                removeStatusMessages();
                addStatusMessage(data.message || 'Processing...');
            },
            tool_result(data) {
                removeStatusMessages();
                const summary = data.summary || 'Done';
                addStatusMessage(`✓ ${data.tool}: ${summary}`);
                // Remove after a short delay
                setTimeout(removeStatusMessages, 1500);
            },
            tool_calls(data) {
                pendingToolCalls = data.calls;
            },
            error(data) {
                addErrorMessage(data.message || 'An error occurred');
            },
            done(data) {
                finalizeStreamingMessage(data.message_id, pendingToolCalls);
                setStreaming(false);
                // Refresh chat list to pick up new title
                loadChats();
            },
        });

    } catch (err) {
        addErrorMessage(`Network error: ${err.message}`);
    } finally {
        setStreaming(false);
    }
}

// ── Edit & Resubmit ─────────────────────────────────────────────────────

function startEditMessage(messageId, content) {
    const input = $('#message-input');
    input.value = content;
    input.focus();
    autoResizeInput(input);

    // Replace send button behavior temporarily
    const sendBtn = $('#send-btn');
    sendBtn.onclick = () => submitEditedMessage(messageId);
    sendBtn.title = 'Resubmit';
}

async function submitEditedMessage(messageId) {
    const input = $('#message-input');
    const content = input.value.trim();
    if (!content || isStreaming) return;
    let pendingToolCalls = null;

    // Restore normal send behavior
    const sendBtn = $('#send-btn');
    sendBtn.onclick = sendMessage;
    sendBtn.title = 'Send';

    input.value = '';
    autoResizeInput(input);
    setStreaming(true);

    // Reload messages up to the edited point
    // (The backend deletes from messageId onward)
    try {
        const response = await API.editMessage(currentChatId, messageId, content);
        if (!response.ok) {
            const err = await response.json();
            addErrorMessage(err.error || 'Failed to edit message');
            setStreaming(false);
            return;
        }

        // Reload messages to show cleaned-up history, then stream
        await loadMessages(currentChatId);

        // Add the new user message
        const container = $('#messages');
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.textContent = content;
        container.appendChild(userDiv);

        const streamingEl = addStreamingMessage();

        await readSSEStream(response, {
            token(data) {
                appendToStreamingMessage(data.text || '');
            },
            status(data) {
                removeStatusMessages();
                addStatusMessage(data.message || 'Processing...');
            },
            tool_result(data) {
                removeStatusMessages();
            },
            tool_calls(data) {
                pendingToolCalls = data.calls;
            },
            error(data) {
                addErrorMessage(data.message || 'An error occurred');
            },
            done(data) {
                finalizeStreamingMessage(data.message_id, pendingToolCalls);
                setStreaming(false);
                loadChats();
            },
        });

    } catch (err) {
        addErrorMessage(`Network error: ${err.message}`);
    } finally {
        setStreaming(false);
    }
}

// ── Input Handling ──────────────────────────────────────────────────────

function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeInput(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function setStreaming(value) {
    isStreaming = value;
    const sendBtn = $('#send-btn');
    const input = $('#message-input');
    if (sendBtn) sendBtn.disabled = value;
    if (input) input.disabled = value;
}

// ── Sidebar ─────────────────────────────────────────────────────────────

function toggleSidebar() {
    const sidebar = $('#sidebar');
    if (sidebar) sidebar.classList.toggle('collapsed');
}

// ── Settings Modal ──────────────────────────────────────────────────────

async function openSettings() {
    const modal = $('#settings-modal');
    modal.classList.remove('hidden');

    // Load current settings
    const settings = await API.getSettings();

    $('#setting-api-key').value = '';
    $('#setting-api-key').placeholder = settings.gemini_api_key_set
        ? settings.gemini_api_key_masked
        : 'Enter your API key';
    $('#api-key-status').textContent = settings.gemini_api_key_set
        ? 'API key is set'
        : 'No API key configured';

    $('#setting-model-agent').value = settings.model_agent || '';
    $('#setting-model-embedding').value = settings.model_embedding || '';
    $('#setting-model-summary').value = settings.model_summary || '';
    $('#setting-rate-limit').value = settings.gemini_rate_limit || '3000';
    $('#setting-system-prompt').value = settings.system_prompt || '';

    // Load parameter settings
    $('#setting-query-k-similar').value = settings.query_k_similar || '10';
    $('#setting-query-k-theme-intra').value = settings.query_k_theme_intra || '5';
    $('#setting-query-k-theme-cross').value = settings.query_k_theme_cross || '5';
    $('#setting-query-k-random').value = settings.query_k_random || '3';
    $('#setting-cluster-k-intra').value = settings.cluster_k_intra || '5';
    $('#setting-cluster-k-cross').value = settings.cluster_k_cross || '15';
    $('#setting-cluster-min-folder-size').value = settings.cluster_min_folder_size || '20';
    $('#setting-cluster-label-terms').value = settings.cluster_label_terms || '3';

    // Load directories
    await loadDirectories();
}

function closeSettings() {
    $('#settings-modal').classList.add('hidden');
}

function switchTab(btn) {
    const tabId = btn.dataset.tab;

    // Update tab buttons
    $$('.modal-tabs .tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    // Update tab panes
    $$('.tab-pane').forEach(p => p.classList.add('hidden'));
    $(`#${tabId}`).classList.remove('hidden');
}

function toggleApiKeyVisibility() {
    const input = $('#setting-api-key');
    input.type = input.type === 'password' ? 'text' : 'password';
}

async function saveApiSettings() {
    const data = {};

    const apiKey = $('#setting-api-key').value.trim();
    if (apiKey) data.gemini_api_key = apiKey;

    const modelAgent = $('#setting-model-agent').value.trim();
    if (modelAgent) data.model_agent = modelAgent;

    const modelEmbed = $('#setting-model-embedding').value.trim();
    if (modelEmbed) data.model_embedding = modelEmbed;

    const modelSummary = $('#setting-model-summary').value.trim();
    if (modelSummary) data.model_summary = modelSummary;

    const rateLimit = $('#setting-rate-limit').value.trim();
    if (rateLimit !== '') data.gemini_rate_limit = rateLimit;

    if (Object.keys(data).length === 0) return;

    await API.updateSettings(data);
    $('#api-key-status').textContent = apiKey ? 'API key updated' : 'Settings saved';
}

async function saveSystemPrompt() {
    const prompt = $('#setting-system-prompt').value;
    await API.updateSettings({ system_prompt: prompt });
}

async function resetSystemPrompt() {
    const result = await API.resetSystemPrompt();
    if (result.system_prompt) {
        $('#setting-system-prompt').value = result.system_prompt;
    }
}

async function saveParamSettings() {
    const settings = {
        query_k_similar: $('#setting-query-k-similar').value,
        query_k_theme_intra: $('#setting-query-k-theme-intra').value,
        query_k_theme_cross: $('#setting-query-k-theme-cross').value,
        query_k_random: $('#setting-query-k-random').value,
        cluster_k_intra: $('#setting-cluster-k-intra').value,
        cluster_k_cross: $('#setting-cluster-k-cross').value,
        cluster_min_folder_size: $('#setting-cluster-min-folder-size').value,
        cluster_label_terms: $('#setting-cluster-label-terms').value,
    };

    const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
    });

    if (response.ok) {
        alert('Parameters saved successfully');
    }
}

// ── Data Directories ────────────────────────────────────────────────────

async function loadDirectories() {
    const dirs = await API.listDirectories();
    const list = $('#dir-list');
    if (!list) return;

    if (dirs.length === 0) {
        list.innerHTML = '<p class="hint">No data directories configured.</p>';
        return;
    }

    list.innerHTML = dirs.map(d => `
        <div class="dir-item">
            <span class="dir-type ${d.dir_type}">${d.dir_type}</span>
            <span class="dir-path" title="${escapeHtml(d.path)}">${escapeHtml(d.path)}</span>
            <button class="btn-danger" onclick="removeDirectory(${d.id})">✕</button>
        </div>
    `).join('');
}

async function addDirectory() {
    const path = $('#new-dir-path').value.trim();
    const dirType = $('#new-dir-type').value;
    if (!path) return;

    try {
        await API.addDirectory(path, dirType);
        $('#new-dir-path').value = '';
        await loadDirectories();
    } catch (err) {
        alert('Failed to add directory: ' + err.message);
    }
}

async function removeDirectory(dirId) {
    await API.deleteDirectory(dirId);
    await loadDirectories();
}

async function triggerIngestion() {
    await API.triggerIngestion();
    closeSettings();
    monitorIngestion();
}

async function refreshOutput() {
    await API.refreshOutput();
    closeSettings();
    monitorIngestion();
}

// ── Ingestion Monitoring ────────────────────────────────────────────────

function monitorIngestion() {
    const overlay = $('#ingestion-overlay');
    const progressBar = $('#ingestion-progress-bar');
    const statusMsg = $('#ingestion-status-message');
    const detail = $('#ingestion-detail');

    // Open SSE connection for ingestion status
    const evtSource = new EventSource('/api/ingestion/status');
    let hasShownOverlay = false;

    evtSource.addEventListener('ingestion_status', (e) => {
        const data = JSON.parse(e.data);

        // Show overlay on first real status update
        if (!hasShownOverlay && data.phase !== 'idle') {
            overlay.classList.remove('hidden');
            hasShownOverlay = true;
        }

        statusMsg.textContent = data.message || 'Processing...';
        detail.textContent = data.current_dir || '';

        if (data.phase === 'discovery') {
            progressBar.classList.add('indeterminate');
            progressBar.style.width = '';
        } else if (data.phase === 'embedding' && data.new_files > 0) {
            progressBar.classList.remove('indeterminate');
            const pct = Math.round((data.current / data.new_files) * 100);
            progressBar.style.width = pct + '%';
        }
    });

    evtSource.addEventListener('ingestion_complete', (e) => {
        const data = JSON.parse(e.data);
        evtSource.close();

        if (hasShownOverlay) {
            statusMsg.textContent = data.message || 'Indexing complete';
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';
            detail.textContent = '';

            // Hide overlay after a brief moment
            setTimeout(() => {
                overlay.classList.add('hidden');
                progressBar.style.width = '0%';
            }, 1200);
        }

        // Refresh stats
        loadStats();
    });

    evtSource.onerror = () => {
        evtSource.close();
        // If ingestion never started, just hide
        if (!hasShownOverlay) {
            overlay.classList.add('hidden');
        }
    };
}

// ── Clustering ──────────────────────────────────────────────────────────

async function triggerClustering() {
    try {
        const response = await fetch("/api/clustering/trigger", { method: "POST" });
        const data = await response.json();

        if (response.ok) {
            // Successfully started
            document.getElementById("clustering-overlay").classList.remove("hidden");
            monitorClustering();
        } else if (response.status === 409) {
            // Already running or ingestion in progress — show overlay and monitor
            if (data.error && data.error.includes("already running")) {
                document.getElementById("clustering-overlay").classList.remove("hidden");
                monitorClustering();
            } else {
                alert(data.error || "Cannot start clustering right now");
            }
        } else {
            alert(data.error || "Failed to start clustering");
        }
    } catch (err) {
        alert("Failed to trigger clustering: " + err.message);
    }
}

function monitorClustering() {
    const overlay = document.getElementById("clustering-overlay");
    const progressBar = document.getElementById("clustering-progress-bar");
    const statusMsg = document.getElementById("clustering-status-message");
    const detail = document.getElementById("clustering-detail");

    const evtSource = new EventSource("/api/clustering/status");
    let hasShownOverlay = false;

    evtSource.addEventListener("clustering_status", (e) => {
        const data = JSON.parse(e.data);

        if (!hasShownOverlay && data.phase !== "idle") {
            overlay.classList.remove("hidden");
            hasShownOverlay = true;
        }

        statusMsg.textContent = data.message || "Processing...";

        // Update progress bar
        if (data.total && data.total > 0) {
            progressBar.classList.remove("indeterminate");
            const pct = Math.round((data.current / data.total) * 100);
            progressBar.style.width = pct + "%";
        } else {
            progressBar.classList.add("indeterminate");
            progressBar.style.width = "";
        }

        // Show phase info in detail
        if (data.phase) {
            const phaseLabels = {
                starting: "Initializing...",
                cross_folder: "Cross-folder clustering",
                intra_folder: "Intra-folder clustering",
                labeling: "Generating cluster labels",
            };
            detail.textContent = phaseLabels[data.phase] || data.phase;
        }
    });

    evtSource.addEventListener("clustering_complete", (e) => {
        const data = JSON.parse(e.data);
        evtSource.close();

        if (hasShownOverlay || !overlay.classList.contains("hidden")) {
            if (data.phase === "error") {
                statusMsg.textContent = data.message || "Clustering failed";
                detail.textContent = "";
                setTimeout(() => {
                    overlay.classList.add("hidden");
                    progressBar.classList.remove("indeterminate");
                    progressBar.style.width = "0%";
                    statusMsg.textContent = "Starting...";
                    detail.textContent = "";
                }, 3000);
            } else {
                statusMsg.textContent = data.message || "Clustering complete";
                progressBar.classList.remove("indeterminate");
                progressBar.style.width = "100%";
                detail.textContent = "";

                setTimeout(() => {
                    overlay.classList.add("hidden");
                    progressBar.style.width = "0%";
                    statusMsg.textContent = "Starting...";
                    detail.textContent = "";
                }, 2000);
            }
        }

        // Refresh stats
        loadStats();
    });

    evtSource.onerror = () => {
        evtSource.close();
        if (!overlay.classList.contains("hidden")) {
            setTimeout(() => {
                overlay.classList.add("hidden");
                progressBar.classList.remove("indeterminate");
                progressBar.style.width = "0%";
                statusMsg.textContent = "Starting...";
                detail.textContent = "";
            }, 1000);
        }
    };
}

// ── Stats ───────────────────────────────────────────────────────────────

async function loadStats() {
    try {
        const stats = await API.getStats();
        const el = $('#db-stats');
        if (el) {
            const total = (stats.training_prompts || 0) + (stats.generated_prompts || 0);
            el.textContent = total > 0
                ? `${stats.training_prompts} training · ${stats.generated_prompts} generated`
                : '';
        }

        const statsEl = document.getElementById("header-stats");
        if (statsEl) {
            const parts = [];
            const totalDocs = (stats.training_prompts || 0) + (stats.generated_prompts || 0);
            if (totalDocs > 0) {
                parts.push(`<span class="stat-item" title="Training: ${stats.training_prompts || 0}, Generated: ${stats.generated_prompts || 0}">📄 ${totalDocs}</span>`);
            }
            if (stats.total_clusters > 0) {
                parts.push(`<span class="stat-item" title="${stats.total_clusters} theme clusters">🏷️ ${stats.total_clusters}</span>`);
            }
            statsEl.innerHTML = parts.join('<span class="stat-sep">·</span>');
        }
    } catch (err) {
        // Stats are non-critical, silently fail
        console.warn("Failed to load stats:", err);
    }
}

// ── Dataset Map ─────────────────────────────────────────────────────────

let _datasetMapData = null;
let _datasetMapFoldersRendered = 0;
const DATASET_MAP_PAGE_SIZE = 10;
let _datasetMapObserver = null;
let _globalClustersVisible = false;

async function openDatasetMap() {
    document.getElementById("dataset-map-modal").classList.remove("hidden");
    const container = document.getElementById("dataset-map-content");
    container.innerHTML = '<div class="loading-text">Loading dataset map...</div>';

    // Reset tab to first
    document.querySelectorAll('.dataset-map-tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.dataset-map-tab[data-map-tab="map-themes"]').classList.add('active');
    _globalClustersVisible = false;

    try {
        const response = await fetch("/api/dataset-map");
        _datasetMapData = await response.json();
        renderDatasetMap(_datasetMapData, container);
    } catch (err) {
        container.innerHTML = '<div class="error-text">Failed to load dataset map</div>';
    }
}

function closeDatasetMap() {
    document.getElementById("dataset-map-modal").classList.add("hidden");
    // Clean up observer
    if (_datasetMapObserver) {
        _datasetMapObserver.disconnect();
        _datasetMapObserver = null;
    }
}

function switchDatasetMapTab(btn) {
    const tabId = btn.dataset.mapTab;

    // Update tab buttons
    document.querySelectorAll('.dataset-map-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    // Update tab panes
    document.querySelectorAll('.dataset-map-pane').forEach(p => p.classList.remove('active'));
    const pane = document.getElementById(tabId);
    if (pane) pane.classList.add('active');

    // If switching to folders tab, set up infinite scroll
    if (tabId === 'map-folders') {
        setupFolderInfiniteScroll();
    }
}

function renderDatasetMap(data, container) {
    let html = '';

    // Update the Folders tab button with the folder count
    const foldersTab = document.querySelector('.dataset-map-tab[data-map-tab="map-folders"]');
    if (foldersTab) {
        const folderCount = (data.folders && data.folders.length) || 0;
        foldersTab.textContent = `Folders (${folderCount})`;
    }

    // ── Themes pane ──
    html += '<div id="map-themes" class="dataset-map-pane active">';
    if (data.cross_folder_themes && data.cross_folder_themes.length > 0) {
        html += '<div class="dataset-section">';
        html += '<div class="theme-list">';
        for (const theme of data.cross_folder_themes) {
            html += `<div class="theme-tag">
                <span class="theme-label">${escapeHtml(theme.label)}</span>
                <span class="theme-count">${theme.prompt_count}</span>
            </div>`;
        }
        html += '</div></div>';
    } else {
        html += '<div class="empty-dataset">No cross-folder themes yet. Run clustering first.</div>';
    }
    html += '</div>';

    // ── Folders pane ──
    html += '<div id="map-folders" class="dataset-map-pane">';
    if (data.folders && data.folders.length > 0) {
        html += '<div class="global-cluster-toggle">';
        html += '<button class="btn-toggle-all-clusters" onclick="toggleAllClusters(this)">Show all clusters</button>';
        html += '</div>';
        html += '<div class="dataset-section" id="folder-list-container"></div>';
        html += '<div id="folder-scroll-sentinel" class="scroll-sentinel" style="display:none;"><span class="spinner-small"></span> Loading more folders...</div>';
    } else {
        html += '<div class="empty-dataset">No folders available. Ingest some data first.</div>';
    }
    html += '</div>';

    if (!data.cross_folder_themes?.length && !data.folders?.length) {
        container.innerHTML = '<div class="empty-dataset">No data available. Ingest some data and run clustering first.</div>';
        return;
    }

    container.innerHTML = html;

    // Render initial batch of folders
    _datasetMapFoldersRendered = 0;
    renderMoreFolders();
}

function renderMoreFolders() {
    if (!_datasetMapData || !_datasetMapData.folders) return;

    const folders = _datasetMapData.folders;
    const listContainer = document.getElementById('folder-list-container');
    const sentinel = document.getElementById('folder-scroll-sentinel');
    if (!listContainer) return;

    const start = _datasetMapFoldersRendered;
    const end = Math.min(start + DATASET_MAP_PAGE_SIZE, folders.length);

    for (let i = start; i < end; i++) {
        const folder = folders[i];
        const card = document.createElement('div');
        card.className = 'folder-card';

        let headerHtml = `<div class="folder-header" onclick="toggleFolderClusters(this)">
            <span class="folder-name">${escapeHtml(folder.name)}</span>
            <span class="folder-meta">
                <span class="source-badge source-${folder.source_type}">${folder.source_type}</span>
                <span class="doc-count">${folder.total_prompts} docs</span>
            </span>
        </div>`;

        let themesHtml = '<div class="folder-themes">';
        if (folder.intra_themes && folder.intra_themes.length > 0) {
            themesHtml += '<div class="theme-list">';
            for (const theme of folder.intra_themes) {
                themesHtml += `<div class="theme-tag">
                    <span class="theme-label">${escapeHtml(theme.label)}</span>
                    <span class="theme-count">${theme.prompt_count}</span>
                </div>`;
            }
            themesHtml += '</div>';
        } else {
            themesHtml += '<p class="no-themes">No themes extracted yet</p>';
        }
        themesHtml += '</div>';

        card.innerHTML = headerHtml + themesHtml;

        // If global clusters are currently shown, make this new card visible too
        if (_globalClustersVisible) {
            card.querySelector('.folder-themes').classList.add('visible');
        }

        listContainer.appendChild(card);
    }

    _datasetMapFoldersRendered = end;

    // Show/hide sentinel
    if (sentinel) {
        sentinel.style.display = (end < folders.length) ? 'block' : 'none';
    }
}

function toggleFolderClusters(headerEl) {
    const card = headerEl.closest('.folder-card');
    const themes = card.querySelector('.folder-themes');
    if (!themes) return;
    themes.classList.toggle('visible');
}

function toggleAllClusters(btn) {
    _globalClustersVisible = !_globalClustersVisible;
    btn.classList.toggle('active', _globalClustersVisible);
    btn.textContent = _globalClustersVisible ? 'Hide all clusters' : 'Show all clusters';

    // Apply to all currently rendered folder cards
    document.querySelectorAll('#folder-list-container .folder-themes').forEach(el => {
        el.classList.toggle('visible', _globalClustersVisible);
    });
}

function setupFolderInfiniteScroll() {
    // Clean up previous observer
    if (_datasetMapObserver) {
        _datasetMapObserver.disconnect();
        _datasetMapObserver = null;
    }

    const sentinel = document.getElementById('folder-scroll-sentinel');
    if (!sentinel) return;

    // The scrollable container is the modal-body
    const scrollContainer = document.getElementById('dataset-map-content');
    if (!scrollContainer) return;

    _datasetMapObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                renderMoreFolders();
                // If all loaded, disconnect
                if (_datasetMapData && _datasetMapFoldersRendered >= _datasetMapData.folders.length) {
                    _datasetMapObserver.disconnect();
                    _datasetMapObserver = null;
                }
            }
        }
    }, {
        root: scrollContainer,
        rootMargin: '100px',
    });

    _datasetMapObserver.observe(sentinel);
}

// ── Utilities ───────────────────────────────────────────────────────────

function scrollToBottom() {
    const container = $('#messages');
    if (container) {
        requestAnimationFrame(() => {
            container.scrollTop = container.scrollHeight;
        });
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
