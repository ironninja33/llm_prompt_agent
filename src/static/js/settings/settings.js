/**
 * LLM Prompt Agent — Settings modal logic.
 *
 * Depends on: api.js, app.js ($, $$, escapeHtml, monitorIngestion)
 */

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

    // Lazy-load ComfyUI settings when that tab is activated
    if (tabId === 'tab-comfyui' && typeof loadComfyUISettings === 'function') {
        loadComfyUISettings();
    }
    // Lazy-load Look & Feel settings
    if (tabId === 'tab-look' && typeof loadLookFeelSettings === 'function') {
        loadLookFeelSettings();
    }
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

// ── Look & Feel Settings ─────────────────────────────────────────────────

async function loadLookFeelSettings() {
    try {
        const settings = await API.getSettings();
        const thumbChat = $('#setting-thumb-chat');
        const thumbBrowser = $('#setting-thumb-browser');
        if (thumbChat) thumbChat.value = settings.thumbnail_size_chat || 'large';
        if (thumbBrowser) thumbBrowser.value = settings.thumbnail_size_browser || 'medium';
    } catch (err) {
        console.warn('Failed to load look & feel settings:', err);
    }
}

async function saveLookFeelSettings() {
    const thumbChat = ($('#setting-thumb-chat') || {}).value || 'large';
    const thumbBrowser = ($('#setting-thumb-browser') || {}).value || 'medium';

    await API.updateSettings({
        thumbnail_size_chat: thumbChat,
        thumbnail_size_browser: thumbBrowser,
    });

    // Dispatch event so active pages can update their grid CSS
    window.dispatchEvent(new CustomEvent('thumbnail-size-changed', {
        detail: { chat: thumbChat, browser: thumbBrowser },
    }));

    alert('Look & Feel settings saved');
}
