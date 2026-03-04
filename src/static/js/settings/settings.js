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

    // Load query parameter settings
    $('#setting-query-k-similar').value = settings.query_k_similar || '10';
    $('#setting-query-k-theme-intra').value = settings.query_k_theme_intra || '5';
    $('#setting-query-k-theme-cross').value = settings.query_k_theme_cross || '5';
    $('#setting-query-k-random').value = settings.query_k_random || '3';

    // Load clustering settings
    $('#setting-cluster-k-cross').value = settings.cluster_k_cross || '15';
    $('#setting-cluster-min-folder-size').value = settings.cluster_min_folder_size || '20';
    $('#setting-cluster-label-terms').value = settings.cluster_label_terms || '3';

    // Load adaptive K tier tables
    const defaultTraining = [{"max_prompts": 40, "k": 2}, {"max_prompts": 80, "k": 3}, {"max_prompts": 150, "k": 4}, {"max_prompts": null, "k": 5}];
    const defaultOutput = [{"max_prompts": 30, "k": 3}, {"max_prompts": 100, "k": 7}, {"max_prompts": 300, "k": 10}, {"max_prompts": null, "k": 15}];

    let trainingTiers = defaultTraining;
    let outputTiers = defaultOutput;
    try {
        if (settings.adaptive_k_training) trainingTiers = JSON.parse(settings.adaptive_k_training);
        if (settings.adaptive_k_output) outputTiers = JSON.parse(settings.adaptive_k_output);
    } catch (e) {
        console.warn('Failed to parse adaptive K tiers:', e);
    }
    renderAdaptiveKTable('adaptive-k-training', trainingTiers);
    renderAdaptiveKTable('adaptive-k-output', outputTiers);

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

async function saveQuerySettings() {
    const data = {
        query_k_similar: $('#setting-query-k-similar').value,
        query_k_theme_intra: $('#setting-query-k-theme-intra').value,
        query_k_theme_cross: $('#setting-query-k-theme-cross').value,
        query_k_random: $('#setting-query-k-random').value,
    };

    const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });

    if (response.ok) {
        alert('Query settings saved');
    }
}

async function saveClusterSettings() {
    const trainingTiers = collectAdaptiveKTiers('adaptive-k-training');
    const outputTiers = collectAdaptiveKTiers('adaptive-k-output');

    const data = {
        adaptive_k_training: JSON.stringify(trainingTiers),
        adaptive_k_output: JSON.stringify(outputTiers),
        cluster_k_cross: $('#setting-cluster-k-cross').value,
        cluster_min_folder_size: $('#setting-cluster-min-folder-size').value,
        cluster_label_terms: $('#setting-cluster-label-terms').value,
    };

    const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });

    if (response.ok) {
        alert('Clustering settings saved');
    }
}

// ── Adaptive K Tier Tables ───────────────────────────────────────────────

function renderAdaptiveKTable(containerId, tiers) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '<table class="adaptive-k-table"><thead><tr>'
        + '<th>Max Prompts</th><th>Clusters (k)</th><th></th>'
        + '</tr></thead><tbody>';

    for (let i = 0; i < tiers.length; i++) {
        const tier = tiers[i];
        const isCatchAll = tier.max_prompts === null;
        html += '<tr>';
        if (isCatchAll) {
            html += `<td><span class="adaptive-k-infinity">\u221e</span></td>`;
        } else {
            html += `<td><input type="number" class="adaptive-k-input" value="${tier.max_prompts}" min="1" step="1"></td>`;
        }
        html += `<td><input type="number" class="adaptive-k-input" value="${tier.k}" min="1" max="50" step="1"></td>`;
        if (isCatchAll) {
            html += '<td></td>';
        } else {
            html += `<td><button class="btn-remove-tier" onclick="removeAdaptiveKTier('${containerId}', ${i})" title="Remove tier">\u00d7</button></td>`;
        }
        html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

function addAdaptiveKTier(containerId) {
    const tiers = collectAdaptiveKTiers(containerId);
    // Insert before the catch-all (last entry)
    const catchAll = tiers[tiers.length - 1];
    const prevMax = tiers.length >= 2 ? tiers[tiers.length - 2].max_prompts : 0;
    const newMax = prevMax ? prevMax + 50 : 50;
    const newK = catchAll.k > 1 ? catchAll.k - 1 : 1;
    tiers.splice(tiers.length - 1, 0, { max_prompts: newMax, k: newK });
    renderAdaptiveKTable(containerId, tiers);
}

function removeAdaptiveKTier(containerId, index) {
    const tiers = collectAdaptiveKTiers(containerId);
    // Don't allow removing the catch-all
    if (tiers[index].max_prompts === null) return;
    tiers.splice(index, 1);
    renderAdaptiveKTable(containerId, tiers);
}

function collectAdaptiveKTiers(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return [];

    const rows = container.querySelectorAll('tbody tr');
    const tiers = [];

    rows.forEach(row => {
        const inputs = row.querySelectorAll('input.adaptive-k-input');
        const infinitySpan = row.querySelector('.adaptive-k-infinity');

        if (infinitySpan) {
            // Catch-all row: only k input
            const k = parseInt(inputs[0].value, 10) || 5;
            tiers.push({ max_prompts: null, k });
        } else if (inputs.length === 2) {
            const maxPrompts = parseInt(inputs[0].value, 10) || 50;
            const k = parseInt(inputs[1].value, 10) || 3;
            tiers.push({ max_prompts: maxPrompts, k });
        }
    });

    return tiers;
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
