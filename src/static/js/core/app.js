/**
 * LLM Prompt Agent — Main application entry point.
 *
 * Provides shared state, DOM helpers, and initialization.
 * Depends on: api.js, sse.js, markdown.js (loaded before this file)
 * Loaded before: sidebar.js, chatPane.js, settings.js, datasetMap.js
 */

// ── State ───────────────────────────────────────────────────────────────
let currentChatId = null;
let isStreaming = false;
let chats = [];

// ── DOM Helpers ─────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Initialization ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Detect refine params early — skip auto-select to avoid racing with refine handler
    const hasRefineParams = new URLSearchParams(window.location.search).has('refine');

    // Each init task is independent — one failure must not block the others.
    // loadChats() is critical for page render so it gets retry with backoff.
    if (typeof loadChats === 'function') {
        let loaded = false;
        for (let attempt = 1; attempt <= 3 && !loaded; attempt++) {
            try {
                await loadChats(hasRefineParams);
                loaded = true;
            } catch (err) {
                console.warn(`loadChats attempt ${attempt}/3 failed:`, err);
                if (attempt < 3) await new Promise(r => setTimeout(r, 1000 * attempt));
            }
        }
    }

    try { loadStats(); } catch (e) { console.warn('loadStats failed:', e); }

    // Register poll modules and start unified polling
    PollManager.register('comfyui', (data) => {
        updateComfyUIStatusIndicator(data.ok, data.queue_size);
    });
    PollManager.register('ingestion', (data) => {
        if (data.phase !== 'idle' && !data.complete) {
            showIngestionOverlay(data);
        } else if (data.complete && data.phase !== 'idle') {
            hideIngestionOverlay();
            loadStats();
        }
    });
    PollManager.register('clustering', (data) => {
        if (data.phase !== 'idle' && !data.complete) {
            showClusteringOverlay(data);
        } else if (data.complete && data.phase !== 'idle') {
            hideClusteringOverlay(data);
            loadStats();
        }
    });
    PollManager.start();

    // Handle browser-to-chat refine URL params: ?refine=<prompt>&attach=<jobId/imgId>
    if (hasRefineParams) {
        try { await _handleRefineParams(); } catch (e) { console.warn('_handleRefineParams failed:', e); }
    }
});

async function _handleRefineParams() {
    const params = new URLSearchParams(window.location.search);
    const refinePrompt = params.get('refine');
    if (!refinePrompt) return;

    // Clear URL params without reload
    history.replaceState(null, '', window.location.pathname);

    // Read source image settings stashed by the browser page
    let refineSettings = null;
    try {
        const raw = sessionStorage.getItem('refineSettings');
        if (raw) {
            refineSettings = JSON.parse(raw);
            sessionStorage.removeItem('refineSettings');
        }
    } catch (e) { /* ignore */ }

    // Set refine context if the function exists (chat page only)
    if (typeof setRefineContext !== 'function') return;

    const chatId = params.get('chat');
    // Select existing chat or create a new one
    if (chatId && typeof selectChat === 'function') {
        await selectChat(chatId);
    } else if (typeof createNewChat === 'function') {
        await createNewChat();
    }

    // params.get() already URL-decodes the value — no double decode
    setRefineContext(refinePrompt);

    // Persist source image settings as the chat's generation defaults
    if (refineSettings && typeof setRefineGenerationSettings === 'function') {
        setRefineGenerationSettings(refineSettings);
    }

    // Handle optional attachment
    const attach = params.get('attach');
    if (attach && typeof addAttachmentFromBrowser === 'function') {
        addAttachmentFromBrowser(attach);
    }
}

// ── Tab Visibility Recovery ──────────────────────────────────────────────
// Browsers throttle/drop SSE and fetch streams for background tabs.
// When the user switches back, refresh any stale state.
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;

    if (typeof refreshStaleChatStream === 'function') {
        refreshStaleChatStream();
    }
});

// ── ComfyUI Status ──────────────────────────────────────────────────────

function updateComfyUIStatusIndicator(ok, queueSize) {
    const container = $('#comfyui-status');
    if (!container) return;
    const queueEl = container.querySelector('.comfyui-queue');
    if (!queueEl) return;

    if (ok) {
        container.classList.add('online');
        container.classList.remove('offline');
        if (queueSize > 0) {
            queueEl.textContent = queueSize;
            queueEl.classList.remove('hidden');
            container.title = `ComfyUI: ${queueSize} job${queueSize !== 1 ? 's' : ''} queued`;
        } else {
            queueEl.classList.add('hidden');
            container.title = 'ComfyUI: connected, queue empty';
        }
    } else {
        container.classList.add('offline');
        container.classList.remove('online');
        queueEl.classList.add('hidden');
        container.title = 'ComfyUI: offline';
    }
}

// ── Stats ───────────────────────────────────────────────────────────────

async function loadStats() {
    try {
        const stats = await API.getStats();

        const statsEl = document.getElementById("header-stats");
        if (statsEl) {
            const parts = [];
            const training = stats.training_prompts || 0;
            const generated = stats.generated_prompts || 0;
            const totalDocs = training + generated;
            if (totalDocs > 0) {
                parts.push(`<span class="stat-item" title="Training images: ${training}\nGenerated images: ${generated}">🖼️ ${totalDocs}</span>`);
            }
            const crossClusters = stats.cross_folder_clusters || 0;
            const intraClusters = stats.intra_folder_clusters || 0;
            const totalClusters = crossClusters + intraClusters;
            if (totalClusters > 0) {
                parts.push(`<span class="stat-item" title="Cross-cutting clusters: ${crossClusters}\nIntra-folder clusters: ${intraClusters}">🏷️ ${totalClusters}</span>`);
            }
            statsEl.innerHTML = parts.join('<span class="stat-sep">·</span>');
        }
    } catch (err) {
        // Stats are non-critical, silently fail
        console.warn("Failed to load stats:", err);
    }
}

// ── Ingestion Monitoring ────────────────────────────────────────────────

let _ingestionOverlayShown = false;

function showIngestionOverlay(data) {
    const overlay = $('#ingestion-overlay');
    const progressBar = $('#ingestion-progress-bar');
    const statusMsg = $('#ingestion-status-message');
    const detail = $('#ingestion-detail');
    if (!overlay) return;

    if (!_ingestionOverlayShown) {
        overlay.classList.remove('hidden');
        _ingestionOverlayShown = true;
    }

    if (statusMsg) statusMsg.textContent = data.message || 'Processing...';
    if (detail) detail.textContent = data.current_dir || '';

    if (progressBar) {
        if (data.phase === 'discovery') {
            progressBar.classList.add('indeterminate');
            progressBar.style.width = '';
        } else if (data.phase === 'embedding' && data.new_files > 0) {
            progressBar.classList.remove('indeterminate');
            const pct = Math.round((data.current / data.new_files) * 100);
            progressBar.style.width = pct + '%';
        }
    }
}

function hideIngestionOverlay() {
    const overlay = $('#ingestion-overlay');
    const progressBar = $('#ingestion-progress-bar');
    const statusMsg = $('#ingestion-status-message');
    const detail = $('#ingestion-detail');

    if (_ingestionOverlayShown && overlay) {
        if (statusMsg) statusMsg.textContent = 'Indexing complete';
        if (progressBar) {
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';
        }
        if (detail) detail.textContent = '';

        setTimeout(() => {
            overlay.classList.add('hidden');
            if (progressBar) progressBar.style.width = '0%';
        }, 1200);
        _ingestionOverlayShown = false;
    }
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

let _clusteringOverlayShown = false;

function showClusteringOverlay(data) {
    const overlay = document.getElementById("clustering-overlay");
    const progressBar = document.getElementById("clustering-progress-bar");
    const statusMsg = document.getElementById("clustering-status-message");
    const detail = document.getElementById("clustering-detail");
    if (!overlay) return;

    if (!_clusteringOverlayShown) {
        overlay.classList.remove("hidden");
        _clusteringOverlayShown = true;
    }

    if (statusMsg) statusMsg.textContent = data.message || "Processing...";

    if (progressBar) {
        if (data.total && data.total > 0) {
            progressBar.classList.remove("indeterminate");
            const pct = Math.round((data.current / data.total) * 100);
            progressBar.style.width = pct + "%";
        } else {
            progressBar.classList.add("indeterminate");
            progressBar.style.width = "";
        }
    }

    if (detail && data.phase) {
        const phaseLabels = {
            starting: "Initializing...",
            cross_folder: "Cross-folder clustering",
            intra_folder: "Intra-folder clustering",
            summarizing: "Generating LLM summaries",
        };
        detail.textContent = phaseLabels[data.phase] || data.phase;
    }
}

function hideClusteringOverlay(data) {
    const overlay = document.getElementById("clustering-overlay");
    const progressBar = document.getElementById("clustering-progress-bar");
    const statusMsg = document.getElementById("clustering-status-message");
    const detail = document.getElementById("clustering-detail");
    if (!overlay) return;

    if (_clusteringOverlayShown || !overlay.classList.contains("hidden")) {
        const isError = data && data.phase === "error";
        if (statusMsg) statusMsg.textContent = isError
            ? (data.message || "Clustering failed")
            : (data?.message || "Clustering complete");
        if (progressBar && !isError) {
            progressBar.classList.remove("indeterminate");
            progressBar.style.width = "100%";
        }
        if (detail) detail.textContent = "";

        setTimeout(() => {
            overlay.classList.add("hidden");
            if (progressBar) {
                progressBar.classList.remove("indeterminate");
                progressBar.style.width = "0%";
            }
            if (statusMsg) statusMsg.textContent = "Starting...";
            if (detail) detail.textContent = "";
        }, isError ? 3000 : 2000);
        _clusteringOverlayShown = false;
    }
}

function monitorClustering() {
    // Legacy — clustering is now monitored via PollManager.
    // This function is kept so triggerClustering() doesn't break.
    // The PollManager 'clustering' handler already handles overlay updates.
}
