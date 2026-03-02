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
    // Each init task is independent — one failure must not block the others.
    // loadChats() is critical for page render so it gets retry with backoff.
    if (typeof loadChats === 'function') {
        let loaded = false;
        for (let attempt = 1; attempt <= 3 && !loaded; attempt++) {
            try {
                await loadChats();
                loaded = true;
            } catch (err) {
                console.warn(`loadChats attempt ${attempt}/3 failed:`, err);
                if (attempt < 3) await new Promise(r => setTimeout(r, 1000 * attempt));
            }
        }
    }

    try { loadStats(); } catch (e) { console.warn('loadStats failed:', e); }
    try { startComfyUIPolling(); } catch (e) { console.warn('startComfyUIPolling failed:', e); }
    try { monitorIngestion(); } catch (e) { console.warn('monitorIngestion failed:', e); }

    // Handle browser-to-chat refine URL params: ?refine=<prompt>&attach=<jobId/imgId>
    try { _handleRefineParams(); } catch (e) { console.warn('_handleRefineParams failed:', e); }
});

function _handleRefineParams() {
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
    if (typeof setRefineContext === 'function') {
        const chatId = params.get('chat');
        // Select existing chat or create a new one
        const chatReady = chatId && typeof selectChat === 'function'
            ? selectChat(chatId)
            : (typeof createNewChat === 'function' ? createNewChat() : Promise.resolve());

        chatReady.then(() => {
            setRefineContext(decodeURIComponent(refinePrompt));

            // Persist source image settings as the chat's generation defaults
            if (refineSettings && typeof setRefineGenerationSettings === 'function') {
                setRefineGenerationSettings(refineSettings);
            }

            // Handle optional attachment
            const attach = params.get('attach');
            if (attach && typeof addAttachmentFromBrowser === 'function') {
                addAttachmentFromBrowser(attach);
            }
        });
    }
}

// ── Tab Visibility Recovery ──────────────────────────────────────────────
// Browsers throttle/drop SSE and fetch streams for background tabs.
// When the user switches back, refresh any stale state.
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;

    if (typeof refreshStaleGenerationPollers === 'function') {
        refreshStaleGenerationPollers();
    }
    if (typeof refreshStaleChatStream === 'function') {
        refreshStaleChatStream();
    }
});

// ── ComfyUI Status ──────────────────────────────────────────────────────

let _comfyStatusInterval = null;

async function pollComfyUIStatus() {
    const container = $('#comfyui-status');
    if (!container) return;
    const dot = container.querySelector('.comfyui-dot');
    const queueEl = container.querySelector('.comfyui-queue');
    if (!dot || !queueEl) return;

    try {
        const status = await API.getComfyUIStatus();
        if (status.ok) {
            container.classList.add('online');
            container.classList.remove('offline');
            if (status.queue_size > 0) {
                queueEl.textContent = status.queue_size;
                queueEl.classList.remove('hidden');
                container.title = `ComfyUI: ${status.queue_size} job${status.queue_size !== 1 ? 's' : ''} queued`;
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
    } catch (_) {
        container.classList.add('offline');
        container.classList.remove('online');
        queueEl.classList.add('hidden');
        container.title = 'ComfyUI: offline';
    }
}

function startComfyUIPolling() {
    pollComfyUIStatus();
    if (!_comfyStatusInterval) {
        _comfyStatusInterval = setInterval(pollComfyUIStatus, 10000);
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
        // Check for per-folder k overrides before triggering
        let resetOverrides = false;
        try {
            const checkRes = await fetch("/api/settings/custom-cluster-k");
            const checkData = await checkRes.json();
            if (checkData.has_overrides) {
                const choice = confirm(
                    "Some folders have custom cluster sizes.\n\n" +
                    "OK = Keep custom sizes for those folders\n" +
                    "Cancel = Reset all to the global default"
                );
                if (!choice) {
                    // User chose to reset
                    await fetch("/api/settings/reset-custom-cluster-k", { method: "POST" });
                }
            }
        } catch (e) {
            // Non-critical — proceed with clustering anyway
            console.warn("Failed to check custom cluster k overrides:", e);
        }

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
