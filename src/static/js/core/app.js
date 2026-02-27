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
    if (typeof loadChats === 'function') await loadChats();
    loadStats();
    monitorIngestion();
});

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
