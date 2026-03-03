/**
 * LLM Prompt Agent — Dataset map modal.
 *
 * Depends on: app.js (escapeHtml)
 */

// ── State ────────────────────────────────────────────────────────────────
let _datasetMapData = null;
let _datasetMapFoldersRendered = 0;
const DATASET_MAP_PAGE_SIZE = 10;
let _folderScrollBound = null;   // bound scroll handler ref for cleanup
let _globalClustersVisible = false;

// ── Dataset Map Modal ───────────────────────────────────────────────────

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
    teardownFolderScroll();
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

    if (tabId === 'map-folders') {
        setupFolderInfiniteScroll();
    } else {
        teardownFolderScroll();
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

function teardownFolderScroll() {
    if (_folderScrollBound) {
        const sc = document.getElementById('dataset-map-content');
        if (sc) sc.removeEventListener('scroll', _folderScrollBound);
        _folderScrollBound = null;
    }
}

function setupFolderInfiniteScroll() {
    teardownFolderScroll();

    const scrollContainer = document.getElementById('dataset-map-content');
    if (!scrollContainer) return;
    if (!_datasetMapData || !_datasetMapData.folders) return;

    let ticking = false;
    _folderScrollBound = () => {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(() => {
            ticking = false;
            if (!_datasetMapData || _datasetMapFoldersRendered >= _datasetMapData.folders.length) {
                teardownFolderScroll();
                return;
            }
            const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
            if (scrollTop + clientHeight >= scrollHeight - 200) {
                renderMoreFolders();
            }
        });
    };

    scrollContainer.addEventListener('scroll', _folderScrollBound, { passive: true });

    // Fill viewport if content is too short to scroll
    function fillIfNeeded() {
        if (!_datasetMapData || _datasetMapFoldersRendered >= _datasetMapData.folders.length) return;
        if (scrollContainer.scrollHeight <= scrollContainer.clientHeight) {
            renderMoreFolders();
            requestAnimationFrame(fillIfNeeded);
        }
    }
    requestAnimationFrame(fillIfNeeded);
}
