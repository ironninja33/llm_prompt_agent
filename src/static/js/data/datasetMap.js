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
    _statsLoaded = false;

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

    if (tabId === 'map-stats') {
        _loadStatsPane();
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
        html += '<div class="cross-theme-list">';
        for (const theme of data.cross_folder_themes) {
            let contribHtml = '';
            if (theme.contributing_folders && theme.contributing_folders.length > 0) {
                const MAX_VISIBLE = 3;
                const allChips = theme.contributing_folders.map(f => {
                    const name = f.folder_path || '';
                    const category = name.includes('__') ? name.split('__')[0] : '';
                    const display = name.includes('__') ? name.split('__')[1] : name;
                    const srcType = f.source_type || 'training';
                    const srcLabel = srcType === 'training' ? 'TRN' : 'OUT';
                    const sourceBadge = `<span class="source-badge source-${srcType}">${srcLabel}</span>`;
                    const categoryPart = category ? `<span class="category-badge">${escapeHtml(category)}</span> ` : '';
                    return `<span class="contrib-folder">${sourceBadge} ${categoryPart}${escapeHtml(display)}</span>`;
                });
                const visibleChips = allChips.slice(0, MAX_VISIBLE).join('');
                const moreHtml = allChips.length > MAX_VISIBLE
                    ? `<span class="contrib-more">+${allChips.length - MAX_VISIBLE} more<span class="contrib-more-tooltip">${allChips.slice(MAX_VISIBLE).join('')}</span></span>`
                    : '';
                contribHtml = `<span class="contributing-folders">${visibleChips}${moreHtml}</span>`;
            }
            html += `<div class="cross-theme-row">
                <div class="cross-theme-summary">${escapeHtml(theme.label)}</div>
                <div class="cross-theme-meta">
                    <span class="theme-count">${theme.prompt_count} prompts</span>
                    ${contribHtml}
                </div>
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
        html += '<input type="text" id="dataset-map-search" class="dataset-map-search" placeholder="Filter folders..." oninput="filterDatasetMap(this.value)">';
        html += '</div>';
        html += '<div class="dataset-section" id="folder-list-container"></div>';
        html += '<div id="folder-scroll-sentinel" class="scroll-sentinel" style="display:none;"><span class="spinner-small"></span> Loading more folders...</div>';
    } else {
        html += '<div class="empty-dataset">No folders available. Ingest some data first.</div>';
    }
    html += '</div>';

    // ── Stats pane (loaded on demand) ──
    html += '<div id="map-stats" class="dataset-map-pane">';
    html += '<div class="loading-text">Loading stats...</div>';
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

    const fragment = document.createDocumentFragment();
    for (let i = start; i < end; i++) {
        fragment.appendChild(_createFolderCard(folders[i]));
    }
    listContainer.appendChild(fragment);

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

function _renderAllRemainingFolders() {
    if (!_datasetMapData || !_datasetMapData.folders) return;
    while (_datasetMapFoldersRendered < _datasetMapData.folders.length) {
        renderMoreFolders();
    }
}

let _filterDebounceTimer = null;

function filterDatasetMap(query) {
    clearTimeout(_filterDebounceTimer);
    _filterDebounceTimer = setTimeout(() => _applyDatasetMapFilter(query), 200);
}

function _applyDatasetMapFilter(query) {
    if (!_datasetMapData || !_datasetMapData.folders) return;

    const q = query.toLowerCase().trim();
    const listContainer = document.getElementById('folder-list-container');
    if (!listContainer) return;

    if (!q) {
        // No filter — reset to paginated rendering
        _datasetMapFoldersRendered = 0;
        listContainer.innerHTML = '';
        renderMoreFolders();
        setupFolderInfiniteScroll();
        return;
    }

    // Filter at the data level instead of rendering all then hiding
    teardownFolderScroll();
    const matching = _datasetMapData.folders.filter(folder => {
        const name = (folder.display_name || folder.name || '').toLowerCase();
        const category = (folder.category || '').toLowerCase();
        const summary = (folder.summary || '').toLowerCase();
        const themes = (folder.intra_themes || []).map(t => (t.label || '').toLowerCase()).join(' ');
        return name.includes(q) || category.includes(q) || summary.includes(q) || themes.includes(q);
    });

    // Re-render only matching folders using DocumentFragment
    const fragment = document.createDocumentFragment();
    for (const folder of matching) {
        fragment.appendChild(_createFolderCard(folder));
    }
    listContainer.innerHTML = '';
    listContainer.appendChild(fragment);
    _datasetMapFoldersRendered = _datasetMapData.folders.length; // Mark as fully rendered

    // Hide sentinel during filtered view
    const sentinel = document.getElementById('folder-scroll-sentinel');
    if (sentinel) sentinel.style.display = 'none';
}

function _createFolderCard(folder) {
    const card = document.createElement('div');
    card.className = 'folder-card';

    const displayName = folder.display_name || folder.name;
    const categoryBadge = folder.category
        ? `<span class="category-badge">${escapeHtml(folder.category)}</span>`
        : '';
    let headerHtml = `<div class="folder-header" onclick="toggleFolderClusters(this)">
        <div class="folder-name-group">
            <span class="folder-name">${escapeHtml(displayName)}</span>
            ${categoryBadge}
            <button class="folder-edit-btn" title="Rename folder" onclick="event.stopPropagation(); startFolderRename(this, '${escapeHtml(folder.name)}')"><svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M11.13 1.47a1.5 1.5 0 0 1 2.12 0l1.28 1.28a1.5 1.5 0 0 1 0 2.12L5.9 13.5a1 1 0 0 1-.42.26l-3.5 1.17a.5.5 0 0 1-.63-.63l1.17-3.5a1 1 0 0 1 .26-.42L11.13 1.47zM12.2 2.53l-8.1 8.1-.78 2.35 2.35-.78 8.1-8.1-1.57-1.57z"/></svg></button>
            ${folder.summary ? `<span class="folder-summary">${escapeHtml(folder.summary)}</span>` : ''}
        </div>
        <span class="folder-meta">
            <span class="source-badge source-${folder.source_type}">${folder.source_type}</span>
            <span class="doc-count">${folder.total_prompts} docs</span>
        </span>
    </div>`;

    let themesHtml = '<div class="folder-themes">';
    if (folder.intra_themes && folder.intra_themes.length > 0) {
        themesHtml += '<div class="intra-theme-list">';
        for (const theme of folder.intra_themes) {
            themesHtml += `<div class="intra-theme-row">
                <span class="theme-count">${theme.prompt_count}</span>
                <span class="theme-label">${escapeHtml(theme.label)}</span>
            </div>`;
        }
        themesHtml += '</div>';
    } else {
        themesHtml += '<p class="no-themes">No themes extracted yet</p>';
    }
    themesHtml += '</div>';

    card.innerHTML = headerHtml + themesHtml;

    if (_globalClustersVisible) {
        card.querySelector('.folder-themes').classList.add('visible');
    }

    return card;
}

// ── Stats pane ──────────────────────────────────────────────────────────

let _statsLoaded = false;

async function _loadStatsPane() {
    if (_statsLoaded) return;
    const pane = document.getElementById('map-stats');
    if (!pane) return;

    try {
        const res = await fetch('/api/metrics/stats');
        const stats = await res.json();
        _statsLoaded = true;
        pane.innerHTML = _renderStatsPane(stats);
    } catch (e) {
        pane.innerHTML = '<div class="error-text">Failed to load stats</div>';
    }
}

function _renderStatsPane(stats) {
    let html = '<div class="stats-pane-content">';

    // ── Generation overview ──
    html += '<div class="stats-section">';
    html += '<h3 class="stats-section-title">Generation Overview</h3>';
    html += '<div class="stats-grid">';
    html += `<div class="stats-card"><div class="stats-value">${stats.total_generations.toLocaleString()}</div><div class="stats-label">Total Generations</div></div>`;
    html += `<div class="stats-card"><div class="stats-value">${stats.session_count.toLocaleString()}</div><div class="stats-label">Sessions</div></div>`;
    html += `<div class="stats-card"><div class="stats-value">${stats.avg_session_generations}</div><div class="stats-label">Avg per Session</div></div>`;
    html += '</div></div>';

    // ── Deletions ──
    const del = stats.deletions_by_reason || {};
    const totalDeletions = Object.values(del).reduce((s, v) => s + v, 0);
    html += '<div class="stats-section">';
    html += '<h3 class="stats-section-title">Deletions</h3>';
    html += '<div class="stats-grid">';
    html += `<div class="stats-card"><div class="stats-value">${totalDeletions.toLocaleString()}</div><div class="stats-label">Total Deleted</div></div>`;
    for (const [reason, count] of Object.entries(del).sort((a, b) => b[1] - a[1])) {
        html += `<div class="stats-card"><div class="stats-value">${count.toLocaleString()}</div><div class="stats-label">${escapeHtml(reason)}</div></div>`;
    }
    html += '</div></div>';

    // ── Top lineage ──
    if (stats.top_lineage && stats.top_lineage.length > 0) {
        html += '<div class="stats-section">';
        html += '<h3 class="stats-section-title">Top Prompt Lineage (by regeneration depth)</h3>';
        html += '<div class="stats-lineage-list">';
        for (const item of stats.top_lineage) {
            html += `<div class="stats-lineage-row">
                <span class="stats-lineage-depth">${item.depth}x</span>
                <span class="stats-lineage-prompt">${escapeHtml(item.prompt)}</span>
            </div>`;
        }
        html += '</div></div>';
    }

    html += '</div>';
    return html;
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

// ── Folder rename ────────────────────────────────────────────────────────

function startFolderRename(btn, rawName) {
    const nameGroup = btn.closest('.folder-name-group');
    if (!nameGroup) return;

    // Save original HTML so we can restore on cancel
    const originalHtml = nameGroup.innerHTML;

    nameGroup.innerHTML = `
        <input type="text" class="folder-rename-input" value="${escapeHtml(rawName)}" />
        <button class="folder-rename-ok" onclick="executeFolderRename(this, '${escapeHtml(rawName)}')">OK</button>
        <button class="folder-rename-cancel" onclick="cancelFolderRename(this)">Cancel</button>
    `;
    nameGroup._originalHtml = originalHtml;

    const input = nameGroup.querySelector('.folder-rename-input');
    input.focus();
    input.select();
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            executeFolderRename(input, rawName);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            cancelFolderRename(input);
        }
    });
}

function cancelFolderRename(el) {
    const nameGroup = el.closest('.folder-name-group');
    if (nameGroup && nameGroup._originalHtml) {
        nameGroup.innerHTML = nameGroup._originalHtml;
        delete nameGroup._originalHtml;
    }
}

async function executeFolderRename(el, oldName) {
    const nameGroup = el.closest('.folder-name-group');
    if (!nameGroup) return;

    const input = nameGroup.querySelector('.folder-rename-input');
    const newName = (input ? input.value : '').trim();

    if (!newName || newName === oldName) {
        cancelFolderRename(el);
        return;
    }

    // Disable controls while request is in-flight
    const okBtn = nameGroup.querySelector('.folder-rename-ok');
    const cancelBtn = nameGroup.querySelector('.folder-rename-cancel');
    if (input) input.disabled = true;
    if (okBtn) okBtn.disabled = true;
    if (cancelBtn) cancelBtn.disabled = true;

    try {
        const resp = await fetch('/api/folder/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ old_name: oldName, new_name: newName }),
        });
        const result = await resp.json();
        if (result.ok) {
            // Refresh the entire dataset map to reflect the change
            const container = document.getElementById("dataset-map-content");
            const response = await fetch("/api/dataset-map");
            _datasetMapData = await response.json();
            _datasetMapFoldersRendered = 0;
            renderDatasetMap(_datasetMapData, container);
            // Re-activate the folders tab
            const foldersTab = document.querySelector('.dataset-map-tab[data-map-tab="map-folders"]');
            if (foldersTab) switchDatasetMapTab(foldersTab);
        } else {
            alert('Rename failed: ' + (result.error || 'Unknown error'));
            cancelFolderRename(el);
        }
    } catch (err) {
        alert('Rename failed: ' + err.message);
        cancelFolderRename(el);
    }
}
