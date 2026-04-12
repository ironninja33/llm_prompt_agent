/**
 * Browser navigation — breadcrumb and path management.
 *
 * Depends on: api.js, app.js, browserState.js, browserGrid.js
 */

/**
 * Render clickable breadcrumb from path segments.
 */
function renderBreadcrumb(breadcrumb, recursiveImageCount) {
    const el = $('#browser-breadcrumb');
    if (!el) return;
    el.innerHTML = '';

    // Left group: path segments
    const pathGroup = document.createElement('div');
    pathGroup.className = 'breadcrumb-path';

    breadcrumb.forEach((seg, idx) => {
        if (idx > 0) {
            const sep = document.createElement('span');
            sep.className = 'breadcrumb-sep';
            sep.textContent = '›';
            pathGroup.appendChild(sep);
        }

        const link = document.createElement('span');
        link.className = 'breadcrumb-segment';

        if (seg.category) {
            const badge = document.createElement('span');
            badge.className = 'category-badge';
            badge.textContent = seg.category;
            link.appendChild(badge);
            link.appendChild(document.createTextNode(' '));
            const nameSpan = document.createElement('span');
            nameSpan.textContent = seg.display_name || seg.name;
            link.appendChild(nameSpan);
        } else {
            link.textContent = seg.display_name || seg.name;
        }

        if (idx === breadcrumb.length - 1) {
            link.classList.add('active');
        } else {
            link.onclick = () => navigateToPath(seg.name, null, seg.path);
        }

        pathGroup.appendChild(link);
    });

    el.appendChild(pathGroup);

    // Right group: image count + action buttons
    const actionsGroup = document.createElement('div');
    actionsGroup.className = 'breadcrumb-actions';

    if (recursiveImageCount > 0) {
        const count = document.createElement('span');
        count.className = 'breadcrumb-image-count';
        count.textContent = `${recursiveImageCount} image${recursiveImageCount !== 1 ? 's' : ''}`;
        actionsGroup.appendChild(count);
    }

    // Parameter display dropdown
    if (typeof initBrowserParamDisplay === 'function') {
        const paramContainer = document.createElement('div');
        paramContainer.style.display = 'inline-block';
        actionsGroup.appendChild(paramContainer);
        initBrowserParamDisplay(paramContainer);
    }

    // Auto-refresh toggle
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'cbdd-btn browser-refresh-toggle' + (BrowserState.autoRefresh ? ' active' : '');
    refreshBtn.type = 'button';
    refreshBtn.title = 'Auto-refresh on new images';
    refreshBtn.innerHTML = '\u27f3';
    const badge = document.createElement('span');
    badge.className = 'refresh-badge hidden';
    refreshBtn.appendChild(badge);
    actionsGroup.appendChild(refreshBtn);

    refreshBtn.addEventListener('click', () => {
        BrowserState.autoRefresh = !BrowserState.autoRefresh;
        refreshBtn.classList.toggle('active', BrowserState.autoRefresh);

        if (BrowserState.autoRefresh && BrowserState.pendingNewCount > 0) {
            BrowserState.pendingNewCount = 0;
            BrowserState.offset = 0;
            loadBrowserContents();
        }
        if (typeof _updateRefreshBadge === 'function') _updateRefreshBadge();
        API.updateSettings({ browser_auto_refresh: String(BrowserState.autoRefresh) }).catch(() => {});
    });

    // Sort dropdown
    const sortContainer = document.createElement('div');
    sortContainer.style.display = 'inline-block';
    actionsGroup.appendChild(sortContainer);
    _createSortDropdown(sortContainer);

    // Thumbnail size dropdown
    const sizeContainer = document.createElement('div');
    sizeContainer.style.display = 'inline-block';
    actionsGroup.appendChild(sizeContainer);
    _createSizeDropdown(sizeContainer);

    el.appendChild(actionsGroup);
}

const _SORT_OPTIONS = [
    { key: 'date', label: 'Date' },
    { key: 'seed', label: 'Seed' },
];

const _SIZE_OPTIONS = [
    { key: 'small', label: 'Small' },
    { key: 'medium', label: 'Medium' },
    { key: 'large', label: 'Large' },
];

function _createSizeDropdown(container) {
    const current = BrowserState.thumbnailSize || 'medium';

    const wrapper = document.createElement('div');
    wrapper.className = 'cbdd-wrapper';

    const btn = document.createElement('button');
    btn.className = 'cbdd-btn';
    btn.type = 'button';
    const currentLabel = _SIZE_OPTIONS.find(o => o.key === current)?.label || 'Medium';
    btn.innerHTML = `<span class="cbdd-label">${currentLabel}</span><span class="cbdd-arrow">▾</span>`;

    const panel = document.createElement('div');
    panel.className = 'cbdd-panel hidden';

    const items = document.createElement('div');
    items.className = 'cbdd-items';
    for (const opt of _SIZE_OPTIONS) {
        const row = document.createElement('div');
        row.className = 'cbdd-item cbdd-item-select' + (opt.key === current ? ' cbdd-item-active' : '');
        row.dataset.key = opt.key;
        row.textContent = opt.label;
        items.appendChild(row);
    }
    panel.appendChild(items);
    wrapper.appendChild(btn);
    wrapper.appendChild(panel);
    container.appendChild(wrapper);

    // Toggle panel
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.toggle('hidden');
    });

    // Click outside to close
    function onDocClick(e) {
        if (!wrapper.contains(e.target)) panel.classList.add('hidden');
    }
    document.addEventListener('click', onDocClick);

    // Item click: apply, close, save
    items.addEventListener('click', (e) => {
        const row = e.target.closest('.cbdd-item-select');
        if (!row) return;
        const size = row.dataset.key;

        // Update active state
        items.querySelectorAll('.cbdd-item-select').forEach(r => r.classList.remove('cbdd-item-active'));
        row.classList.add('cbdd-item-active');

        // Update button label
        btn.querySelector('.cbdd-label').textContent = row.textContent;

        // Close panel
        panel.classList.add('hidden');

        // Apply
        BrowserState.thumbnailSize = size;
        window.dispatchEvent(new CustomEvent('thumbnail-size-changed', { detail: { browser: size } }));

        // Persist to DB
        API.updateSettings({ thumbnail_size_browser: size }).catch(() => {});
    });
}

function _createSortDropdown(container) {
    const current = BrowserState.sortMode || 'date';
    const currentDir = BrowserState.sortDirection || 'desc';

    const wrapper = document.createElement('div');
    wrapper.className = 'cbdd-wrapper';

    const btn = document.createElement('button');
    btn.className = 'cbdd-btn';
    btn.type = 'button';
    const currentLabel = _SORT_OPTIONS.find(o => o.key === current)?.label || 'Date';
    const arrow = currentDir === 'asc' ? ' \u2191' : ' \u2193';
    btn.innerHTML = `<span class="cbdd-label">${currentLabel}${arrow}</span><span class="cbdd-arrow">\u25be</span>`;

    const panel = document.createElement('div');
    panel.className = 'cbdd-panel hidden';

    const items = document.createElement('div');
    items.className = 'cbdd-items';
    for (const opt of _SORT_OPTIONS) {
        const row = document.createElement('div');
        row.className = 'cbdd-item cbdd-item-select' + (opt.key === current ? ' cbdd-item-active' : '');
        row.dataset.key = opt.key;
        row.textContent = opt.label;
        items.appendChild(row);
    }
    panel.appendChild(items);
    wrapper.appendChild(btn);
    wrapper.appendChild(panel);
    container.appendChild(wrapper);

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.toggle('hidden');
    });

    document.addEventListener('click', (e) => {
        if (!wrapper.contains(e.target)) panel.classList.add('hidden');
    });

    items.addEventListener('click', (e) => {
        const row = e.target.closest('.cbdd-item-select');
        if (!row) return;
        const mode = row.dataset.key;

        if (mode === BrowserState.sortMode) {
            // Same option: toggle direction
            BrowserState.sortDirection = BrowserState.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            // New mode: apply default direction
            BrowserState.sortMode = mode;
            BrowserState.sortDirection = mode === 'date' ? 'desc' : 'asc';
        }

        items.querySelectorAll('.cbdd-item-select').forEach(r => r.classList.remove('cbdd-item-active'));
        row.classList.add('cbdd-item-active');

        const dirArrow = BrowserState.sortDirection === 'asc' ? ' \u2191' : ' \u2193';
        const label = _SORT_OPTIONS.find(o => o.key === BrowserState.sortMode)?.label || 'Date';
        btn.querySelector('.cbdd-label').textContent = label + dirArrow;
        panel.classList.add('hidden');

        BrowserState.offset = 0;
        API.updateSettings({
            browser_sort_mode: BrowserState.sortMode,
            browser_sort_direction: BrowserState.sortDirection,
        }).catch(() => {});
        loadBrowserContents();
    });
}

/**
 * Navigate to a path in the browser.
 * @param {string} name - Display name (not used for navigation)
 * @param {string|null} absPath - Absolute path (unused, kept for compatibility)
 * @param {string|null} virtualPath - Virtual path for API
 */
async function navigateToPath(name, absPath, virtualPath) {
    // Build virtual path if navigating from a directory listing
    if (virtualPath === undefined || virtualPath === null) {
        // Clicking a directory card: append name to current path
        virtualPath = BrowserState.currentPath
            ? `${BrowserState.currentPath}/${name}`
            : name;
    }

    BrowserState.currentPath = virtualPath;
    BrowserState.offset = 0;
    BrowserState.hasMore = false;
    BrowserState.isSearchActive = false;
    BrowserState.searchQuery = '';
    BrowserState.pollTimestamp = Date.now() / 1000;
    BrowserState.lastCompletionSeq = null;
    BrowserState.newestCreatedAt = null;
    BrowserState.pendingNewCount = 0;

    // Persist path in URL hash (F5) and sessionStorage (cross-page nav)
    window.location.hash = virtualPath ? encodeURIComponent(virtualPath) : '';
    try { sessionStorage.setItem('browserPath', virtualPath || ''); } catch (_) {}

    // Clear search input
    const searchInput = $('#browser-search-input');
    if (searchInput) searchInput.value = '';

    await loadBrowserContents();
}

/**
 * Load browser contents for the current path.
 */
async function loadBrowserContents() {
    if (BrowserState.isLoading) return;
    BrowserState.isLoading = true;

    const loadingEl = $('#browser-loading');
    if (loadingEl) loadingEl.classList.remove('hidden');

    try {
        const result = await API.browserListing(
            BrowserState.currentPath,
            BrowserState.offset,
            BrowserState.limit,
            BrowserState.sortMode,
            BrowserState.sortDirection
        );

        // Render breadcrumb
        if (result.breadcrumb) {
            renderBreadcrumb(result.breadcrumb, result.recursive_image_count || 0);
        }

        // Render grid
        if (BrowserState.offset === 0) {
            renderBrowserGrid(result.directories, result.images);
        } else {
            appendBrowserItems(result.images);
        }

        BrowserState.hasMore = result.has_more || false;

        // Track newest created_at for incremental polling
        if (result.images) {
            for (const img of result.images) {
                if (img.created_at && (!BrowserState.newestCreatedAt || img.created_at > BrowserState.newestCreatedAt)) {
                    BrowserState.newestCreatedAt = img.created_at;
                }
            }
        }

    } catch (err) {
        console.error('Failed to load browser contents:', err);
    } finally {
        BrowserState.isLoading = false;
        if (loadingEl) loadingEl.classList.add('hidden');
    }
}

/**
 * Load next page (infinite scroll).
 */
async function loadNextPage() {
    if (!BrowserState.hasMore || BrowserState.isLoading) return;

    BrowserState.offset += BrowserState.limit;

    if (BrowserState.isSearchActive) {
        await performSearchPage();
    } else {
        await loadBrowserContents();
    }
}

/**
 * Load next page of search results.
 */
async function performSearchPage() {
    if (BrowserState.isLoading) return;
    BrowserState.isLoading = true;

    const loadingEl = $('#browser-loading');
    if (loadingEl) loadingEl.classList.remove('hidden');

    try {
        const result = await API.browserSearch(
            BrowserState.searchQuery,
            BrowserState.searchMode,
            BrowserState.offset,
            BrowserState.limit
        );

        appendBrowserItems(result.images);
        BrowserState.hasMore = result.has_more || false;

    } catch (err) {
        console.error('Failed to load search page:', err);
    } finally {
        BrowserState.isLoading = false;
        if (loadingEl) loadingEl.classList.add('hidden');
    }
}
