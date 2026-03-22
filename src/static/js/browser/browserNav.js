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

    el.appendChild(actionsGroup);
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
            BrowserState.limit
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

        // Show "Suggest subfolders" button if applicable
        if (typeof maybeShowReorgButton === 'function') {
            maybeShowReorgButton(
                BrowserState.currentPath,
                result.total_image_count || 0,
                (result.directories && result.directories.length > 0)
            );
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
