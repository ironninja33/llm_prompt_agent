/**
 * Browser search — keyword and embedding search.
 *
 * Depends on: api.js, app.js, browserState.js, browserGrid.js, browserNav.js
 */

let _searchDebounceTimer = null;

/**
 * Perform a search and render results.
 */
async function performSearch(query, mode) {
    if (!query || !query.trim()) {
        clearSearch();
        return;
    }

    BrowserState.isSearchActive = true;
    BrowserState.searchQuery = query.trim();
    BrowserState.searchMode = mode || BrowserState.searchMode;
    BrowserState.offset = 0;
    BrowserState.hasMore = false;

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

        // Render search breadcrumb
        renderBreadcrumb([
            { name: 'Root', path: '' },
            { name: `Search: "${BrowserState.searchQuery}"`, path: '' },
        ]);

        renderBrowserGrid([], result.images);
        BrowserState.hasMore = result.has_more || false;

    } catch (err) {
        console.error('Search failed:', err);
    } finally {
        BrowserState.isLoading = false;
        if (loadingEl) loadingEl.classList.add('hidden');
    }
}

/**
 * Clear search and return to directory browsing.
 */
function clearSearch() {
    BrowserState.isSearchActive = false;
    BrowserState.searchQuery = '';
    BrowserState.offset = 0;

    const searchInput = $('#browser-search-input');
    if (searchInput) searchInput.value = '';

    loadBrowserContents();
}

/**
 * Toggle between keyword and embedding search mode.
 */
function toggleSearchMode() {
    if (BrowserState.searchMode === 'keyword') {
        BrowserState.searchMode = 'embedding';
    } else {
        BrowserState.searchMode = 'keyword';
    }

    const modeBtn = $('#browser-search-mode');
    if (modeBtn) {
        modeBtn.textContent = BrowserState.searchMode === 'keyword' ? 'KW' : 'EM';
        modeBtn.classList.toggle('active-embedding', BrowserState.searchMode === 'embedding');
    }

    // Save preference
    API.updateSettings({ search_mode: BrowserState.searchMode }).catch(() => {});

    // Re-search if there's an active query
    if (BrowserState.searchQuery) {
        performSearch(BrowserState.searchQuery, BrowserState.searchMode);
    }
}

/**
 * Setup search input handlers.
 */
function initBrowserSearch() {
    const input = $('#browser-search-input');
    if (!input) return;

    // Debounced input
    input.addEventListener('input', () => {
        clearTimeout(_searchDebounceTimer);
        const query = input.value.trim();
        if (!query) {
            clearSearch();
            return;
        }
        _searchDebounceTimer = setTimeout(() => {
            performSearch(query, BrowserState.searchMode);
        }, 400);
    });

    // Enter key
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(_searchDebounceTimer);
            const query = input.value.trim();
            if (query) {
                performSearch(query, BrowserState.searchMode);
            }
        }
        if (e.key === 'Escape') {
            clearSearch();
            input.blur();
        }
    });
}
