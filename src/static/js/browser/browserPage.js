/**
 * Browser page entry point — initialisation, infinite scroll, events.
 *
 * Depends on: api.js, app.js, browserState.js, browserGrid.js,
 *             browserNav.js, browserSearch.js, browserPolling.js
 */

document.addEventListener('DOMContentLoaded', async () => {
    // Load user preferences
    try {
        const settings = await API.getSettings();
        if (settings.search_mode) {
            BrowserState.searchMode = settings.search_mode;
        }
        if (settings.thumbnail_size_browser) {
            BrowserState.thumbnailSize = settings.thumbnail_size_browser;
        }
        if (settings.browser_sort_mode) {
            BrowserState.sortMode = settings.browser_sort_mode;
        }
        if (settings.browser_sort_direction) {
            BrowserState.sortDirection = settings.browser_sort_direction;
        }
        if (settings.browser_auto_refresh != null) {
            BrowserState.autoRefresh = settings.browser_auto_refresh !== 'false';
        }
        if (settings.browser_display_params) {
            try { BrowserState.displayParams = JSON.parse(settings.browser_display_params); } catch (e) { /* use default */ }
        }
    } catch (e) {
        // Use defaults
    }

    // Apply thumbnail size class
    applyThumbnailSize(BrowserState.thumbnailSize);

    // Update search mode button
    const modeBtn = $('#browser-search-mode');
    if (modeBtn) {
        modeBtn.textContent = BrowserState.searchMode === 'keyword' ? 'KW' : 'EM';
        modeBtn.classList.toggle('active-embedding', BrowserState.searchMode === 'embedding');
    }

    // Setup search handlers
    initBrowserSearch();

    // Restore path: URL hash (F5) > sessionStorage (cross-page nav)
    let savedPath = window.location.hash
        ? decodeURIComponent(window.location.hash.slice(1))
        : '';
    if (!savedPath) {
        try { savedPath = sessionStorage.getItem('browserPath') || ''; } catch (_) {}
    }
    if (savedPath) {
        window.location.hash = encodeURIComponent(savedPath);
    }
    BrowserState.currentPath = savedPath;
    BrowserState.pollTimestamp = Date.now() / 1000;
    await loadBrowserContents();

    // Setup infinite scroll with IntersectionObserver
    const sentinel = $('#browser-scroll-sentinel');
    if (sentinel) {
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting && BrowserState.hasMore && !BrowserState.isLoading) {
                loadNextPage();
            }
        }, { rootMargin: '200px' });
        observer.observe(sentinel);
    }

    // Start polling for new files
    startBrowserPolling();

    // Listen for thumbnail size changes from settings
    window.addEventListener('thumbnail-size-changed', (e) => {
        const size = e.detail?.browser;
        if (size) {
            BrowserState.thumbnailSize = size;
            applyThumbnailSize(size);
        }
    });

});

/**
 * Apply thumbnail size CSS class to the grid.
 */
function applyThumbnailSize(size) {
    const grid = $('#browser-grid');
    if (!grid) return;
    grid.classList.remove('thumb-small', 'thumb-medium', 'thumb-large');
    grid.classList.add(`thumb-${size || 'medium'}`);
}

/**
 * Toggle sidebar visibility (browser version for hamburger menu).
 */
function toggleSidebar() {
    const sidebar = $('.browser-sidebar');
    if (sidebar) {
        sidebar.classList.toggle('hidden');
    }
}
