/**
 * Browser polling — check for new files periodically.
 *
 * Uses setTimeout chain (not setInterval) to prevent overlapping polls.
 * Backs off when agent is busy to reduce contention.
 *
 * Depends on: api.js, browserState.js, browserGrid.js
 */

/**
 * Start polling for new files via PollManager.
 */
function startBrowserPolling() {
    PollManager.register('browser', (data) => {
        if (BrowserState.isSearchActive || BrowserState.isLoading || BrowserState.deletePending) {
            return;
        }

        // Generation completion (path-scoped via server)
        const serverSeq = data.completion_seq || 0;
        if (BrowserState.lastCompletionSeq === null) {
            // First poll after page load or navigation — sync baseline
            BrowserState.lastCompletionSeq = serverSeq;
        } else if (serverSeq !== BrowserState.lastCompletionSeq) {
            BrowserState.lastCompletionSeq = serverSeq;
            BrowserState.offset = 0;
            BrowserState.pollTimestamp = Date.now() / 1000;
            loadBrowserContents();
            return;
        }

        // Non-generation file changes (mtime-based)
        if (data.has_new_files) {
            BrowserState.offset = 0;
            BrowserState.pollTimestamp = Date.now() / 1000;
            loadBrowserContents();
        }
    }, () => ({
        browser_path: BrowserState.currentPath || '',
        browser_since: String(BrowserState.pollTimestamp || 0),
    }));
}

/**
 * Stop polling.
 */
function stopBrowserPolling() {
    PollManager.unregister('browser');
}

/**
 * Prepend new items to the top of the grid.
 */
function prependBrowserItems(images) {
    const grid = $('#browser-grid');
    if (!grid || !images || images.length === 0) return;

    const fragment = document.createDocumentFragment();
    images.forEach(imageData => {
        fragment.appendChild(createBrowserImageThumbnail(imageData));
    });

    // Find first image item (after any directory cards)
    const firstImgItem = grid.querySelector('.browser-img-item');
    if (firstImgItem) {
        grid.insertBefore(fragment, firstImgItem);
    } else {
        grid.appendChild(fragment);
    }

    // Hide empty state if it was showing
    const emptyEl = $('#browser-empty');
    if (emptyEl) emptyEl.classList.add('hidden');
}
