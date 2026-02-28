/**
 * Browser polling — check for new files periodically.
 *
 * Depends on: api.js, browserState.js, browserGrid.js
 */

/**
 * Start polling for new files every 5 seconds.
 */
function startBrowserPolling() {
    stopBrowserPolling();

    BrowserState.pollTimer = setInterval(async () => {
        if (BrowserState.isSearchActive || BrowserState.isLoading || BrowserState.deletePending) return;

        try {
            const result = await API.browserPoll(
                BrowserState.currentPath,
                BrowserState.pollTimestamp
            );

            if (result.has_new_files) {
                // Reload the full listing to show new files in correct sort order
                BrowserState.offset = 0;
                BrowserState.pollTimestamp = Date.now() / 1000;
                await loadBrowserContents();
            }
        } catch (err) {
            // Silently ignore poll failures
        }
    }, 5000);
}

/**
 * Stop polling.
 */
function stopBrowserPolling() {
    if (BrowserState.pollTimer) {
        clearInterval(BrowserState.pollTimer);
        BrowserState.pollTimer = null;
    }
}

/**
 * Prepend new items to the top of the grid.
 */
function prependBrowserItems(images) {
    const grid = $('#browser-grid');
    if (!grid || !images || images.length === 0) return;

    // Find first image item (after any directory cards)
    const firstImgItem = grid.querySelector('.browser-img-item');

    images.forEach(imageData => {
        const el = createBrowserImageThumbnail(imageData);
        if (firstImgItem) {
            grid.insertBefore(el, firstImgItem);
        } else {
            grid.appendChild(el);
        }
    });

    // Hide empty state if it was showing
    const emptyEl = $('#browser-empty');
    if (emptyEl) emptyEl.classList.add('hidden');
}
