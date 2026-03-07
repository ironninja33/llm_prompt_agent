/**
 * Browser polling — check for new files periodically.
 *
 * Uses setTimeout chain (not setInterval) to prevent overlapping polls.
 * Backs off when agent is busy to reduce contention.
 *
 * Depends on: api.js, browserState.js, browserGrid.js
 */

const POLL_INTERVAL_NORMAL = 10000;   // 10s default
const POLL_INTERVAL_BUSY = 30000;     // 30s when agent is streaming

/**
 * Start polling for new files.
 */
function startBrowserPolling() {
    stopBrowserPolling();
    _schedulePoll(POLL_INTERVAL_NORMAL);
}

/**
 * Stop polling.
 */
function stopBrowserPolling() {
    if (BrowserState.pollTimer) {
        clearTimeout(BrowserState.pollTimer);
        BrowserState.pollTimer = null;
    }
}

function _schedulePoll(delay) {
    BrowserState.pollTimer = setTimeout(async () => {
        BrowserState.pollTimer = null;

        if (BrowserState.isSearchActive || BrowserState.isLoading || BrowserState.deletePending) {
            _schedulePoll(POLL_INTERVAL_NORMAL);
            return;
        }

        let nextDelay = POLL_INTERVAL_NORMAL;
        try {
            const result = await API.browserPoll(
                BrowserState.currentPath,
                BrowserState.pollTimestamp
            );

            // Adaptive backoff when agent is busy
            if (result.agent_busy) {
                nextDelay = POLL_INTERVAL_BUSY;
            }

            if (result.has_new_files) {
                BrowserState.offset = 0;
                BrowserState.pollTimestamp = Date.now() / 1000;
                await loadBrowserContents();
            }
        } catch (err) {
            // Silently ignore poll failures
        }

        _schedulePoll(nextDelay);
    }, delay);
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
