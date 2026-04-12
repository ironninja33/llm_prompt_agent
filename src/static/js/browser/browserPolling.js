/**
 * Browser polling — check for new files periodically, with smart insertion.
 *
 * When auto-refresh is ON and no overlay is open, new images are inserted
 * at the correct grid position based on the current sort order.
 * When auto-refresh is OFF (or an overlay is open), new images are buffered
 * and a badge shows the pending count. Full re-render on toggle-on or
 * overlay close.
 *
 * Depends on: api.js, browserState.js, browserGrid.js, browserNav.js
 */

let _insertionInProgress = false;

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
            BrowserState.lastCompletionSeq = serverSeq;
        } else if (serverSeq !== BrowserState.lastCompletionSeq) {
            BrowserState.lastCompletionSeq = serverSeq;
            // Advance pollTimestamp so the mtime check doesn't re-trigger
            // for the same files on the next poll cycle.
            BrowserState.pollTimestamp = Date.now() / 1000;

            if (!BrowserState.autoRefresh) {
                _fetchAndBuffer();
                return;
            }

            // Don't re-render while an overlay or viewer is open
            if (_isOverlayOpen()) {
                _fetchAndBuffer();
                return;
            }

            _fetchAndInsert();
            return;
        }

        // Non-generation file changes (mtime-based)
        if (data.has_new_files) {
            if (!BrowserState.autoRefresh) {
                // Suppress mtime refresh; advance timestamp to stop re-triggering
                BrowserState.pollTimestamp = Date.now() / 1000;
                return;
            }
            BrowserState.offset = 0;
            BrowserState.pollTimestamp = Date.now() / 1000;
            loadBrowserContents();
        }
    }, () => ({
        browser_path: BrowserState.currentPath || '',
        browser_since: String(BrowserState.pollTimestamp || 0),
    }));

    // Flush buffered images when overlays close
    window.addEventListener('overlay-closed', _flushPending);
}

/**
 * Stop polling.
 */
function stopBrowserPolling() {
    PollManager.unregister('browser');
    window.removeEventListener('overlay-closed', _flushPending);
}

// ── Helpers ──────────────────────────────────────────────────────────────

function _isOverlayOpen() {
    const genOverlay = document.getElementById('generation-overlay');
    const viewer = document.getElementById('fullsize-viewer');
    return (genOverlay && !genOverlay.classList.contains('hidden'))
        || (viewer && !viewer.classList.contains('hidden'));
}

function _flushPending() {
    if (BrowserState.pendingNewCount > 0 && BrowserState.autoRefresh) {
        BrowserState.pendingNewCount = 0;
        BrowserState.offset = 0;
        BrowserState.pollTimestamp = Date.now() / 1000;
        loadBrowserContents();
        _updateRefreshBadge();
    }
}

async function _fetchAndBuffer() {
    if (!BrowserState.newestCreatedAt || !BrowserState.currentPath) return;
    try {
        const result = await API.browserNewImages(
            BrowserState.currentPath,
            BrowserState.newestCreatedAt
        );
        if (result.images && result.images.length > 0) {
            BrowserState.pendingNewCount += result.images.length;
            BrowserState.newestCreatedAt = _maxCreatedAt(result.images, BrowserState.newestCreatedAt);
            _updateRefreshBadge();
        }
    } catch (e) {
        console.warn('Failed to fetch new images for buffer:', e);
    }
}

async function _fetchAndInsert() {
    if (_insertionInProgress) return;

    if (!BrowserState.newestCreatedAt || !BrowserState.currentPath) {
        // No baseline — fall back to full re-render
        BrowserState.offset = 0;
        BrowserState.pollTimestamp = Date.now() / 1000;
        loadBrowserContents();
        return;
    }

    _insertionInProgress = true;
    try {
        const result = await API.browserNewImages(
            BrowserState.currentPath,
            BrowserState.newestCreatedAt
        );
        if (result.images && result.images.length > 0) {
            if (result.images.length > BrowserState.limit) {
                // Too many new images — full re-render is safer
                BrowserState.offset = 0;
                BrowserState.pollTimestamp = Date.now() / 1000;
                loadBrowserContents();
            } else {
                _smartInsertImages(result.images);
            }
            BrowserState.newestCreatedAt = _maxCreatedAt(result.images, BrowserState.newestCreatedAt);
        }
    } catch (e) {
        console.warn('Smart insert failed, falling back to full re-render:', e);
        BrowserState.offset = 0;
        loadBrowserContents();
    } finally {
        _insertionInProgress = false;
    }
}

function _smartInsertImages(images) {
    const grid = $('#browser-grid');
    if (!grid) return;

    for (const imageData of images) {
        // Skip duplicates
        if (grid.querySelector(`.browser-img-item[data-image-id="${imageData.id}"]`)) continue;

        const newEl = createBrowserImageThumbnail(imageData);
        const insertionPoint = _findInsertionPoint(grid, imageData);
        if (insertionPoint) {
            grid.insertBefore(newEl, insertionPoint);
        } else {
            grid.appendChild(newEl);
        }
    }

    if (typeof updateBrowserParamDisplay === 'function') updateBrowserParamDisplay();

    const emptyEl = $('#browser-empty');
    if (emptyEl) emptyEl.classList.add('hidden');
}

function _findInsertionPoint(grid, imageData) {
    const items = grid.querySelectorAll('.browser-img-item');
    if (items.length === 0) return null;

    const sortMode = BrowserState.sortMode || 'date';
    const sortDir = BrowserState.sortDirection || 'desc';

    for (const item of items) {
        if (_shouldInsertBefore(item, imageData, sortMode, sortDir)) {
            return item;
        }
    }
    return null; // append at end
}

function _shouldInsertBefore(existingEl, newImageData, sortMode, sortDir) {
    if (sortMode === 'seed') {
        const existingSeed = parseInt(existingEl.dataset.sortSeed || '0', 10);
        const newSeed = parseInt(newImageData.settings?.seed || '0', 10);
        if (newSeed !== existingSeed) {
            return sortDir === 'asc' ? newSeed < existingSeed : newSeed > existingSeed;
        }
        // Tiebreaker: date (opposite direction)
        const existingDate = existingEl.dataset.sortDate || '';
        const newDate = newImageData.created_at || '';
        return sortDir === 'asc' ? newDate > existingDate : newDate < existingDate;
    }
    // Default: date sort
    const existingDate = existingEl.dataset.sortDate || '';
    const newDate = newImageData.created_at || '';
    return sortDir === 'desc' ? newDate > existingDate : newDate < existingDate;
}

function _maxCreatedAt(images, current) {
    let max = current || '';
    for (const img of images) {
        if (img.created_at && img.created_at > max) {
            max = img.created_at;
        }
    }
    return max;
}

/**
 * Update the pending-count badge on the auto-refresh toggle button.
 */
function _updateRefreshBadge() {
    const badge = document.querySelector('.refresh-badge');
    if (!badge) return;
    if (BrowserState.pendingNewCount > 0 && !BrowserState.autoRefresh) {
        badge.textContent = String(BrowserState.pendingNewCount);
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }
}
