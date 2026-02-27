/**
 * Shared thumbnail creation — reusable across chat and browser pages.
 *
 * All action buttons use callbacks so consumers can wire up page-specific
 * behavior (e.g. chat refine context, browser selection).
 *
 * Depends on: api.js (API), app.js ($, $$), fullSizeViewer.js (openFullSizeViewer)
 */

// ── Thumbnail item ──────────────────────────────────────────────────────

/**
 * Create a single thumbnail item with image and action icons.
 *
 * @param {Object} job - Job object from API
 * @param {Object} img - Image object { id, filename, ... }
 * @param {Object} [options] - Callback options
 * @param {Function|null} [options.onRegenerate]          - (job, img) => void; null hides button
 * @param {Function|null} [options.onRefine]              - (job, img) => void; null hides button
 * @param {Function|null} [options.onRefineWithAttachment] - (job, img) => void; null hides menu option
 * @param {Function|null} [options.onAttach]              - (job, img) => void; null hides button
 * @param {Function|null} [options.onDelete]              - (job, img, item) => void; null hides button
 * @returns {HTMLElement}
 */
function createThumbnailItem(job, img, options = {}) {
    const item = document.createElement('div');
    item.className = 'gen-thumbnail-item';
    item.dataset.jobId = job.id;

    // Store job/img references on the DOM element so the viewer can
    // collect all images from the parent grid for cross-job navigation.
    item._viewerJob = job;
    item._viewerImg = img;

    // Thumbnail image (clickable for full-size view)
    const thumb = document.createElement('img');
    thumb.className = 'gen-thumbnail-img';
    thumb.src = `/api/generate/thumbnail/${job.id}/${img.id}`;
    thumb.alt = 'Generated image';
    thumb.loading = 'lazy';
    thumb.onclick = () => {
        if (item.classList.contains('thumbnail-missing')) return;
        if (typeof openFullSizeViewer === 'function') {
            // Gather all non-missing, non-failed, non-pending images from the grid
            const grid = item.closest('.gen-thumbnail-grid');
            if (grid) {
                const allItems = grid.querySelectorAll(
                    '.gen-thumbnail-item:not(.gen-thumbnail-pending):not(.gen-thumbnail-failed):not(.thumbnail-missing)'
                );
                const viewerItems = [];
                let startIndex = 0;
                allItems.forEach((el, idx) => {
                    if (el._viewerJob && el._viewerImg) {
                        viewerItems.push({ job: el._viewerJob, img: el._viewerImg });
                        if (el === item) startIndex = viewerItems.length - 1;
                    }
                });
                if (viewerItems.length > 0) {
                    openFullSizeViewer(viewerItems, startIndex);
                    return;
                }
            }
            // Fallback: open with just this image
            openFullSizeViewer(job, img);
        }
    };

    // Handle missing/broken images
    thumb.onerror = () => {
        replaceThumbnailWithPlaceholder(item, thumb);
    };
    // Also detect SVG placeholder returned by backend (content-type mismatch)
    thumb.onload = () => {
        if (thumb.naturalWidth === 256 && thumb.naturalHeight === 256 && thumb.src.includes('/thumbnail/')) {
            fetch(thumb.src, { method: 'HEAD' }).then(res => {
                if (res.headers.get('content-type')?.includes('svg')) {
                    replaceThumbnailWithPlaceholder(item, thumb);
                }
            }).catch(() => {});
        }
    };

    item.appendChild(thumb);

    // Action icons row
    const actions = document.createElement('div');
    actions.className = 'gen-thumbnail-actions';

    // Regenerate icon
    if (options.onRegenerate !== null) {
        const regenBtn = document.createElement('button');
        regenBtn.className = 'gen-action-btn';
        regenBtn.title = 'Regenerate with same settings';
        regenBtn.innerHTML = '🔄';
        regenBtn.onclick = (e) => {
            e.stopPropagation();
            if (options.onRegenerate) options.onRegenerate(job, img);
        };
        actions.appendChild(regenBtn);
    }

    // Refine icon (with optional hover menu for refine+attach)
    if (options.onRefine !== null) {
        const hasRefineAttach = options.onRefineWithAttachment != null;

        if (hasRefineAttach) {
            // Wrap button + popover in a container
            const wrap = document.createElement('div');
            wrap.className = 'gen-refine-wrap';

            const menu = document.createElement('div');
            menu.className = 'gen-refine-menu';
            const menuItem = document.createElement('button');
            menuItem.className = 'gen-refine-menu-item';
            menuItem.textContent = 'Refine + attach';
            menuItem.onclick = (e) => {
                e.stopPropagation();
                options.onRefineWithAttachment(job, img);
            };
            menu.appendChild(menuItem);
            wrap.appendChild(menu);

            const refineBtn = document.createElement('button');
            refineBtn.className = 'gen-action-btn gen-refine-btn';
            refineBtn.title = 'Refine this prompt';
            refineBtn.innerHTML = '✏️';
            refineBtn.onclick = (e) => {
                e.stopPropagation();
                if (options.onRefine) options.onRefine(job, img);
            };
            wrap.appendChild(refineBtn);

            actions.appendChild(wrap);
        } else {
            const refineBtn = document.createElement('button');
            refineBtn.className = 'gen-action-btn gen-refine-btn';
            refineBtn.title = 'Refine this prompt';
            refineBtn.innerHTML = '✏️';
            refineBtn.onclick = (e) => {
                e.stopPropagation();
                if (options.onRefine) options.onRefine(job, img);
            };
            actions.appendChild(refineBtn);
        }
    }

    // Attach icon
    if (options.onAttach !== null) {
        const attachBtn = document.createElement('button');
        attachBtn.className = 'gen-action-btn gen-attach-btn';
        attachBtn.title = 'Add as attachment';
        attachBtn.innerHTML = '📎';
        attachBtn.onclick = (e) => {
            e.stopPropagation();
            if (options.onAttach) options.onAttach(job, img);
        };
        actions.appendChild(attachBtn);
    }

    // Delete icon
    if (options.onDelete !== null) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'gen-action-btn gen-delete-btn';
        deleteBtn.title = 'Delete image';
        deleteBtn.innerHTML = '✕';
        deleteBtn.onclick = async (e) => {
            e.stopPropagation();
            if (options.onDelete) options.onDelete(job, img, item);
        };
        actions.appendChild(deleteBtn);
    }

    item.appendChild(actions);
    return item;
}

// ── Failed card ──────────────────────────────────────────────────────────

/**
 * Create a thumbnail-sized failed card with error icon.
 *
 * @param {Object} job - Job object from API
 * @param {Object} [options] - Callback options
 * @param {Function|null} [options.onRegenerate] - (job) => void; null hides button
 * @param {Function|null} [options.onDelete]     - (job, item) => void; null hides button
 * @returns {HTMLElement}
 */
function createFailedCard(job, options = {}) {
    const item = document.createElement('div');
    item.className = 'gen-thumbnail-item gen-thumbnail-failed';
    item.dataset.jobId = job.id;

    const content = document.createElement('div');
    content.className = 'gen-failed-content';
    content.innerHTML = `<span class="gen-failed-icon">✕</span>
                         <span class="gen-failed-text">Failed</span>`;
    item.appendChild(content);

    // Action icons (always visible on failed cards)
    const actions = document.createElement('div');
    actions.className = 'gen-thumbnail-actions';

    if (options.onRegenerate !== null) {
        const regenBtn = document.createElement('button');
        regenBtn.className = 'gen-action-btn';
        regenBtn.title = 'Retry with same settings';
        regenBtn.innerHTML = '🔄';
        regenBtn.onclick = (e) => {
            e.stopPropagation();
            if (options.onRegenerate) options.onRegenerate(job);
        };
        actions.appendChild(regenBtn);
    }

    if (options.onDelete !== null) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'gen-action-btn gen-delete-btn';
        deleteBtn.title = 'Delete failed job';
        deleteBtn.innerHTML = '✕';
        deleteBtn.onclick = async (e) => {
            e.stopPropagation();
            if (options.onDelete) options.onDelete(job, item);
        };
        actions.appendChild(deleteBtn);
    }

    item.appendChild(actions);
    return item;
}

// ── Missing/broken image handling ────────────────────────────────────────

/**
 * Replace a broken/missing thumbnail <img> with a placeholder div.
 */
function replaceThumbnailWithPlaceholder(item, imgEl) {
    if (item.classList.contains('thumbnail-missing')) return;
    item.classList.add('thumbnail-missing');

    const placeholder = document.createElement('div');
    placeholder.className = 'gen-thumbnail-placeholder';
    placeholder.innerHTML = '<span class="gen-placeholder-icon">✕</span><span class="gen-placeholder-text">Missing</span>';
    item.replaceChild(placeholder, imgEl);

    // Disable the attach button for this item
    const attachBtn = item.querySelector('.gen-attach-btn');
    if (attachBtn) {
        attachBtn.disabled = true;
        attachBtn.title = 'Image not available';
    }

    // Check if ALL thumbnails in this bubble are now missing
    checkAllThumbnailsMissing(item);
}

/**
 * After marking a thumbnail as missing, check if all thumbnails in the
 * parent grid are missing. If so, add a class for styling.
 */
function checkAllThumbnailsMissing(item) {
    const grid = item.closest('.gen-thumbnail-grid');
    if (!grid) return;
    const allItems = grid.querySelectorAll('.gen-thumbnail-item:not(.gen-thumbnail-failed)');
    const missingItems = grid.querySelectorAll('.gen-thumbnail-item.thumbnail-missing');
    if (allItems.length > 0 && allItems.length === missingItems.length) {
        grid.closest('.message.generation')?.classList.add('gen-all-missing');
    }
}

// ── Progress ─────────────────────────────────────────────────────────────

/**
 * Create circular SVG progress indicator.
 * @param {number} progress - 0.0 to 1.0
 * @param {number} [queuePosition=0] - Queue position (1+ shows #N, 0 = not queued)
 * @returns {string} HTML string
 */
function createCircularProgress(progress, queuePosition = 0) {
    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - progress);

    let label;
    if (queuePosition > 0 && progress === 0) {
        label = `#${queuePosition}`;
    } else if (progress > 0) {
        label = Math.round(progress * 100) + '%';
    } else {
        label = '⏳';
    }
    return `<svg class="gen-progress-svg" viewBox="0 0 80 80">
        <circle class="gen-progress-bg" cx="40" cy="40" r="${radius}" />
        <circle class="gen-progress-fill" cx="40" cy="40" r="${radius}"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${offset}" />
        <text class="gen-progress-text" x="40" y="40" text-anchor="middle" dominant-baseline="central">${label}</text>
    </svg>`;
}

// ── Grid status ──────────────────────────────────────────────────────────

/**
 * Compute aggregate status from a thumbnail grid's contents.
 * Returns data without mutating the DOM — callers decide how to use it.
 *
 * @param {HTMLElement} grid - A .gen-thumbnail-grid element
 * @returns {{ completed: number, missing: number, failed: number, pending: number, statusText: string, phase: string }}
 */
function getGridStatus(grid) {
    const completed = grid.querySelectorAll(
        '.gen-thumbnail-item:not(.gen-thumbnail-pending):not(.gen-thumbnail-failed):not(.thumbnail-missing)'
    ).length;
    const missing = grid.querySelectorAll('.gen-thumbnail-item.thumbnail-missing').length;
    const failed = grid.querySelectorAll('.gen-thumbnail-failed').length;
    const pending = grid.querySelectorAll('.gen-thumbnail-pending').length;

    const parts = [];
    if (completed > 0) parts.push(`${completed} image${completed !== 1 ? 's' : ''}`);
    if (missing > 0) parts.push(`${missing} missing`);
    if (failed > 0) parts.push(`${failed} failed`);
    if (pending > 0) parts.push(`generating…`);

    const statusText = parts.join(', ') || 'Waiting…';

    // Set phase based on priority: running > failed > completed
    let phase = 'pending';
    if (pending > 0) phase = 'running';
    else if (completed > 0) phase = 'completed';
    else if (failed > 0) phase = 'failed';

    return { completed, missing, failed, pending, statusText, phase };
}
