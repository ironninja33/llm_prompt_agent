/**
 * Generation bubble — displays generated image thumbnails in the chat.
 *
 * One bubble per message_id.  Multiple generation attempts for the same
 * message are merged into a single bubble: completed thumbnails, failed
 * cards (with regenerate icons), and active progress spinners all live
 * in the same grid.
 *
 * Depends on: api.js, app.js ($, $$, escapeHtml, currentChatId, isStreaming),
 *             chatPane.js (setStreaming, scrollToBottom),
 *             generationOverlay.js (openGenerationOverlay)
 */

// Track active SSE connections for cleanup
let _activeGenerationPollers = {};

// ── Bubble creation ──────────────────────────────────────────────────────

/**
 * Create a generation bubble for a single job (real-time submission).
 * For merged multi-job bubbles on reload, see _createMergedBubble().
 * @param {Object} job - Job object from API
 * @returns {HTMLElement}
 */
function createGenerationBubble(job) {
    const bubble = document.createElement('div');
    bubble.className = 'message generation';
    bubble.dataset.jobId = job.id;
    bubble.dataset.messageId = job.message_id || '';

    const header = document.createElement('div');
    header.className = 'gen-bubble-header';
    header.innerHTML = `<span class="gen-bubble-label">Generated Images</span>
                        <span class="gen-bubble-status"></span>`;
    bubble.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'gen-thumbnail-grid';
    bubble.appendChild(grid);

    _appendJobItems(grid, job);
    _updateBubbleStatusFromGrid(grid);

    return bubble;
}

/**
 * Create a merged bubble combining multiple jobs for the same message.
 * Used by loadGenerationBubbles() on chat reload.
 */
function _createMergedBubble(jobs) {
    const bubble = document.createElement('div');
    bubble.className = 'message generation';
    bubble.dataset.messageId = jobs[0].message_id || '';
    bubble.dataset.jobId = jobs[0].id;

    const header = document.createElement('div');
    header.className = 'gen-bubble-header';
    header.innerHTML = `<span class="gen-bubble-label">Generated Images</span>
                        <span class="gen-bubble-status"></span>`;
    bubble.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'gen-thumbnail-grid';
    bubble.appendChild(grid);

    for (const job of jobs) {
        _appendJobItems(grid, job);
    }
    _updateBubbleStatusFromGrid(grid);

    return bubble;
}

/**
 * Append grid items for a single job (thumbnails, failed card, or spinners).
 * Does NOT update the bubble header — caller is responsible for that.
 */
function _appendJobItems(grid, job) {
    if (job.status === 'completed' && job.images && job.images.length > 0) {
        job.images.forEach(img => {
            grid.appendChild(_createThumbnailItem(job, img));
        });
    } else if (job.status === 'failed') {
        grid.appendChild(_createFailedCard(job));
    } else {
        _appendProgressSpinners(grid, job);
        _startProgressPolling(job.id);
    }
}

/**
 * Append progress spinner placeholders for a job to the grid.
 * Each spinner is tagged with data-job-id for targeted updates.
 */
function _appendProgressSpinners(grid, job) {
    const numImages = job.settings?.num_images || 1;
    for (let i = 0; i < numImages; i++) {
        const item = document.createElement('div');
        item.className = 'gen-thumbnail-item gen-thumbnail-pending';
        item.dataset.jobId = job.id;
        item.innerHTML = _createCircularProgress(0);
        grid.appendChild(item);
    }
}

/**
 * Insert a generation bubble after the correct message in the chat.
 * Finds the message element with matching message_id and inserts after it.
 */
function insertGenerationBubble(bubble, messageId) {
    const container = $('#messages');
    if (!container) return;

    // Find the message element with this message ID
    if (messageId) {
        const msgEl = container.querySelector(`[data-message-id="${messageId}"]`);
        if (msgEl) {
            // Insert after that message (or after existing generation bubble for same message)
            let insertAfter = msgEl;
            let next = msgEl.nextElementSibling;
            while (next && next.classList.contains('generation') && next.dataset.messageId === String(messageId)) {
                insertAfter = next;
                next = next.nextElementSibling;
            }
            insertAfter.after(bubble);
            scrollToBottom();
            return;
        }
    }

    // Fallback: append to end
    container.appendChild(bubble);
    scrollToBottom();
}

// ── Thumbnail items ──────────────────────────────────────────────────────

/**
 * Create a single thumbnail item with image and action icons.
 */
function _createThumbnailItem(job, img) {
    const item = document.createElement('div');
    item.className = 'gen-thumbnail-item';
    item.dataset.jobId = job.id;

    // Thumbnail image (clickable for full-size view)
    const thumb = document.createElement('img');
    thumb.className = 'gen-thumbnail-img';
    thumb.src = `/api/generate/thumbnail/${job.id}/${img.id}`;
    thumb.alt = 'Generated image';
    thumb.loading = 'lazy';
    thumb.onclick = () => {
        if (item.classList.contains('thumbnail-missing')) return;
        if (typeof openFullSizeViewer === 'function') {
            openFullSizeViewer(job, img);
        }
    };

    // Handle missing/broken images
    thumb.onerror = () => {
        _replaceThumbnailWithPlaceholder(item, thumb);
    };
    // Also detect SVG placeholder returned by backend (content-type mismatch)
    thumb.onload = () => {
        if (thumb.naturalWidth === 256 && thumb.naturalHeight === 256 && thumb.src.includes('/thumbnail/')) {
            fetch(thumb.src, { method: 'HEAD' }).then(res => {
                if (res.headers.get('content-type')?.includes('svg')) {
                    _replaceThumbnailWithPlaceholder(item, thumb);
                }
            }).catch(() => {});
        }
    };

    item.appendChild(thumb);

    // Action icons row
    const actions = document.createElement('div');
    actions.className = 'gen-thumbnail-actions';

    // Regenerate icon
    const regenBtn = document.createElement('button');
    regenBtn.className = 'gen-action-btn';
    regenBtn.title = 'Regenerate with same settings';
    regenBtn.innerHTML = '🔄';
    regenBtn.onclick = (e) => {
        e.stopPropagation();
        openGenerationOverlay({
            prompt: job.settings?.positive_prompt || '',
            chatId: job.chat_id,
            messageId: job.message_id,
            settings: job.settings
        });
    };
    actions.appendChild(regenBtn);

    // Attach icon
    const attachBtn = document.createElement('button');
    attachBtn.className = 'gen-action-btn gen-attach-btn';
    attachBtn.title = 'Add as attachment';
    attachBtn.innerHTML = '📎';
    attachBtn.onclick = (e) => {
        e.stopPropagation();
        if (typeof addAttachment === 'function') {
            addAttachment({
                type: 'generated',
                jobId: job.id,
                imageId: img.id,
                thumbnailUrl: `/api/generate/thumbnail/${job.id}/${img.id}`,
                fullUrl: `/api/generate/image/${job.id}/${img.id}`,
            });
        }
    };
    actions.appendChild(attachBtn);

    // Delete icon
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'gen-action-btn gen-delete-btn';
    deleteBtn.title = 'Delete image';
    deleteBtn.innerHTML = '✕';
    deleteBtn.onclick = async (e) => {
        e.stopPropagation();
        try {
            await API.deleteGeneratedImage(job.id, img.id);
            const grid = item.closest('.gen-thumbnail-grid');
            item.remove();
            if (grid) {
                _updateBubbleStatusFromGrid(grid);
                // If grid is now empty, remove the entire bubble
                if (grid.querySelectorAll('.gen-thumbnail-item').length === 0) {
                    const bubble = grid.closest('.message.generation');
                    if (bubble) bubble.remove();
                }
            }
        } catch (err) {
            console.error('Failed to delete image:', err);
        }
    };
    actions.appendChild(deleteBtn);

    item.appendChild(actions);
    return item;
}

// ── Failed card ──────────────────────────────────────────────────────────

/**
 * Create a thumbnail-sized failed card with error icon and regenerate button.
 * Same aspect ratio and size as a real thumbnail so they sit in the grid.
 */
function _createFailedCard(job) {
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

    const regenBtn = document.createElement('button');
    regenBtn.className = 'gen-action-btn';
    regenBtn.title = 'Retry with same settings';
    regenBtn.innerHTML = '🔄';
    regenBtn.onclick = (e) => {
        e.stopPropagation();
        openGenerationOverlay({
            prompt: job.settings?.positive_prompt || '',
            chatId: job.chat_id,
            messageId: job.message_id,
            settings: job.settings
        });
    };
    actions.appendChild(regenBtn);

    item.appendChild(actions);
    return item;
}

// ── Missing/broken image handling ────────────────────────────────────────

/**
 * Replace a broken/missing thumbnail <img> with a placeholder div.
 */
function _replaceThumbnailWithPlaceholder(item, imgEl) {
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
    _checkAllThumbnailsMissing(item);
}

/**
 * After marking a thumbnail as missing, check if all thumbnails in the
 * parent bubble are missing. If so, add a class for styling.
 */
function _checkAllThumbnailsMissing(item) {
    const grid = item.closest('.gen-thumbnail-grid');
    if (!grid) return;
    const allItems = grid.querySelectorAll('.gen-thumbnail-item:not(.gen-thumbnail-failed)');
    const missingItems = grid.querySelectorAll('.gen-thumbnail-item.thumbnail-missing');
    if (allItems.length > 0 && allItems.length === missingItems.length) {
        grid.closest('.message.generation')?.classList.add('gen-all-missing');
    }
}

// ── Progress & status ────────────────────────────────────────────────────

/**
 * Create circular SVG progress indicator.
 * @param {number} progress - 0.0 to 1.0
 */
function _createCircularProgress(progress) {
    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - progress);

    return `<svg class="gen-progress-svg" viewBox="0 0 80 80">
        <circle class="gen-progress-bg" cx="40" cy="40" r="${radius}" />
        <circle class="gen-progress-fill" cx="40" cy="40" r="${radius}"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${offset}" />
        <text class="gen-progress-text" x="40" y="44" text-anchor="middle">
            ${progress > 0 ? Math.round(progress * 100) + '%' : '⏳'}
        </text>
    </svg>`;
}

/**
 * Update the bubble header status by scanning grid contents.
 * Computes aggregate counts of completed, failed, and pending items.
 */
function _updateBubbleStatusFromGrid(grid) {
    const bubble = grid.closest('.message.generation');
    if (!bubble) return;
    const statusEl = bubble.querySelector('.gen-bubble-status');
    if (!statusEl) return;

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

    statusEl.textContent = parts.join(', ') || 'Waiting…';

    // Set status class based on priority: running > failed > completed
    let phase = 'pending';
    if (pending > 0) phase = 'running';
    else if (completed > 0) phase = 'completed';
    else if (failed > 0) phase = 'failed';
    statusEl.className = `gen-bubble-status gen-status-${phase}`;
}

// ── SSE polling ──────────────────────────────────────────────────────────

/**
 * Start SSE polling for a generation job.
 */
function _startProgressPolling(jobId) {
    const evtSource = new EventSource(`/api/generate/${jobId}/status`);
    _activeGenerationPollers[jobId] = evtSource;

    evtSource.addEventListener('generation_status', (e) => {
        const data = JSON.parse(e.data);
        _handleProgressUpdate(data);
    });

    evtSource.addEventListener('generation_complete', (e) => {
        const data = JSON.parse(e.data);
        _handleGenerationComplete(data);
        evtSource.close();
        delete _activeGenerationPollers[jobId];
    });

    evtSource.onerror = () => {
        evtSource.close();
        delete _activeGenerationPollers[jobId];
    };
}

/**
 * Handle a progress update from SSE.
 * Finds spinners for the job by data-job-id attribute.
 */
function _handleProgressUpdate(data) {
    const items = document.querySelectorAll(
        `.gen-thumbnail-pending[data-job-id="${data.job_id}"]`
    );
    if (items.length > 0 && data.progress > 0) {
        const currentIdx = Math.min(data.current_image || 0, items.length - 1);
        items[currentIdx].innerHTML = _createCircularProgress(data.progress);
    }

    // Update aggregate status in the bubble header
    if (items.length > 0) {
        const grid = items[0].closest('.gen-thumbnail-grid');
        if (grid) {
            const bubble = grid.closest('.message.generation');
            const statusEl = bubble?.querySelector('.gen-bubble-status');
            if (statusEl) {
                const statusMsg = data.phase === 'running'
                    ? `Generating… ${Math.round(data.progress * 100)}%`
                    : 'Queued…';
                statusEl.textContent = statusMsg;
                statusEl.className = `gen-bubble-status gen-status-${data.phase}`;
            }
        }
    }
}

/**
 * Handle generation complete from SSE.
 * Replaces the spinners for this job with completed thumbnails or a failed card.
 */
async function _handleGenerationComplete(data) {
    // Find pending items for this job
    const pendingItems = document.querySelectorAll(
        `.gen-thumbnail-pending[data-job-id="${data.job_id}"]`
    );

    try {
        const jobs = await API.getChatGenerations(currentChatId);
        const job = jobs.find(j => j.id === data.job_id);

        if (job && pendingItems.length > 0) {
            const grid = pendingItems[0].closest('.gen-thumbnail-grid');

            // Remove spinners for this job
            pendingItems.forEach(item => item.remove());

            if (data.phase === 'completed' && job.images && job.images.length > 0) {
                job.images.forEach(img => {
                    grid.appendChild(_createThumbnailItem(job, img));
                });
            } else {
                grid.appendChild(_createFailedCard(job));
            }

            _updateBubbleStatusFromGrid(grid);
        } else if (!job && pendingItems.length > 0) {
            // Job not found in API — render a generic failed card
            const grid = pendingItems[0].closest('.gen-thumbnail-grid');
            pendingItems.forEach(item => item.remove());
            const failedItem = document.createElement('div');
            failedItem.className = 'gen-thumbnail-item gen-thumbnail-failed';
            failedItem.dataset.jobId = data.job_id;
            failedItem.innerHTML = `<div class="gen-failed-content">
                <span class="gen-failed-icon">✕</span>
                <span class="gen-failed-text">Failed</span>
            </div>`;
            grid.appendChild(failedItem);
            _updateBubbleStatusFromGrid(grid);
        }
    } catch (err) {
        console.error('Failed to fetch generation results:', err);
    }

    // Re-enable chat input if no more active jobs
    _checkAndUnlockChat();
}

/**
 * Check if there are any active generation jobs and unlock chat if not.
 */
function _checkAndUnlockChat() {
    if (Object.keys(_activeGenerationPollers).length === 0) {
        setStreaming(false);
    }
}

// ── Overflow viewer ──────────────────────────────────────────────────────

/**
 * Show all thumbnails in a scrollable overlay (for overflow).
 */
function _showAllThumbnails(job) {
    const overlay = document.createElement('div');
    overlay.className = 'gen-thumbnails-overlay';
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.remove();
    };

    const content = document.createElement('div');
    content.className = 'gen-thumbnails-overlay-content';

    const header = document.createElement('div');
    header.className = 'gen-thumbnails-overlay-header';
    header.innerHTML = `<h3>All Generated Images (${job.images.length})</h3>
                        <button class="btn-icon" onclick="this.closest('.gen-thumbnails-overlay').remove()">&times;</button>`;
    content.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'gen-thumbnail-grid gen-thumbnail-grid-full';
    job.images.forEach(img => {
        grid.appendChild(_createThumbnailItem(job, img));
    });
    content.appendChild(grid);

    overlay.appendChild(content);
    document.body.appendChild(overlay);
}

// ── Chat reload ──────────────────────────────────────────────────────────

/**
 * Load and render generation bubbles for a chat (on chat reload).
 * Groups jobs by message_id so each message gets one merged bubble.
 */
async function loadGenerationBubbles(chatId) {
    try {
        const jobs = await API.getChatGenerations(chatId);
        if (!jobs || jobs.length === 0) return;

        // Group jobs by message_id
        const grouped = {};
        for (const job of jobs) {
            const key = String(job.message_id || `orphan_${job.id}`);
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(job);
        }

        for (const [msgId, groupJobs] of Object.entries(grouped)) {
            // Skip if bubble already exists for this message
            const realMsgId = msgId.startsWith('orphan_') ? null : msgId;
            if (realMsgId) {
                const existing = document.querySelector(
                    `.message.generation[data-message-id="${realMsgId}"]`
                );
                if (existing) continue;
            }

            // Single job → use simple bubble; multiple → merged bubble
            const bubble = groupJobs.length === 1
                ? createGenerationBubble(groupJobs[0])
                : _createMergedBubble(groupJobs);

            insertGenerationBubble(bubble, groupJobs[0].message_id);
        }
    } catch (err) {
        console.error('Failed to load generation bubbles:', err);
    }
}

/**
 * Clean up all active pollers (called on chat switch).
 */
function cleanupGenerationPollers() {
    Object.values(_activeGenerationPollers).forEach(es => es.close());
    _activeGenerationPollers = {};
}

// ── Event Listeners ─────────────────────────────────────────────────────

window.addEventListener('generation-submitted', (e) => {
    const job = e.detail;
    if (!job || !job.id) return;

    // Lock chat input
    setStreaming(true);

    const msgId = String(job.message_id || job.messageId || '');

    // Check if a bubble already exists for this message
    let existingBubble = null;
    if (msgId) {
        existingBubble = document.querySelector(
            `.message.generation[data-message-id="${msgId}"]`
        );
    }

    if (existingBubble) {
        // Append spinners to the existing bubble's grid
        const grid = existingBubble.querySelector('.gen-thumbnail-grid');
        if (grid) {
            _appendProgressSpinners(grid, job);
            _startProgressPolling(job.id);
            _updateBubbleStatusFromGrid(grid);
            scrollToBottom();
        }
    } else {
        // Create and insert a new bubble
        const bubble = createGenerationBubble(job);
        insertGenerationBubble(bubble, job.message_id || job.messageId);
    }
});
