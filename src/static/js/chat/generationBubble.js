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
 *             generationOverlay.js (openGenerationOverlay),
 *             thumbnails.js (createThumbnailItem, createFailedCard,
 *                            createCircularProgress, getGridStatus)
 */

// Track active SSE connections for cleanup
let _activeGenerationPollers = {};

// ── Chat-specific thumbnail callbacks ────────────────────────────────────

/** Default callbacks for thumbnail action buttons in chat context. */
function _chatThumbnailOptions(job) {
    return {
        onRegenerate: (job, img) => {
            openGenerationOverlay({
                prompt: job.settings?.positive_prompt || '',
                chatId: job.chat_id,
                messageId: job.message_id,
                settings: job.settings
            });
        },
        onRefine: (job, img) => {
            const prompt = job.settings?.positive_prompt || '';
            if (prompt && typeof setRefineContext === 'function') {
                setRefineContext(prompt);
            }
        },
        onAttach: (job, img) => {
            if (typeof addAttachment === 'function') {
                addAttachment({
                    type: 'generated',
                    jobId: job.id,
                    imageId: img.id,
                    thumbnailUrl: `/api/generate/thumbnail/${job.id}/${img.id}`,
                    fullUrl: `/api/generate/image/${job.id}/${img.id}`,
                });
            }
        },
        onDelete: async (job, img, item) => {
            try {
                await API.deleteGeneratedImage(job.id, img.id);
                const grid = item.closest('.gen-thumbnail-grid');
                item.remove();
                if (grid) {
                    _updateBubbleStatusFromGrid(grid);
                    if (grid.querySelectorAll('.gen-thumbnail-item').length === 0) {
                        const bubble = grid.closest('.message.generation');
                        if (bubble) bubble.remove();
                    }
                }
            } catch (err) {
                console.error('Failed to delete image:', err);
            }
        },
    };
}

/** Default callbacks for failed card action buttons in chat context. */
function _chatFailedOptions(job) {
    return {
        onRegenerate: (job) => {
            openGenerationOverlay({
                prompt: job.settings?.positive_prompt || '',
                chatId: job.chat_id,
                messageId: job.message_id,
                settings: job.settings
            });
        },
        onDelete: async (job, item) => {
            try {
                await API.deleteGenerationJob(job.id);
                const grid = item.closest('.gen-thumbnail-grid');
                item.remove();
                if (grid) {
                    _updateBubbleStatusFromGrid(grid);
                    if (grid.querySelectorAll('.gen-thumbnail-item').length === 0) {
                        const bubble = grid.closest('.message.generation');
                        if (bubble) bubble.remove();
                    }
                }
            } catch (err) {
                console.error('Failed to delete job:', err);
            }
        },
    };
}

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
 * Store the last completed job's generation settings on the bubble element.
 * Used as defaults when opening the generation overlay from prompt blocks
 * after a page refresh (when _lastGenerationSettings is lost).
 */
function _storeLastJobSettings(bubble, job) {
    if (!bubble || !job || !job.settings) return;
    bubble._lastJobSettings = {
        base_model: job.settings.base_model || '',
        loras: job.settings.loras || [],
        output_folder: job.settings.output_folder || '',
    };
}

/**
 * Append grid items for a single job (thumbnails, failed card, or spinners).
 * Does NOT update the bubble header — caller is responsible for that.
 */
function _appendJobItems(grid, job) {
    if (job.status === 'completed' && job.images && job.images.length > 0) {
        job.images.forEach(img => {
            grid.appendChild(createThumbnailItem(job, img, _chatThumbnailOptions(job)));
        });
        // Store this completed job's settings on the bubble for later use as defaults
        const bubble = grid.closest('.message.generation');
        if (bubble) _storeLastJobSettings(bubble, job);
    } else if (job.status === 'failed') {
        grid.appendChild(createFailedCard(job, _chatFailedOptions(job)));
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
        item.innerHTML = createCircularProgress(0);
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

// ── Bubble status helper ─────────────────────────────────────────────────

/**
 * Update the bubble header status by scanning grid contents.
 * Wraps the shared getGridStatus() and applies it to the bubble DOM.
 */
function _updateBubbleStatusFromGrid(grid) {
    const bubble = grid.closest('.message.generation');
    if (!bubble) return;
    const statusEl = bubble.querySelector('.gen-bubble-status');
    if (!statusEl) return;

    const status = getGridStatus(grid);
    statusEl.textContent = status.statusText;
    statusEl.className = `gen-bubble-status gen-status-${status.phase}`;
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
        items[currentIdx].innerHTML = createCircularProgress(data.progress);
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

            // Capture the insertion point: the first spinner's position.
            // We insert new items before the first spinner, preserving
            // submission order even when jobs complete out of order.
            const insertBeforeRef = pendingItems[0];

            if (data.phase === 'completed' && job.images && job.images.length > 0) {
                job.images.forEach(img => {
                    grid.insertBefore(
                        createThumbnailItem(job, img, _chatThumbnailOptions(job)),
                        insertBeforeRef
                    );
                });
                // Store this completed job's settings on the bubble
                const bubble = grid.closest('.message.generation');
                if (bubble) _storeLastJobSettings(bubble, job);
            } else {
                grid.insertBefore(
                    createFailedCard(job, _chatFailedOptions(job)),
                    insertBeforeRef
                );
            }

            // Remove spinners for this job (after insertion to keep ref valid)
            pendingItems.forEach(item => item.remove());

            _updateBubbleStatusFromGrid(grid);
        } else if (!job && pendingItems.length > 0) {
            // Job not found in API — render a generic failed card
            const grid = pendingItems[0].closest('.gen-thumbnail-grid');
            const insertBeforeRef = pendingItems[0];
            const failedItem = document.createElement('div');
            failedItem.className = 'gen-thumbnail-item gen-thumbnail-failed';
            failedItem.dataset.jobId = data.job_id;
            failedItem.innerHTML = `<div class="gen-failed-content">
                <span class="gen-failed-icon">✕</span>
                <span class="gen-failed-text">Failed</span>
            </div>`;
            grid.insertBefore(failedItem, insertBeforeRef);
            pendingItems.forEach(item => item.remove());
            _updateBubbleStatusFromGrid(grid);
        }
    } catch (err) {
        console.error('Failed to fetch generation results:', err);
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
        grid.appendChild(createThumbnailItem(job, img, _chatThumbnailOptions(job)));
    });
    content.appendChild(grid);

    overlay.appendChild(content);
    document.body.appendChild(overlay);
}

// ── Retrieve stored settings for a message ──────────────────────────────

/**
 * Get the last generation settings stored on the bubble for a given message ID.
 * Searches for generation bubbles associated with the message and returns
 * the _lastJobSettings from the last one (most recent settings).
 * Returns null if no bubble or no stored settings found.
 * @param {string|number} messageId
 * @returns {Object|null} - { base_model, loras, output_folder } or null
 */
function getGenerationBubbleSettings(messageId) {
    if (!messageId) return null;
    const bubbles = document.querySelectorAll(
        `.message.generation[data-message-id="${messageId}"]`
    );
    // Walk backwards to find the last bubble with stored settings
    for (let i = bubbles.length - 1; i >= 0; i--) {
        if (bubbles[i]._lastJobSettings) {
            return bubbles[i]._lastJobSettings;
        }
    }
    return null;
}

/**
 * Get the most recent generation settings from ANY generation bubble in the chat.
 * Searches all generation bubbles in DOM order (last = most recent) and returns
 * the _lastJobSettings from the last one that has stored settings.
 * Used as a fallback when a new assistant message has suggested prompts but
 * the generated images are on an earlier message.
 * Works after page reload because loadGenerationBubbles() re-fetches jobs
 * from the API and _storeLastJobSettings() repopulates _lastJobSettings.
 * @returns {Object|null} - { base_model, loras, output_folder } or null
 */
function getLastChatGenerationSettings() {
    const bubbles = document.querySelectorAll('.message.generation');
    for (let i = bubbles.length - 1; i >= 0; i--) {
        if (bubbles[i]._lastJobSettings) {
            return bubbles[i]._lastJobSettings;
        }
    }
    return null;
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
