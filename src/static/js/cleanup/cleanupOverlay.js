/**
 * Cleanup overlay — main controller for the cleanup assistant.
 *
 * Manages open/close, parse trigger, wave state, and coordinates
 * sidebar + grid + near-dupes modules.
 *
 * Depends on: api.js ($, $$), cleanupGrid.js, cleanupSidebar.js, multiSelect.js
 */

// ── State ────────────────────────────────────────────────────────
let _cleanupOpen = false;
let _cleanupCurrentFolder = null;
let _cleanupCurrentWave = 1;
let _cleanupSortOrder = 'desc';  // 'asc' or 'desc'
// Cleanup polling is handled by PollManager (no local timers needed)

// ── Acted-upon group tracking (sessionStorage) ──────────────────
const ACTED_GROUPS_KEY = 'cleanup_acted_groups';

function _groupKey(imageIds) {
    return imageIds.slice().sort((a, b) => a - b).join(',');
}

function _groupsFingerprint(dupeGroups) {
    const keys = dupeGroups.map(g => _groupKey(g.image_ids));
    keys.sort();
    let hash = 0;
    const str = keys.join('|');
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash.toString(36);
}

function _loadActedGroups() {
    try {
        const raw = sessionStorage.getItem(ACTED_GROUPS_KEY);
        const state = raw ? JSON.parse(raw) : {};
        return { fingerprint: state.fingerprint || '', groups: state.groups || [], visited: state.visited || [] };
    } catch { return { fingerprint: '', groups: [], visited: [] }; }
}

function _saveActedGroups(state) {
    sessionStorage.setItem(ACTED_GROUPS_KEY, JSON.stringify(state));
}

function _markGroupActed(groupKey) {
    const state = _loadActedGroups();
    if (!state.groups.includes(groupKey)) {
        state.groups.push(groupKey);
        _saveActedGroups(state);
    }
}

function _markGroupVisited(groupKey) {
    const state = _loadActedGroups();
    if (!state.visited.includes(groupKey) && !state.groups.includes(groupKey)) {
        state.visited.push(groupKey);
        _saveActedGroups(state);
    }
}

function _isGroupActed(groupKey) {
    const state = _loadActedGroups();
    return state.groups.includes(groupKey);
}

function _isGroupVisited(groupKey) {
    const state = _loadActedGroups();
    return state.visited.includes(groupKey);
}

// ── Open / Close ─────────────────────────────────────────────────

async function openCleanupOverlay() {
    const modal = $('#cleanup-overlay');
    if (!modal) return;
    modal.classList.remove('hidden');
    _cleanupOpen = true;

    // Reset state
    _cleanupCurrentFolder = null;
    _cleanupCurrentWave = 1;

    // Set active wave tab
    $$('.cleanup-wave-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.wave === '1');
    });

    // Check for pending images and block UI until parsed
    const needsParse = await _checkAndBlockForParse();
    if (needsParse) {
        // _checkAndBlockForParse already showed the blocker and will
        // call _onParseComplete when done. Don't load the grid yet.
        return;
    }

    // No pending images — load everything immediately
    await _loadCleanupContent();
}

async function _loadCleanupContent() {
    await Promise.all([
        loadCleanupSidebar(),
        loadCleanupGrid(),  // combined endpoint also returns wave counts
    ]);

    // Check if scoring is in progress
    _checkScoringStatus();

    // Wire up Score All + Sort buttons
    const scoreBtn = $('#cleanup-score-all-btn');
    if (scoreBtn) scoreBtn.onclick = _startScoring;
    const sortBtn = $('#cleanup-sort-btn');
    if (sortBtn) sortBtn.onclick = _toggleSortOrder;

    // Init multi-select on grid
    _initCleanupMultiSelect();
}

function closeCleanupOverlay() {
    const modal = $('#cleanup-overlay');
    if (!modal) return;
    modal.classList.add('hidden');
    _cleanupOpen = false;

    // Unregister poll handlers
    PollManager.unregister('cleanup_parse');
    PollManager.unregister('cleanup_scoring');
}

// ── Parse (blocking) ─────────────────────────────────────────────

/**
 * Check for pending images. If any exist, show the full-screen blocker,
 * trigger a parse, and poll until done. Returns true if a parse was needed.
 */
async function _checkAndBlockForParse() {
    try {
        const res = await fetch('/api/cleanup/parse-status');
        const status = await res.json();

        if (status.running) {
            _showParseBlocker(status);
            _startParsePolling();
            return true;
        }

        if (status.total > 0) {
            // There are pending images — trigger parse
            _showParseBlocker({ running: true, parsed: 0, total: status.total });
            await fetch('/api/cleanup/parse', { method: 'POST' });
            _startParsePolling();
            return true;
        }

        // Nothing to parse
        _hideParseBlocker();
        return false;
    } catch (e) {
        console.error('Failed to check parse status:', e);
        _hideParseBlocker();
        return false;
    }
}

function _showParseBlocker(status) {
    const blocker = $('#cleanup-parse-blocker');
    if (!blocker) return;
    blocker.classList.remove('hidden');

    const fill = $('#cleanup-parse-blocker-fill');
    const label = $('#cleanup-parse-blocker-label');
    const title = $('#cleanup-parse-blocker-title');

    const pct = status.total > 0 ? (status.parsed / status.total * 100) : 0;
    if (fill) fill.style.width = `${pct}%`;
    if (label) label.textContent = `${status.parsed} / ${status.total} images`;
    if (title) title.textContent = status.parsed > 0
        ? 'Parsing image metadata...'
        : 'Indexing images for cleanup...';
}

function _hideParseBlocker() {
    const blocker = $('#cleanup-parse-blocker');
    if (blocker) blocker.classList.add('hidden');
}

function _startParsePolling() {
    PollManager.register('cleanup_parse', (data) => {
        if (!_cleanupOpen) {
            PollManager.unregister('cleanup_parse');
            return;
        }
        if (data.running) {
            _showParseBlocker(data);
        } else {
            PollManager.unregister('cleanup_parse');
            _hideParseBlocker();
            _loadCleanupContent();
        }
    });
}

// ── Scoring (ImageReward) ─────────────────────────────────────────

async function _checkScoringStatus() {
    try {
        const res = await fetch('/api/cleanup/scoring-progress');
        const data = await res.json();
        if (data.running) {
            _showScoringBlocker(data);
            _startScoringPolling();
        }
    } catch (e) {
        console.error('Failed to check scoring status:', e);
    }
}

async function _startScoring() {
    try {
        const res = await fetch('/api/cleanup/score', { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            console.error('Scoring error:', data.error);
            return;
        }
        _showScoringBlocker({ phase: 'loading_model', scored: 0, total: 0 });
        _startScoringPolling();
    } catch (e) {
        console.error('Failed to start scoring:', e);
    }
}

function _showScoringBlocker(progress) {
    const blocker = $('#cleanup-scoring-blocker');
    if (!blocker) return;
    blocker.classList.remove('hidden');

    const fill = $('#cleanup-scoring-blocker-fill');
    const label = $('#cleanup-scoring-blocker-label');
    const title = $('#cleanup-scoring-blocker-title');

    const pct = progress.total > 0 ? (progress.scored / progress.total * 100) : 0;
    if (fill) fill.style.width = `${pct}%`;

    let titleText = 'Scoring images...';
    if (progress.phase === 'loading_model') titleText = 'Loading ImageReward model...';
    else if (progress.phase === 'unloading') titleText = 'Unloading model...';
    else if (progress.phase === 'scoring') titleText = `Scoring: ${progress.current_folder || ''}`;

    if (title) title.textContent = titleText;
    if (label) label.textContent = progress.total > 0
        ? `${progress.scored} / ${progress.total} images`
        : 'Preparing...';
}

function _hideScoringBlocker() {
    const blocker = $('#cleanup-scoring-blocker');
    if (blocker) blocker.classList.add('hidden');
}

function _startScoringPolling() {
    PollManager.register('cleanup_scoring', (data) => {
        if (!_cleanupOpen) {
            PollManager.unregister('cleanup_scoring');
            return;
        }
        if (data.running) {
            _showScoringBlocker(data);
        } else {
            PollManager.unregister('cleanup_scoring');
            _hideScoringBlocker();
            if (data.phase === 'completed') {
                loadCleanupGrid();
            }
        }
    });
}

// ── Sort ──────────────────────────────────────────────────────────

function _toggleSortOrder() {
    _cleanupSortOrder = _cleanupSortOrder === 'desc' ? 'asc' : 'desc';
    const btn = $('#cleanup-sort-btn');
    if (btn) btn.innerHTML = _cleanupSortOrder === 'desc' ? 'Score &#9660;' : 'Score &#9650;';
    loadCleanupGrid();
}

// ── Wave switching ───────────────────────────────────────────────

function switchCleanupWave(wave) {
    const tab = document.querySelector(`.cleanup-wave-tab[data-wave="${wave}"]`);
    if (tab && tab.classList.contains('locked')) return;

    _cleanupCurrentWave = wave;
    $$('.cleanup-wave-tab').forEach(t => {
        t.classList.toggle('active', parseInt(t.dataset.wave) === wave);
    });
    loadCleanupGrid();
}

// ── Multi-select (legacy — kept for compatibility) ──────────────
// The three-state dupe tagging system replaces multi-select for cleanup.

function _initCleanupMultiSelect() {}
function cleanupSelectAll() {}
function cleanupDeselectAll() {}

// ── Actions ──────────────────────────────────────────────────────
// All delete/keep actions are now handled per-group via _dupeDeleteTagged.
// These stubs kept for any remaining HTML references.

function _updateCleanupFreed(freedBytes) {
    const el = $('#cleanup-freed-info');
    if (!el) return;
    const currentFreed = parseFloat(el.dataset.freed || '0');
    const newFreed = currentFreed + freedBytes;
    el.dataset.freed = newFreed;
    const mb = (newFreed / (1024 * 1024)).toFixed(1);
    el.textContent = `${mb} MB freed this session`;
}

// ── Delete blocker ───────────────────────────────────────────────

function _showDeleteBlocker(deleted, total, freedBytes) {
    const blocker = $('#cleanup-delete-blocker');
    if (!blocker) return;
    blocker.classList.remove('hidden');

    const fill = $('#cleanup-delete-blocker-fill');
    const label = $('#cleanup-delete-blocker-label');

    const pct = total > 0 ? (deleted / total * 100) : 0;
    if (fill) fill.style.width = `${pct}%`;

    const mb = (freedBytes / (1024 * 1024)).toFixed(1);
    if (label) {
        label.textContent = deleted > 0
            ? `${deleted} / ${total} images — ${mb} MB freed`
            : `0 / ${total} images`;
    }
}

function _hideDeleteBlocker() {
    const blocker = $('#cleanup-delete-blocker');
    if (blocker) blocker.classList.add('hidden');
}

/**
 * Delete images in chunks with a blocking progress overlay.
 * Returns aggregate {deleted_count, freed_bytes, errors}.
 */
async function _deleteInChunks(imageIds) {
    const total = imageIds.length;
    const chunkSize = 20;
    let deletedCount = 0;
    let freedBytes = 0;
    const errors = [];

    _showDeleteBlocker(0, total, 0);

    for (let i = 0; i < total; i += chunkSize) {
        const chunk = imageIds.slice(i, i + chunkSize);
        try {
            const res = await fetch('/api/cleanup/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_ids: chunk }),
            });
            const result = await res.json();
            deletedCount += result.deleted_count || 0;
            freedBytes += result.freed_bytes || 0;
            if (result.errors) errors.push(...result.errors);
        } catch (e) {
            console.error('Delete chunk failed:', e);
            errors.push({ error: e.message });
        }
        _showDeleteBlocker(deletedCount, total, freedBytes);
    }

    _hideDeleteBlocker();
    return { deleted_count: deletedCount, freed_bytes: freedBytes, errors };
}

// ── Grid loading ─────────────────────────────────────────────────

async function loadCleanupGrid() {
    const grid = $('#cleanup-grid');
    if (!grid) return;

    grid.innerHTML = '<div class="loading-text">Loading...</div>';

    // Single combined call: computes dupe detection once, returns
    // triage images, near-dupe groups, and wave counts together.
    const triageData = await _fetchTriageData();
    console.log('[cleanup] triageData:', triageData);
    if (!triageData) {
        grid.innerHTML = '<div class="loading-text">Failed to load data.</div>';
        return;
    }

    const images = triageData.images || [];
    const dupeGroups = triageData.dupe_groups || [];
    console.log(`[cleanup] wave=${_cleanupCurrentWave} images=${images.length} dupeGroups=${dupeGroups.length}`);

    // Update wave count badges from combined response
    if (triageData.wave_counts) {
        for (const [wave, count] of Object.entries(triageData.wave_counts)) {
            const el = $(`#cleanup-wave${wave}-count`);
            if (el) el.textContent = count;
        }
    }

    grid.innerHTML = '';

    // Wave group: show ALL wave images (option A — don't filter out dupe members)
    if (images.length > 0) {
        const card = _buildWaveGroup(images);
        grid.appendChild(card);
        console.log(`[cleanup] built wave group with ${images.length} images`);
    }

    // Render near-dupe groups below the wave group
    const dupeCount = _renderNearDupes(dupeGroups);

    // Update total info
    const totalEl = $('#cleanup-total-info');
    if (totalEl) {
        totalEl.textContent = `${images.length} images in Wave ${_cleanupCurrentWave}`;
    }

    // Show empty state only if no triage images AND no near-dupe groups
    if (images.length === 0 && !dupeCount) {
        grid.innerHTML = '<div class="loading-text">No images in this wave.</div>';
    }
}

async function _fetchTriageData() {
    try {
        const params = new URLSearchParams();
        if (_cleanupCurrentFolder) params.set('folder', _cleanupCurrentFolder);
        params.set('wave', _cleanupCurrentWave);
        params.set('sort', _cleanupSortOrder);

        const res = await fetch(`/api/cleanup/triage-data?${params}`);
        return await res.json();
    } catch (e) {
        console.error('Failed to load triage data:', e);
        return null;
    }
}

function _buildWaveGroup(images) {
    const card = document.createElement('div');
    card.className = 'cleanup-dupe-group cleanup-wave-group';

    const header = document.createElement('div');
    header.className = 'cleanup-dupe-header';
    header.innerHTML = `<span>Wave ${_cleanupCurrentWave} Images (${images.length})</span><span class="cleanup-scroll-pct"></span>`;

    const body = document.createElement('div');
    body.className = 'cleanup-dupe-body visible';
    body._members = images.map(img => ({
        image_id: img.image_id,
        job_id: img.job_id,
        file_path: img.file_path,
        file_size: img.file_size,
        keep_score: img.keep_score,
    }));
    body._bestPickId = null;  // no best pick for wave groups
    body._originalImageIds = images.map(img => img.image_id);
    body._shownCount = 0;

    // Check if this group was previously acted upon
    if (_isGroupActed(_groupKey(body._originalImageIds))) {
        card.classList.add('cleanup-group-acted');
    }

    // Action bar
    const actionBar = document.createElement('div');
    actionBar.className = 'dupe-group-actions visible';
    actionBar.innerHTML = `
        <span class="dupe-group-status"></span>
        <div class="dupe-group-btns">
            <button class="btn-small btn-danger dupe-delete-all-btn">Delete All</button>
            <button class="btn-small dupe-delete-tagged-btn" disabled>Delete Tagged</button>
            <button class="btn-small dupe-mark-rest-btn">Mark Rest as Delete</button>
            <button class="btn-small dupe-clear-btn">Clear Tags</button>
        </div>
    `;
    actionBar.querySelector('.dupe-delete-all-btn').onclick = () => _dupeDeleteAll(body);
    actionBar.querySelector('.dupe-delete-tagged-btn').onclick = () => _dupeDeleteTagged(body);
    actionBar.querySelector('.dupe-mark-rest-btn').onclick = () => _dupeMarkRestAsDelete(body);
    actionBar.querySelector('.dupe-clear-btn').onclick = () => _dupeClearTags(body);

    // Header toggles collapse (starts expanded); accordion closes others
    header.onclick = () => {
        _closeOtherGroups(body);
        body.classList.toggle('visible');
        actionBar.classList.toggle('visible', body.classList.contains('visible'));
    };

    // Assemble DOM tree before populating (functions traverse parentElement)
    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(actionBar);

    // Populate first batch + scroll
    _appendDupeMembers(body, 20);
    _setupDupeScroll(body);
    _updateDupeGroupStatus(body);

    return card;
}

// ── Accordion: close other groups ─────────────────────────────────

function _closeOtherGroups(exceptBody) {
    const allBodies = document.querySelectorAll('.cleanup-dupe-body.visible');
    allBodies.forEach(b => {
        if (b === exceptBody) return;
        b.classList.remove('visible');
        const actionBar = b.parentElement.querySelector('.dupe-group-actions');
        if (actionBar) actionBar.classList.remove('visible');
    });
}

// ── Near-duplicates ──────────────────────────────────────────────

function _appendDupeMembers(body, count) {
    const members = body._members;
    const start = body._shownCount;
    const end = Math.min(start + count, members.length);

    for (let i = start; i < end; i++) {
        const m = members[i];
        const job = { id: m.job_id };
        const imgObj = {
            id: m.image_id,
            file_path: m.file_path,
            file_size: m.file_size,
        };

        const item = createThumbnailItem(job, imgObj, {
            onRegenerate: null,
            onRefine: null,
            onAttach: null,
            onDelete: null,
        });

        item.dataset.imageId = m.image_id;

        // Disable full-size viewer — clicks cycle through tag states
        const thumb = item.querySelector('.gen-thumbnail-img');
        if (thumb) thumb.onclick = (e) => {
            e.stopPropagation();
            _cycleDupeTag(item, body);
        };

        // Pre-tag best pick as keep (dupe groups only)
        if (m.image_id === body._bestPickId) {
            item.dataset.dupeTag = 'keep';
            item.classList.add('dupe-keep');
        }

        // Add score badge (keep_score for wave groups, quality_overall for dupe groups)
        const scoreVal = m.keep_score ?? m.quality_overall;
        if (scoreVal != null) {
            const badge = document.createElement('span');
            badge.className = 'cleanup-score-badge';
            if (scoreVal < 0.55) badge.classList.add('score-low');
            else if (scoreVal < 0.70) badge.classList.add('score-mid');
            else badge.classList.add('score-high');
            badge.textContent = scoreVal.toFixed(2);
            badge.title = m.keep_score != null
                ? `Keep score: ${m.keep_score.toFixed(4)}`
                : `Quality: ${scoreVal.toFixed(4)}`;
            item.appendChild(badge);
        }

        body.appendChild(item);
    }

    body._shownCount = end;
}

/** Cycle an item through neutral → keep → delete → neutral */
function _cycleDupeTag(item, body) {
    const current = item.dataset.dupeTag || 'neutral';
    let next;
    if (current === 'neutral') next = 'keep';
    else if (current === 'keep') next = 'delete';
    else next = 'neutral';

    item.classList.remove('dupe-keep', 'dupe-delete');
    item.dataset.dupeTag = next;
    if (next === 'keep') item.classList.add('dupe-keep');
    else if (next === 'delete') item.classList.add('dupe-delete');

    _updateDupeGroupStatus(body);
}

/** Update the status line in a dupe group's action bar */
function _updateDupeGroupStatus(body) {
    const statusEl = body.parentElement.querySelector('.dupe-group-status');
    if (!statusEl) return;

    const items = body.querySelectorAll('.gen-thumbnail-item');
    let keeps = 0, deletes = 0, neutral = 0;
    items.forEach(item => {
        const tag = item.dataset.dupeTag || 'neutral';
        if (tag === 'keep') keeps++;
        else if (tag === 'delete') deletes++;
        else neutral++;
    });

    // Also count unloaded members as neutral
    neutral += body._members.length - body._shownCount;

    const parts = [];
    if (keeps) parts.push(`${keeps} kept`);
    if (deletes) parts.push(`${deletes} to delete`);
    if (neutral) parts.push(`${neutral} untagged`);
    statusEl.textContent = parts.join(', ');

    // Enable/disable delete button
    const deleteBtn = body.parentElement.querySelector('.dupe-delete-tagged-btn');
    if (deleteBtn) deleteBtn.disabled = deletes === 0;
}

/** Mark all neutral (untagged) items in a dupe group as delete */
function _dupeMarkRestAsDelete(body) {
    // Load all remaining members first
    if (body._shownCount < body._members.length) {
        _appendDupeMembers(body, body._members.length - body._shownCount);
    }
    body.querySelectorAll('.gen-thumbnail-item').forEach(item => {
        const tag = item.dataset.dupeTag || 'neutral';
        if (tag === 'neutral') {
            item.dataset.dupeTag = 'delete';
            item.classList.add('dupe-delete');
        }
    });
    _updateDupeGroupStatus(body);
}

/** Clear all tags in a dupe group back to neutral (re-tag best pick as keep) */
function _dupeClearTags(body) {
    body.querySelectorAll('.gen-thumbnail-item').forEach(item => {
        item.dataset.dupeTag = 'neutral';
        item.classList.remove('dupe-keep', 'dupe-delete');
    });
    // Re-tag best pick
    const bestItem = body.querySelector(`[data-image-id="${body._bestPickId}"]`);
    if (bestItem) {
        bestItem.dataset.dupeTag = 'keep';
        bestItem.classList.add('dupe-keep');
    }
    _updateDupeGroupStatus(body);
}

/** Delete ALL images in a group */
async function _dupeDeleteAll(body) {
    const total = body._members.length;
    if (total === 0) return;

    if (!confirm(`Delete ALL ${total} image${total > 1 ? 's' : ''} in this group? This cannot be undone.`)) return;

    const ids = body._members.map(m => m.image_id);

    try {
        const result = await _deleteInChunks(ids);

        _updateCleanupFreed(result.freed_bytes);

        // Mark group as acted upon
        if (body._originalImageIds) {
            _markGroupActed(_groupKey(body._originalImageIds));
        }

        // Remove the entire group card
        const isWaveGroup = body.parentElement.classList.contains('cleanup-wave-group');
        if (isWaveGroup) {
            body.innerHTML = '';
            body._members = [];
            body._shownCount = 0;
            const header = body.parentElement.querySelector('.cleanup-dupe-header span');
            if (header) header.textContent = `Wave ${_cleanupCurrentWave} Images (0)`;
        } else {
            body.parentElement.remove();
        }

        await loadCleanupWaveCounts();
        await loadCleanupSidebar();
    } catch (e) {
        console.error('Delete all failed:', e);
    }
}

/** Delete all items tagged as 'delete' in a dupe group */
async function _dupeDeleteTagged(body) {
    const items = body.querySelectorAll('.gen-thumbnail-item[data-dupe-tag="delete"]');
    if (items.length === 0) return;

    const ids = Array.from(items).map(el => parseInt(el.dataset.imageId, 10));
    if (!confirm(`Delete ${ids.length} image${ids.length > 1 ? 's' : ''} from this group? This cannot be undone.`)) return;

    try {
        const result = await _deleteInChunks(ids);

        // Remove deleted items from DOM
        items.forEach(el => el.remove());

        // Remove from members array
        const deletedSet = new Set(ids);
        body._members = body._members.filter(m => !deletedSet.has(m.image_id));
        body._shownCount = body.querySelectorAll('.gen-thumbnail-item').length;

        _updateDupeGroupStatus(body);
        _updateCleanupFreed(result.freed_bytes);

        // Mark group as acted upon
        if (body._originalImageIds) {
            _markGroupActed(_groupKey(body._originalImageIds));
            body.parentElement.classList.add('cleanup-group-acted');
        }

        // Update header count
        const header = body.parentElement.querySelector('.cleanup-dupe-header span');
        const isWaveGroup = body.parentElement.classList.contains('cleanup-wave-group');
        if (header) {
            if (isWaveGroup) {
                header.textContent = `Wave ${_cleanupCurrentWave} Images (${body._members.length})`;
            } else {
                header.textContent = `${body._members.length} images`;
            }
        }

        // Remove dupe groups with < 2 members; wave groups can go to 0
        if (body._members.length < 2 && !isWaveGroup) {
            body.parentElement.remove();
        }

        await loadCleanupWaveCounts();
        await loadCleanupSidebar();
    } catch (e) {
        console.error('Dupe delete failed:', e);
    }
}

function _setupDupeScroll(body) {
    body.addEventListener('scroll', () => {
        // Update scroll percentage indicator (relative to total members, not just loaded)
        const pctEl = body.parentElement.querySelector('.cleanup-scroll-pct');
        if (pctEl) {
            const maxScroll = body.scrollHeight - body.clientHeight;
            const total = body._members.length;
            if (maxScroll > 0 && total > 1) {
                const scrollFraction = body.scrollTop / maxScroll;
                const viewedIndex = Math.round(scrollFraction * (body._shownCount - 1));
                const pct = Math.round((viewedIndex / (total - 1)) * 100);
                pctEl.textContent = pct > 0 ? `${pct}%` : '';
            } else {
                pctEl.textContent = '';
            }
        }

        // Lazy-load next batch when near bottom
        if (body._dupeLoading) return;
        if (body._shownCount >= body._members.length) return;
        const remaining = body.scrollHeight - body.scrollTop - body.clientHeight;
        if (remaining < 200) {
            body._dupeLoading = true;
            _appendDupeMembers(body, 20);
            // Let the layout settle before allowing another batch,
            // otherwise the reflow from appending re-triggers scroll
            // and cascades until everything is loaded.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => { body._dupeLoading = false; });
            });
        }
    }, { passive: true });
}

/**
 * Render near-duplicate groups from pre-fetched data.
 * Returns the number of groups rendered.
 */
function _renderNearDupes(groups) {
    const container = $('#cleanup-near-dupes');
    if (!container) return 0;

    if (!groups || groups.length === 0) {
        container.classList.add('hidden');
        return 0;
    }

    container.classList.remove('hidden');

    // Validate acted-groups fingerprint; clear if groups changed
    const fingerprint = _groupsFingerprint(groups);
    const actedState = _loadActedGroups();
    if (actedState.fingerprint !== fingerprint) {
        _saveActedGroups({ fingerprint, groups: [], visited: [] });
    }

    const totalDupeImages = groups.reduce((sum, g) => sum + g.image_ids.length, 0);
    container.innerHTML = `<h3 style="font-size:0.88rem;color:var(--text-secondary);margin:0 0 8px">Near-Duplicate Groups (${groups.length}) — ${totalDupeImages} images</h3>`;

    for (const group of groups) {
        const card = document.createElement('div');
        card.className = 'cleanup-dupe-group';

        const header = document.createElement('div');
        header.className = 'cleanup-dupe-header';
        header.innerHTML = `
            <span>${group.image_ids.length} images</span>
            <span class="cleanup-scroll-pct"></span>
            <div class="cleanup-dupe-folders">
                ${group.folders.map(f => `<span class="cleanup-dupe-folder-badge">${escapeHtml(f)}</span>`).join('')}
            </div>
        `;
        const body = document.createElement('div');
        body.className = 'cleanup-dupe-body';
        body._members = group.members || group.image_ids.map(id => ({ image_id: id, job_id: id }));
        body._bestPickId = group.best_pick_id;
        body._originalImageIds = group.image_ids.slice();
        body._shownCount = 0;

        // Check if this group was previously acted upon or visited
        const gKey = _groupKey(body._originalImageIds);
        if (_isGroupActed(gKey)) {
            card.classList.add('cleanup-group-acted');
        } else if (_isGroupVisited(gKey)) {
            card.classList.add('cleanup-group-visited');
        }

        // Action bar with status + buttons
        const actionBar = document.createElement('div');
        actionBar.className = 'dupe-group-actions';
        actionBar.innerHTML = `
            <span class="dupe-group-status"></span>
            <div class="dupe-group-btns">
                <button class="btn-small btn-danger dupe-delete-all-btn">Delete All</button>
                <button class="btn-small dupe-delete-tagged-btn" disabled>Delete Tagged</button>
                <button class="btn-small dupe-mark-rest-btn">Mark Rest as Delete</button>
                <button class="btn-small dupe-clear-btn">Clear Tags</button>
            </div>
        `;
        actionBar.querySelector('.dupe-delete-all-btn').onclick = () => _dupeDeleteAll(body);
        actionBar.querySelector('.dupe-delete-tagged-btn').onclick = () => _dupeDeleteTagged(body);
        actionBar.querySelector('.dupe-mark-rest-btn').onclick = () => _dupeMarkRestAsDelete(body);
        actionBar.querySelector('.dupe-clear-btn').onclick = () => _dupeClearTags(body);

        header.onclick = () => {
            // Populate first batch on first expand
            if (body._shownCount === 0) {
                _appendDupeMembers(body, 20);
                _setupDupeScroll(body);
                _updateDupeGroupStatus(body);
            }
            _closeOtherGroups(body);
            body.classList.toggle('visible');
            actionBar.classList.toggle('visible', body.classList.contains('visible'));
            // Mark as visited when expanded (if not already acted upon)
            if (body.classList.contains('visible') && !card.classList.contains('cleanup-group-acted')) {
                _markGroupVisited(_groupKey(body._originalImageIds));
                card.classList.add('cleanup-group-visited');
            }
        };
        card.appendChild(header);
        card.appendChild(body);
        card.appendChild(actionBar);
        container.appendChild(card);
    }
    return groups.length;
}

// ── Wave counts ──────────────────────────────────────────────────

async function loadCleanupWaveCounts() {
    try {
        const params = new URLSearchParams();
        if (_cleanupCurrentFolder) params.set('folder', _cleanupCurrentFolder);
        const res = await fetch(`/api/cleanup/wave-counts?${params}`);
        const data = await res.json();

        for (const [wave, count] of Object.entries(data.counts)) {
            const el = $(`#cleanup-wave${wave}-count`);
            if (el) el.textContent = count;
        }
    } catch (e) {
        console.error('Failed to load wave counts:', e);
    }
}
