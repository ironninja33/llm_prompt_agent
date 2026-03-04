/**
 * Browser reorg — suggest and execute subfolder reorganization.
 *
 * Renders a "Suggest subfolders" button in the breadcrumb bar for any
 * non-root directory. On click, fetches cluster-based suggestions and
 * shows a reorg overlay with a cluster-count input and recluster button.
 *
 * Depends on: api.js, app.js, browserState.js, browserNav.js
 */

let _reorgOverlayEl = null;

/**
 * Check if the "Suggest subfolders" button should be shown and render it.
 * Called after loadBrowserContents() finishes.
 *
 * @param {string} virtualPath - current browser virtual path
 * @param {number} imageCount - total images in this directory
 * @param {boolean} hasSubdirs - whether the directory has subdirectories
 */
function maybeShowReorgButton(virtualPath, imageCount, hasSubdirs) {
    const actions = document.querySelector('#browser-breadcrumb .breadcrumb-actions');
    if (!actions) return;

    // Remove any existing reorg button
    const existing = actions.querySelector('.reorg-suggest-btn');
    if (existing) existing.remove();

    // Show for all non-root paths
    if (!virtualPath) return;

    const btn = document.createElement('button');
    btn.className = 'btn-small reorg-suggest-btn';
    btn.textContent = 'Suggest subfolders';
    btn.title = 'Organize images into subfolders based on content clusters';
    btn.onclick = () => _openReorgOverlay(virtualPath);
    actions.appendChild(btn);
}

/**
 * Extract the concept name from a virtual path.
 * Matches ingestion logic: concept is the first subdirectory within a root.
 */
function _conceptFromPath(virtualPath) {
    const parts = virtualPath.trim().replace(/^\/+|\/+$/g, '').split('/');
    return parts.length >= 2 ? parts[1] : parts[0];
}

async function _openReorgOverlay(virtualPath) {
    // Create overlay if needed
    if (!_reorgOverlayEl) {
        _reorgOverlayEl = document.createElement('div');
        _reorgOverlayEl.id = 'reorg-overlay';
        _reorgOverlayEl.className = 'modal hidden';
        _reorgOverlayEl.innerHTML = `
            <div class="modal-backdrop" onclick="_closeReorgOverlay()"></div>
            <div class="modal-dialog reorg-dialog">
                <div class="modal-header">
                    <h2>Organize into Subfolders</h2>
                    <button class="btn-icon" onclick="_closeReorgOverlay()">×</button>
                </div>
                <div class="modal-body">
                    <div class="reorg-controls">
                        <label for="reorg-k-input">Clusters:</label>
                        <input type="number" id="reorg-k-input" min="2" max="50" value="5">
                        <button class="btn-small" id="reorg-recluster-btn" onclick="_reclusterFolder()">Recluster</button>
                    </div>
                    <div id="reorg-content">
                        <div class="loading-text">Loading suggestions...</div>
                    </div>
                </div>
                <div class="reorg-footer" style="padding:12px 20px;border-top:1px solid var(--border);display:flex;justify-content:flex-end;gap:8px">
                    <button class="btn-small" onclick="_closeReorgOverlay()">Cancel</button>
                    <button class="btn-small btn-keep" id="reorg-execute-btn" style="display:none" onclick="_executeReorg()">Execute Reorg</button>
                </div>
                <div id="reorg-progress-blocker" class="reorg-blocker hidden">
                    <div class="reorg-blocker-content">
                        <div class="progress-bar-container">
                            <div id="reorg-progress-bar" class="progress-bar indeterminate"></div>
                        </div>
                        <p id="reorg-progress-message">Reclustering...</p>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(_reorgOverlayEl);
    }

    _reorgOverlayEl.classList.remove('hidden');
    _reorgOverlayEl.dataset.path = virtualPath;

    const content = $('#reorg-content');
    content.innerHTML = '<div class="loading-text">Loading suggestions...</div>';
    $('#reorg-execute-btn').style.display = 'none';

    // Fetch per-folder k from settings, fall back to adaptive default
    const concept = _conceptFromPath(virtualPath);
    try {
        const allSettings = await API.getSettings();
        const perFolderK = allSettings[`cluster_k_intra:${concept}`];
        if (perFolderK) {
            $('#reorg-k-input').value = parseInt(perFolderK, 10);
        } else {
            // Server uses adaptive K at clustering time; show a reasonable default
            $('#reorg-k-input').value = 5;
        }
    } catch (e) {
        console.warn('Failed to load k setting:', e);
        $('#reorg-k-input').value = 5;
    }

    // Fetch suggestions
    await _loadReorgSuggestions(virtualPath);
}

async function _loadReorgSuggestions(virtualPath) {
    const content = $('#reorg-content');
    content.innerHTML = '<div class="loading-text">Loading suggestions...</div>';
    $('#reorg-execute-btn').style.display = 'none';

    try {
        const res = await fetch(`/api/browser/reorg/suggest?path=${encodeURIComponent(virtualPath)}`);
        const data = await res.json();

        if (data.error) {
            content.innerHTML = `<div class="error-text">${escapeHtml(data.error)}</div>`;
            return;
        }

        if (!data.subfolders || data.subfolders.length === 0) {
            content.innerHTML = '<div class="loading-text">No clusters found for this folder. Change the cluster count and click Recluster.</div>';
            return;
        }

        _renderReorgSuggestions(content, data.subfolders);
        $('#reorg-execute-btn').style.display = 'inline-block';
    } catch (e) {
        console.error('Failed to load reorg suggestions:', e);
        content.innerHTML = '<div class="error-text">Failed to load suggestions.</div>';
    }
}

function _closeReorgOverlay() {
    if (_reorgOverlayEl) _reorgOverlayEl.classList.add('hidden');
}

function _renderReorgSuggestions(container, subfolders) {
    container.innerHTML = '';

    const desc = document.createElement('p');
    desc.style.cssText = 'font-size:0.85rem;color:var(--text-secondary);margin:0 0 12px';
    desc.textContent = `${subfolders.length} proposed subfolders. Click a name to rename it.`;
    container.appendChild(desc);

    for (const sf of subfolders) {
        const card = document.createElement('div');
        card.className = 'reorg-subfolder-card';
        card.dataset.originalName = sf.name;

        // Header: editable name + image count
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:8px';

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.value = sf.name;
        nameInput.className = 'reorg-name-input';
        nameInput.style.cssText = 'background:var(--bg-input);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text-primary);font-size:0.85rem;font-family:var(--font-mono);width:200px';
        nameInput.title = `Original cluster label: ${sf.label}`;
        header.appendChild(nameInput);

        const countSpan = document.createElement('span');
        countSpan.style.cssText = 'font-size:0.78rem;color:var(--text-muted)';
        countSpan.textContent = `${sf.image_count} images`;
        header.appendChild(countSpan);
        card.appendChild(header);

        // Preview thumbnails
        if (sf.sample_previews && sf.sample_previews.length > 0) {
            const previews = document.createElement('div');
            previews.style.cssText = 'display:flex;gap:4px';
            for (const p of sf.sample_previews) {
                const img = document.createElement('img');
                img.src = `/api/generate/thumbnail/${p.job_id}/${p.image_id}`;
                img.style.cssText = 'width:60px;height:60px;object-fit:cover;border-radius:4px';
                img.loading = 'lazy';
                img.alt = '';
                previews.appendChild(img);
            }
            card.appendChild(previews);
        }

        // Store image IDs for execution
        card._imageIds = sf.image_ids;

        container.appendChild(card);
    }
}

async function _reclusterFolder() {
    const path = _reorgOverlayEl?.dataset.path;
    if (!path) return;

    const kInput = $('#reorg-k-input');
    const k = parseInt(kInput.value, 10);
    if (!k || k < 2) {
        alert('Cluster count must be at least 2.');
        return;
    }

    // Show progress blocker
    const blocker = $('#reorg-progress-blocker');
    const progressBar = $('#reorg-progress-bar');
    const progressMsg = $('#reorg-progress-message');
    const reclusterBtn = $('#reorg-recluster-btn');

    blocker.classList.remove('hidden');
    progressBar.className = 'progress-bar indeterminate';
    progressMsg.textContent = 'Reclustering...';
    reclusterBtn.disabled = true;

    try {
        // Trigger recluster
        const res = await fetch('/api/browser/reorg/recluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, k }),
        });
        const data = await res.json();

        if (data.error) {
            blocker.classList.add('hidden');
            reclusterBtn.disabled = false;
            alert(`Recluster failed: ${data.error}`);
            return;
        }

        // Monitor clustering progress via SSE
        const evtSource = new EventSource('/api/clustering/status');

        evtSource.addEventListener('clustering_status', (e) => {
            const status = JSON.parse(e.data);
            progressMsg.textContent = status.message || 'Reclustering...';
            if (status.total && status.total > 0) {
                progressBar.classList.remove('indeterminate');
                const pct = Math.round((status.current / status.total) * 100);
                progressBar.style.width = pct + '%';
            }
        });

        evtSource.addEventListener('clustering_complete', async (e) => {
            evtSource.close();
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';
            progressMsg.textContent = 'Done! Loading suggestions...';

            // Re-fetch suggestions
            await _loadReorgSuggestions(path);

            blocker.classList.add('hidden');
            reclusterBtn.disabled = false;
        });

        evtSource.onerror = () => {
            evtSource.close();
            blocker.classList.add('hidden');
            reclusterBtn.disabled = false;
        };

    } catch (e) {
        console.error('Recluster failed:', e);
        blocker.classList.add('hidden');
        reclusterBtn.disabled = false;
        alert('Recluster failed. Check console for details.');
    }
}

async function _executeReorg() {
    const path = _reorgOverlayEl?.dataset.path;
    if (!path) return;

    const cards = document.querySelectorAll('.reorg-subfolder-card');
    const subfolders = [];

    for (const card of cards) {
        const input = card.querySelector('.reorg-name-input');
        const name = input?.value?.trim();
        if (!name || !card._imageIds || card._imageIds.length === 0) continue;
        subfolders.push({ name, image_ids: card._imageIds });
    }

    if (subfolders.length === 0) return;

    if (!confirm(`Move images into ${subfolders.length} subfolders? Files will be physically moved on disk.`)) return;

    const btn = $('#reorg-execute-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Moving...';
    }

    try {
        const res = await fetch('/api/browser/reorg/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, subfolders }),
        });
        const result = await res.json();

        if (result.error) {
            alert(`Error: ${result.error}`);
            return;
        }

        alert(`Moved ${result.moved} images. ${result.errors?.length || 0} errors.`);
        _closeReorgOverlay();

        // Refresh browser view
        await loadBrowserContents();

    } catch (e) {
        console.error('Reorg failed:', e);
        alert('Reorg failed. Check console for details.');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Execute Reorg';
        }
    }
}
