/**
 * Full-size image viewer with generation settings sidebar.
 * Opens as a fullscreen overlay when clicking a thumbnail.
 *
 * Supports viewing multiple images from different generation jobs
 * (e.g. a merged bubble with images from several 1-image generations).
 */

let _viewerItems = [];       // Array of {job, img} entries
let _viewerImageIndex = 0;   // Current index into _viewerItems
let _viewerMissingSet = new Set();  // Track indices of missing images

/**
 * Open the full-size viewer.
 *
 * Accepts two calling conventions:
 *   1. openFullSizeViewer(items, initialIndex) — array of {job, img} + index
 *   2. openFullSizeViewer(job, img) — legacy single-job call (auto-wrapped)
 *
 * @param {Array|Object} itemsOrJob - Array of {job, img} or a single job object
 * @param {number|Object} initialIndexOrImg - Starting index or the specific image
 */
function openFullSizeViewer(itemsOrJob, initialIndexOrImg) {
    _viewerMissingSet = new Set();

    if (Array.isArray(itemsOrJob)) {
        // New-style: array of {job, img} entries + starting index
        _viewerItems = itemsOrJob;
        _viewerImageIndex = typeof initialIndexOrImg === 'number' ? initialIndexOrImg : 0;
    } else {
        // Legacy: single job + img object
        const job = itemsOrJob;
        const img = initialIndexOrImg;
        if (job.images && job.images.length > 1) {
            // Job has multiple images — build flat list from this job
            _viewerItems = job.images.map(i => ({ job, img: i }));
            _viewerImageIndex = job.images.findIndex(i => i.id === img.id);
            if (_viewerImageIndex < 0) _viewerImageIndex = 0;
        } else {
            // Single image
            _viewerItems = [{ job, img }];
            _viewerImageIndex = 0;
        }
    }

    if (_viewerItems.length === 0) return;

    const overlay = document.getElementById('fullsize-viewer');
    overlay.classList.remove('hidden');

    _renderViewerImage();
    _renderViewerSidebar();
    _updateViewerNavigation();

    // Add keyboard listener
    document.addEventListener('keydown', _viewerKeyHandler);
}

function closeFullSizeViewer() {
    const overlay = document.getElementById('fullsize-viewer');
    overlay.classList.add('hidden');
    _viewerItems = [];
    _viewerMissingSet = new Set();
    document.removeEventListener('keydown', _viewerKeyHandler);
}

function _viewerKeyHandler(e) {
    if (e.key === 'Escape') closeFullSizeViewer();
    if (e.key === 'ArrowLeft') viewerPrev();
    if (e.key === 'ArrowRight') viewerNext();
}

/**
 * Find the next valid (non-missing) image index in the given direction.
 * Returns -1 if none found.
 */
function _findNextValidIndex(fromIndex, direction) {
    if (_viewerItems.length === 0) return -1;
    let idx = fromIndex + direction;
    while (idx >= 0 && idx < _viewerItems.length) {
        if (!_viewerMissingSet.has(idx)) return idx;
        idx += direction;
    }
    return -1;
}

function viewerPrev() {
    const idx = _findNextValidIndex(_viewerImageIndex, -1);
    if (idx >= 0) {
        _viewerImageIndex = idx;
        _renderViewerImage();
        _renderViewerSidebar();
        _updateViewerNavigation();
    }
}

function viewerNext() {
    const idx = _findNextValidIndex(_viewerImageIndex, 1);
    if (idx >= 0) {
        _viewerImageIndex = idx;
        _renderViewerImage();
        _renderViewerSidebar();
        _updateViewerNavigation();
    }
}

function _renderViewerImage() {
    const imgEl = document.getElementById('viewer-image');
    if (!imgEl || _viewerItems.length === 0) return;

    const entry = _viewerItems[_viewerImageIndex];
    if (!entry) return;
    const { job, img } = entry;

    // Remove any previous placeholder
    const container = imgEl.parentElement;
    const existingPlaceholder = container?.querySelector('.viewer-missing-placeholder');
    if (existingPlaceholder) existingPlaceholder.remove();
    imgEl.classList.remove('hidden');

    // Hide old image while the new one loads to prevent stale flash
    imgEl.classList.add('hidden');
    imgEl.src = `/api/generate/image/${job.id}/${img.id}`;
    imgEl.alt = `Generated image ${_viewerImageIndex + 1}`;

    // Handle load failure — show placeholder in viewer
    imgEl.onerror = () => {
        _viewerMissingSet.add(_viewerImageIndex);
        _showViewerMissingPlaceholder(imgEl);
        _updateViewerNavigation();
    };

    // Show image once loaded; detect SVG placeholder returned by backend
    imgEl.onload = () => {
        imgEl.classList.remove('hidden');
        if (imgEl.naturalWidth === 256 && imgEl.naturalHeight === 256) {
            fetch(imgEl.src, { method: 'HEAD' }).then(res => {
                if (res.headers.get('content-type')?.includes('svg')) {
                    _viewerMissingSet.add(_viewerImageIndex);
                    _showViewerMissingPlaceholder(imgEl);
                    _updateViewerNavigation();
                }
            }).catch(() => {});
        }
    };

    // Update counter
    const counter = document.getElementById('viewer-counter');
    if (counter) {
        counter.textContent = `${_viewerImageIndex + 1} / ${_viewerItems.length}`;
    }
}

/**
 * Show a "missing image" placeholder in the viewer area.
 */
function _showViewerMissingPlaceholder(imgEl) {
    imgEl.classList.add('hidden');
    const container = imgEl.parentElement;
    if (!container) return;

    // Avoid duplicates
    if (container.querySelector('.viewer-missing-placeholder')) return;

    const placeholder = document.createElement('div');
    placeholder.className = 'viewer-missing-placeholder';
    placeholder.innerHTML = '<span class="viewer-missing-icon">✕</span>'
        + '<span class="viewer-missing-text">Image file not found</span>';
    container.appendChild(placeholder);
}

function _renderViewerSidebar() {
    const sidebar = document.getElementById('viewer-sidebar-content');
    if (!sidebar || _viewerItems.length === 0) return;

    const entry = _viewerItems[_viewerImageIndex];
    if (!entry) return;
    const s = entry.job.settings || {};

    // Build settings table
    let html = '<table class="viewer-settings-table">';

    // Positive prompt
    html += `<tr><td class="viewer-label">Prompt</td></tr>`;
    html += `<tr><td class="viewer-value viewer-prompt">${escapeHtml(s.positive_prompt || '')}</td></tr>`;

    // Negative prompt
    if (s.negative_prompt) {
        html += `<tr><td class="viewer-label">Negative</td></tr>`;
        html += `<tr><td class="viewer-value viewer-prompt-neg">${escapeHtml(s.negative_prompt)}</td></tr>`;
    }

    // Base model
    if (s.base_model) {
        html += `<tr><td class="viewer-label">Model</td></tr>`;
        html += `<tr><td class="viewer-value">${escapeHtml(s.base_model)}</td></tr>`;
    }

    // LoRAs
    let loras = s.loras;
    if (typeof loras === 'string') {
        try { loras = JSON.parse(loras); } catch(e) { loras = []; }
    }
    if (loras && loras.length > 0) {
        html += `<tr><td class="viewer-label">LoRAs</td></tr>`;
        const loraList = loras.map(l => {
            if (typeof l === 'object') return `${escapeHtml(l.name)} (${l.strength || 1.0})`;
            return escapeHtml(l);
        }).join('<br>');
        html += `<tr><td class="viewer-value">${loraList}</td></tr>`;
    }

    // Output folder
    if (s.output_folder) {
        html += `<tr><td class="viewer-label">Folder</td></tr>`;
        html += `<tr><td class="viewer-value">${escapeHtml(s.output_folder)}</td></tr>`;
    }

    // Seed
    html += `<tr><td class="viewer-label">Seed</td></tr>`;
    html += `<tr><td class="viewer-value">${s.seed != null ? s.seed : '-1'}</td></tr>`;

    // Number of images
    html += `<tr><td class="viewer-label">Images</td></tr>`;
    html += `<tr><td class="viewer-value">${s.num_images || 1}</td></tr>`;

    html += '</table>';
    sidebar.innerHTML = html;
}

function _updateViewerNavigation() {
    const prevBtn = document.getElementById('viewer-prev');
    const nextBtn = document.getElementById('viewer-next');
    if (prevBtn) prevBtn.disabled = _findNextValidIndex(_viewerImageIndex, -1) < 0;
    if (nextBtn) nextBtn.disabled = _findNextValidIndex(_viewerImageIndex, 1) < 0;
}
