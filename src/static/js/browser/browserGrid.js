/**
 * Browser grid rendering — directories and image thumbnails.
 *
 * Depends on: api.js, app.js, thumbnails.js, browserState.js, browserNav.js
 */

/**
 * Create a directory thumbnail card with 2x2 preview grid.
 */
function createDirectoryThumbnail(dir) {
    const card = document.createElement('div');
    card.className = 'browser-dir-card';
    card.onclick = () => navigateToPath(dir.name, dir.path);

    // 2x2 preview grid
    const preview = document.createElement('div');
    preview.className = 'browser-dir-preview';
    const previews = dir.previews || [];
    for (let i = 0; i < 4; i++) {
        if (i < previews.length) {
            const img = document.createElement('img');
            img.src = `/api/generate/thumbnail/${previews[i].job_id}/${previews[i].image_id}`;
            img.alt = '';
            img.loading = 'lazy';
            preview.appendChild(img);
        } else {
            const empty = document.createElement('div');
            empty.className = 'dir-preview-empty';
            preview.appendChild(empty);
        }
    }
    card.appendChild(preview);

    // Info below
    const info = document.createElement('div');
    info.className = 'browser-dir-info';
    const name = document.createElement('div');
    name.className = 'browser-dir-name';
    name.textContent = dir.display_name || dir.name;
    name.title = dir.name;
    info.appendChild(name);

    if (dir.category) {
        const badge = document.createElement('span');
        badge.className = 'category-badge';
        badge.textContent = dir.category;
        info.appendChild(badge);
    }

    const meta = document.createElement('div');
    meta.className = 'browser-dir-meta';
    const count = dir.image_count || 0;
    meta.textContent = count > 0 ? `${count} image${count !== 1 ? 's' : ''}` : 'Directory';
    info.appendChild(meta);

    card.appendChild(info);
    return card;
}

/**
 * Create a browser image thumbnail using the shared createThumbnailItem function.
 */
function createBrowserImageThumbnail(imageData) {
    // Build a job-like and img-like object for the shared thumbnail component
    const job = {
        id: imageData.job_id,
        status: imageData.status || 'completed',
        settings: imageData.settings || {},
    };
    const img = {
        id: imageData.id,
        filename: imageData.filename,
        file_path: imageData.file_path,
    };

    const item = createThumbnailItem(job, img, {
        onRegenerate: (j, i) => {
            // Open generation overlay with extracted settings
            if (typeof openGenerationOverlay === 'function') {
                openGenerationOverlay({
                    prompt: (j.settings || {}).positive_prompt || '',
                    settings: j.settings || {},
                });
            }
        },
        onRefine: (j, i) => {
            openRefineDialog(j, i);
        },
        onRefineWithAttachment: null,  // Attachment toggle lives in the refine dialog
        onAttach: null,  // Hidden in browser context
        onDelete: async (j, i, el) => {
            BrowserState.deletePending = true;
            try {
                const result = await API.deleteGeneratedImage(j.id, i.id);
                if (result.error) {
                    console.error('Failed to delete image:', result.error);
                    return;
                }
                // Remove the .browser-img-item wrapper so the grid reflows
                const wrapper = el.closest('.browser-img-item');
                (wrapper || el).remove();
            } catch (err) {
                console.error('Failed to delete image:', err);
            } finally {
                BrowserState.deletePending = false;
            }
        },
    });

    // Wrap in browser-specific container with extra info
    const wrapper = document.createElement('div');
    wrapper.className = 'browser-img-item';
    wrapper.appendChild(item);

    // Filename
    const info = document.createElement('div');
    info.className = 'browser-img-info';
    info.textContent = imageData.filename || '';
    info.title = imageData.file_path || imageData.filename || '';
    wrapper.appendChild(info);

    // File size + creation date
    const meta = document.createElement('div');
    meta.className = 'browser-img-meta';

    const sizeEl = document.createElement('span');
    if (imageData.file_size) {
        const kb = Math.round(imageData.file_size / 1024);
        sizeEl.textContent = kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb} KB`;
    }
    meta.appendChild(sizeEl);

    const dateEl = document.createElement('span');
    if (imageData.created_at) {
        const dt = new Date(imageData.created_at.replace(' ', 'T') + 'Z');
        if (!isNaN(dt)) {
            dateEl.textContent = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
        }
    }
    meta.appendChild(dateEl);

    wrapper.appendChild(meta);

    return wrapper;
}

/**
 * Full render into #browser-grid.
 */
function renderBrowserGrid(directories, images) {
    const grid = $('#browser-grid');
    if (!grid) return;
    grid.innerHTML = '';
    grid.classList.add('gen-thumbnail-grid');

    const fragment = document.createDocumentFragment();

    // Render directories first
    if (directories && directories.length > 0) {
        directories.forEach(dir => {
            fragment.appendChild(createDirectoryThumbnail(dir));
        });
    }

    // Render images
    if (images && images.length > 0) {
        images.forEach(imageData => {
            fragment.appendChild(createBrowserImageThumbnail(imageData));
        });
    }

    grid.appendChild(fragment);

    // Show empty state if nothing
    const emptyEl = $('#browser-empty');
    if (emptyEl) {
        if ((!directories || directories.length === 0) && (!images || images.length === 0)) {
            emptyEl.classList.remove('hidden');
        } else {
            emptyEl.classList.add('hidden');
        }
    }
}

/**
 * Append items for infinite scroll.
 */
function appendBrowserItems(images) {
    const grid = $('#browser-grid');
    if (!grid || !images || images.length === 0) return;

    const fragment = document.createDocumentFragment();
    images.forEach(imageData => {
        fragment.appendChild(createBrowserImageThumbnail(imageData));
    });
    grid.appendChild(fragment);
}

/**
 * Add a progress placeholder for a generation in progress.
 */
function addBrowserProgressPlaceholder(jobId, settings) {
    const grid = $('#browser-grid');
    if (!grid) return;

    const numImages = (settings && settings.num_images) || 1;
    for (let i = 0; i < numImages; i++) {
        const item = document.createElement('div');
        item.className = 'browser-img-item';
        item.dataset.jobId = jobId;

        const pending = document.createElement('div');
        pending.className = 'gen-thumbnail-item gen-thumbnail-pending';
        pending.dataset.jobId = jobId;
        pending.innerHTML = typeof createCircularProgress === 'function'
            ? createCircularProgress(0, 0)
            : '<span>⏳</span>';
        item.appendChild(pending);

        const info = document.createElement('div');
        info.className = 'browser-img-info';
        info.textContent = 'Generating…';
        item.appendChild(info);

        // Insert at the start of images (after directory cards)
        const firstImgItem = grid.querySelector('.browser-img-item');
        if (firstImgItem) {
            grid.insertBefore(item, firstImgItem);
        } else {
            grid.appendChild(item);
        }
    }

    // Hide empty state
    const emptyEl = $('#browser-empty');
    if (emptyEl) emptyEl.classList.add('hidden');
}

/**
 * Store source image settings in sessionStorage for the refine flow.
 * Picked up by app.js _handleRefineParams() on the chat page.
 */
function _storeRefineSettings(settings) {
    if (!settings) return;
    try {
        sessionStorage.setItem('refineSettings', JSON.stringify({
            base_model: settings.base_model || '',
            loras: settings.loras || [],
            output_folder: settings.output_folder || '',
            seed: settings.seed != null ? settings.seed : -1,
        }));
    } catch (e) {
        // sessionStorage unavailable — settings won't carry over
    }
}
