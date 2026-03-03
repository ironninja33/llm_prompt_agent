/**
 * Cleanup sidebar — folder list with image counts and cleanup scores.
 *
 * Depends on: cleanupOverlay.js (_cleanupCurrentFolder, loadCleanupGrid, loadCleanupWaveCounts)
 */

async function loadCleanupSidebar() {
    const list = $('#cleanup-folder-list');
    if (!list) return;

    try {
        const res = await fetch('/api/cleanup/folders');
        const data = await res.json();

        list.innerHTML = '';

        // "All" item
        const allItem = document.createElement('li');
        allItem.className = 'cleanup-folder-item' + (_cleanupCurrentFolder === null ? ' active' : '');
        allItem.dataset.folder = '';
        allItem.innerHTML = '<span class="cleanup-folder-name">All</span>';
        allItem.onclick = () => _selectCleanupFolder(null);
        list.appendChild(allItem);

        // Folder items
        for (const folder of data.folders) {
            const name = folder.output_folder || '(root)';
            const li = document.createElement('li');
            li.className = 'cleanup-folder-item' + (_cleanupCurrentFolder === name ? ' active' : '');
            li.dataset.folder = name;

            // Determine bar class
            const pct = folder.cleanup_pct || 0;
            let barClass = 'low';
            if (pct >= 50) barClass = 'high';
            else if (pct >= 25) barClass = 'medium';

            li.innerHTML = `
                <span class="cleanup-folder-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
                <div class="cleanup-folder-meta">
                    <span class="cleanup-folder-count">${folder.image_count}</span>
                    <div class="cleanup-folder-bar">
                        <div class="cleanup-folder-bar-fill ${barClass}" style="width:${pct}%"></div>
                    </div>
                </div>
            `;
            li.onclick = () => _selectCleanupFolder(name);
            list.appendChild(li);
        }
    } catch (e) {
        console.error('Failed to load cleanup sidebar:', e);
    }
}

function _selectCleanupFolder(folder) {
    _cleanupCurrentFolder = folder;

    // Update active class
    $$('.cleanup-folder-item').forEach(item => {
        const f = item.dataset.folder || null;
        item.classList.toggle('active', f === (folder || ''));
    });

    // Reload grid and counts
    loadCleanupGrid();
    loadCleanupWaveCounts();
}
