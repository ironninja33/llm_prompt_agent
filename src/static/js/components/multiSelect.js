/**
 * MultiSelect — reusable click-to-select behavior for thumbnail grids.
 *
 * Usage:
 *   const ms = new MultiSelect(containerEl, {
 *     itemSelector: '.gen-thumbnail-item',
 *     idAttribute: 'data-image-id',
 *     sizeAttribute: 'data-file-size',
 *     selectedClass: 'selected',
 *     onSelectionChanged: (selectedIds, totalSize) => { ... }
 *   });
 *
 * Behavior:
 *   - Click item: toggle selection
 *   - Shift+click: select range from last click to current
 *   - ms.selectAll() / ms.deselectAll()
 *   - ms.getSelectedIds() → Set<int>
 *   - ms.destroy() — remove event listeners
 */
class MultiSelect {
    constructor(container, options = {}) {
        this.container = container;
        this.itemSelector = options.itemSelector || '.gen-thumbnail-item';
        this.idAttribute = options.idAttribute || 'data-image-id';
        this.sizeAttribute = options.sizeAttribute || 'data-file-size';
        this.selectedClass = options.selectedClass || 'selected';
        this.onSelectionChanged = options.onSelectionChanged || null;

        this._selected = new Set();
        this._lastClickedIndex = -1;

        this._onClick = this._handleClick.bind(this);
        this.container.addEventListener('click', this._onClick);
    }

    _getItems() {
        return Array.from(this.container.querySelectorAll(this.itemSelector));
    }

    _getItemId(item) {
        const raw = item.getAttribute(this.idAttribute);
        return raw ? parseInt(raw, 10) : null;
    }

    _getItemSize(item) {
        const raw = item.getAttribute(this.sizeAttribute);
        return raw ? parseInt(raw, 10) : 0;
    }

    _handleClick(e) {
        const item = e.target.closest(this.itemSelector);
        if (!item || !this.container.contains(item)) return;

        // Don't intercept clicks on action buttons or dupe group items
        if (e.target.closest('.gen-thumbnail-actions') ||
            e.target.closest('button') ||
            e.target.closest('a') ||
            e.target.closest('.cleanup-dupe-body')) return;

        const items = this._getItems();
        const clickedIndex = items.indexOf(item);
        if (clickedIndex === -1) return;

        const id = this._getItemId(item);
        if (id === null) return;

        if (e.shiftKey && this._lastClickedIndex >= 0) {
            // Range select
            const start = Math.min(this._lastClickedIndex, clickedIndex);
            const end = Math.max(this._lastClickedIndex, clickedIndex);
            for (let i = start; i <= end; i++) {
                const rangeId = this._getItemId(items[i]);
                if (rangeId !== null) {
                    this._selected.add(rangeId);
                    items[i].classList.add(this.selectedClass);
                }
            }
        } else {
            // Toggle single
            if (this._selected.has(id)) {
                this._selected.delete(id);
                item.classList.remove(this.selectedClass);
            } else {
                this._selected.add(id);
                item.classList.add(this.selectedClass);
            }
        }

        this._lastClickedIndex = clickedIndex;
        this._emitChange();
    }

    _emitChange() {
        if (!this.onSelectionChanged) return;
        const totalSize = this._computeTotalSize();
        this.onSelectionChanged(new Set(this._selected), totalSize);
    }

    _computeTotalSize() {
        let total = 0;
        const items = this._getItems();
        for (const item of items) {
            const id = this._getItemId(item);
            if (id !== null && this._selected.has(id)) {
                total += this._getItemSize(item);
            }
        }
        return total;
    }

    /** Select all visible items. */
    selectAll() {
        const items = this._getItems();
        for (const item of items) {
            const id = this._getItemId(item);
            if (id !== null) {
                this._selected.add(id);
                item.classList.add(this.selectedClass);
            }
        }
        this._emitChange();
    }

    /** Deselect all items. */
    deselectAll() {
        const items = this._getItems();
        for (const item of items) {
            item.classList.remove(this.selectedClass);
        }
        this._selected.clear();
        this._lastClickedIndex = -1;
        this._emitChange();
    }

    /** Get all selected IDs as a Set<int>. */
    getSelectedIds() {
        return new Set(this._selected);
    }

    /** Get count of selected items. */
    getSelectedCount() {
        return this._selected.size;
    }

    /** Remove a set of IDs from selection (e.g. after deletion). */
    removeIds(ids) {
        for (const id of ids) {
            this._selected.delete(id);
        }
        // Also remove visual class from any remaining DOM elements
        const items = this._getItems();
        for (const item of items) {
            const id = this._getItemId(item);
            if (id !== null && ids.has(id)) {
                item.classList.remove(this.selectedClass);
            }
        }
        this._emitChange();
    }

    /** Clean up event listeners. */
    destroy() {
        this.container.removeEventListener('click', this._onClick);
        this._selected.clear();
    }
}
