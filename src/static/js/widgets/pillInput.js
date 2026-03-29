/**
 * Pill input widget (multi-select with removable pills).
 *
 * Wraps a searchable dropdown. When an item is selected it appears as
 * a removable pill above the search input. Items already selected are
 * hidden from the dropdown.
 *
 * Usage:
 *   const pills = createPillInput({
 *     container: document.getElementById('my-container'),
 *     placeholder: 'Search loras...',
 *     items: ['lora1.safetensors', 'lora2.safetensors'],
 *     maxVisible: 15,
 *     truncateAt: 30,
 *     onChange: (selectedItems) => { console.log('Selected:', selectedItems); },
 *     value: ['current_lora'],  // optional initial values
 *   });
 *
 *   pills.setItems(newItems);
 *   pills.getValue();  // returns array of selected items
 *   pills.setValue(['item1', 'item2']);
 *   pills.destroy();
 *
 * Depends on: app.js (escapeHtml), searchableDropdown.js
 */

function createPillInput(options) {
    const {
        container,
        placeholder = 'Search...',
        items: initialItems = [],
        maxVisible,
        truncateAt = 30,
        onChange = null,
        value: initialValue = [],
        minRequired = 0,
        errorMessage = 'At least one selection required',
    } = options;

    let allItems = [...initialItems];
    let selected = [...initialValue];

    // ── Build DOM ───────────────────────────────────────────────
    const wrapper = document.createElement('div');
    wrapper.className = 'pill-wrapper';

    const pillsContainer = document.createElement('div');
    pillsContainer.className = 'pill-container';

    const dropdownContainer = document.createElement('div');
    dropdownContainer.className = 'pill-dropdown-host';

    const errorEl = document.createElement('div');
    errorEl.className = 'pill-error-msg hidden';
    errorEl.textContent = errorMessage;

    wrapper.appendChild(pillsContainer);
    wrapper.appendChild(dropdownContainer);
    wrapper.appendChild(errorEl);
    container.appendChild(wrapper);

    // ── Internal dropdown ───────────────────────────────────────
    const dropdown = createSearchableDropdown({
        container: dropdownContainer,
        placeholder,
        items: availableItems(),
        maxVisible,
        onSelect: (value) => {
            addPill(value);
            // Reset the input text after selection
            dropdown.setValue('');
            const remaining = availableItems();
            dropdown.setItems(remaining);

            // Queue scroll target so next open centers near the just-selected item
            queueScrollTarget(value);
        },
    });

    /** Set scroll target to the nearest remaining neighbor of a just-selected value. */
    function queueScrollTarget(value) {
        const remaining = availableItems();
        const neighbor = findNeighbor(value, remaining);
        if (neighbor) dropdown.setScrollTarget(neighbor);
    }

    // If initialized with pre-selected values, center near the last one on first open
    if (selected.length > 0) {
        queueScrollTarget(selected[selected.length - 1]);
    }

    // ── Helpers ─────────────────────────────────────────────────
    function availableItems() {
        const sel = new Set(selected);
        return allItems.filter(item => !sel.has(item));
    }

    /** Find the nearest still-available item to `value` in the master list. */
    function findNeighbor(value, remaining) {
        if (remaining.length === 0) return null;
        const idx = allItems.indexOf(value);
        if (idx === -1) return remaining[0];

        // Search outward from the original position for the closest remaining item
        const remainingSet = new Set(remaining);
        for (let offset = 1; offset < allItems.length; offset++) {
            if (idx + offset < allItems.length && remainingSet.has(allItems[idx + offset])) {
                return allItems[idx + offset];
            }
            if (idx - offset >= 0 && remainingSet.has(allItems[idx - offset])) {
                return allItems[idx - offset];
            }
        }
        return remaining[0];
    }

    function truncate(text) {
        if (text.length <= truncateAt) return escapeHtml(text);
        return escapeHtml(text.slice(0, truncateAt)) + '&hellip;';
    }

    function renderPills() {
        pillsContainer.innerHTML = selected.map(item => `
            <span class="pill" title="${escapeHtml(item)}">
                <span class="pill-text">${truncate(item)}</span>
                <button class="pill-remove" data-value="${escapeHtml(item)}" type="button">&times;</button>
            </span>
        `).join('');
    }

    function addPill(value) {
        if (selected.includes(value)) return;
        selected.push(value);
        renderPills();
        // Auto-clear validation error if we've met the minimum
        if (minRequired > 0 && selected.length >= minRequired) {
            _clearError();
        }
        if (onChange) onChange([...selected]);
    }

    function _showError() {
        wrapper.classList.add('pill-error');
        errorEl.classList.remove('hidden');
    }

    function _clearError() {
        wrapper.classList.remove('pill-error');
        errorEl.classList.add('hidden');
    }

    function removePill(value) {
        selected = selected.filter(v => v !== value);
        renderPills();
        dropdown.setItems(availableItems());
        if (onChange) onChange([...selected]);
    }

    // ── Events ──────────────────────────────────────────────────
    pillsContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('.pill-remove');
        if (btn) {
            e.preventDefault();
            removePill(btn.dataset.value);
        }
    });

    // Initial render
    renderPills();

    // ── Public API ──────────────────────────────────────────────
    return {
        setItems(newItems) {
            allItems = [...newItems];
            dropdown.setItems(availableItems());
        },

        getValue() {
            return [...selected];
        },

        setValue(values) {
            selected = [...values];
            renderPills();
            dropdown.setItems(availableItems());
            dropdown.setValue('');
        },

        validate() {
            if (minRequired > 0 && selected.length < minRequired) {
                _showError();
                return false;
            }
            _clearError();
            return true;
        },

        clearError() {
            _clearError();
        },

        destroy() {
            dropdown.destroy();
            wrapper.remove();
        },
    };
}
