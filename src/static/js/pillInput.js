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

    wrapper.appendChild(pillsContainer);
    wrapper.appendChild(dropdownContainer);
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
            dropdown.setItems(availableItems());
        },
    });

    // ── Helpers ─────────────────────────────────────────────────
    function availableItems() {
        const sel = new Set(selected);
        return allItems.filter(item => !sel.has(item));
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
        if (onChange) onChange([...selected]);
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

        destroy() {
            dropdown.destroy();
            wrapper.remove();
        },
    };
}
