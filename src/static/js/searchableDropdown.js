/**
 * Searchable dropdown widget.
 *
 * Creates a scrollable dropdown with type-to-filter functionality.
 * Single-select mode — returns the selected value.
 *
 * Usage:
 *   const dropdown = createSearchableDropdown({
 *     container: document.getElementById('my-container'),
 *     placeholder: 'Select a model...',
 *     items: ['item1', 'item2', ...],
 *     maxVisible: 15,
 *     onSelect: (value) => { console.log('Selected:', value); },
 *     value: 'current_value',  // optional initial value
 *   });
 *
 *   dropdown.setItems(['new1', 'new2']);
 *   dropdown.getValue();
 *   dropdown.setValue('item1');
 *   dropdown.destroy();
 *
 * Depends on: app.js (escapeHtml)
 */

function createSearchableDropdown(options) {
    const {
        container,
        placeholder = 'Search...',
        items: initialItems = [],
        maxVisible = 10,
        onSelect = null,
        value: initialValue = '',
    } = options;

    let items = [...initialItems];
    let currentValue = initialValue;
    let isOpen = false;
    let highlightIndex = -1;
    let _mouseDownOnDropdown = false;

    // ── Build DOM ───────────────────────────────────────────────
    const wrapper = document.createElement('div');
    wrapper.className = 'sd-wrapper';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'sd-input';
    input.placeholder = placeholder;
    input.value = currentValue;
    input.autocomplete = 'off';

    const arrow = document.createElement('span');
    arrow.className = 'sd-arrow';
    arrow.textContent = '▾';

    const dropdownList = document.createElement('div');
    dropdownList.className = 'sd-dropdown hidden';

    wrapper.appendChild(input);
    wrapper.appendChild(arrow);
    wrapper.appendChild(dropdownList);
    container.appendChild(wrapper);

    // ── Apply max-height based on maxVisible ────────────────────
    dropdownList.style.maxHeight = `calc(${maxVisible} * 2.2rem)`;

    // ── Render filtered items ───────────────────────────────────
    function renderItems(filter = '') {
        const lowerFilter = filter.toLowerCase();
        const filtered = items.filter(item =>
            item.toLowerCase().includes(lowerFilter)
        );

        if (filtered.length === 0) {
            dropdownList.innerHTML = '<div class="sd-item sd-no-results">No results</div>';
            highlightIndex = -1;
            return;
        }

        dropdownList.innerHTML = filtered.map((item, idx) => {
            const escaped = escapeHtml(item);
            let display = escaped;

            // Highlight matching text
            if (lowerFilter) {
                const matchStart = item.toLowerCase().indexOf(lowerFilter);
                if (matchStart !== -1) {
                    const before = escapeHtml(item.slice(0, matchStart));
                    const match = escapeHtml(item.slice(matchStart, matchStart + lowerFilter.length));
                    const after = escapeHtml(item.slice(matchStart + lowerFilter.length));
                    display = `${before}<mark class="sd-highlight">${match}</mark>${after}`;
                }
            }

            const activeClass = idx === highlightIndex ? ' sd-item-active' : '';
            return `<div class="sd-item${activeClass}" data-index="${idx}" data-value="${escaped}">${display}</div>`;
        }).join('');
    }

    // ── Show / hide dropdown ────────────────────────────────────
    function showDropdown() {
        if (isOpen) return;
        isOpen = true;
        highlightIndex = -1;
        renderItems(input.value);
        dropdownList.classList.remove('hidden');
    }

    function hideDropdown() {
        isOpen = false;
        highlightIndex = -1;
        dropdownList.classList.add('hidden');
    }

    // ── Select an item ──────────────────────────────────────────
    function selectItem(value) {
        currentValue = value;
        input.value = value;
        hideDropdown();
        if (onSelect) onSelect(value);
    }

    // ── Event handlers ──────────────────────────────────────────
    input.addEventListener('focus', () => {
        showDropdown();
    });

    input.addEventListener('input', () => {
        highlightIndex = -1;
        renderItems(input.value);
        if (!isOpen) showDropdown();
    });

    input.addEventListener('keydown', (e) => {
        if (!isOpen) return;

        const visibleItems = dropdownList.querySelectorAll('.sd-item:not(.sd-no-results):not(.sd-more)');
        const count = visibleItems.length;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            highlightIndex = Math.min(highlightIndex + 1, count - 1);
            updateHighlight(visibleItems);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            highlightIndex = Math.max(highlightIndex - 1, 0);
            updateHighlight(visibleItems);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (highlightIndex >= 0 && highlightIndex < count) {
                selectItem(visibleItems[highlightIndex].dataset.value);
            }
        } else if (e.key === 'Escape') {
            e.preventDefault();
            hideDropdown();
            input.blur();
        }
    });

    function updateHighlight(visibleItems) {
        visibleItems.forEach((el, i) => {
            el.classList.toggle('sd-item-active', i === highlightIndex);
        });
        // Scroll highlighted item into view
        if (highlightIndex >= 0 && visibleItems[highlightIndex]) {
            visibleItems[highlightIndex].scrollIntoView({ block: 'nearest' });
        }
    }

    // Click on items in dropdown
    dropdownList.addEventListener('mousedown', (e) => {
        _mouseDownOnDropdown = true;
        // mousedown instead of click so it fires before blur
        const itemEl = e.target.closest('.sd-item:not(.sd-no-results):not(.sd-more)');
        if (itemEl) {
            e.preventDefault();
            selectItem(itemEl.dataset.value);
        }
    });

    // Hide on blur (with a small delay so click can register)
    input.addEventListener('blur', () => {
        setTimeout(() => {
            if (!_mouseDownOnDropdown) hideDropdown();
            _mouseDownOnDropdown = false;
        }, 150);
    });

    // Toggle dropdown on arrow click
    arrow.addEventListener('mousedown', (e) => {
        e.preventDefault();
        if (isOpen) {
            hideDropdown();
            input.blur();
        } else {
            input.focus();
        }
    });

    // ── Public API ──────────────────────────────────────────────
    return {
        setItems(newItems) {
            items = [...newItems];
            if (isOpen) renderItems(input.value);
        },

        getValue() {
            return currentValue;
        },

        setValue(val) {
            currentValue = val;
            input.value = val;
        },

        destroy() {
            wrapper.remove();
        },

        /** Expose the input element for external focus management */
        getInputElement() {
            return input;
        },
    };
}
