/**
 * Checkbox dropdown widget — toggle button that opens a dropdown with
 * checkboxes, "Select all" / "Deselect all" controls, and click-outside
 * to close.
 *
 * Usage:
 *   const dd = createCheckboxDropdown({
 *     container: document.getElementById('my-container'),
 *     label: 'Params',
 *     options: [
 *       { key: 'sampler', label: 'Sampler' },
 *       { key: 'steps', label: 'Steps' },
 *     ],
 *     onChange: (checkedKeys) => { console.log(checkedKeys); },
 *     initialChecked: ['sampler'],
 *   });
 *
 *   dd.getChecked();    // ['sampler']
 *   dd.setChecked([]);  // uncheck all
 *   dd.destroy();
 *
 * Depends on: app.js (escapeHtml)
 */

function createCheckboxDropdown(options) {
    const {
        container,
        label = 'Options',
        options: items = [],
        onChange = null,
        initialChecked = [],
    } = options;

    let checked = new Set(initialChecked);
    let isOpen = false;

    // ── Build DOM ───────────────────────────────────────────────
    const wrapper = document.createElement('div');
    wrapper.className = 'cbdd-wrapper';

    const btn = document.createElement('button');
    btn.className = 'cbdd-btn';
    btn.type = 'button';
    btn.innerHTML = `<span class="cbdd-label">${escapeHtml(label)}</span><span class="cbdd-arrow">▾</span>`;

    const panel = document.createElement('div');
    panel.className = 'cbdd-panel hidden';

    // Controls row
    const controls = document.createElement('div');
    controls.className = 'cbdd-controls';
    const selectAll = document.createElement('button');
    selectAll.type = 'button';
    selectAll.className = 'cbdd-link';
    selectAll.textContent = 'Select all';
    const deselectAll = document.createElement('button');
    deselectAll.type = 'button';
    deselectAll.className = 'cbdd-link';
    deselectAll.textContent = 'Deselect all';
    controls.appendChild(selectAll);
    controls.appendChild(deselectAll);
    panel.appendChild(controls);

    // Checkbox items
    const itemList = document.createElement('div');
    itemList.className = 'cbdd-items';
    for (const item of items) {
        const row = document.createElement('label');
        row.className = 'cbdd-item';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.dataset.key = item.key;
        cb.checked = checked.has(item.key);
        const span = document.createElement('span');
        span.textContent = item.label;
        row.appendChild(cb);
        row.appendChild(span);
        itemList.appendChild(row);
    }
    panel.appendChild(itemList);

    wrapper.appendChild(btn);
    wrapper.appendChild(panel);
    container.appendChild(wrapper);

    // ── Events ──────────────────────────────────────────────────
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        isOpen = !isOpen;
        panel.classList.toggle('hidden', !isOpen);
    });

    // Click outside to close
    function _onDocClick(e) {
        if (!wrapper.contains(e.target)) {
            isOpen = false;
            panel.classList.add('hidden');
        }
    }
    document.addEventListener('click', _onDocClick);

    // Checkbox change
    itemList.addEventListener('change', (e) => {
        const cb = e.target;
        if (cb.type !== 'checkbox') return;
        if (cb.checked) {
            checked.add(cb.dataset.key);
        } else {
            checked.delete(cb.dataset.key);
        }
        _notify();
    });

    selectAll.addEventListener('click', () => {
        for (const item of items) checked.add(item.key);
        itemList.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
        _notify();
    });

    deselectAll.addEventListener('click', () => {
        checked.clear();
        itemList.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
        _notify();
    });

    function _notify() {
        if (onChange) onChange([...checked]);
    }

    // ── Public API ──────────────────────────────────────────────
    return {
        getChecked() {
            return [...checked];
        },

        setChecked(keys) {
            checked = new Set(keys);
            itemList.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = checked.has(cb.dataset.key);
            });
        },

        destroy() {
            document.removeEventListener('click', _onDocClick);
            wrapper.remove();
        },
    };
}
