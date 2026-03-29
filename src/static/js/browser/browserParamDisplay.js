/**
 * Browser parameter display — shows generation parameters below image
 * thumbnails based on user-selected checkbox options.
 *
 * Depends on: app.js ($, escapeHtml), checkboxDropdown.js, browserState.js
 */

const _PARAM_OPTIONS = [
    { key: 'sampler', label: 'Sampler' },
    { key: 'scheduler', label: 'Scheduler' },
    { key: 'cfg_scale', label: 'CFG' },
    { key: 'steps', label: 'Steps' },
    { key: 'base_model', label: 'Model' },
    { key: 'loras', label: 'LoRA(s)' },
    { key: 'seed', label: 'Seed' },
];

let _paramDropdown = null;

/**
 * Initialize the parameter display dropdown in the given container.
 * Restores previous selection from sessionStorage.
 */
function initBrowserParamDisplay(container) {
    const stored = sessionStorage.getItem('browserDisplayParams');
    const initial = stored ? JSON.parse(stored) : [];

    if (typeof BrowserState !== 'undefined') {
        BrowserState.displayParams = initial;
    }

    _paramDropdown = createCheckboxDropdown({
        container,
        label: 'Params',
        options: _PARAM_OPTIONS,
        initialChecked: initial,
        onChange: (checkedKeys) => {
            if (typeof BrowserState !== 'undefined') {
                BrowserState.displayParams = checkedKeys;
            }
            sessionStorage.setItem('browserDisplayParams', JSON.stringify(checkedKeys));
            updateBrowserParamDisplay();
        },
    });
}

/**
 * Re-render parameter rows on all visible browser image thumbnails.
 * Called when checkbox selection changes or new images are loaded.
 */
function updateBrowserParamDisplay() {
    const checkedParams = (typeof BrowserState !== 'undefined')
        ? BrowserState.displayParams || []
        : [];

    document.querySelectorAll('.browser-img-params').forEach(el => {
        if (checkedParams.length === 0) {
            el.innerHTML = '';
            return;
        }

        let settings;
        try {
            settings = JSON.parse(el.dataset.settings || '{}');
        } catch { settings = {}; }

        const rows = checkedParams.map(key => {
            const value = settings[key];
            if (value == null || value === '') return '';
            const labelObj = _PARAM_OPTIONS.find(o => o.key === key);
            const label = labelObj ? labelObj.label : key;

            // Format LoRAs: array of {name, strength} or strings → comma-separated names
            let displayVal;
            if (key === 'loras' && Array.isArray(value)) {
                if (value.length === 0) return '';
                displayVal = value.map(l => typeof l === 'object' ? l.name : l).join(', ');
            } else {
                displayVal = String(value);
            }
            return `<div class="param-row">
                <span class="param-name">${escapeHtml(label)}</span>
                <span class="param-value" title="${escapeHtml(displayVal)}">${escapeHtml(displayVal)}</span>
            </div>`;
        }).filter(Boolean);

        el.innerHTML = rows.join('');
    });
}
