/**
 * Browser parameter display — shows generation parameters below image
 * thumbnails based on user-selected checkbox options.
 *
 * Depends on: app.js ($, escapeHtml), checkboxDropdown.js, browserState.js
 */

const _PARAM_OPTIONS = [
    { key: 'filename', label: 'Filename', source: 'layout' },
    { key: 'file_size', label: 'File Size', source: 'layout' },
    { key: 'created_at', label: 'Created', source: 'layout' },
    { key: 'dimensions', label: 'Dimensions', source: 'meta' },
    { key: 'sampler', label: 'Sampler', source: 'settings' },
    { key: 'scheduler', label: 'Scheduler', source: 'settings' },
    { key: 'cfg_scale', label: 'CFG', source: 'settings' },
    { key: 'steps', label: 'Steps', source: 'settings' },
    { key: 'base_model', label: 'Model', source: 'settings' },
    { key: 'loras', label: 'LoRA(s)', source: 'settings' },
    { key: 'seed', label: 'Seed', source: 'settings' },
];

let _paramDropdown = null;

/**
 * Initialize the parameter display dropdown in the given container.
 * Restores previous selection from sessionStorage.
 */
function initBrowserParamDisplay(container) {
    const initial = (typeof BrowserState !== 'undefined' && BrowserState.displayParams.length > 0)
        ? BrowserState.displayParams
        : [];

    _paramDropdown = createCheckboxDropdown({
        container,
        label: 'Properties',
        options: _PARAM_OPTIONS,
        initialChecked: initial,
        onChange: (checkedKeys) => {
            if (typeof BrowserState !== 'undefined') {
                BrowserState.displayParams = checkedKeys;
            }
            API.updateSettings({ browser_display_params: JSON.stringify(checkedKeys) }).catch(() => {});
            updateBrowserParamDisplay();
        },
    });
}

/**
 * Re-render parameter rows on all visible browser image thumbnails.
 * Called when checkbox selection changes or new images are loaded.
 */
function updateBrowserParamDisplay() {
    const checked = (typeof BrowserState !== 'undefined')
        ? BrowserState.displayParams || []
        : [];

    const checkedSet = new Set(checked);

    document.querySelectorAll('.browser-img-item').forEach(wrapper => {
        // Toggle layout elements: filename row and meta row
        const infoEl = wrapper.querySelector('.browser-img-info');
        const metaEl = wrapper.querySelector('.browser-img-meta');

        if (infoEl) {
            infoEl.classList.toggle('hidden', !checkedSet.has('filename'));
        }
        if (metaEl) {
            // Show the meta row if either file_size or created_at is checked
            const showSize = checkedSet.has('file_size');
            const showDate = checkedSet.has('created_at');
            metaEl.classList.toggle('hidden', !showSize && !showDate);
            // Toggle individual spans within meta
            const spans = metaEl.querySelectorAll('span');
            if (spans[0]) spans[0].classList.toggle('hidden', !showSize);
            if (spans[1]) spans[1].classList.toggle('hidden', !showDate);
        }

        // Render key/value rows for non-layout properties
        const paramsEl = wrapper.querySelector('.browser-img-params');
        if (!paramsEl) return;

        // Filter to only non-layout checked keys
        const kvKeys = checked.filter(k => {
            const opt = _PARAM_OPTIONS.find(o => o.key === k);
            return opt && opt.source !== 'layout';
        });

        if (kvKeys.length === 0) {
            paramsEl.innerHTML = '';
            return;
        }

        let settings, meta;
        try { settings = JSON.parse(paramsEl.dataset.settings || '{}'); } catch { settings = {}; }
        try { meta = JSON.parse(paramsEl.dataset.meta || '{}'); } catch { meta = {}; }

        const rows = kvKeys.map(key => {
            const optDef = _PARAM_OPTIONS.find(o => o.key === key);
            if (!optDef) return '';
            const label = optDef.label;

            let displayVal;
            if (optDef.source === 'meta') {
                if (key === 'dimensions') {
                    if (!meta.width || !meta.height) return '';
                    displayVal = `${meta.width} \u00d7 ${meta.height}`;
                }
                if (!displayVal) return '';
            } else {
                const value = settings[key];
                if (value == null || value === '') return '';
                if (key === 'loras' && Array.isArray(value)) {
                    if (value.length === 0) return '';
                    displayVal = value.map(l => typeof l === 'object' ? l.name : l).join(', ');
                } else {
                    displayVal = String(value);
                }
            }

            return `<div class="param-row">
                <span class="param-name">${escapeHtml(label)}</span>
                <span class="param-value" title="${escapeHtml(displayVal)}">${escapeHtml(displayVal)}</span>
            </div>`;
        }).filter(Boolean);

        paramsEl.innerHTML = rows.join('');
    });
}
