/**
 * ComfyUI settings tab logic.
 *
 * Manages the ComfyUI configuration pane inside the settings modal:
 * connection testing, model dropdown, dual workflow upload (API + UI), and save.
 *
 * Depends on: api.js (API), app.js ($, escapeHtml),
 *             searchableDropdown.js (createSearchableDropdown)
 */

let _comfyuiModelDropdown = null; // Searchable dropdown instance

// ── Load settings into the ComfyUI tab ──────────────────────────────────

async function loadComfyUISettings() {
    try {
        const settings = await API.getSettings();

        // Populate fields
        $('#setting-comfyui-url').value = settings.comfyui_base_url || 'http://localhost:8188';
        $('#setting-comfyui-negative').value = settings.comfyui_default_negative || '';

        // Initialize or update model dropdown
        _initModelDropdown(settings.comfyui_default_model || '');

        // Load workflow info
        await _loadWorkflowInfo();

        // Wire up file input change handlers (once)
        _initWorkflowFileInputs();

        // Check connection status
        await _checkConnectionStatus();
    } catch (err) {
        console.warn('Failed to load ComfyUI settings:', err);
    }
}

// ── Model dropdown ──────────────────────────────────────────────────────

function _initModelDropdown(currentModel) {
    const container = $('#comfyui-model-dropdown');
    if (!container) return;

    // Destroy previous instance if exists
    if (_comfyuiModelDropdown) {
        _comfyuiModelDropdown.destroy();
        _comfyuiModelDropdown = null;
    }

    _comfyuiModelDropdown = createSearchableDropdown({
        container,
        placeholder: 'Select a diffusion model...',
        items: [],
        value: currentModel,
        onSelect: (value) => {
            // Value is stored; will be saved when user clicks Save
        },
    });
}

// ── Connection status ───────────────────────────────────────────────────

async function _checkConnectionStatus() {
    const dot = $('#comfyui-status-dot');
    const text = $('#comfyui-status-text');
    if (!dot || !text) return;

    try {
        const result = await API.checkComfyUIHealth();
        if (result.ok) {
            dot.className = 'status-dot online';
            text.textContent = 'Connected';
            // Refresh model list since we're connected
            await _refreshModelList();
        } else {
            dot.className = 'status-dot offline';
            text.textContent = result.message || 'Not connected';
        }
    } catch (err) {
        dot.className = 'status-dot offline';
        text.textContent = 'Not connected';
    }
}

// ── Test Connection button ──────────────────────────────────────────────

async function testComfyUIConnection() {
    const dot = $('#comfyui-status-dot');
    const text = $('#comfyui-status-text');
    if (!dot || !text) return;

    // First, save the URL so the backend uses the latest value
    const url = $('#setting-comfyui-url').value.trim();
    if (url) {
        try {
            await API.updateSettings({ comfyui_base_url: url });
        } catch (_) {
            // proceed anyway to test
        }
    }

    dot.className = 'status-dot offline';
    text.textContent = 'Testing...';

    try {
        const result = await API.checkComfyUIHealth();
        if (result.ok) {
            dot.className = 'status-dot online';
            text.textContent = 'Connected';
            await _refreshModelList();
        } else {
            dot.className = 'status-dot error';
            text.textContent = result.message || 'Connection failed';
        }
    } catch (err) {
        dot.className = 'status-dot error';
        text.textContent = 'Connection failed: ' + err.message;
    }
}

// ── Workflow file upload (dual: API + UI) ───────────────────────────────

let _workflowFileInputsWired = false;

function _initWorkflowFileInputs() {
    if (_workflowFileInputsWired) return;

    const apiInput = $('#workflow-api-file-input');
    const uiInput = $('#workflow-ui-file-input');

    if (apiInput) {
        apiInput.addEventListener('change', _handleAPIWorkflowFileChange);
    }
    if (uiInput) {
        uiInput.addEventListener('change', _handleUIWorkflowFileChange);
    }

    _workflowFileInputsWired = true;
}

async function _handleAPIWorkflowFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusEl = $('#workflow-api-status');
    if (statusEl) {
        statusEl.textContent = 'Uploading…';
        statusEl.className = 'hint';
    }

    try {
        const result = await API.uploadWorkflowAPI(file);
        if (result.error || result.status === 'error') {
            if (statusEl) {
                statusEl.textContent = `✗ ${result.error || result.message}`;
                statusEl.className = 'hint hint-error';
            }
        } else {
            _updateAPIWorkflowDisplay(result.filename, result.workflow_name, true);
            if (statusEl) {
                const msg = result.status === 'unchanged'
                    ? `✓ ${result.message}`
                    : `✓ Uploaded: ${result.workflow_name || result.filename}`;
                statusEl.textContent = msg;
                statusEl.className = 'hint hint-success';
            }
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = '✗ Upload failed: ' + err.message;
            statusEl.className = 'hint hint-error';
        }
    }

    // Reset the file input so the same file can be re-selected
    e.target.value = '';
}

async function _handleUIWorkflowFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusEl = $('#workflow-ui-status');
    if (statusEl) {
        statusEl.textContent = 'Uploading…';
        statusEl.className = 'hint';
    }

    try {
        const result = await API.uploadWorkflowUI(file);
        if (result.error || result.status === 'error') {
            if (statusEl) {
                statusEl.textContent = `✗ ${result.error || result.message}`;
                statusEl.className = 'hint hint-error';
            }
        } else {
            _updateUIWorkflowDisplay(result.filename, true);
            if (statusEl) {
                const msg = result.status === 'unchanged'
                    ? `✓ ${result.message}`
                    : `✓ Uploaded: ${result.filename}`;
                statusEl.textContent = msg;
                statusEl.className = 'hint hint-success';
            }
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = '✗ Upload failed: ' + err.message;
            statusEl.className = 'hint hint-error';
        }
    }

    // Reset the file input so the same file can be re-selected
    e.target.value = '';
}

async function deleteUploadedWorkflow() {
    const apiStatusEl = $('#workflow-api-status');
    const uiStatusEl = $('#workflow-ui-status');

    try {
        await API.deleteWorkflow();
        _updateAPIWorkflowDisplay('', null, false);
        _updateUIWorkflowDisplay('', false);
        if (apiStatusEl) {
            apiStatusEl.textContent = 'Workflows removed';
            apiStatusEl.className = 'hint';
        }
        if (uiStatusEl) {
            uiStatusEl.textContent = '';
            uiStatusEl.className = 'hint';
        }
    } catch (err) {
        if (apiStatusEl) {
            apiStatusEl.textContent = '✗ Failed to remove: ' + err.message;
            apiStatusEl.className = 'hint hint-error';
        }
    }
}

// ── Load workflow info ──────────────────────────────────────────────────

async function _loadWorkflowInfo() {
    try {
        const info = await API.getWorkflowInfo();

        // Update API workflow display
        _updateAPIWorkflowDisplay(
            info.api_filename || info.filename || '',
            info.workflow_name || null,
            info.has_api_workflow || info.has_workflow || false,
        );

        // Update UI workflow display
        _updateUIWorkflowDisplay(
            info.ui_filename || '',
            info.has_ui_workflow || false,
        );

        const apiStatusEl = $('#workflow-api-status');
        if (apiStatusEl && (info.has_api_workflow || info.has_workflow)) {
            if (info.valid) {
                apiStatusEl.textContent = `✓ ${info.workflow_name || 'Valid workflow'}`;
                apiStatusEl.className = 'hint hint-success';
            } else {
                apiStatusEl.textContent = '✗ No matching workflow definition';
                apiStatusEl.className = 'hint hint-error';
            }
        }
    } catch (err) {
        console.warn('Failed to load workflow info:', err);
    }
}

function _updateAPIWorkflowDisplay(filename, workflowName, hasWorkflow) {
    const filenameEl = $('#workflow-api-filename');
    const deleteBtn = $('#workflow-api-delete-btn');

    if (filenameEl) {
        filenameEl.textContent = hasWorkflow
            ? (filename || 'Uploaded API workflow')
            : 'No API workflow uploaded';
        filenameEl.title = filename || '';
    }

    if (deleteBtn) {
        deleteBtn.style.display = hasWorkflow ? '' : 'none';
    }
}

function _updateUIWorkflowDisplay(filename, hasWorkflow) {
    const filenameEl = $('#workflow-ui-filename');
    const deleteBtn = $('#workflow-ui-delete-btn');

    if (filenameEl) {
        filenameEl.textContent = hasWorkflow
            ? (filename || 'Uploaded UI workflow')
            : 'No UI workflow uploaded';
        filenameEl.title = filename || '';
    }

    if (deleteBtn) {
        deleteBtn.style.display = hasWorkflow ? '' : 'none';
    }
}

// ── Save all ComfyUI settings ───────────────────────────────────────────

async function saveComfyUISettings() {
    const data = {};

    const url = $('#setting-comfyui-url').value.trim();
    if (url) data.comfyui_base_url = url;

    if (_comfyuiModelDropdown) {
        const model = _comfyuiModelDropdown.getValue();
        if (model) data.comfyui_default_model = model;
    }

    const negative = $('#setting-comfyui-negative').value;
    data.comfyui_default_negative = negative;

    try {
        await API.updateSettings(data);
        const dot = $('#comfyui-status-dot');
        const text = $('#comfyui-status-text');
        // Brief confirmation
        const prevText = text ? text.textContent : '';
        if (text) text.textContent = 'Settings saved ✓';
        setTimeout(() => {
            if (text) text.textContent = prevText;
        }, 2000);
    } catch (err) {
        alert('Failed to save ComfyUI settings: ' + err.message);
    }
}

// ── Refresh model list from ComfyUI ────────────────────────────────────

async function _refreshModelList() {
    if (!_comfyuiModelDropdown) return;

    try {
        const models = await API.getComfyUIModels('diffusion_models');
        if (Array.isArray(models)) {
            _comfyuiModelDropdown.setItems(models);
        }
    } catch (err) {
        console.warn('Failed to load ComfyUI models:', err);
    }
}
