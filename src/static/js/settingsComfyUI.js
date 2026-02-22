/**
 * ComfyUI settings tab logic.
 *
 * Manages the ComfyUI configuration pane inside the settings modal:
 * connection testing, model dropdown, workflow upload, and save.
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
        $('#setting-comfyui-negative').value = settings.comfyui_default_negative_prompt || '';

        // Initialize or update model dropdown
        _initModelDropdown(settings.comfyui_default_model || '');

        // Load workflow info
        await _loadWorkflowInfo();

        // Wire up file input change handler (once)
        _initWorkflowFileInput();

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

// ── API Workflow file upload ────────────────────────────────────────────

let _workflowFileInputWired = false;
let _uiWorkflowFileInputWired = false;

function _initWorkflowFileInput() {
    if (!_workflowFileInputWired) {
        const input = $('#workflow-file-input');
        if (input) {
            input.addEventListener('change', _handleWorkflowFileChange);
            _workflowFileInputWired = true;
        }
    }
    if (!_uiWorkflowFileInputWired) {
        const uiInput = $('#ui-workflow-file-input');
        if (uiInput) {
            uiInput.addEventListener('change', _handleUIWorkflowFileChange);
            _uiWorkflowFileInputWired = true;
        }
    }
}

async function _handleWorkflowFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusEl = $('#workflow-status');
    if (statusEl) {
        statusEl.textContent = 'Uploading…';
        statusEl.className = 'hint';
    }

    try {
        const result = await API.uploadWorkflow(file);
        if (result.error) {
            if (statusEl) {
                statusEl.textContent = `✗ ${result.error}`;
                statusEl.className = 'hint hint-error';
            }
        } else {
            _updateWorkflowDisplay(result.filename, result.workflow_name, true);
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

async function deleteUploadedWorkflow() {
    const statusEl = $('#workflow-status');

    try {
        await API.deleteWorkflow();
        _updateWorkflowDisplay('', null, false);
        _updateUIWorkflowDisplay('', false);
        if (statusEl) {
            statusEl.textContent = 'Workflows removed';
            statusEl.className = 'hint';
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = '✗ Failed to remove: ' + err.message;
            statusEl.className = 'hint hint-error';
        }
    }
}

// ── UI Workflow file upload ─────────────────────────────────────────────

async function _handleUIWorkflowFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusEl = $('#ui-workflow-status');
    if (statusEl) {
        statusEl.textContent = 'Uploading…';
        statusEl.className = 'hint';
    }

    try {
        const result = await API.uploadUIWorkflow(file);
        if (result.error) {
            if (statusEl) {
                statusEl.textContent = `✗ ${result.error}`;
                statusEl.className = 'hint hint-error';
            }
        } else {
            _updateUIWorkflowDisplay(result.filename, true);
            if (statusEl) {
                statusEl.textContent = `✓ ${result.message || result.filename}`;
                statusEl.className = 'hint hint-success';
            }
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = '✗ Upload failed: ' + err.message;
            statusEl.className = 'hint hint-error';
        }
    }

    e.target.value = '';
}

async function deleteUploadedUIWorkflow() {
    const statusEl = $('#ui-workflow-status');

    try {
        await API.deleteUIWorkflow();
        _updateUIWorkflowDisplay('', false);
        if (statusEl) {
            statusEl.textContent = 'UI workflow removed';
            statusEl.className = 'hint';
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = '✗ Failed to remove: ' + err.message;
            statusEl.className = 'hint hint-error';
        }
    }
}

// ── Load workflow info ──────────────────────────────────────────────────

async function _loadWorkflowInfo() {
    try {
        const info = await API.getWorkflowInfo();

        // API workflow
        _updateWorkflowDisplay(
            info.filename || '',
            info.workflow_name || null,
            info.has_workflow || false,
        );

        const statusEl = $('#workflow-status');
        if (statusEl && info.has_workflow) {
            if (info.valid) {
                statusEl.textContent = `✓ ${info.workflow_name || 'Valid API workflow'}`;
                statusEl.className = 'hint hint-success';
            } else {
                statusEl.textContent = '✗ No matching workflow definition';
                statusEl.className = 'hint hint-error';
            }
        }

        // UI workflow
        _updateUIWorkflowDisplay(
            info.ui_filename || '',
            info.has_ui_workflow || false,
        );

        const uiStatusEl = $('#ui-workflow-status');
        if (uiStatusEl && info.has_ui_workflow) {
            uiStatusEl.textContent = `✓ ${info.ui_filename}`;
            uiStatusEl.className = 'hint hint-success';
        }
    } catch (err) {
        console.warn('Failed to load workflow info:', err);
    }
}

function _updateWorkflowDisplay(filename, workflowName, hasWorkflow) {
    const filenameEl = $('#workflow-filename');
    const deleteBtn = $('#workflow-delete-btn');

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
    const filenameEl = $('#ui-workflow-filename');
    const deleteBtn = $('#ui-workflow-delete-btn');

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
    data.comfyui_default_negative_prompt = negative;

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
