/**
 * Generation overlay — configurable image generation dialog.
 * Reusable: can be opened from prompt blocks, thumbnail regenerate icons,
 * or future image browser.
 *
 * Depends on: api.js (API), app.js ($, escapeHtml),
 *             searchableDropdown.js (createSearchableDropdown),
 *             pillInput.js (createPillInput)
 */

// Per-chat session-level last-used settings (not persisted to DB).
// Maps chatId → settings object. Ensures switching threads doesn't
// bleed settings from one chat into another.
let _lastGenerationSettingsPerChat = {};

// Widget instances (created once, reused across opens)
let _genModelDropdown = null;
let _genLoraInput = null;

// Grid tab widget instances
let _gridModelPills = null;
let _gridLoraPills = null;
let _gridSamplerPills = null;
let _gridSchedulerPills = null;
let _gridWidgetsInitialized = false;

// Cached sampler options from ComfyUI
let _cachedSamplerOptions = null;

// Active tab: 'single' or 'grid'
let _activeGenTab = 'single';

// Previous grid session data (for seed reuse)
let _previousGridSeeds = null;

// Internal state for the current overlay open
let _currentGenChatId = null;
let _currentGenMessageId = null;

// Seed from a previous generation (shown as a clickable hint)
let _previousGenSeed = null;

// Parent job ID for regeneration lineage tracking
let _parentJobId = null;

// Cache for fetched lists (avoid re-fetching every open)
let _cachedModels = null;
let _cachedLoras = null;
let _cachedFolders = null;

// Cached most-recent job settings (for "pull from latest" buttons)
let _cachedRecentSettings = null;


// ── Open the overlay ────────────────────────────────────────────────────

/**
 * Open the generation overlay.
 * @param {Object|string} options - Options object or just the prompt string.
 * @param {string} options.prompt - Positive prompt text to pre-fill
 * @param {string|null} [options.chatId] - Chat ID for submission
 * @param {number|null} [options.messageId] - Message ID the prompt came from
 * @param {Object|null} [options.settings] - Full settings to pre-fill (for regenerate)
 * @param {Object|null} [options.defaultSettings] - Partial defaults (base_model, loras, output_folder) from generation bubble
 */
async function openGenerationOverlay(options) {
    // Accept a plain string as shorthand
    if (typeof options === 'string') {
        options = { prompt: options };
    }

    const {
        prompt = '',
        chatId = null,
        messageId = null,
        settings = null,
        defaultSettings = null,
        parentJobId = null,
        noChatContext = false,
    } = options;

    _parentJobId = parentJobId;

    // Store chat/message context; fall back to global currentChatId
    _currentGenChatId = noChatContext ? null : (chatId || (typeof currentChatId !== 'undefined' ? currentChatId : null));
    _currentGenMessageId = messageId || null;

    // Show modal
    const overlay = $('#generation-overlay');
    if (!overlay) return;
    overlay.classList.remove('hidden');

    // Reset to single tab
    _activeGenTab = 'single';
    $$('.gen-tab').forEach(b => b.classList.toggle('active', b.dataset.genTab === 'single'));
    $('#gen-tab-single')?.classList.remove('hidden');
    $('#gen-tab-grid')?.classList.add('hidden');

    // Initialise widgets on first open
    _initGenWidgets();

    // Pre-fetch most-recent job settings for "pull from latest" buttons
    API.getMostRecentGenerationSettings().then(s => { _cachedRecentSettings = s; });

    // Load model/lora/folder lists (folders always refreshed, models/loras cached)
    _cachedFolders = null;
    await _loadGenerationData();

    // Determine what values to fill.
    // Priority: 1) explicit settings (regenerate), 2) per-chat session memory,
    // 3) defaultSettings from bubbles (fallback after reload), 4) first-time defaults.
    // Session memory is updated on every submission so it always reflects the
    // user's most recent choices.  Bubble-based defaultSettings only carry
    // base_model/loras/output_folder and are useful after a page reload when
    // session memory has been lost.
    const chatSessionSettings = _currentGenChatId
        ? _lastGenerationSettingsPerChat[_currentGenChatId] || null
        : null;

    if (settings) {
        // Regenerate mode: remember the actual seed but default input to -1
        const actualSeed = settings.seed != null ? settings.seed : null;
        _previousGenSeed = (actualSeed != null && actualSeed !== -1) ? actualSeed : null;
        // Filter unavailable models/loras against cached lists
        const filtered = { ...settings, seed: -1 };
        if (filtered.base_model && _cachedModels && !_cachedModels.includes(filtered.base_model)) {
            filtered.base_model = '';
        }
        if (Array.isArray(filtered.loras) && _cachedLoras) {
            filtered.loras = filtered.loras.filter(l => {
                const name = typeof l === 'object' ? l.name : l;
                return _cachedLoras.includes(name);
            });
        }
        _fillOverlayFields(filtered);
    } else if (chatSessionSettings) {
        // Session memory from this chat: always up-to-date within the session
        // since it's written on every submission. Use new prompt + last settings.
        const srcSeed = chatSessionSettings.seed;
        _previousGenSeed = (srcSeed != null && srcSeed !== -1) ? srcSeed : null;
        _fillOverlayFields({ ...chatSessionSettings, positive_prompt: prompt, seed: -1 });
    } else if (defaultSettings) {
        // Fallback after page reload: bubble DOM stores base_model, loras,
        // output_folder from the last completed job. Session memory is lost
        // on reload so this provides continuity for those three fields.
        _previousGenSeed = null;
        _fillOverlayFields({
            positive_prompt: prompt,
            base_model: defaultSettings.base_model || '',
            loras: defaultSettings.loras || [],
            output_folder: defaultSettings.output_folder || '',
            seed: -1,
            num_images: 1,
        });
    } else {
        // First time: defaults + prompt; try to get default model from settings
        _previousGenSeed = null;
        _fillOverlayDefaults(prompt);
    }

    // Show/hide the "use previous seed" hint button
    _updateSeedHint();

    // Focus the prompt textarea
    const textarea = $('#gen-prompt');
    if (textarea) textarea.focus();
}

// ── Open from header button ──────────────────────────────────────────────

async function openHeaderGenerate() {
    let latestSettings = null;
    try {
        latestSettings = await API.getLatestGenerationSettings();
    } catch (_) {}

    openGenerationOverlay({
        prompt: '',
        chatId: null,
        messageId: null,
        settings: null,
        defaultSettings: latestSettings ? {
            base_model: latestSettings.base_model || '',
            loras: latestSettings.loras || [],
            output_folder: latestSettings.output_folder || '',
        } : null,
        parentJobId: null,
        noChatContext: true,
    });
}

// ── Refine settings from browser ─────────────────────────────────────────

/**
 * Store source image settings as the current chat's generation defaults.
 * Called from app.js when navigating from browser refine/refine+attach.
 * These settings are used when the user later opens the generation overlay
 * from a prompt block in this chat.
 */
function setRefineGenerationSettings(settings) {
    const chatId = typeof currentChatId !== 'undefined' ? currentChatId : null;
    if (!chatId || !settings) return;
    _lastGenerationSettingsPerChat[chatId] = {
        base_model: settings.base_model || '',
        loras: settings.loras || [],
        output_folder: settings.output_folder || '',
        seed: settings.seed != null ? settings.seed : -1,
    };
}

// ── Close the overlay ───────────────────────────────────────────────────

function closeGenerationOverlay() {
    const overlay = $('#generation-overlay');
    if (overlay) overlay.classList.add('hidden');
    _parentJobId = null;
    window.dispatchEvent(new CustomEvent('overlay-closed'));
}

// ── Random seed ─────────────────────────────────────────────────────────

function generateRandomSeed() {
    const seed = Math.floor(Math.random() * 1125899906842624);
    const input = $('#gen-seed');
    if (input) input.value = seed;
}

// ── Use previous seed ───────────────────────────────────────────────────

/**
 * Apply the previous generation's seed value to the seed input.
 * Called when user clicks the "Use seed: …" hint button.
 */
function usePreviousSeed() {
    if (_previousGenSeed == null) return;
    const input = $('#gen-seed');
    if (input) input.value = _previousGenSeed;
}

/**
 * Show or hide the "use previous seed" hint button below the seed input.
 */
function _updateSeedHint() {
    const hintBtn = $('#gen-seed-hint');
    if (!hintBtn) return;

    if (_previousGenSeed != null) {
        hintBtn.textContent = `Use seed: ${_previousGenSeed}`;
        hintBtn.classList.remove('hidden');
    } else {
        hintBtn.classList.add('hidden');
    }
}

// ── Tab switching ──────────────────────────────────────────────────────

async function switchGenTab(btn) {
    const tab = btn.dataset.genTab;
    if (tab === _activeGenTab) return;
    _activeGenTab = tab;

    // Update tab buttons
    $$('.gen-tab').forEach(b => b.classList.toggle('active', b.dataset.genTab === tab));

    // Toggle panes
    $('#gen-tab-single').classList.toggle('hidden', tab !== 'single');
    $('#gen-tab-grid').classList.toggle('hidden', tab !== 'grid');

    // On first switch to grid, initialize grid widgets and populate
    if (tab === 'grid') {
        _initGridWidgets();
        await _loadGridData();
        await _restoreGridSession();
        // Sync values from single tab that grid doesn't have its own source for
        _syncSingleToGrid();
        _updateGridComboCount();
    }
}

/** Copy single-tab values to empty grid-tab fields. */
function _syncSingleToGrid() {
    // Output folder
    const singleFolder = ($('#gen-output-folder') || {}).value || '';
    const gridFolder = $('#gen-grid-output-folder');
    if (gridFolder && !gridFolder.value && singleFolder) {
        gridFolder.value = singleFolder;
    }

    // Prompt: always overwrite grid prompt from single tab if single has content
    // (the single tab prompt reflects what the user clicked — agent prompt block,
    // regenerate image, etc. — and should take priority over stale session data)
    const singlePrompt = ($('#gen-prompt') || {}).value || '';
    const gridPrompt = $('#gen-grid-prompt');
    if (gridPrompt && singlePrompt) {
        gridPrompt.value = singlePrompt;
    }

    // Model: copy from single tab dropdown if grid pills are empty
    if (_gridModelPills && _gridModelPills.getValue().length === 0) {
        const singleModel = _genModelDropdown ? _genModelDropdown.getValue() : '';
        if (singleModel) {
            _gridModelPills.setValue([singleModel]);
        }
    }

    // LoRAs: copy from single tab pills if grid pills are empty
    if (_gridLoraPills && _gridLoraPills.getValue().length === 0) {
        const singleLoras = _genLoraInput ? _genLoraInput.getValue() : [];
        if (singleLoras.length > 0) {
            _gridLoraPills.setValue(singleLoras);
        }
    }
}

// ── Grid widget initialization ─────────────────────────────────────────

function _initGridWidgets() {
    if (_gridWidgetsInitialized) return;

    const modelContainer = $('#gen-grid-models');
    if (modelContainer) {
        _gridModelPills = createPillInput({
            container: modelContainer,
            placeholder: 'Search models…',
            items: _cachedModels || [],
            truncateAt: 35,
            minRequired: 1,
            errorMessage: 'At least one model required',
            onChange: _updateGridComboCount,
        });
    }

    const loraContainer = $('#gen-grid-loras');
    if (loraContainer) {
        _gridLoraPills = createPillInput({
            container: loraContainer,
            placeholder: 'Search LoRAs…',
            items: _cachedLoras || [],
            truncateAt: 35,
            onChange: _updateGridComboCount,
        });
    }

    const samplerContainer = $('#gen-grid-samplers');
    if (samplerContainer) {
        _gridSamplerPills = createPillInput({
            container: samplerContainer,
            placeholder: 'Search samplers…',
            items: [],
            truncateAt: 40,
            minRequired: 1,
            errorMessage: 'At least one sampler required',
            onChange: _updateGridComboCount,
        });
    }

    const schedulerContainer = $('#gen-grid-schedulers');
    if (schedulerContainer) {
        _gridSchedulerPills = createPillInput({
            container: schedulerContainer,
            placeholder: 'Search schedulers…',
            items: [],
            truncateAt: 30,
            minRequired: 1,
            errorMessage: 'At least one scheduler required',
            onChange: _updateGridComboCount,
        });
    }

    // Wire up range inputs to update combo count
    ['gen-grid-cfg-start', 'gen-grid-cfg-stop', 'gen-grid-cfg-step',
     'gen-grid-steps-start', 'gen-grid-steps-stop', 'gen-grid-steps-step',
     'gen-grid-num-seeds'].forEach(id => {
        const el = $('#' + id);
        if (el) el.addEventListener('input', _updateGridComboCount);
    });

    // Wire up seed reuse checkbox
    const reuseCheckbox = $('#gen-grid-reuse-seeds');
    if (reuseCheckbox) {
        reuseCheckbox.addEventListener('change', _handleSeedReuseToggle);
    }

    _gridWidgetsInitialized = true;
}

async function _loadGridData() {
    // Populate sampler/scheduler options
    if (!_cachedSamplerOptions) {
        try {
            _cachedSamplerOptions = await API.getSamplerOptions();
        } catch (err) {
            console.warn('Failed to load sampler options:', err);
            _cachedSamplerOptions = { samplers: [], schedulers: [] };
        }
    }
    if (_gridSamplerPills) _gridSamplerPills.setItems(_cachedSamplerOptions.samplers || []);
    if (_gridSchedulerPills) _gridSchedulerPills.setItems(_cachedSamplerOptions.schedulers || []);

    // Update model/lora lists if available
    if (_gridModelPills && _cachedModels) _gridModelPills.setItems(_cachedModels);
    if (_gridLoraPills && _cachedLoras) _gridLoraPills.setItems(_cachedLoras);
}

// ── Grid session restore/save ──────────────────────────────────────────

async function _restoreGridSession() {
    const stored = sessionStorage.getItem('gridGenSettings');
    if (!stored) {
        // No session — fill defaults from settings
        await _fillGridDefaults();
        _previousGridSeeds = null;
        _updateSeedReuseUI();
        return;
    }

    try {
        const gs = JSON.parse(stored);

        // Prompt
        const prompt = $('#gen-grid-prompt');
        if (prompt) prompt.value = gs.positive_prompt || '';

        // Pill inputs
        if (_gridModelPills && gs.base_models) _gridModelPills.setValue(gs.base_models);
        if (_gridLoraPills && gs.loras) _gridLoraPills.setValue(gs.loras);
        if (_gridSamplerPills && gs.samplers) _gridSamplerPills.setValue(gs.samplers);
        if (_gridSchedulerPills && gs.schedulers) _gridSchedulerPills.setValue(gs.schedulers);

        // Output folder
        const folder = $('#gen-grid-output-folder');
        if (folder) folder.value = gs.output_folder || '';

        // Ranges
        if (gs.cfg_range) {
            const s = gs.cfg_range;
            $('#gen-grid-cfg-start').value = s.start ?? '';
            $('#gen-grid-cfg-stop').value = s.stop ?? '';
            $('#gen-grid-cfg-step').value = s.step ?? 0;
        }
        if (gs.steps_range) {
            const s = gs.steps_range;
            $('#gen-grid-steps-start').value = s.start ?? '';
            $('#gen-grid-steps-stop').value = s.stop ?? '';
            $('#gen-grid-steps-step').value = s.step ?? 0;
        }

        // Seeds
        const seedsInput = $('#gen-grid-num-seeds');
        if (seedsInput) seedsInput.value = gs.num_seeds || 1;

        // Previous seeds for reuse
        _previousGridSeeds = gs.seeds_used || null;
        _updateSeedReuseUI();

    } catch (err) {
        console.warn('Failed to restore grid session:', err);
        await _fillGridDefaults();
    }
}

async function _fillGridDefaults() {
    try {
        const appSettings = await API.getSettings();
        const cfg = parseFloat(appSettings.comfyui_default_cfg) || '';
        const steps = parseInt(appSettings.comfyui_default_steps, 10) || '';

        $('#gen-grid-cfg-start').value = cfg;
        $('#gen-grid-cfg-stop').value = cfg;
        $('#gen-grid-cfg-step').value = 0;
        $('#gen-grid-steps-start').value = steps;
        $('#gen-grid-steps-stop').value = steps;
        $('#gen-grid-steps-step').value = 0;

        // Pre-select current default sampler/scheduler
        if (appSettings.comfyui_default_sampler && _gridSamplerPills) {
            _gridSamplerPills.setValue([appSettings.comfyui_default_sampler]);
        }
        if (appSettings.comfyui_default_scheduler && _gridSchedulerPills) {
            _gridSchedulerPills.setValue([appSettings.comfyui_default_scheduler]);
        }
        // Pre-select default model
        if (appSettings.comfyui_default_model && _gridModelPills) {
            _gridModelPills.setValue([appSettings.comfyui_default_model]);
        }
    } catch (_) {}

    // Copy prompt from single tab if it has content
    const singlePrompt = ($('#gen-prompt') || {}).value || '';
    const gridPrompt = $('#gen-grid-prompt');
    if (gridPrompt && !gridPrompt.value && singlePrompt) {
        gridPrompt.value = singlePrompt;
    }
}

function _saveGridSession(gridSettings, seedsUsed) {
    const data = { ...gridSettings, seeds_used: seedsUsed };
    sessionStorage.setItem('gridGenSettings', JSON.stringify(data));
}

// ── Seed reuse logic ───────────────────────────────────────────────────

function _updateSeedReuseUI() {
    const row = $('#gen-grid-reuse-row');
    const checkbox = $('#gen-grid-reuse-seeds');
    const label = $('#gen-grid-reuse-text');
    const seedsInput = $('#gen-grid-num-seeds');

    if (!row || !checkbox || !label || !seedsInput) return;

    if (!_previousGridSeeds || _previousGridSeeds.length === 0) {
        row.style.display = 'none';
        checkbox.checked = false;
        seedsInput.disabled = false;
        return;
    }

    const prevCount = _previousGridSeeds.length;
    const currentCount = parseInt(seedsInput.value, 10) || 1;
    row.style.display = '';

    if (currentCount === prevCount) {
        checkbox.disabled = false;
        label.textContent = `Reuse ${prevCount} seed${prevCount > 1 ? 's' : ''} from previous run`;
    } else {
        checkbox.disabled = true;
        checkbox.checked = false;
        seedsInput.disabled = false;
        label.textContent = `Previous run used ${prevCount} seed${prevCount > 1 ? 's' : ''} — set count to ${prevCount} to reuse`;
    }
}

function _handleSeedReuseToggle() {
    const checkbox = $('#gen-grid-reuse-seeds');
    const seedsInput = $('#gen-grid-num-seeds');
    if (!checkbox || !seedsInput) return;

    if (checkbox.checked && _previousGridSeeds) {
        seedsInput.value = _previousGridSeeds.length;
        seedsInput.disabled = true;
    } else {
        seedsInput.disabled = false;
    }
    _updateGridComboCount();
}

// ── Grid combination counter ───────────────────────────────────────────

function _expandRangeCount(startId, stopId, stepId, isFloat) {
    const start = parseFloat($('#' + startId)?.value);
    const stop = parseFloat($('#' + stopId)?.value);
    const step = parseFloat($('#' + stepId)?.value);

    if (isNaN(start)) return 1;  // Use default
    if (isNaN(stop) || isNaN(step) || step === 0) return 1;
    if (stop < start) return 1;

    return Math.floor((stop - start) / step + 1.0001);
}

function _updateGridComboCount() {
    const models = _gridModelPills ? _gridModelPills.getValue().length : 1;
    const loras = _gridLoraPills ? Math.max(_gridLoraPills.getValue().length, 1) : 1;
    const samplers = _gridSamplerPills ? _gridSamplerPills.getValue().length : 1;
    const schedulers = _gridSchedulerPills ? _gridSchedulerPills.getValue().length : 1;
    const cfgCount = _expandRangeCount('gen-grid-cfg-start', 'gen-grid-cfg-stop', 'gen-grid-cfg-step', true);
    const stepsCount = _expandRangeCount('gen-grid-steps-start', 'gen-grid-steps-stop', 'gen-grid-steps-step', false);
    const seeds = parseInt($('#gen-grid-num-seeds')?.value, 10) || 1;

    const combos = Math.max(models, 0) * loras * Math.max(samplers, 0) * Math.max(schedulers, 0) * cfgCount * stepsCount;
    const total = combos * seeds;

    const comboEl = $('#gen-grid-combo-count');
    const totalEl = $('#gen-grid-total-count');
    if (comboEl) comboEl.textContent = `${combos} combination${combos !== 1 ? 's' : ''}`;
    if (totalEl) {
        totalEl.textContent = `${total} total image${total !== 1 ? 's' : ''}`;
        totalEl.classList.toggle('gen-grid-warn', total > 50);
    }

    // Update seed reuse UI on count change
    _updateSeedReuseUI();
}

// ── Submit generation ───────────────────────────────────────────────────

async function submitGeneration() {
    if (_activeGenTab === 'grid') {
        return _submitGridGeneration();
    }
    return _submitSingleGeneration();
}

async function _submitSingleGeneration() {
    const btn = $('#gen-submit-btn');
    if (!btn) return;

    // Prevent double-click
    if (btn.disabled) return;
    btn.disabled = true;
    btn.textContent = 'Submitting…';

    try {
        const settings = _gatherOverlayValues();

        // Persist to per-chat session memory
        if (_currentGenChatId) {
            _lastGenerationSettingsPerChat[_currentGenChatId] = { ...settings };
        }

        // Build chat/message IDs
        const chatId = _currentGenChatId;
        const messageId = _currentGenMessageId;

        let result;
        if (!chatId) {
            // Browser mode: no chat context
            result = await API.browserGenerate(settings, _parentJobId);
        } else {
            result = await API.submitGeneration(chatId, messageId, settings, _parentJobId);
        }

        // Close overlay
        closeGenerationOverlay();

        // Dispatch custom event so generation bubbles can react
        window.dispatchEvent(new CustomEvent('generation-submitted', {
            detail: { ...result, chatId, messageId, settings },
        }));
    } catch (err) {
        console.error('Generation submission failed:', err);
        alert('Failed to submit generation: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

async function _submitGridGeneration() {
    const btn = $('#gen-submit-btn');
    if (!btn) return;

    // Validate required pill inputs
    let valid = true;
    if (_gridModelPills && !_gridModelPills.validate()) valid = false;
    if (_gridSamplerPills && !_gridSamplerPills.validate()) valid = false;
    if (_gridSchedulerPills && !_gridSchedulerPills.validate()) valid = false;
    if (!valid) return;

    if (btn.disabled) return;
    btn.disabled = true;
    btn.textContent = 'Submitting…';

    try {
        const gridSettings = _gatherGridValues();

        const chatId = _currentGenChatId;
        const messageId = _currentGenMessageId;

        let result;
        if (!chatId) {
            result = await API.browserGridGenerate(gridSettings, _parentJobId);
        } else {
            result = await API.submitGridGeneration(chatId, messageId, gridSettings, _parentJobId);
        }

        // Save session (including seeds_used from response)
        _previousGridSeeds = result.seeds_used || null;
        _saveGridSession(gridSettings, result.seeds_used || []);

        closeGenerationOverlay();

        // Dispatch events for each job
        if (result.jobs) {
            for (const job of result.jobs) {
                window.dispatchEvent(new CustomEvent('generation-submitted', {
                    detail: { ...job, chatId, messageId },
                }));
            }
        }
    } catch (err) {
        console.error('Grid generation submission failed:', err);
        alert('Failed to submit grid generation: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate';
    }
}

function _gatherGridValues() {
    const reuseCheckbox = $('#gen-grid-reuse-seeds');
    const reusingSeeds = reuseCheckbox?.checked && _previousGridSeeds;

    return {
        positive_prompt: ($('#gen-grid-prompt') || {}).value || '',
        base_models: _gridModelPills ? _gridModelPills.getValue() : [],
        loras: _gridLoraPills ? _gridLoraPills.getValue() : [],
        samplers: _gridSamplerPills ? _gridSamplerPills.getValue() : [],
        schedulers: _gridSchedulerPills ? _gridSchedulerPills.getValue() : [],
        output_folder: ($('#gen-grid-output-folder') || {}).value || '',
        cfg_range: {
            start: parseFloat($('#gen-grid-cfg-start')?.value) || null,
            stop: parseFloat($('#gen-grid-cfg-stop')?.value) || null,
            step: parseFloat($('#gen-grid-cfg-step')?.value) || 0,
        },
        steps_range: {
            start: parseInt($('#gen-grid-steps-start')?.value, 10) || null,
            stop: parseInt($('#gen-grid-steps-stop')?.value, 10) || null,
            step: parseInt($('#gen-grid-steps-step')?.value, 10) || 0,
        },
        num_seeds: parseInt($('#gen-grid-num-seeds')?.value, 10) || 1,
        seeds: reusingSeeds ? _previousGridSeeds : null,
    };
}

// ── Internal: initialise widgets ────────────────────────────────────────

function _initGenWidgets() {
    // Only create once
    if (_genModelDropdown) return;

    const modelContainer = $('#gen-model-dropdown');
    if (modelContainer) {
        _genModelDropdown = createSearchableDropdown({
            container: modelContainer,
            placeholder: 'Select a diffusion model…',
            items: [],
            value: '',
        });
    }

    const loraContainer = $('#gen-lora-input');
    if (loraContainer) {
        _genLoraInput = createPillInput({
            container: loraContainer,
            placeholder: 'Search LoRAs…',
            items: [],
            truncateAt: 35,
        });
    }

    // "Pull from latest" icon buttons
    document.querySelectorAll('.gen-pull-latest').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!_cachedRecentSettings) return;
            _applyLatestField(btn.dataset.field, _cachedRecentSettings);
        });
    });
}

// ── Internal: load models, loras, and output folders ────────────────────

async function _loadGenerationData() {
    const promises = [];

    // Models
    if (!_cachedModels) {
        promises.push(
            API.getComfyUIModels('diffusion_models')
                .then(models => {
                    if (Array.isArray(models)) {
                        _cachedModels = models;
                        if (_genModelDropdown) _genModelDropdown.setItems(models);
                    }
                })
                .catch(err => console.warn('Failed to load models:', err))
        );
    } else if (_genModelDropdown) {
        _genModelDropdown.setItems(_cachedModels);
    }

    // LoRAs
    if (!_cachedLoras) {
        promises.push(
            API.getComfyUIModels('loras')
                .then(loras => {
                    if (Array.isArray(loras)) {
                        _cachedLoras = loras;
                        if (_genLoraInput) _genLoraInput.setItems(loras);
                    }
                })
                .catch(err => console.warn('Failed to load loras:', err))
        );
    } else if (_genLoraInput) {
        _genLoraInput.setItems(_cachedLoras);
    }

    // Output folders
    if (!_cachedFolders) {
        promises.push(
            API.getComfyUIOutputFolders()
                .then(folders => {
                    if (Array.isArray(folders)) {
                        _cachedFolders = folders;
                        _populateFolderDatalist(folders);
                    }
                })
                .catch(err => console.warn('Failed to load output folders:', err))
        );
    } else {
        _populateFolderDatalist(_cachedFolders);
    }

    if (promises.length > 0) {
        await Promise.allSettled(promises);
    }
}

function _populateFolderDatalist(folders) {
    const datalist = $('#gen-folder-list');
    if (!datalist) return;
    datalist.innerHTML = folders.map(f => `<option value="${escapeHtml(f)}">`).join('');
}

// ── Internal: apply a single field from most-recent job settings ────────

function _applyLatestField(field, settings) {
    switch (field) {
        case 'base_model':
            if (_genModelDropdown && settings.base_model) {
                _genModelDropdown.setValue(settings.base_model);
            }
            break;
        case 'loras':
            if (_genLoraInput) {
                const raw = Array.isArray(settings.loras) ? settings.loras : [];
                const normalized = raw.map(l => typeof l === 'object' ? l.name : l);
                _genLoraInput.setValue(normalized);
            }
            break;
        case 'output_folder': {
            const fi = document.getElementById('gen-output-folder');
            if (fi) fi.value = settings.output_folder || '';
            break;
        }
    }
}

// ── Internal: fill overlay fields ───────────────────────────────────────

function _fillOverlayFields(settings) {
    const textarea = $('#gen-prompt');
    if (textarea) textarea.value = settings.positive_prompt || '';

    if (_genModelDropdown && settings.base_model) {
        _genModelDropdown.setValue(settings.base_model);
    }

    if (_genLoraInput) {
        const rawLoras = Array.isArray(settings.loras) ? settings.loras : [];
        // Normalize: agent tool stores {name, strength} objects, overlay uses strings
        const normalized = rawLoras.map(l => typeof l === 'object' ? l.name : l);
        _genLoraInput.setValue(normalized);
    }

    const folderInput = $('#gen-output-folder');
    if (folderInput) folderInput.value = settings.output_folder || '';

    const seedInput = $('#gen-seed');
    if (seedInput) seedInput.value = settings.seed != null ? settings.seed : -1;

    const numInput = $('#gen-num-images');
    if (numInput) numInput.value = settings.num_images || 1;
}

async function _fillOverlayDefaults(prompt) {
    const textarea = $('#gen-prompt');
    if (textarea) textarea.value = prompt || '';

    // Try to get the default model and sampler settings from app settings
    try {
        const appSettings = await API.getSettings();
        if (_genModelDropdown && appSettings.comfyui_default_model) {
            _genModelDropdown.setValue(appSettings.comfyui_default_model);
        }
        // Apply default sampler settings if fields exist
        const samplerInput = $('#gen-sampler');
        if (samplerInput && appSettings.comfyui_default_sampler) {
            samplerInput.value = appSettings.comfyui_default_sampler;
        }
        const cfgInput = $('#gen-cfg');
        if (cfgInput && appSettings.comfyui_default_cfg) {
            cfgInput.value = appSettings.comfyui_default_cfg;
        }
        const schedulerInput = $('#gen-scheduler');
        if (schedulerInput && appSettings.comfyui_default_scheduler) {
            schedulerInput.value = appSettings.comfyui_default_scheduler;
        }
        const stepsInput = $('#gen-steps');
        if (stepsInput && appSettings.comfyui_default_steps) {
            stepsInput.value = appSettings.comfyui_default_steps;
        }
    } catch (_) {
        // Non-critical
    }

    if (_genLoraInput) _genLoraInput.setValue([]);

    const folderInput = $('#gen-output-folder');
    if (folderInput) folderInput.value = '';

    const seedInput = $('#gen-seed');
    if (seedInput) seedInput.value = -1;

    const numInput = $('#gen-num-images');
    if (numInput) numInput.value = 1;
}

// ── Internal: gather values from the form ───────────────────────────────

function _gatherOverlayValues() {
    return {
        positive_prompt: ($('#gen-prompt') || {}).value || '',
        base_model: _genModelDropdown ? _genModelDropdown.getValue() : '',
        loras: _genLoraInput ? _genLoraInput.getValue() : [],
        output_folder: ($('#gen-output-folder') || {}).value || '',
        seed: parseInt(($('#gen-seed') || {}).value, 10) || -1,
        num_images: parseInt(($('#gen-num-images') || {}).value, 10) || 1,
    };
}
