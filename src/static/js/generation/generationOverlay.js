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

// Internal state for the current overlay open
let _currentGenChatId = null;
let _currentGenMessageId = null;

// Seed from a previous generation (shown as a clickable hint)
let _previousGenSeed = null;

// Cache for fetched lists (avoid re-fetching every open)
let _cachedModels = null;
let _cachedLoras = null;
let _cachedFolders = null;

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
    } = options;

    // Store chat/message context; fall back to global currentChatId
    _currentGenChatId = chatId || (typeof currentChatId !== 'undefined' ? currentChatId : null);
    _currentGenMessageId = messageId || null;

    // Show modal
    const overlay = $('#generation-overlay');
    if (!overlay) return;
    overlay.classList.remove('hidden');

    // Initialise widgets on first open
    _initGenWidgets();

    // Load model/lora/folder lists (folders always refreshed, models/loras cached)
    _cachedFolders = null;
    await _loadGenerationData();

    // Determine what values to fill.
    // Priority: 1) explicit settings (regenerate), 2) defaultSettings from
    // current chat's bubbles, 3) per-chat session memory, 4) first-time defaults.
    // This ensures switching threads uses the correct thread's settings.
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
    } else if (defaultSettings) {
        // Use stored bubble settings from the current chat (base_model, loras, output_folder).
        // This takes priority over session memory to ensure the current thread's
        // settings are used even if the user generated in a different thread recently.
        _previousGenSeed = null;
        _fillOverlayFields({
            positive_prompt: prompt,
            base_model: defaultSettings.base_model || '',
            loras: defaultSettings.loras || [],
            output_folder: defaultSettings.output_folder || '',
            seed: -1,
            num_images: 1,
        });
    } else if (chatSessionSettings) {
        // Returning user on same chat: use last settings from this chat + new prompt.
        // Preserve the source image's seed as a clickable hint.
        const srcSeed = chatSessionSettings.seed;
        _previousGenSeed = (srcSeed != null && srcSeed !== -1) ? srcSeed : null;
        _fillOverlayFields({ ...chatSessionSettings, positive_prompt: prompt, seed: -1 });
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

// ── Submit generation ───────────────────────────────────────────────────

async function submitGeneration() {
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
            result = await API.browserGenerate(settings);
        } else {
            result = await API.submitGeneration(chatId, messageId, settings);
        }

        // Close overlay
        closeGenerationOverlay();

        // Dispatch custom event so Phase 8 (generation bubbles) can react
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

// ── Internal: fill overlay fields ───────────────────────────────────────

function _fillOverlayFields(settings) {
    const textarea = $('#gen-prompt');
    if (textarea) textarea.value = settings.positive_prompt || '';

    if (_genModelDropdown && settings.base_model) {
        _genModelDropdown.setValue(settings.base_model);
    }

    if (_genLoraInput) {
        _genLoraInput.setValue(Array.isArray(settings.loras) ? settings.loras : []);
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
