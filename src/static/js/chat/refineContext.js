/**
 * Refine Context — "reply-to" quote bar for prompt refinement.
 *
 * When the user clicks a refine icon on a prompt block or generated image,
 * the prompt text is stored and a quote bar appears above the input textarea.
 * On send, the prompt is prepended to the user's message in a structured format.
 *
 * Provides: setRefineContext(), clearRefineContext(), getRefineContext(),
 *           hasRefineContext(), buildRefineMessage()
 *
 * Depends on: app.js ($)
 */

let _refinePrompt = null;  // The full prompt text being refined

/**
 * Set the refine context and show the quote bar.
 * @param {string} promptText - The full prompt to refine
 */
function setRefineContext(promptText) {
    if (!promptText || !promptText.trim()) return;

    _refinePrompt = promptText.trim();
    _renderRefineBar();

    // Focus the input textarea so user can start typing immediately
    const input = $('#message-input');
    if (input) input.focus();
}

/**
 * Clear the refine context and hide the quote bar.
 */
function clearRefineContext() {
    _refinePrompt = null;
    _renderRefineBar();
}

/**
 * Get the current refine prompt text (or null).
 * @returns {string|null}
 */
function getRefineContext() {
    return _refinePrompt;
}

/**
 * Check if there's an active refine context.
 * @returns {boolean}
 */
function hasRefineContext() {
    return _refinePrompt !== null;
}

/**
 * Build the combined message for sending to the LLM.
 * Prepends the refine context to the user's typed text.
 * @param {string} userText - The text the user typed in the input
 * @returns {string} - The combined message
 */
function buildRefineMessage(userText) {
    if (!_refinePrompt) return userText;

    const instruction = userText.trim() || 'Suggest improvements to this prompt';

    return `[Refining prompt]: "${_refinePrompt}"\n\nRequested changes: ${instruction}`;
}

/**
 * Render or hide the refine bar based on current state.
 */
function _renderRefineBar() {
    const bar = document.getElementById('refine-bar');
    if (!bar) return;

    if (!_refinePrompt) {
        bar.classList.add('hidden');
        bar.innerHTML = '';
        return;
    }

    bar.classList.remove('hidden');

    // Truncate display text for the bar
    const maxLen = 120;
    const displayText = _refinePrompt.length > maxLen
        ? _refinePrompt.substring(0, maxLen) + '…'
        : _refinePrompt;

    bar.innerHTML = '';

    const icon = document.createElement('span');
    icon.className = 'refine-bar-icon';
    icon.textContent = '✏️';
    bar.appendChild(icon);

    const label = document.createElement('span');
    label.className = 'refine-bar-label';
    label.textContent = 'Refining:';
    bar.appendChild(label);

    const text = document.createElement('span');
    text.className = 'refine-bar-text';
    text.textContent = displayText;
    text.title = _refinePrompt;  // Full text on hover
    bar.appendChild(text);

    const dismiss = document.createElement('button');
    dismiss.className = 'refine-bar-dismiss';
    dismiss.innerHTML = '&times;';
    dismiss.title = 'Cancel refinement';
    dismiss.onclick = (e) => {
        e.preventDefault();
        clearRefineContext();
    };
    bar.appendChild(dismiss);
}
