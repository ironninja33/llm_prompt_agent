/**
 * Stream Registry — per-chat streaming state manager.
 *
 * Tracks active agent SSE streams per chat, buffering tokens so that
 * switching away from a chat and back restores the in-progress response.
 *
 * On page refresh or browser reopen, the registry is empty (all JS state
 * is lost). This is correct: the SSE connection is also lost, so no
 * streams are active. selectChat() will call setStreaming(false) to
 * ensure the input is unlocked.
 *
 * Depends on: nothing (loaded before chatPane.js, sidebar.js)
 */

const _activeStreams = {};

const StreamRegistry = {
    /**
     * Register a new active stream for a chat.
     * @param {string} chatId
     */
    register(chatId) {
        _activeStreams[chatId] = {
            buffer: '',
            toolCalls: [],
            messageId: null,
            phase: 'streaming',
            abortController: null,
            userText: '',
            userMessageId: null,
        };
    },

    /**
     * Append token text to the stream buffer.
     * @param {string} chatId
     * @param {string} text
     */
    appendText(chatId, text) {
        if (_activeStreams[chatId]) {
            _activeStreams[chatId].buffer += text;
        }
    },

    /**
     * Get the accumulated text buffer for a chat's stream.
     * @param {string} chatId
     * @returns {string}
     */
    getBuffer(chatId) {
        return _activeStreams[chatId]?.buffer || '';
    },

    /**
     * Store tool call introspection data.
     * @param {string} chatId
     * @param {Array} calls
     */
    setToolCalls(chatId, calls) {
        if (_activeStreams[chatId]) {
            _activeStreams[chatId].toolCalls = calls;
        }
    },

    /**
     * Get stored tool calls for a chat's stream.
     * @param {string} chatId
     * @returns {Array}
     */
    getToolCalls(chatId) {
        return _activeStreams[chatId]?.toolCalls || [];
    },

    /**
     * Mark a stream as done with the final message ID.
     * @param {string} chatId
     * @param {number} messageId
     */
    finalize(chatId, messageId) {
        if (_activeStreams[chatId]) {
            _activeStreams[chatId].messageId = messageId;
            _activeStreams[chatId].phase = 'done';
        }
    },

    /**
     * Mark a stream as errored.
     * @param {string} chatId
     */
    setError(chatId) {
        if (_activeStreams[chatId]) {
            _activeStreams[chatId].phase = 'error';
        }
    },

    /**
     * Remove a stream entry (call after finalizing/handling).
     * @param {string} chatId
     */
    cleanup(chatId) {
        delete _activeStreams[chatId];
    },

    /**
     * Check if a chat has an active (in-progress) stream.
     * @param {string} chatId
     * @returns {boolean}
     */
    isActive(chatId) {
        const s = _activeStreams[chatId];
        return !!(s && s.phase === 'streaming');
    },

    /**
     * Get the phase of a chat's stream.
     * @param {string} chatId
     * @returns {string|null}
     */
    getPhase(chatId) {
        return _activeStreams[chatId]?.phase || null;
    },

    /**
     * Check if any chat has an active stream.
     * @returns {boolean}
     */
    hasAnyActive() {
        return Object.values(_activeStreams).some(s => s.phase === 'streaming');
    },

    /**
     * Get list of chat IDs with active streams.
     * @returns {string[]}
     */
    getActiveChats() {
        return Object.keys(_activeStreams).filter(id => _activeStreams[id].phase === 'streaming');
    },

    // ── Cancel support ───────────────────────────────────────────

    setAbortController(chatId, ctrl) {
        if (_activeStreams[chatId]) _activeStreams[chatId].abortController = ctrl;
    },

    getAbortController(chatId) {
        return _activeStreams[chatId]?.abortController || null;
    },

    setUserText(chatId, text) {
        if (_activeStreams[chatId]) _activeStreams[chatId].userText = text;
    },

    getUserText(chatId) {
        return _activeStreams[chatId]?.userText || '';
    },

    setUserMessageId(chatId, id) {
        if (_activeStreams[chatId]) _activeStreams[chatId].userMessageId = id;
    },

    getUserMessageId(chatId) {
        return _activeStreams[chatId]?.userMessageId || null;
    },

    /**
     * Abort the stream for a chat (cancel in progress).
     * @param {string} chatId
     */
    abort(chatId) {
        const s = _activeStreams[chatId];
        if (!s) return;
        if (s.abortController) s.abortController.abort();
        s.phase = 'cancelled';
    },
};
