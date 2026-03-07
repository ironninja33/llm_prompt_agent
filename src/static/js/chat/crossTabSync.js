/**
 * Cross-tab synchronization via BroadcastChannel.
 *
 * Broadcasts chat streaming events so other tabs can show the live stream.
 * Feature-gated: works only if BroadcastChannel is available.
 *
 * Depends on: chatPane.js (appendToStreamingMessage, finalizeStreamingMessage, etc.)
 */

const CrossTabSync = (() => {
    if (!('BroadcastChannel' in window)) {
        // No-op stub for browsers without BroadcastChannel
        return {
            broadcast: () => {},
            init: () => {},
        };
    }

    const channel = new BroadcastChannel('llm_prompt_agent_chat');
    let _tokenBuffer = '';
    let _tokenFlushTimer = null;
    const TOKEN_DEBOUNCE_MS = 150;

    function broadcast(type, data) {
        if (type === 'stream_token') {
            // Buffer tokens and send in batches
            _tokenBuffer += (data.text || '');
            if (!_tokenFlushTimer) {
                _tokenFlushTimer = setTimeout(() => {
                    channel.postMessage({ type: 'stream_token', chatId: data.chatId, text: _tokenBuffer });
                    _tokenBuffer = '';
                    _tokenFlushTimer = null;
                }, TOKEN_DEBOUNCE_MS);
            }
            return;
        }

        // Flush any pending tokens before sending other events
        if (_tokenFlushTimer && (type === 'stream_done' || type === 'stream_error')) {
            clearTimeout(_tokenFlushTimer);
            if (_tokenBuffer) {
                channel.postMessage({ type: 'stream_token', chatId: data.chatId, text: _tokenBuffer });
                _tokenBuffer = '';
            }
            _tokenFlushTimer = null;
        }

        channel.postMessage({ type, ...data });
    }

    function init() {
        channel.onmessage = (event) => {
            const msg = event.data;
            if (!msg || !msg.type) return;

            switch (msg.type) {
                case 'stream_started':
                    _onRemoteStreamStarted(msg);
                    break;
                case 'stream_token':
                    _onRemoteStreamToken(msg);
                    break;
                case 'stream_done':
                    _onRemoteStreamDone(msg);
                    break;
                case 'chat_created':
                case 'chat_deleted':
                    _onRemoteChatListChanged(msg);
                    break;
            }
        };
    }

    function _onRemoteStreamStarted(msg) {
        // Only show streaming cursor if viewing that chat and not already streaming locally
        if (currentChatId !== msg.chatId) return;
        if (StreamRegistry.isActive(msg.chatId)) return;

        // Add streaming message element
        addStreamingMessage();
    }

    function _onRemoteStreamToken(msg) {
        if (currentChatId !== msg.chatId) return;
        if (StreamRegistry.isActive(msg.chatId)) return;

        const el = document.getElementById('streaming-message');
        if (!el) {
            // Stream started in another tab before this tab opened — create element
            addStreamingMessage();
        }
        appendToStreamingMessage(msg.text || '');
    }

    function _onRemoteStreamDone(msg) {
        if (currentChatId !== msg.chatId) return;
        if (StreamRegistry.isActive(msg.chatId)) return;

        // Reload full messages from DB to get the finalized response
        loadMessages(msg.chatId);
    }

    function _onRemoteChatListChanged(_msg) {
        // Refresh sidebar
        if (typeof loadChats === 'function') {
            loadChats();
        }
    }

    return { broadcast, init };
})();
