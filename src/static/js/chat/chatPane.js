/**
 * LLM Prompt Agent — Message rendering and sending.
 *
 * Depends on: api.js, sse.js, markdown.js,
 *             app.js ($, $$, escapeHtml, currentChatId, isStreaming),
 *             sidebar.js (loadChats, createNewChat)
 */

// ── Messages ────────────────────────────────────────────────────────────

async function loadMessages(chatId) {
    const messages = await API.getMessages(chatId);
    renderMessages(messages);
    // Load generation bubbles after messages are rendered
    await loadGenerationBubbles(chatId);
}

function renderMessages(messages) {
    const container = $('#messages');
    if (!container) return;

    if (messages.length === 0) {
        renderEmptyState();
        return;
    }

    $('#empty-state')?.classList.add('hidden');

    container.innerHTML = '';
    messages.forEach((msg, idx) => {
        // Determine if this user message should show a delete button:
        // it's the last user msg AND either the very last message or followed by an error
        const isLastUserMsg = msg.role === 'user'
            && (idx === messages.length - 1
                || (idx === messages.length - 2 && messages[idx + 1].is_error));
        const el = createMessageElement(msg, isLastUserMsg);
        container.appendChild(el);
    });

    scrollToBottom();
}

function renderEmptyState() {
    const container = $('#messages');
    if (!container) return;

    container.innerHTML = `
        <div id="empty-state" class="empty-state">
            <h2>Welcome to LLM Prompt Agent</h2>
            <p>Start a conversation to generate creative image prompts.</p>
            <p class="hint">The agent can search your training data and generated outputs to create new, tailored prompts.</p>
        </div>
    `;
}

function createMessageElement(msg, isLastUserMsg = false) {
    const div = document.createElement('div');
    div.dataset.messageId = msg.id;

    // Error messages (persisted from agent loop failures)
    if (msg.is_error) {
        div.className = 'message error';
        div.textContent = msg.content;
        return div;
    }

    div.className = `message ${msg.role}`;

    if (msg.role === 'assistant') {
        div.innerHTML = renderMarkdown(msg.content);

        // Render persisted tool calls (loaded from DB on reload)
        if (msg.tool_calls && msg.tool_calls.length > 0) {
            const section = buildToolCallsSection(msg.tool_calls);
            div.insertBefore(section, div.firstChild);
        }
    } else {
        div.textContent = msg.content;
    }

    // Show attachment thumbnails in user messages (if metadata has them)
    if (msg.role === 'user' && msg.attachment_urls && msg.attachment_urls.length > 0) {
        const attDiv = document.createElement('div');
        attDiv.className = 'message-attachments';
        msg.attachment_urls.forEach(url => {
            const img = document.createElement('img');
            img.src = url;
            img.alt = 'Attached image';
            img.onclick = () => {
                if (typeof openFullSizeViewer === 'function') {
                    openFullSizeViewer([{ fullUrl: url }], 0);
                } else {
                    window.open(url, '_blank');
                }
            };
            attDiv.appendChild(img);
        });
        div.appendChild(attDiv);
    }

    // Delete button on the last user message (when dangling or followed by error)
    if (isLastUserMsg && msg.role === 'user') {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = '✕';
        deleteBtn.title = 'Delete message';
        deleteBtn.onclick = () => deleteLastUserMessage(msg.id);
        div.appendChild(deleteBtn);

        const editBtn = document.createElement('button');
        editBtn.className = 'edit-btn';
        editBtn.textContent = '✎';
        editBtn.title = 'Edit and resubmit';
        editBtn.onclick = () => startEditMessage(msg.id, msg.content);
        div.appendChild(editBtn);
    }

    return div;
}

function addStreamingMessage() {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'message assistant streaming';
    div.id = 'streaming-message';
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function appendToStreamingMessage(text) {
    const el = $('#streaming-message');
    if (!el) return;

    // Accumulate raw text in a data attribute
    const rawText = (el.dataset.rawText || '') + text;
    el.dataset.rawText = rawText;

    // Render as markdown
    el.innerHTML = renderMarkdown(rawText);
    scrollToBottom();
}

function finalizeStreamingMessage(messageId, toolCalls = null) {
    const el = $('#streaming-message');
    if (!el) {
        return;
    }

    el.classList.remove('streaming');
    el.id = '';
    if (messageId) {
        el.dataset.messageId = messageId;
    }

    // Insert tool calls introspection section if present
    if (toolCalls && toolCalls.length > 0) {
        const section = buildToolCallsSection(toolCalls);
        el.insertBefore(section, el.firstChild);
    }
}

function finalizeStreamingAsError(errorText) {
    const el = $('#streaming-message');
    if (el) {
        el.classList.remove('streaming', 'assistant');
        el.classList.add('error');
        el.id = '';
        el.textContent = errorText;
    } else {
        addErrorMessage(errorText);
    }
    // Add edit and delete buttons to the preceding user message
    addDeleteButtonToLastUserMessage();
    addEditButtonToLastUserMessage();
}

function finalizeStreamingAsCutoff(errorText) {
    const el = $('#streaming-message');
    if (el) {
        // Keep partial content, just stop streaming animation
        el.classList.remove('streaming');
        el.classList.add('message-cutoff');
        el.id = '';
    }
    // Add red error bubble below
    addErrorMessage(errorText);
    // Add edit/resend affordance to the user message
    addDeleteButtonToLastUserMessage();
    addEditButtonToLastUserMessage();
}

function addEditButtonToLastUserMessage() {
    const msgs = $$('.message.user');
    const lastUser = msgs[msgs.length - 1];
    if (!lastUser || lastUser.querySelector('.edit-btn')) return;

    const editBtn = document.createElement('button');
    editBtn.className = 'edit-btn';
    editBtn.textContent = '✎';
    editBtn.title = 'Edit and resubmit';
    editBtn.onclick = async () => {
        const messages = await API.getMessages(currentChatId);
        const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
        if (lastUserMsg) {
            startEditMessage(lastUserMsg.id, lastUserMsg.content);
        }
    };
    lastUser.appendChild(editBtn);
}

function addDeleteButtonToLastUserMessage() {
    const msgs = $$('.message.user');
    const lastUser = msgs[msgs.length - 1];
    if (!lastUser || lastUser.querySelector('.delete-btn')) return;

    const msgId = lastUser.dataset.messageId;
    if (!msgId) return;

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = '✕';
    deleteBtn.title = 'Delete message';
    deleteBtn.onclick = () => deleteLastUserMessage(parseInt(msgId));
    lastUser.appendChild(deleteBtn);
}

function buildToolCallsSection(toolCalls) {
    const details = document.createElement('details');
    details.className = 'tool-calls-section';

    // Sort by sequence to guarantee execution order
    const sorted = [...toolCalls].sort((a, b) => (a.sequence || 0) - (b.sequence || 0));

    // Group by iteration
    const iterations = new Map();
    sorted.forEach(call => {
        const iter = call.iteration || 1;
        if (!iterations.has(iter)) iterations.set(iter, []);
        iterations.get(iter).push(call);
    });

    const iterCount = iterations.size;
    const summary = document.createElement('summary');
    summary.className = 'tool-calls-toggle';
    const iterLabel = iterCount > 1 ? ` across ${iterCount} iterations` : '';
    summary.innerHTML =
        `<span class="tool-calls-icon">🔧</span>` +
        `<span>${sorted.length} tool call${sorted.length !== 1 ? 's' : ''}${iterLabel}</span>` +
        `<span class="tool-calls-chevron">▶</span>`;
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'tool-calls-list';

    iterations.forEach((calls, iterNum) => {
        // Show iteration header when there are multiple iterations
        if (iterCount > 1) {
            const iterHeader = document.createElement('div');
            iterHeader.className = 'tool-calls-iteration-header';
            iterHeader.textContent = `Iteration ${iterNum}`;
            list.appendChild(iterHeader);
        }

        calls.forEach(call => {
            const item = document.createElement('div');
            item.className = 'tool-call-item';

            const header = document.createElement('div');
            header.className = 'tool-call-header';
            header.innerHTML = `<span class="tool-call-name">${escapeHtml(call.tool)}</span>`;
            item.appendChild(header);

            const detailsDiv = document.createElement('div');
            detailsDiv.className = 'tool-call-details';

            // Args
            const argsDiv = document.createElement('div');
            argsDiv.className = 'tool-call-args';
            const argsLabel = document.createElement('span');
            argsLabel.className = 'tool-call-label';
            argsLabel.textContent = 'Args:';
            const argsPre = document.createElement('pre');
            argsPre.textContent = JSON.stringify(call.args, null, 2);
            argsDiv.appendChild(argsLabel);
            argsDiv.appendChild(argsPre);
            detailsDiv.appendChild(argsDiv);

            // Result
            const resultDiv = document.createElement('div');
            resultDiv.className = 'tool-call-result';
            const resultLabel = document.createElement('span');
            resultLabel.className = 'tool-call-label';
            resultLabel.textContent = 'Result:';
            const resultPre = document.createElement('pre');
            resultPre.textContent = JSON.stringify(call.result, null, 2);
            resultDiv.appendChild(resultLabel);
            resultDiv.appendChild(resultPre);
            detailsDiv.appendChild(resultDiv);

            item.appendChild(detailsDiv);
            list.appendChild(item);
        });
    });

    details.appendChild(list);
    return details;
}

function addStatusMessage(text) {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'tool-status';
    div.innerHTML = `<div class="spinner"></div><span>${escapeHtml(text)}</span>`;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function removeStatusMessages() {
    $$('.tool-status').forEach(el => el.remove());
}

function addErrorMessage(text) {
    const container = $('#messages');
    const div = document.createElement('div');
    div.className = 'message error';
    div.textContent = text;
    container.appendChild(div);
    scrollToBottom();
}

async function deleteLastUserMessage(messageId) {
    if (!currentChatId) return;

    // Check if deleting this message would leave the chat empty
    const allMsgs = $$('.message');
    const userEl = document.querySelector(`.message[data-message-id="${messageId}"]`);
    const remainingAfter = allMsgs.length
        - (userEl ? 1 : 0)
        - (userEl?.nextElementSibling?.classList.contains('error') ? 1 : 0);

    if (remainingAfter === 0) {
        // No messages will remain — delete the entire chat
        await deleteChat(currentChatId);
        return;
    }

    await API.deleteMessage(currentChatId, messageId);

    // Remove the user message element from DOM
    if (userEl) {
        // Also remove any following error message
        const next = userEl.nextElementSibling;
        if (next && next.classList.contains('error')) {
            next.remove();
        }
        userEl.remove();
    }
}

// ── Send Message ────────────────────────────────────────────────────────

async function sendMessage() {
    const input = $('#message-input');
    const rawContent = input.value.trim();
    if (!rawContent) return;

    // Block if THIS chat already has an active stream
    if (StreamRegistry.isActive(currentChatId)) return;

    let pendingToolCalls = null;

    // If refine context is active, build the combined message
    const hasRefine = typeof hasRefineContext === 'function' && hasRefineContext();
    const content = hasRefine ? buildRefineMessage(rawContent) : rawContent;
    const refinePrompt = hasRefine ? getRefineContext() : '';

    // Create chat if none selected
    if (!currentChatId) {
        await createNewChat();
    }

    // Capture chatId at send time — this won't change even if user switches chats
    const sendChatId = currentChatId;

    // Snapshot attachments before clearing
    const attachmentSnapshot = typeof getAttachments === 'function'
        ? getAttachments().slice() : [];
    const attachmentThumbs = typeof getAttachmentThumbnails === 'function'
        ? getAttachmentThumbnails() : [];
    const hasAtts = attachmentSnapshot.length > 0;

    // Clear input, attachments, and refine context immediately
    input.value = '';
    autoResizeInput(input);
    if (hasAtts && typeof clearAttachments === 'function') clearAttachments();
    if (hasRefine && typeof clearRefineContext === 'function') clearRefineContext();

    // Add user message to UI
    const container = $('#messages');
    $('#empty-state')?.classList.add('hidden');

    const userDiv = document.createElement('div');
    userDiv.className = 'message user';
    userDiv.textContent = content;

    // If refining, show a small indicator of what prompt is being refined
    if (hasRefine) {
        const refineIndicator = document.createElement('div');
        refineIndicator.className = 'message-refine-indicator';
        const truncated = refinePrompt.length > 80
            ? refinePrompt.substring(0, 80) + '…'
            : refinePrompt;
        refineIndicator.innerHTML = `<span class="refine-indicator-label">Refining:</span> <span class="refine-indicator-text">${escapeHtml(truncated)}</span>`;
        userDiv.insertBefore(refineIndicator, userDiv.firstChild);
    }

    // Show attachment thumbnails with upload progress overlays
    let attPlaceholders = [];
    if (hasAtts) {
        const attDiv = document.createElement('div');
        attDiv.className = 'message-attachments';
        attachmentThumbs.forEach((url, idx) => {
            const wrap = document.createElement('div');
            wrap.className = 'message-attachment-wrap uploading';
            const img = document.createElement('img');
            img.src = url;
            img.alt = 'Attached image';
            wrap.appendChild(img);
            // Progress overlay (spinner + progress bar)
            const overlay = document.createElement('div');
            overlay.className = 'attachment-upload-overlay';
            const spinner = document.createElement('div');
            spinner.className = 'attachment-upload-spinner';
            overlay.appendChild(spinner);
            const progressBar = document.createElement('div');
            progressBar.className = 'attachment-progress-bar';
            const progressFill = document.createElement('div');
            progressFill.className = 'attachment-progress-fill';
            progressBar.appendChild(progressFill);
            overlay.appendChild(progressBar);
            wrap.appendChild(overlay);
            attDiv.appendChild(wrap);
            attPlaceholders.push(wrap);
        });
        userDiv.appendChild(attDiv);
    }

    container.appendChild(userDiv);

    // Register stream and lock input for this chat
    StreamRegistry.register(sendChatId);
    const abortController = new AbortController();
    StreamRegistry.setAbortController(sendChatId, abortController);
    StreamRegistry.setUserText(sendChatId, rawContent);
    renderChatList();  // Update sidebar to show streaming indicator
    setStreaming(true);
    scrollToBottom();

    try {
        // Prepare and send attachments, or send text-only
        let response;
        if (hasAtts) {
            const formData = await prepareAttachmentsForSend(
                sendChatId, content, attachmentSnapshot,
                (idx) => {
                    // Mark individual attachment as prepared
                    if (attPlaceholders[idx]) {
                        attPlaceholders[idx].classList.remove('uploading');
                    }
                },
                (idx, loaded, total) => {
                    // Update progress bar during generated image fetch
                    if (attPlaceholders[idx]) {
                        const fill = attPlaceholders[idx].querySelector('.attachment-progress-fill');
                        if (fill && total > 0) {
                            fill.style.width = Math.round((loaded / total) * 100) + '%';
                        }
                    }
                }
            );
            // All attachments prepared — show uploading state on all
            attPlaceholders.forEach(wrap => {
                wrap.classList.remove('prepared');
                wrap.classList.add('uploading-server');
            });
            response = await API.sendMessageWithAttachments(sendChatId, formData, { signal: abortController.signal });
            // Upload complete — clear all upload states
            attPlaceholders.forEach(wrap => {
                wrap.classList.remove('uploading-server');
            });
        } else {
            response = await API.sendMessage(sendChatId, content, { signal: abortController.signal });
        }

        if (!response.ok) {
            const err = await response.json();
            StreamRegistry.setError(sendChatId);
            if (currentChatId === sendChatId) {
                finalizeStreamingAsError(err.error || 'Failed to send message');
                setStreaming(false);
            }
            StreamRegistry.cleanup(sendChatId);
            return;
        }

        // Only add streaming element if still viewing this chat
        if (currentChatId === sendChatId) {
            addStreamingMessage();
        }

        // Broadcast stream start to other tabs
        CrossTabSync.broadcast('stream_started', { chatId: sendChatId });

        const streamResult = await readSSEStream(response, {
            user_saved(data) {
                StreamRegistry.setUserMessageId(sendChatId, data.message_id);
                // Tag the DOM element so delete/edit buttons can reference it
                if (currentChatId === sendChatId) {
                    const userMsgs = $$('.message.user');
                    const lastUser = userMsgs[userMsgs.length - 1];
                    if (lastUser) lastUser.dataset.messageId = data.message_id;
                }
            },
            token(data) {
                const text = data.text || '';
                // Always buffer in registry (survives chat switching)
                StreamRegistry.appendText(sendChatId, text);
                // Only update DOM if user is still viewing this chat
                if (currentChatId === sendChatId) {
                    appendToStreamingMessage(text);
                }
                CrossTabSync.broadcast('stream_token', { chatId: sendChatId, text });
            },
            status(data) {
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                    addStatusMessage(data.message || 'Processing...');
                }
            },
            tool_result(data) {
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                    const summary = data.summary || 'Done';
                    addStatusMessage(`✓ ${data.tool}: ${summary}`);
                    setTimeout(removeStatusMessages, 1500);
                }
            },
            tool_calls(data) {
                pendingToolCalls = data.calls;
                StreamRegistry.setToolCalls(sendChatId, data.calls);
            },
            generation_submitted(data) {
                if (currentChatId === sendChatId && data.job) {
                    handleStreamingGeneration(data.job);
                }
            },
            error(data) {
                StreamRegistry.setError(sendChatId);
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                    finalizeStreamingAsError(data.message || 'An error occurred');
                }
                StreamRegistry.cleanup(sendChatId);
                CrossTabSync.broadcast('stream_error', { chatId: sendChatId });
                renderChatList();
            },
            done(data) {
                StreamRegistry.finalize(sendChatId, data.message_id);
                if (currentChatId === sendChatId) {
                    finalizeStreamingMessage(data.message_id, pendingToolCalls);
                    finalizeStreamingGenerationBubbles(data.message_id);
                    setStreaming(false);
                }
                StreamRegistry.cleanup(sendChatId);
                CrossTabSync.broadcast('stream_done', { chatId: sendChatId, messageId: data.message_id });
                // Refresh chat list to pick up new title
                loadChats();
            },
        });

        // Handle premature stream close (connection dropped without done/error)
        if (!streamResult.complete && StreamRegistry.isActive(sendChatId)) {
            StreamRegistry.setError(sendChatId);
            if (currentChatId === sendChatId) {
                removeStatusMessages();
                // Agent may still be running — wait briefly then check DB
                setTimeout(async () => {
                    try {
                        const messages = await API.getMessages(sendChatId);
                        const lastMsg = messages[messages.length - 1];
                        if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.is_partial) {
                            await loadMessages(sendChatId);
                        } else {
                            finalizeStreamingAsCutoff('Connection lost — response may be incomplete.');
                        }
                    } catch {
                        finalizeStreamingAsCutoff('Connection lost. Please try again.');
                    }
                    setStreaming(false);
                }, 1500);
            }
            StreamRegistry.cleanup(sendChatId);
            renderChatList();
        }

    } catch (err) {
        if (err.name === 'AbortError') return;  // Cancelled by user
        StreamRegistry.setError(sendChatId);
        if (currentChatId === sendChatId) {
            removeStatusMessages();
            finalizeStreamingAsError(`Network error: ${err.message}`);
        }
        StreamRegistry.cleanup(sendChatId);
        renderChatList();
    } finally {
        // Ensure input is unlocked if still viewing this chat
        if (currentChatId === sendChatId) {
            setStreaming(false);
        }
    }
}

// ── Edit & Resubmit ─────────────────────────────────────────────────────

function startEditMessage(messageId, content) {
    const input = $('#message-input');
    input.value = content;
    input.focus();
    autoResizeInput(input);

    // Replace send button behavior temporarily
    const sendBtn = $('#send-btn');
    sendBtn.onclick = () => submitEditedMessage(messageId);
    sendBtn.title = 'Resubmit';
}

async function submitEditedMessage(messageId) {
    const input = $('#message-input');
    const content = input.value.trim();
    if (!content) return;

    // Block if THIS chat already has an active stream
    if (StreamRegistry.isActive(currentChatId)) return;

    let pendingToolCalls = null;

    // Restore normal send behavior
    const sendBtn = $('#send-btn');
    sendBtn.onclick = sendMessage;
    sendBtn.title = 'Send';

    const sendChatId = currentChatId;

    input.value = '';
    autoResizeInput(input);
    StreamRegistry.register(sendChatId);
    const abortController = new AbortController();
    StreamRegistry.setAbortController(sendChatId, abortController);
    StreamRegistry.setUserText(sendChatId, content);
    renderChatList();  // Update sidebar to show streaming indicator
    setStreaming(true);

    try {
        const response = await API.editMessage(sendChatId, messageId, content, { signal: abortController.signal });
        if (!response.ok) {
            const err = await response.json();
            StreamRegistry.setError(sendChatId);
            if (currentChatId === sendChatId) {
                finalizeStreamingAsError(err.error || 'Failed to edit message');
                setStreaming(false);
            }
            StreamRegistry.cleanup(sendChatId);
            return;
        }

        if (currentChatId === sendChatId) {
            // Reload messages to show cleaned-up history, then stream
            // (the new user message is already persisted by the backend
            //  before streaming begins, so loadMessages picks it up)
            await loadMessages(sendChatId);

            addStreamingMessage();
        }

        CrossTabSync.broadcast('stream_started', { chatId: sendChatId });

        const editStreamResult = await readSSEStream(response, {
            user_saved(data) {
                StreamRegistry.setUserMessageId(sendChatId, data.message_id);
                if (currentChatId === sendChatId) {
                    const userMsgs = $$('.message.user');
                    const lastUser = userMsgs[userMsgs.length - 1];
                    if (lastUser) lastUser.dataset.messageId = data.message_id;
                }
            },
            token(data) {
                const text = data.text || '';
                StreamRegistry.appendText(sendChatId, text);
                if (currentChatId === sendChatId) {
                    appendToStreamingMessage(text);
                }
                CrossTabSync.broadcast('stream_token', { chatId: sendChatId, text });
            },
            status(data) {
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                    addStatusMessage(data.message || 'Processing...');
                }
            },
            tool_result(data) {
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                }
            },
            tool_calls(data) {
                pendingToolCalls = data.calls;
                StreamRegistry.setToolCalls(sendChatId, data.calls);
            },
            generation_submitted(data) {
                if (currentChatId === sendChatId && data.job) {
                    handleStreamingGeneration(data.job);
                }
            },
            error(data) {
                StreamRegistry.setError(sendChatId);
                if (currentChatId === sendChatId) {
                    removeStatusMessages();
                    finalizeStreamingAsError(data.message || 'An error occurred');
                }
                StreamRegistry.cleanup(sendChatId);
                CrossTabSync.broadcast('stream_error', { chatId: sendChatId });
                renderChatList();
            },
            done(data) {
                StreamRegistry.finalize(sendChatId, data.message_id);
                if (currentChatId === sendChatId) {
                    finalizeStreamingMessage(data.message_id, pendingToolCalls);
                    finalizeStreamingGenerationBubbles(data.message_id);
                    setStreaming(false);
                }
                StreamRegistry.cleanup(sendChatId);
                CrossTabSync.broadcast('stream_done', { chatId: sendChatId, messageId: data.message_id });
                loadChats();
            },
        });

        // Handle premature stream close
        if (!editStreamResult.complete && StreamRegistry.isActive(sendChatId)) {
            StreamRegistry.setError(sendChatId);
            if (currentChatId === sendChatId) {
                removeStatusMessages();
                setTimeout(async () => {
                    try {
                        const messages = await API.getMessages(sendChatId);
                        const lastMsg = messages[messages.length - 1];
                        if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.is_partial) {
                            await loadMessages(sendChatId);
                        } else {
                            finalizeStreamingAsCutoff('Connection lost — response may be incomplete.');
                        }
                    } catch {
                        finalizeStreamingAsCutoff('Connection lost. Please try again.');
                    }
                    setStreaming(false);
                }, 1500);
            }
            StreamRegistry.cleanup(sendChatId);
            renderChatList();
        }

    } catch (err) {
        if (err.name === 'AbortError') return;  // Cancelled by user
        StreamRegistry.setError(sendChatId);
        if (currentChatId === sendChatId) {
            removeStatusMessages();
            finalizeStreamingAsError(`Network error: ${err.message}`);
        }
        StreamRegistry.cleanup(sendChatId);
        renderChatList();
    } finally {
        if (currentChatId === sendChatId) {
            setStreaming(false);
        }
    }
}

// ── Input Handling ──────────────────────────────────────────────────────

function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeInput(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function cancelStream() {
    const chatId = currentChatId;
    if (!StreamRegistry.isActive(chatId)) return;

    // 1. Abort the HTTP stream
    StreamRegistry.abort(chatId);

    // 2. Remove assistant streaming bubble
    const streamEl = $('#streaming-message');
    if (streamEl) streamEl.remove();

    // 3. Remove status messages
    removeStatusMessages();

    // 4. Get user text and put it back in input
    const userText = StreamRegistry.getUserText(chatId);
    const input = $('#message-input');
    if (input) {
        input.value = userText;
        autoResizeInput(input);
    }

    // 5. Remove last user message bubble from DOM
    const userMessages = $$('.message.user');
    const lastUser = userMessages[userMessages.length - 1];
    if (lastUser) lastUser.remove();

    // 6. Delete user message from DB
    const userMsgId = StreamRegistry.getUserMessageId(chatId);
    if (userMsgId) API.cancelMessage(chatId, userMsgId);

    // 7. Cleanup
    StreamRegistry.cleanup(chatId);
    setStreaming(false);
    renderChatList();
}

function setStreaming(value) {
    isStreaming = value;
    const sendBtn = $('#send-btn');
    const input = $('#message-input');

    if (sendBtn) {
        if (value) {
            // Switch to cancel mode
            sendBtn.disabled = false;
            sendBtn.querySelector('.send-icon').textContent = '⊘';
            sendBtn.onclick = cancelStream;
            sendBtn.title = 'Cancel';
            sendBtn.classList.add('btn-cancel');
        } else {
            // Restore send mode
            sendBtn.disabled = false;
            sendBtn.querySelector('.send-icon').textContent = '➤';
            sendBtn.onclick = sendMessage;
            sendBtn.title = 'Send';
            sendBtn.classList.remove('btn-cancel');
        }
    }
    if (input) input.disabled = value;
}

// ── Tab Visibility Recovery ──────────────────────────────────────────

/**
 * Check if an active agent stream completed while the tab was hidden.
 * If the server has a newer assistant message than what the client shows,
 * reload the full message list and clean up the stale stream state.
 */
async function refreshStaleChatStream() {
    const chatId = currentChatId;
    if (!chatId) return;
    if (!StreamRegistry.isActive(chatId)) return;

    try {
        const messages = await API.getMessages(chatId);
        if (!messages || messages.length === 0) return;

        const lastMsg = messages[messages.length - 1];
        if (lastMsg.role === 'assistant') {
            // Stream finished on server while tab was hidden — reload UI
            StreamRegistry.cleanup(chatId);
            removeStatusMessages();
            await loadMessages(chatId);
            setStreaming(false);
            addEditButtonToLastUserMessage();
            renderChatList();
        }
    } catch (err) {
        console.error('Failed to refresh stale chat stream:', err);
    }
}

// ── Scroll ──────────────────────────────────────────────────────────────

function scrollToBottom() {
    const container = $('#messages');
    if (container) {
        requestAnimationFrame(() => {
            container.scrollTop = container.scrollHeight;
        });
    }
}
