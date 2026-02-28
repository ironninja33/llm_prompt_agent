/**
 * Chat Refine Dialog — lightweight modal for refining a prompt from a
 * generated image thumbnail.
 *
 * Shows the prompt text, an "Attach source image" checkbox, and an OK button.
 * On confirm: sets the refine context and optionally adds the image as an
 * attachment.
 *
 * Depends on: app.js ($), refineContext.js (setRefineContext),
 *             attachments.js (addAttachment)
 */

let _chatRefineDialog = null;

function _ensureChatRefineDialog() {
    if (_chatRefineDialog) return _chatRefineDialog;

    const modal = document.createElement('div');
    modal.className = 'modal hidden';
    modal.id = 'chat-refine-dialog';

    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-dialog" style="width:420px">
            <div class="modal-header">
                <h2>Refine Prompt</h2>
                <button class="btn-close" id="chat-refine-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="chat-refine-prompt" id="chat-refine-prompt"></div>
            </div>
            <div class="refine-dialog-controls" id="chat-refine-controls">
                <label class="refine-attach-toggle">
                    <input type="checkbox" id="chat-refine-attach-checkbox">
                    <span>Attach source image</span>
                </label>
            </div>
            <div class="refine-dialog-footer">
                <button class="btn-refine-ok" id="chat-refine-ok">OK</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    modal.querySelector('.modal-backdrop').addEventListener('click', _closeChatRefineDialog);
    modal.querySelector('#chat-refine-close').addEventListener('click', _closeChatRefineDialog);

    _chatRefineDialog = modal;
    return modal;
}

function _closeChatRefineDialog() {
    if (_chatRefineDialog) _chatRefineDialog.classList.add('hidden');
}

/**
 * Open the chat refine dialog for a generated image.
 * @param {Object} job - Generation job object
 * @param {Object} img - Image object { id, ... }
 */
function openChatRefineDialog(job, img) {
    const modal = _ensureChatRefineDialog();
    const promptEl = modal.querySelector('#chat-refine-prompt');
    const checkbox = modal.querySelector('#chat-refine-attach-checkbox');
    const okBtn = modal.querySelector('#chat-refine-ok');

    const prompt = job.settings?.positive_prompt || '';

    // Show truncated prompt preview
    const maxLen = 200;
    promptEl.textContent = prompt.length > maxLen
        ? prompt.substring(0, maxLen) + '…'
        : prompt;
    promptEl.title = prompt;

    checkbox.checked = false;
    modal.classList.remove('hidden');

    // Replace OK handler (clone to remove old listeners)
    const newOk = okBtn.cloneNode(true);
    okBtn.parentNode.replaceChild(newOk, okBtn);
    newOk.addEventListener('click', () => {
        _closeChatRefineDialog();

        // Set refine context
        if (prompt && typeof setRefineContext === 'function') {
            setRefineContext(prompt);
        }

        // Optionally attach the image
        if (checkbox.checked && typeof addAttachment === 'function') {
            addAttachment({
                type: 'generated',
                jobId: job.id,
                imageId: img.id,
                thumbnailUrl: `/api/generate/thumbnail/${job.id}/${img.id}`,
                fullUrl: `/api/generate/image/${job.id}/${img.id}`,
            });
        }
    });
}
