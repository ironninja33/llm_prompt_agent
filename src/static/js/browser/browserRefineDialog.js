/**
 * Browser Refine Chat Picker Dialog.
 *
 * Shows a modal letting the user choose between starting a new chat
 * or inserting into an existing one when clicking Refine.
 * Includes an "attach image" toggle when the source image is available.
 *
 * Depends on: api.js, browserGrid.js (_storeRefineSettings)
 */

let _refineDialog = null;
let _refineAttach = false;

function _ensureRefineDialog() {
    if (_refineDialog) return _refineDialog;

    const modal = document.createElement('div');
    modal.className = 'modal hidden';
    modal.id = 'refine-chat-dialog';

    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-dialog" style="width:420px">
            <div class="modal-header">
                <h2>Refine Prompt</h2>
                <button class="btn-close" id="refine-dialog-close">&times;</button>
            </div>
            <div class="refine-dialog-controls" id="refine-dialog-controls">
                <label class="refine-attach-toggle">
                    <input type="checkbox" id="refine-attach-checkbox">
                    <span>Attach source image</span>
                </label>
            </div>
            <div class="modal-body" style="padding:0">
                <div class="refine-chat-list" id="refine-chat-list"></div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Close on backdrop click
    modal.querySelector('.modal-backdrop').addEventListener('click', _closeRefineDialog);
    modal.querySelector('#refine-dialog-close').addEventListener('click', _closeRefineDialog);

    _refineDialog = modal;
    return modal;
}

function _closeRefineDialog() {
    if (_refineDialog) _refineDialog.classList.add('hidden');
}

/**
 * Open the refine chat picker dialog.
 * @param {object} job  - job-like object with .id and .settings
 * @param {object} img  - image-like object with .id
 */
async function openRefineDialog(job, img) {
    const modal = _ensureRefineDialog();
    const list = modal.querySelector('#refine-chat-list');
    const checkbox = modal.querySelector('#refine-attach-checkbox');
    list.innerHTML = '<div class="refine-chat-loading">Loading chats...</div>';
    checkbox.checked = false;
    modal.classList.remove('hidden');

    const prompt = (job.settings || {}).positive_prompt || '';

    // Build the navigate URL, reading attach checkbox at click time
    function navigate(chatId) {
        _storeRefineSettings(job.settings);
        let url = `/?refine=${encodeURIComponent(prompt)}`;
        if (chatId) url += `&chat=${encodeURIComponent(chatId)}`;
        if (checkbox.checked) url += `&attach=${job.id}/${img.id}`;
        window.location.href = url;
    }

    // Fetch chats
    let chats = [];
    try {
        chats = await API.listChats();
    } catch (e) {
        console.error('Failed to list chats:', e);
    }

    // Render options
    list.innerHTML = '';

    // "New Chat" option always first
    const newOpt = document.createElement('div');
    newOpt.className = 'refine-chat-option new-chat';
    newOpt.innerHTML = '<span class="refine-chat-title">+ New Chat</span>';
    newOpt.addEventListener('click', () => {
        _closeRefineDialog();
        navigate(null);
    });
    list.appendChild(newOpt);

    // Existing chats (top 10)
    const recent = chats.slice(0, 10);
    recent.forEach(chat => {
        const opt = document.createElement('div');
        opt.className = 'refine-chat-option';

        const title = document.createElement('span');
        title.className = 'refine-chat-title';
        title.textContent = chat.title || 'New Chat';

        const date = document.createElement('span');
        date.className = 'refine-chat-date';
        if (chat.updated_at) {
            const dt = new Date(chat.updated_at.replace(' ', 'T') + 'Z');
            if (!isNaN(dt)) {
                date.textContent = dt.toLocaleDateString(undefined, {
                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                });
            }
        }

        opt.appendChild(title);
        opt.appendChild(date);
        opt.addEventListener('click', () => {
            _closeRefineDialog();
            navigate(chat.id);
        });
        list.appendChild(opt);
    });
}
