/**
 * Deletion reason dialog — one-click flow.
 * Clicking a reason immediately calls the callback and closes.
 */

let _deleteDialog = null;
let _deleteCallback = null;

const DELETE_REASONS = [
    { key: 'quality', label: 'Bad generation' },
    { key: 'wrong_direction', label: 'Wrong direction' },
    { key: 'duplicate', label: 'Duplicate' },
    { key: 'space', label: 'Just cleaning up' },
];

function _ensureDeleteDialog() {
    if (_deleteDialog) return _deleteDialog;

    const modal = document.createElement('div');
    modal.className = 'modal hidden';
    modal.id = 'delete-reason-dialog';

    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-dialog" style="width:280px">
            <div class="modal-header">
                <h2>Why delete?</h2>
                <button class="btn-close" id="delete-dialog-close">&times;</button>
            </div>
            <div class="modal-body" style="padding:0">
                <div class="delete-reason-list" id="delete-reason-list"></div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Build reason buttons
    const list = modal.querySelector('#delete-reason-list');
    DELETE_REASONS.forEach(({ key, label }) => {
        const btn = document.createElement('div');
        btn.className = 'delete-reason-option';
        btn.textContent = label;
        btn.addEventListener('click', () => {
            const cb = _deleteCallback;
            _closeDeleteDialog();
            if (cb) cb(key);
        });
        list.appendChild(btn);
    });

    // Close handlers
    modal.querySelector('.modal-backdrop').addEventListener('click', _closeDeleteDialog);
    modal.querySelector('#delete-dialog-close').addEventListener('click', _closeDeleteDialog);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            _closeDeleteDialog();
        }
    });

    _deleteDialog = modal;
    return modal;
}

function _closeDeleteDialog() {
    if (_deleteDialog) _deleteDialog.classList.add('hidden');
    _deleteCallback = null;
}

/**
 * Open the deletion reason dialog. Clicking a reason calls onConfirm(reason) immediately.
 * @param {Function} onConfirm - Called with reason string ('quality'|'wrong_direction'|'duplicate'|'space')
 */
function openDeleteDialog(onConfirm) {
    const modal = _ensureDeleteDialog();
    _deleteCallback = onConfirm;
    modal.classList.remove('hidden');
}
