/**
 * Attachment system — manage image attachments for user messages.
 *
 * Provides: addAttachment(), removeAttachment(), clearAttachments(),
 *           getAttachments(), hasAttachments(), openAttachmentPicker(),
 *           prepareAttachmentsForSend(), getAttachmentThumbnails()
 *
 * Depends on: app.js ($)
 */

let _attachments = [];  // Array of {type, jobId, imageId, thumbnailUrl, fullUrl, file, dataUrl, name}

/**
 * Add an attachment. Called from generation bubble attach icon or file picker.
 * @param {Object} attachment
 *   For generated images: {type: 'generated', jobId, imageId, thumbnailUrl, fullUrl}
 *   For uploaded files:   {type: 'uploaded', file: File, dataUrl: string, name: string}
 */
function addAttachment(attachment) {
    // Prevent duplicates
    if (attachment.type === 'generated') {
        const exists = _attachments.some(
            a => a.type === 'generated' && a.imageId === attachment.imageId
        );
        if (exists) return;
    } else if (attachment.type === 'uploaded' && attachment.file) {
        const exists = _attachments.some(
            a => a.type === 'uploaded' &&
                 a.name === attachment.file.name &&
                 a.file && a.file.size === attachment.file.size
        );
        if (exists) return;
    }

    _attachments.push(attachment);
    _renderAttachmentBar();
}

function removeAttachment(index) {
    _attachments.splice(index, 1);
    _renderAttachmentBar();
}

function clearAttachments() {
    _attachments = [];
    _renderAttachmentBar();
}

function getAttachments() {
    return _attachments;
}

function hasAttachments() {
    return _attachments.length > 0;
}

/**
 * Return an array of thumbnail URLs/dataUrls for the current attachments.
 * Used to render thumbnails in the user message bubble.
 */
function getAttachmentThumbnails() {
    return _attachments.map(att => {
        if (att.type === 'generated') {
            return att.thumbnailUrl;
        } else if (att.dataUrl) {
            return att.dataUrl;
        }
        return null;
    }).filter(Boolean);
}

function _renderAttachmentBar() {
    const bar = document.getElementById('attachment-bar');
    if (!bar) return;

    if (_attachments.length === 0) {
        bar.classList.add('hidden');
        bar.innerHTML = '';
        return;
    }

    bar.classList.remove('hidden');
    bar.innerHTML = '';

    _attachments.forEach((att, idx) => {
        const thumb = document.createElement('div');
        thumb.className = 'attachment-thumb';

        const img = document.createElement('img');
        if (att.type === 'generated') {
            img.src = att.thumbnailUrl;
        } else if (att.dataUrl) {
            img.src = att.dataUrl;
        }
        img.alt = 'Attachment';
        thumb.appendChild(img);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'attachment-remove';
        removeBtn.innerHTML = '&times;';
        removeBtn.title = 'Remove attachment';
        removeBtn.onclick = () => removeAttachment(idx);
        thumb.appendChild(removeBtn);

        bar.appendChild(thumb);
    });
}

/**
 * Open file picker for manual image upload.
 */
function openAttachmentPicker() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    input.onchange = (e) => {
        const files = Array.from(e.target.files);
        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                addAttachment({
                    type: 'uploaded',
                    file: file,
                    dataUrl: ev.target.result,
                    name: file.name,
                });
            };
            reader.readAsDataURL(file);
        });
    };
    input.click();
}

/**
 * Prepare attachments for sending with a message.
 * Returns a FormData object if there are attachments, or null if text-only.
 *
 * @param {string} chatId
 * @param {string} content - Message text
 * @param {Object[]} attachmentSnapshot - Snapshot of attachments captured before clearing
 * @param {Function} [onReady] - Called as onReady(index) when each attachment is prepared
 * @param {Function} [onProgress] - Called as onProgress(index, loaded, total) during fetch
 * @returns {Promise<FormData|null>}
 */
async function prepareAttachmentsForSend(chatId, content, attachmentSnapshot, onReady, onProgress) {
    if (!attachmentSnapshot || attachmentSnapshot.length === 0) return null;

    const formData = new FormData();
    formData.append('content', content);

    for (let i = 0; i < attachmentSnapshot.length; i++) {
        const att = attachmentSnapshot[i];
        if (att.type === 'uploaded' && att.file) {
            try {
                const result = await ImageConvert.processFile(att.file, {
                    onProgress: (phase, fraction) => {
                        if (!onProgress) return;
                        // Map 3 phases to 0–100 synthetic progress
                        const phaseOffsets = { decoding: 0, resizing: 40, encoding: 60 };
                        const phaseWidths  = { decoding: 40, resizing: 20, encoding: 40 };
                        const offset = phaseOffsets[phase] || 0;
                        const width  = phaseWidths[phase]  || 0;
                        onProgress(i, offset + fraction * width, 100);
                    },
                });
                formData.append('attachments', result.blob, result.filename);
            } catch (err) {
                console.warn('Image conversion failed, uploading original:', err);
                formData.append('attachments', att.file, att.file.name);
            }
            if (onReady) onReady(i);
        } else if (att.type === 'generated') {
            // For generated images, fetch the image and add as blob
            try {
                const response = await fetch(att.fullUrl);
                const total = parseInt(response.headers.get('content-length') || '0', 10);

                let blob;
                if (total > 0 && response.body && onProgress) {
                    // Read with progress tracking
                    const reader = response.body.getReader();
                    let loaded = 0;
                    const chunks = [];
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        chunks.push(value);
                        loaded += value.length;
                        onProgress(i, loaded, total);
                    }
                    blob = new Blob(chunks, { type: response.headers.get('content-type') || 'image/png' });
                } else {
                    blob = await response.blob();
                }

                const filename = `generated_${att.jobId}_${att.imageId}.png`;
                formData.append('attachments', blob, filename);
                if (onReady) onReady(i);
            } catch (err) {
                console.error('Failed to fetch generated image for attachment:', err);
                if (onReady) onReady(i);
            }
        }
    }

    return formData;
}
