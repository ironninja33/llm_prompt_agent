/**
 * Minimal markdown-to-HTML converter for chat messages.
 * Handles: headings, bold, italic, inline code, code blocks, links, lists,
 *          paragraphs, and special ```prompt blocks with copy-to-clipboard.
 */

// SVG icon for the copy button
const COPY_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const CHECK_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
// SVG icon for the generate button (image/landscape icon)
const GENERATE_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>`;

/**
 * Copy the text content of a prompt block to the clipboard.
 * Called from onclick on the copy button inside .prompt-block.
 */
function copyPromptText(button) {
    const block = button.closest('.prompt-block');
    if (!block) return;

    const contentEl = block.querySelector('.prompt-block-content');
    if (!contentEl) return;

    const text = contentEl.textContent;

    navigator.clipboard.writeText(text).then(() => {
        // Show success feedback
        button.innerHTML = CHECK_ICON_SVG;
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = COPY_ICON_SVG;
            button.classList.remove('copied');
        }, 1500);
    }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);

        button.innerHTML = CHECK_ICON_SVG;
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = COPY_ICON_SVG;
            button.classList.remove('copied');
        }, 1500);
    });
}

/**
 * Open generation overlay from a prompt block's generate button.
 * Extracts the prompt text and the parent message's ID so that
 * generated images are grouped under the correct chat message.
 */
function _openGenFromPromptBlock(button) {
    const block = button.closest('.prompt-block');
    if (!block) return;
    const prompt = block.querySelector('.prompt-block-content')?.textContent || '';
    const msgEl = block.closest('.message[data-message-id]');
    const messageId = msgEl ? parseInt(msgEl.dataset.messageId, 10) || null : null;
    openGenerationOverlay({ prompt, messageId });
}

function renderMarkdown(text) {
    if (!text) return '';

    // Escape HTML first
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks (```...```) — with special handling for ```prompt blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        if (lang === 'prompt') {
            const trimmed = code.trim();
            return `<div class="prompt-block">`
                + `<div class="prompt-block-header">`
                + `<span class="prompt-block-label">Suggested Prompt</span>`
                + `<span class="prompt-header-actions">`
                + `<button class="prompt-gen-btn" title="Generate image" onclick="_openGenFromPromptBlock(this)">${GENERATE_ICON_SVG}</button>`
                + `<button class="prompt-copy-btn" title="Copy to clipboard" onclick="copyPromptText(this)">${COPY_ICON_SVG}</button>`
                + `</span>`
                + `</div>`
                + `<div class="prompt-block-content">${trimmed}</div>`
                + `</div>`;
        }
        return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold (**...**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic (*...*)
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Links [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Headings (h1–h6) — must come before list/paragraph processing
    html = html.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
    html = html.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
    html = html.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

    // Horizontal rules
    html = html.replace(/^---+$/gm, '<hr>');

    // Unordered lists (lines starting with - or *)
    html = html.replace(/^([\s]*[-*]\s.+(\n|$))+/gm, (match) => {
        const items = match.trim().split('\n').map(line => {
            const content = line.replace(/^\s*[-*]\s/, '');
            return `<li>${content}</li>`;
        }).join('');
        return `<ul>${items}</ul>`;
    });

    // Ordered lists (lines starting with N.)
    html = html.replace(/^(\s*\d+\.\s.+(\n|$))+/gm, (match) => {
        const items = match.trim().split('\n').map(line => {
            const content = line.replace(/^\s*\d+\.\s/, '');
            return `<li>${content}</li>`;
        }).join('');
        return `<ol>${items}</ol>`;
    });

    // Paragraphs — wrap blocks separated by blank lines
    html = html.replace(/\n{2,}/g, '</p><p>');
    if (!html.startsWith('<') || html.startsWith('<strong') || html.startsWith('<em') || html.startsWith('<code>') || html.startsWith('<a ')) {
        html = '<p>' + html + '</p>';
    }

    // Clean up: remove <p> tags wrapping block elements
    html = html.replace(/<p>(<h[1-6]>)/g, '$1');
    html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
    html = html.replace(/<p>(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ol>)/g, '$1');
    html = html.replace(/(<\/ol>)<\/p>/g, '$1');
    html = html.replace(/<p>(<div class="prompt-block">)/g, '$1');
    html = html.replace(/(<\/div>)<\/p>/g, '$1');

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
}
