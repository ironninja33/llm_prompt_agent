/**
 * SSE stream parser — reads a fetch Response as a Server-Sent Events stream.
 *
 * Since we POST to send messages, we can't use the browser's EventSource
 * (which only supports GET). Instead, parse the response body as SSE manually.
 */

async function readSSEStream(response, handlers) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let receivedTerminal = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split on double newline (SSE event boundary)
        const parts = buffer.split('\n\n');
        buffer = parts.pop(); // Keep incomplete last part in buffer

        for (const part of parts) {
            const event = parseSSEEvent(part);
            if (event) {
                if (event.type === 'done' || event.type === 'error') {
                    receivedTerminal = true;
                }
                if (handlers[event.type]) {
                    handlers[event.type](event.data);
                }
            }
        }
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
        const event = parseSSEEvent(buffer);
        if (event) {
            if (event.type === 'done' || event.type === 'error') {
                receivedTerminal = true;
            }
            if (handlers[event.type]) {
                handlers[event.type](event.data);
            }
        }
    }

    return { complete: receivedTerminal };
}

function parseSSEEvent(raw) {
    let eventType = 'message';
    let dataLines = [];

    for (const line of raw.split('\n')) {
        if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
            dataLines.push(line.slice(6));
        } else if (line.startsWith(':')) {
            // Comment/keepalive, ignore
        }
    }

    if (dataLines.length === 0) return null;

    try {
        return { type: eventType, data: JSON.parse(dataLines.join('\n')) };
    } catch {
        return { type: eventType, data: { text: dataLines.join('\n') } };
    }
}
