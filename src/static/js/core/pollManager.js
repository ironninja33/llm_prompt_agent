/**
 * Unified poll manager — flat 2s interval polling for all status modules.
 *
 * Modules register/unregister to control the `m` query param and provide
 * extra query params. Always runs once started (call start() on DOMContentLoaded).
 */

const PollManager = {
    _timer: null,
    _handlers: {},      // module -> callback(data)
    _paramFns: {},      // module -> () => {key: value}
    _interval: 2000,    // flat 2s
    _onAlways: [],      // callbacks that run on every poll response

    /**
     * Register a module for polling.
     * @param {string} module - Module name (matches backend's `m` param values)
     * @param {Function} handler - callback(data) called with the module's response data
     * @param {Function|null} paramsFn - optional () => {key: value} for extra query params
     */
    register(module, handler, paramsFn = null) {
        this._handlers[module] = handler;
        if (paramsFn) this._paramFns[module] = paramsFn;
    },

    unregister(module) {
        delete this._handlers[module];
        delete this._paramFns[module];
    },

    /**
     * Register a callback that receives the entire poll response on every poll.
     * Used for meta fields like _active_generation_jobs.
     */
    onEveryPoll(callback) {
        this._onAlways.push(callback);
    },

    start() {
        if (this._timer) return;
        this._poll();
    },

    stop() {
        clearTimeout(this._timer);
        this._timer = null;
    },

    async _poll() {
        const modules = Object.keys(this._handlers);

        const params = new URLSearchParams();
        if (modules.length) params.set('m', modules.join(','));
        for (const mod of modules) {
            if (this._paramFns[mod]) {
                for (const [k, v] of Object.entries(this._paramFns[mod]())) {
                    params.set(k, v);
                }
            }
        }

        try {
            const resp = await fetch('/api/poll?' + params);
            if (resp.ok) {
                const data = await resp.json();

                for (const mod of modules) {
                    if (data[mod] !== undefined) {
                        try { this._handlers[mod](data[mod]); } catch (e) {
                            console.warn(`PollManager: handler error for ${mod}:`, e);
                        }
                    }
                }

                for (const cb of this._onAlways) {
                    try { cb(data); } catch (e) {
                        console.warn('PollManager: onAlways callback error:', e);
                    }
                }
            }
        } catch {
            // Silent — next poll retries
        }

        this._timer = setTimeout(() => this._poll(), this._interval);
    }
};
