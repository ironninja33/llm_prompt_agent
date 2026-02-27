/**
 * Client-side image conversion — resize and convert lossless formats to JPEG
 * before upload to reduce bandwidth and payload size.
 *
 * Provides: ImageConvert.processFile(), ImageConvert.shouldConvert()
 *
 * No dependencies.
 */

const ImageConvert = (() => {
    const CONVERTIBLE_TYPES = new Set([
        'image/png', 'image/bmp', 'image/tiff', 'image/webp',
    ]);
    const JPEG_TYPES = new Set(['image/jpeg', 'image/jpg']);
    const PASSTHROUGH_TYPES = new Set(['image/gif', 'image/svg+xml']);

    /**
     * Check whether a file should be converted (lossless → JPEG).
     * @param {File} file
     * @returns {boolean}
     */
    function shouldConvert(file) {
        return CONVERTIBLE_TYPES.has(file.type);
    }

    /**
     * Load a File into an HTMLImageElement.
     * @param {File} file
     * @returns {Promise<HTMLImageElement>}
     */
    function _loadImage(file) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(file);
            const img = new Image();
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to decode image'));
            };
            img.src = url;
        });
    }

    /**
     * Draw an image onto a canvas, resizing so no dimension exceeds maxDimension.
     * Never scales up. Returns the canvas.
     * @param {HTMLImageElement} img
     * @param {number} maxDimension
     * @returns {HTMLCanvasElement}
     */
    function _drawToCanvas(img, maxDimension) {
        let { naturalWidth: w, naturalHeight: h } = img;

        if (w > maxDimension || h > maxDimension) {
            const scale = Math.min(maxDimension / w, maxDimension / h);
            w = Math.round(w * scale);
            h = Math.round(h * scale);
        }

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        return canvas;
    }

    /**
     * Encode a canvas to a JPEG Blob.
     * @param {HTMLCanvasElement} canvas
     * @param {number} quality - 0–1
     * @returns {Promise<Blob>}
     */
    function _canvasToBlob(canvas, quality) {
        return new Promise((resolve, reject) => {
            canvas.toBlob(
                blob => blob ? resolve(blob) : reject(new Error('Canvas encoding failed')),
                'image/jpeg',
                quality
            );
        });
    }

    /**
     * Replace the file extension with .jpg.
     * @param {string} filename
     * @returns {string}
     */
    function _changeExtension(filename) {
        const dot = filename.lastIndexOf('.');
        const base = dot > 0 ? filename.substring(0, dot) : filename;
        return base + '.jpg';
    }

    /**
     * Process a file: convert lossless images to JPEG and resize if needed.
     *
     * @param {File} file
     * @param {Object} [options]
     * @param {number} [options.maxDimension=1024] - Max width or height
     * @param {number} [options.jpegQuality=0.9] - JPEG quality 0–1
     * @param {Function} [options.onProgress] - Called as onProgress(phase, fraction)
     *   phase: 'decoding' | 'resizing' | 'encoding', fraction: 0–1
     * @returns {Promise<{blob: Blob, filename: string}>}
     */
    async function processFile(file, options = {}) {
        const maxDimension = options.maxDimension || 1024;
        const jpegQuality = options.jpegQuality != null ? options.jpegQuality : 0.9;
        const onProgress = options.onProgress || (() => {});

        // GIF, SVG, unknown — pass through unchanged
        if (PASSTHROUGH_TYPES.has(file.type) || (!CONVERTIBLE_TYPES.has(file.type) && !JPEG_TYPES.has(file.type))) {
            return { blob: file, filename: file.name };
        }

        // Decode
        onProgress('decoding', 0);
        const img = await _loadImage(file);
        onProgress('decoding', 1);

        const needsResize = img.naturalWidth > maxDimension || img.naturalHeight > maxDimension;

        // JPEG that doesn't need resize — return as-is
        if (JPEG_TYPES.has(file.type) && !needsResize) {
            return { blob: file, filename: file.name };
        }

        // Resize
        onProgress('resizing', 0);
        const canvas = _drawToCanvas(img, maxDimension);
        onProgress('resizing', 1);

        // Encode to JPEG
        onProgress('encoding', 0);
        const blob = await _canvasToBlob(canvas, jpegQuality);
        onProgress('encoding', 1);

        const filename = _changeExtension(file.name);
        return { blob, filename };
    }

    return { processFile, shouldConvert };
})();
