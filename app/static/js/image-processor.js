class ImageProcessor {
    constructor(imageId, canvasId) {
        this.image = document.getElementById(imageId);
        this.canvas = document.getElementById(canvasId);
        this.processedCanvas = document.getElementById('processed-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.processedCtx = this.processedCanvas.getContext('2d');
        
        // Image processing parameters
        this.scale = 1.0;
        this.brightness = 0;
        this.contrast = 100;
        this.rotation = 0;
        
        // View sync state
        this.syncViews = true;
        
        // Initialize
        this.image.onload = () => {
            this.drawImage();
            this.updateProcessedView();
        };
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Zoom with mouse wheel
        [this.canvas, this.processedCanvas].forEach(canvas => {
            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                this.zoom(delta);
            });
        });

        // Pan functionality
        let isDragging = false;
        let startX, startY, translateX = 0, translateY = 0;

        const handleDragStart = (e) => {
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
        };

        const handleDragMove = (e) => {
            if (isDragging) {
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                this.drawImage();
                this.updateProcessedView();
            }
        };

        const handleDragEnd = () => {
            isDragging = false;
        };

        [this.canvas, this.processedCanvas].forEach(canvas => {
            canvas.addEventListener('mousedown', handleDragStart);
            canvas.addEventListener('mousemove', handleDragMove);
            canvas.addEventListener('mouseup', handleDragEnd);
            canvas.addEventListener('mouseleave', handleDragEnd);
        });

        // Sync views toggle
        document.getElementById('syncViews').addEventListener('change', (e) => {
            this.syncViews = e.target.checked;
        });
    }

    drawImage(targetCanvas = this.canvas, targetCtx = this.ctx) {
        // Clear canvas
        targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        
        // Save context state
        targetCtx.save();
        
        // Set canvas size to match image
        targetCanvas.width = this.image.width;
        targetCanvas.height = this.image.height;
        
        // Transform operations
        const centerX = targetCanvas.width / 2;
        const centerY = targetCanvas.height / 2;
        
        targetCtx.translate(centerX, centerY);
        targetCtx.rotate(this.rotation * Math.PI / 180);
        targetCtx.scale(this.scale, this.scale);
        targetCtx.translate(-centerX, -centerY);
        
        // Draw image
        targetCtx.drawImage(this.image, 0, 0);
        
        // Apply brightness/contrast
        const imageData = targetCtx.getImageData(0, 0, targetCanvas.width, targetCanvas.height);
        this.applyBrightnessContrast(imageData.data);
        targetCtx.putImageData(imageData, 0, 0);
        
        // Restore context state
        targetCtx.restore();
    }

    updateProcessedView() {
        if (this.syncViews) {
            this.drawImage(this.processedCanvas, this.processedCtx);
        }
    }

    zoom(delta) {
        this.scale *= delta;
        // Limit zoom range
        this.scale = Math.min(Math.max(0.1, this.scale), 5.0);
        this.drawImage();
        this.updateProcessedView();
        // Update zoom value display
        document.getElementById('zoom-value').textContent = Math.round(this.scale * 100) + '%';
    }

    rotate(degrees) {
        this.rotation = (this.rotation + degrees) % 360;
        this.drawImage();
        this.updateProcessedView();
    }

    setBrightness(value) {
        this.brightness = parseInt(value);
        this.drawImage();
        this.updateProcessedView();
    }

    setContrast(value) {
        this.contrast = parseInt(value);
        this.drawImage();
        this.updateProcessedView();
    }

    applyBrightnessContrast(pixels) {
        const brightness = this.brightness;
        const contrast = this.contrast;
        const factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

        for (let i = 0; i < pixels.length; i += 4) {
            for (let j = 0; j < 3; j++) {
                pixels[i + j] = factor * (pixels[i + j] - 128 + brightness) + 128;
            }
        }
    }

    resetImage() {
        this.scale = 1.0;
        this.brightness = 0;
        this.contrast = 100;
        this.rotation = 0;
        this.drawImage();
        this.updateProcessedView();
        
        // Reset UI elements
        document.getElementById('zoom-value').textContent = '100%';
        document.getElementById('brightness-value').textContent = '0';
        document.getElementById('contrast-value').textContent = '100';
    }
}

// View layout management
function setViewLayout(layout) {
    const viewerLayout = document.querySelector('.viewer-layout');
    viewerLayout.className = 'viewer-layout ' + layout;
    
    // Update active state of layout buttons
    document.querySelectorAll('.layout-controls .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[onclick="setViewLayout('${layout}')"]`).classList.add('active');
}

// Fullscreen management
function toggleFullscreen(viewId) {
    const view = document.getElementById(viewId);
    if (!document.fullscreenElement) {
        view.requestFullscreen();
        view.classList.add('fullscreen');
    } else {
        document.exitFullscreen();
        view.classList.remove('fullscreen');
    }
} 