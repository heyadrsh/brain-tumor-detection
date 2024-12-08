class ModelInfoManager {
    constructor(modelType) {
        this.modelType = modelType;
        this.architectureModal = new bootstrap.Modal(document.getElementById('architectureModal'));
        this.initializeTooltips();
    }

    initializeTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    updateConfidenceScores(scores) {
        const classes = this.modelType === 'mri' 
            ? ['glioma', 'meningioma', 'pituitary', 'no-tumor']
            : ['aneurysm', 'cancer', 'tumor'];

        classes.forEach(className => {
            const score = scores[className] || 0;
            const formattedScore = (score * 100).toFixed(1) + '%';
            
            // Update score text
            const scoreElement = document.getElementById(`score-${className}`);
            if (scoreElement) {
                scoreElement.textContent = formattedScore;
            }
            
            // Update progress bar
            const progressElement = document.getElementById(`progress-${className}`);
            if (progressElement) {
                progressElement.style.width = formattedScore;
                
                // Add color based on confidence level
                this.updateProgressBarColor(progressElement, score);
            }
        });
    }

    updateProgressBarColor(element, score) {
        // Remove existing color classes
        element.classList.remove('bg-danger', 'bg-warning', 'bg-info', 'bg-success');
        
        // Add appropriate color class based on confidence level
        if (score < 0.3) {
            element.classList.add('bg-danger');
        } else if (score < 0.6) {
            element.classList.add('bg-warning');
        } else if (score < 0.8) {
            element.classList.add('bg-info');
        } else {
            element.classList.add('bg-success');
        }
    }

    showArchitecture() {
        this.architectureModal.show();
    }

    hideArchitecture() {
        this.architectureModal.hide();
    }
}

// Function to toggle architecture modal
function toggleModelArchitecture() {
    const modelInfo = window.modelInfo || new ModelInfoManager(
        document.querySelector('.model-info-container').dataset.modelType
    );
    modelInfo.showArchitecture();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize model info manager
    window.modelInfo = new ModelInfoManager(
        document.querySelector('.model-info-container').dataset.modelType
    );

    // Handle architecture modal close
    document.getElementById('architectureModal').addEventListener('hidden.bs.modal', function() {
        window.modelInfo.hideArchitecture();
    });
}); 