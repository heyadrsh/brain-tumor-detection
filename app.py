from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from brain_tumor_detection import BrainTumorCNN, class_names
import io
import base64
from scipy.ndimage import zoom
import nibabel as nib
from skimage import measure
from pathlib import Path
from skimage import filters
from scipy import stats

app = Flask(__name__)

# Initialize model
def load_model():
    model = BrainTumorCNN()
    model.load_state_dict(torch.load('brain_tumor_detection_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)

def apply_clahe(image):
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final)

def apply_denoising(image):
    denoised = cv2.fastNlMeansDenoisingColored(np.array(image))
    return Image.fromarray(denoised)

def get_gradcam(model, image, pred_class):
    model.eval()
    target_layer = model.conv_layers[-4]
    
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)
    
    x = process_image(image)
    x.requires_grad_()
    
    activations = []
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)
    
    handle = target_layer.register_forward_hook(forward_hook)
    
    output = model(x)
    
    if pred_class is None:
        pred_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    score = output[0, pred_class]
    score.backward()
    
    handle.remove()
    
    if not gradients or not activations:
        return None
    
    gradients = gradients[0].cpu().data.numpy()[0]
    activations = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (150, 150))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-7)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img_array = np.array(image.resize((150, 150)))
    superimposed = heatmap * 0.4 + img_array
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed)

def estimate_tumor_size(image, pred_class):
    if pred_class == 2:  # No tumor
        return 0
    
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        return area / (image.size[0] * image.size[1]) * 100
    return 0

def create_3d_reconstruction(slices):
    """Create 3D reconstruction from multiple MRI slices"""
    try:
        print("Starting 3D reconstruction...")
        # Convert slices to numpy array and normalize
        volume = []
        for i, slice_img in enumerate(slices):
            print(f"Processing slice {i+1}/{len(slices)}")
            # Convert to grayscale if not already
            if isinstance(slice_img, Image.Image):
                slice_array = np.array(slice_img.convert('L'))
            else:
                slice_array = np.array(slice_img)
            
            # Ensure 2D array
            if len(slice_array.shape) > 2:
                slice_array = cv2.cvtColor(slice_array, cv2.COLOR_RGB2GRAY)
            
            # Normalize slice
            slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
            volume.append(slice_array)
        
        print("Stacking slices into 3D volume...")
        # Stack slices into 3D volume
        volume = np.stack(volume)
        print(f"Volume shape: {volume.shape}")
        
        print("Creating binary mask...")
        # Create binary mask for tumor region using Otsu's thresholding
        threshold = filters.threshold_otsu(volume)
        binary_volume = volume > threshold
        
        print("Applying Gaussian smoothing...")
        # Apply some preprocessing to smooth the volume
        binary_volume = filters.gaussian(binary_volume, sigma=1)
        
        print("Generating mesh using marching cubes...")
        # Extract surface mesh using marching cubes
        verts, faces, _, _ = measure.marching_cubes(binary_volume, level=0.5)
        
        print("Scaling vertices...")
        # Scale vertices to match original dimensions
        scale_factors = np.array(volume.shape)
        verts = verts * scale_factors / np.max(scale_factors)  # Normalize to keep proportions
        
        print("Creating mesh data structure...")
        # Create mesh data structure for Three.js
        mesh_data = {
            'vertices': verts.tolist(),
            'faces': faces.tolist(),
            'dimensions': volume.shape
        }
        
        print("3D reconstruction completed successfully")
        return mesh_data
        
    except Exception as e:
        print(f"Error in 3D reconstruction: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to create 3D reconstruction: {str(e)}")

def segment_tumor(image):
    """Segment tumor region from MRI slice"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask
    mask = np.zeros_like(gray)
    if contours:
        # Find largest contour (assumed to be tumor)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask

def calculate_volume_in_cc(pixel_count, pixel_spacing_mm=1.0, slice_thickness_mm=1.0):
    """
    Calculate volume in cubic centimeters from pixel count
    Default values based on typical MRI parameters:
    - pixel_spacing_mm: typically 1mm for standard MRI
    - slice_thickness_mm: typically 1mm for standard MRI sequence
    """
    # Convert pixel count to volume in mm³
    volume_mm3 = pixel_count * (pixel_spacing_mm ** 2) * slice_thickness_mm
    
    # Convert mm³ to cc (1 cc = 1000 mm³)
    volume_cc = volume_mm3 / 1000.0
    
    return volume_cc

def calculate_growth_rate(volumes, time_intervals):
    """
    Calculate tumor growth rate using exponential growth model
    volumes: List of tumor volumes in cc
    time_intervals: List of days since first scan
    Returns: growth rate (per day) and predicted volumes
    """
    # Convert to numpy arrays
    volumes = np.array(volumes)
    time_intervals = np.array(time_intervals)
    
    # Take natural log of volumes
    log_volumes = np.log(volumes)
    
    # Perform linear regression on log volumes
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_intervals, log_volumes)
    
    # Calculate growth rate (per day)
    growth_rate = slope  # This is the exponential growth rate
    
    # Generate predictions for next 90 days
    future_times = np.arange(0, 90, 30)  # Predict at 0, 30, 60, 90 days
    predicted_log_volumes = slope * future_times + intercept
    predicted_volumes = np.exp(predicted_log_volumes)
    
    # Calculate doubling time
    doubling_time = np.log(2) / growth_rate if growth_rate > 0 else float('inf')
    
    return {
        'growth_rate_per_day': float(growth_rate),
        'doubling_time_days': float(doubling_time),
        'r_squared': float(r_value**2),
        'predictions': [
            {
                'days': int(t),
                'volume_cc': float(v)
            } for t, v in zip(future_times, predicted_volumes)
        ]
    }

def process_image_by_modality(image, modality):
    """Process image based on imaging modality"""
    if modality == 'MRI':
        # Standard MRI processing
        return process_image(image)
    elif modality == 'CT':
        # CT-specific processing: enhance bone and tissue contrast
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Apply window-level adjustment for better CT visualization
        img_enhanced = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        # Convert back to RGB for consistent processing
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
        return process_image(Image.fromarray(img_rgb))
    elif modality == 'PET':
        # PET-specific processing: enhance metabolic activity regions
        img_array = np.array(image)
        # Enhance high-intensity regions (metabolically active)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2) # Enhance saturation
        img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return process_image(Image.fromarray(img_enhanced))
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def segment_tumor_by_modality(image, modality):
    """Segment tumor based on imaging modality"""
    if modality == 'MRI':
        return segment_tumor(image)
    elif modality == 'CT':
        # CT-specific segmentation
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Apply threshold suitable for CT
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif modality == 'PET':
        # PET-specific segmentation (focus on high uptake regions)
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        # Threshold on saturation channel to detect high metabolic activity
        _, thresh = cv2.threshold(hsv[:,:,1], 127, 255, cv2.THRESH_BINARY)
        return thresh
    else:
        raise ValueError(f"Unsupported modality: {modality}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        image = Image.open(file).convert('RGB')
        
        # Apply image processing if requested
        enhance_contrast = request.form.get('enhance_contrast') == 'true'
        reduce_noise = request.form.get('reduce_noise') == 'true'
        
        if enhance_contrast:
            image = apply_clahe(image)
        if reduce_noise:
            image = apply_denoising(image)
        
        # Make prediction
        with torch.no_grad():
            img_tensor = process_image(image)
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Calculate tumor size
        tumor_size = estimate_tumor_size(image, predicted_class)
        
        # Generate Grad-CAM visualization
        gradcam_image = get_gradcam(model, image, predicted_class)
        
        # Convert images to base64 for response
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        buffered_gradcam = io.BytesIO()
        gradcam_image.save(buffered_gradcam, format="PNG")
        gradcam_str = base64.b64encode(buffered_gradcam.getvalue()).decode()
        
        result = {
            'success': True,
            'class': class_names[predicted_class],
            'confidence': float(max(probabilities[0])) * 100,
            'tumor_size': tumor_size,
            'probabilities': {
                class_name: float(prob) * 100 
                for class_name, prob in zip(class_names, probabilities[0])
            },
            'original_image': img_str,
            'gradcam_image': gradcam_str,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_3d', methods=['POST'])
def analyze_3d():
    try:
        print("Starting 3D analysis...")
        files = request.files.getlist('files')
        if not files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        print(f"Received {len(files)} files")
        # Get MRI parameters from form data (or use defaults)
        pixel_spacing_mm = float(request.form.get('pixel_spacing_mm', 1.0))
        slice_thickness_mm = float(request.form.get('slice_thickness_mm', 1.0))
        
        print(f"Parameters: pixel_spacing={pixel_spacing_mm}mm, slice_thickness={slice_thickness_mm}mm")
        
        # Process each slice
        slices = []
        slice_results = []
        total_tumor_pixels = 0
        
        for i, file in enumerate(sorted(files, key=lambda x: x.filename)):
            print(f"Processing file {i+1}: {file.filename}")
            # Read image
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Get prediction
            with torch.no_grad():
                img_tensor = process_image(image)
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Segment tumor
            tumor_mask = segment_tumor(image)
            tumor_pixels = np.sum(tumor_mask > 0)
            total_tumor_pixels += tumor_pixels
            
            print(f"File {file.filename}: Found {tumor_pixels} tumor pixels")
            
            # Convert mask to base64
            _, buffer = cv2.imencode('.png', tumor_mask)
            mask_b64 = base64.b64encode(buffer).decode()
            
            # Calculate slice-specific volume
            slice_volume_cc = calculate_volume_in_cc(tumor_pixels, pixel_spacing_mm, slice_thickness_mm)
            
            # Store slice information
            slice_results.append({
                'slice_number': len(slices) + 1,
                'class': class_names[predicted_class],
                'confidence': float(max(probabilities[0])) * 100,
                'mask': mask_b64,
                'slice_volume_cc': round(slice_volume_cc, 2)
            })
            
            # Store processed image for 3D reconstruction
            slices.append(image)
        
        print("Creating 3D reconstruction...")
        # Create 3D reconstruction
        reconstruction_data = create_3d_reconstruction(slices)
        
        print("Calculating final metrics...")
        # Calculate total volume metrics
        total_volume_pixels = len(slices) * slices[0].size[0] * slices[0].size[1]
        volume_percentage = (total_tumor_pixels / total_volume_pixels) * 100
        total_volume_cc = calculate_volume_in_cc(total_tumor_pixels, pixel_spacing_mm, slice_thickness_mm)
        
        # Calculate dimensions in cm
        dimensions_cm = {
            'width': (slices[0].size[0] * pixel_spacing_mm) / 10,
            'height': (slices[0].size[1] * pixel_spacing_mm) / 10,
            'depth': (len(slices) * slice_thickness_mm) / 10
        }
        
        result = {
            'success': True,
            'slice_results': slice_results,
            'reconstruction': reconstruction_data,
            'metrics': {
                'total_slices': len(slices),
                'tumor_volume_percentage': round(volume_percentage, 2),
                'tumor_volume_pixels': int(total_tumor_pixels),
                'tumor_volume_cc': round(total_volume_cc, 2),
                'dimensions_cm': {k: round(v, 2) for k, v in dimensions_cm.items()}
            }
        }
        
        print("Analysis completed successfully")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_3d: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_growth_rate', methods=['POST'])
def analyze_growth_rate():
    try:
        print("Starting growth rate analysis...")
        data = request.json
        
        if not data or 'scans' not in data:
            return jsonify({'success': False, 'error': 'No scan data provided'})
        
        scans = data['scans']
        if len(scans) < 2:
            return jsonify({'success': False, 'error': 'At least 2 scans needed for growth rate analysis'})
        
        # Extract volumes and dates
        volumes = [scan['volume_cc'] for scan in scans]
        dates = [datetime.strptime(scan['date'], '%Y-%m-%d') for scan in scans]
        
        # Calculate days since first scan
        first_date = min(dates)
        days_since_first = [(date - first_date).days for date in dates]
        
        # Calculate growth metrics
        growth_metrics = calculate_growth_rate(volumes, days_since_first)
        
        # Prepare scan history
        scan_history = []
        for date, volume, days in zip(dates, volumes, days_since_first):
            scan_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'volume_cc': volume,
                'days_since_first': days
            })
        
        result = {
            'success': True,
            'growth_metrics': growth_metrics,
            'scan_history': scan_history,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'next_recommended_scan': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        }
        
        print("Growth rate analysis completed successfully")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in growth rate analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_multimodal', methods=['POST'])
def analyze_multimodal():
    try:
        print("Starting multi-modal analysis...")
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
            
        file = request.files['file']
        modality = request.form.get('modality', 'MRI')
        print(f"Processing {modality} image")
        
        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process based on modality
        img_tensor = process_image_by_modality(image, modality)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = float(max(probabilities[0])) * 100
        
        # Segment tumor based on modality
        tumor_mask = segment_tumor_by_modality(image, modality)
        
        # Convert mask to base64
        _, buffer = cv2.imencode('.png', tumor_mask)
        mask_b64 = base64.b64encode(buffer).decode()
        
        # Calculate volume
        pixel_spacing_mm = float(request.form.get('pixel_spacing_mm', 1.0))
        slice_thickness_mm = float(request.form.get('slice_thickness_mm', 1.0))
        tumor_pixels = np.sum(tumor_mask > 0)
        volume_cc = calculate_volume_in_cc(tumor_pixels, pixel_spacing_mm, slice_thickness_mm)
        
        # Prepare modality-specific metrics
        metrics = {
            'MRI': {
                'contrast_ratio': float(np.mean(tumor_mask) / np.mean(~tumor_mask)),
                'intensity_std': float(np.std(np.array(image))),
            },
            'CT': {
                'hounsfield_mean': float(np.mean(np.array(image)[tumor_mask > 0])),
                'density_ratio': float(np.mean(tumor_mask) / np.mean(~tumor_mask)),
            },
            'PET': {
                'suv_max': float(np.max(np.array(image)[tumor_mask > 0])),
                'metabolic_volume': float(np.sum(tumor_mask > 0) * pixel_spacing_mm * pixel_spacing_mm * slice_thickness_mm / 1000),
            }
        }
        
        result = {
            'success': True,
            'modality': modality,
            'class': class_names[predicted_class],
            'confidence': confidence,
            'mask': mask_b64,
            'volume_cc': volume_cc,
            'modality_metrics': metrics.get(modality, {}),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("Multi-modal analysis completed successfully")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in multi-modal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 