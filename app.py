from flask import render_template, request, jsonify, url_for, redirect, flash, session, send_file
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import time
import json

try:
    import psutil
except ImportError:
    print("psutil not installed. CPU metrics will be limited.")
    psutil = None

from app import create_app, db, login_manager
from app.utils.report_generator import generate_report
from app.utils.dicom_exporter import export_to_dicom
from app.models import User
from src.brain_tumor_detection import BrainTumorCNN, class_names, image_size
from src.ct_classification import CTNet

app = create_app()

# Model Information
MODEL_INFO = {
    'mri': {
        'version': '2.1.0',
        'last_updated': '2024-02-15',
        'accuracy': 98.86,
        'batch_size': 179
    },
    'ct': {
        'version': '1.8.0',
        'last_updated': '2024-02-15',
        'accuracy': 94.5,
        'batch_size': 1
    }
}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create upload directories if they don't exist
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load MRI model
try:
    mri_model = BrainTumorCNN()
    mri_model.load_state_dict(torch.load('app/models/brain_tumor_model/brain_tumor_detection_model.pth'))
    mri_model.eval()
except Exception as e:
    print(f"Error loading MRI model: {e}")
    mri_model = None

# Load CT model
try:
    ct_model = CTNet(num_classes=3)
    ct_model.load_state_dict(torch.load('app/models/ct_model/best_ct_model.pth'))
    ct_model.eval()
except Exception as e:
    print(f"Error loading CT model: {e}")
    ct_model = None

# Transform for MRI model
mri_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for CT model
ct_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classes for both models
mri_classes = class_names
ct_classes = ['aneurysm', 'cancer', 'tumor']

def get_system_metrics():
    """Get system metrics like CPU and memory usage"""
    metrics = {
        'cpu_usage': None,
        'memory_usage': None
    }
    
    if psutil:
        try:
            metrics['cpu_usage'] = f"{psutil.cpu_percent()}%"
            metrics['memory_usage'] = f"{psutil.virtual_memory().percent}%"
        except Exception as e:
            print(f"Error getting system metrics: {e}")
    
    return metrics

def preprocess_image(image, transform, is_mri=True):
    """Preprocess image to ensure consistent size and format"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    target_size = image_size if is_mri else (224, 224)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

@app.route('/predict', methods=['POST'])
def predict():
    scan_type = request.form.get('scan_type', 'mri')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'success': False})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False})

    try:
        # Start timing
        start_total = time.time()
        preprocessing_start = time.time()

        # Read and preprocess image
        image = Image.open(file).convert('RGB')
        
        if scan_type == 'mri':
            if mri_model is None:
                return jsonify({'error': 'MRI model not loaded', 'success': False})
                
            # MRI Processing
            image_tensor = preprocess_image(image, mri_transform, is_mri=True)
            preprocessing_time = time.time() - preprocessing_start
            
            # Inference timing
            inference_start = time.time()
            with torch.no_grad():
                outputs = mri_model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = mri_classes[predicted.item()]
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence = probabilities[predicted.item()].item() * 100
                
                scores = {}
                for class_name, prob in zip(mri_classes, probabilities):
                    formatted_name = class_name.replace('_', '-')
                    scores[formatted_name] = prob.item()
        else:
            if ct_model is None:
                return jsonify({'error': 'CT model not loaded', 'success': False})
                
            # CT Processing
            image_tensor = preprocess_image(image, ct_transform, is_mri=False)
            preprocessing_time = time.time() - preprocessing_start
            
            inference_start = time.time()
            with torch.no_grad():
                outputs = ct_model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = ct_classes[predicted.item()]
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence = probabilities[predicted.item()].item() * 100
                
                scores = {
                    class_name.lower(): prob.item()
                    for class_name, prob in zip(ct_classes, probabilities)
                }

        # Calculate timing metrics
        inference_time = time.time() - inference_start
        total_time = time.time() - start_total

        # Get system metrics
        system_metrics = get_system_metrics()

        # Get GPU memory usage if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

        # Format prediction
        formatted_prediction = prediction.replace('_', '-')
        
        # Store prediction and confidence in session
        session['last_prediction'] = formatted_prediction
        session['last_confidence'] = confidence
        
        # Get model information
        model_info = MODEL_INFO[scan_type]
        
        return jsonify({
            'success': True,
            'prediction': formatted_prediction,
            'confidence': f"{confidence:.2f}%",
            'scores': scores,
            'metrics': {
                'total_time': f"{total_time:.3f}s",
                'preprocessing': f"{preprocessing_time:.3f}s",
                'inference': f"{inference_time:.3f}s",
            },
            'model_info': {
                'version': model_info['version'],
                'last_updated': model_info['last_updated'],
                'accuracy': model_info['accuracy'],
                'batch_size': model_info['batch_size']
            },
            'hardware': {
                'gpu_memory': f"{gpu_memory:.1f}MB" if gpu_memory is not None else None,
                'cpu_usage': system_metrics['cpu_usage'],
                'memory_usage': system_metrics['memory_usage']
            }
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e), 'success': False})

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('select_scan_type'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('select_scan_type'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/select_scan_type')
@login_required
def select_scan_type():
    return render_template('select_scan.html')

@app.route('/mri_analysis')
@login_required
def mri_analysis():
    return render_template('mri_analysis.html')

@app.route('/ct_analysis')
@login_required
def ct_analysis():
    return render_template('ct_analysis.html')

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report_route():
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        scan_type = data.get('scan_type')
        patient_name = data.get('patient_name')
        
        # Get prediction and confidence from session
        prediction = session.get('last_prediction', 'Unknown')
        confidence = session.get('last_confidence', 0)
        
        # Generate report
        report_path = generate_report(
            patient_name=patient_name,
            scan_type=scan_type,
            prediction=prediction,
            confidence=confidence,
            image_path=image_path
        )
        
        return jsonify({
            'success': True,
            'report_url': url_for('download_report', filename=os.path.basename(report_path))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_dicom', methods=['POST'])
@login_required
def export_dicom_route():
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        patient_name = data.get('patient_name')
        
        # Export to DICOM
        dicom_path = export_to_dicom(
            image_path=image_path,
            patient_name=patient_name
        )
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_dicom', filename=os.path.basename(dicom_path))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    return send_file(
        os.path.join('reports', filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/download_dicom/<filename>')
@login_required
def download_dicom(filename):
    return send_file(
        os.path.join('dicom_exports', filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)