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

from . import db
from .models import User
from .utils.report_generator import generate_report
from .utils.dicom_exporter import export_to_dicom
from src.brain_tumor_detection import BrainTumorCNN, class_names, image_size
from src.ct_classification import CTNet

def init_routes(app):
    # Model Information
    MODEL_INFO = {
        'mri': {
            'name': 'Brain Tumor Detection (MRI)',
            'classes': class_names,
            'model': BrainTumorCNN(),
            'weights_path': 'models/brain_tumor_detection.pth',
            'transform': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        },
        'ct': {
            'name': 'CT Scan Analysis',
            'classes': ['Normal', 'Abnormal'],
            'model': CTNet(),
            'weights_path': 'models/ct_classification.pth',
            'transform': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        }
    }

    @app.route('/health')
    def health_check():
        return jsonify({"status": "healthy"}), 200

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            user = User.query.filter_by(email=email).first()

            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Invalid email or password')

        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if User.query.filter_by(email=email).first():
                flash('Email already exists')
                return redirect(url_for('register'))

            if User.query.filter_by(username=username).first():
                flash('Username already exists')
                return redirect(url_for('register'))

            new_user = User(
                username=username,
                email=email,
                password=password  # This will use the password setter method
            )
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('index'))

    # Add all your other routes here...

    return app