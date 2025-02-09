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

from .models import User
from . import db
from .utils.report_generator import generate_report
from .utils.dicom_exporter import export_to_dicom
from src.brain_tumor_detection import BrainTumorCNN, class_names, image_size
from src.ct_classification import CTNet

def init_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')
        
    # Copy all your routes from app.py here
    # Make sure to properly indent them under init_routes function
    
    return app 