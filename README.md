# Brain Tumor Detection System

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![Flask](https://img.shields.io/badge/flask-2.x-lightgrey)

A sophisticated medical imaging platform that uses deep learning to detect and analyze brain tumors from MRI and CT scans with high accuracy.

## Overview

This application provides healthcare professionals with an intuitive interface to upload and analyze medical brain scans. Using state-of-the-art convolutional neural networks, the system can detect the presence of tumors and provide AI-generated insights to assist medical professionals.

### Demo

<p align="center">
  <img src="public/Brain Tumor Detection Demo.gif" alt="Brain Tumor Detection Demo" width="700">
</p>

> **Note**: The GIF above demonstrates the key features of the application.

## Features

- **Multi-scan Support**: Analyze both MRI and CT scans
- **High Accuracy Detection**: Powered by custom-trained CNN models
- **Interactive UI**: User-friendly interface with real-time analysis
- **AI-powered Suggestions**: Get AI-generated insights based on scan results
- **PDF Report Generation**: Create and download detailed patient reports
- **Secure User Authentication**: Protect patient data with user accounts

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow, PyTorch
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: Flask-Login

## Model Architecture

The brain tumor detection model uses a custom Convolutional Neural Network (CNN) architecture:

- Input layer for processing 224x224 pixel images
- Multiple convolutional layers with ReLU activation
- Max pooling layers for feature extraction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Softmax output layer for multi-class prediction

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/heyadrsh/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # On Windows
   set FLASK_APP=app.py
   set FLASK_ENV=development
   # On macOS/Linux
   export FLASK_APP=app.py
   export FLASK_ENV=development
   ```

5. Initialize the database:
   ```bash
   python recreate_db.py
   ```

6. Run the application:
   ```bash
   python app.py
   ```

7. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. **Register/Login**: Create an account or log in to access the system
2. **Select Scan Type**: Choose between MRI or CT scan analysis
3. **Upload Image**: Upload a brain scan image for analysis
4. **View Results**: See detection results with confidence scores
5. **Get AI Insights**: Request AI-generated suggestions based on the results
6. **Generate Report**: Create a downloadable PDF report for the patient

## Future Enhancements

- Integration with hospital PACS systems
- Support for additional scan types (PET, fMRI)
- Enhanced AI suggestions with more detailed analysis
- Mobile application for on-the-go access
- Cloud deployment for scalability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Aadarsh Kumar - [@heyadrsh](https://github.com/heyadrsh) - heyadrsh@gmail.com

---

<p align="center">
  <i>Built for advancing medical diagnostics through AI</i>
</p>
