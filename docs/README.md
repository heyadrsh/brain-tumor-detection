# Brain Tumor Detection

A deep learning-based web application for detecting brain tumors in MRI scans.

## Features
- Upload and process brain MRI scans
- Real-time tumor detection using PyTorch
- Web interface built with Flask
- Supports common image formats

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd brain-tumor-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model weights:
You'll need to download the model weights separately and place them in the root directory.

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`
3. Upload an MRI scan image
4. View the detection results

## Project Structure
- `app.py`: Flask web application
- `brain_tumor_detection.py`: Core detection logic
- `requirements.txt`: Project dependencies
- `templates/`: HTML templates
- `static/`: Static assets (CSS, JS, images)

## Technologies Used
- Python 3.x
- PyTorch
- Flask
- OpenCV
- NumPy
- Pillow

## License
MIT License
