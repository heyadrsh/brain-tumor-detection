from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime
import os

def generate_report(patient_name, scan_type, prediction, confidence, image_path, output_dir="reports"):
    # Create reports directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/report_{patient_name}_{timestamp}.pdf"
    
    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Add header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Medical Image Analysis Report")
    
    # Add content
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Patient Name: {patient_name}")
    c.drawString(50, 680, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 660, f"Scan Type: {scan_type}")
    c.drawString(50, 640, f"Prediction: {prediction}")
    c.drawString(50, 620, f"Confidence: {confidence:.2f}%")
    
    # Add image if available
    if os.path.exists(image_path):
        c.drawImage(image_path, 50, 300, width=400, height=300)
    
    # Add footer
    c.setFont("Helvetica-Italic", 10)
    c.drawString(50, 50, "This is an AI-generated report and should be reviewed by a medical professional.")
    
    c.save()
    return filename 