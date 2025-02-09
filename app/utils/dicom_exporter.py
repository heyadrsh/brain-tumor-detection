import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
import os
import cv2
import numpy as np

def export_to_dicom(image_path, patient_name, output_dir="dicom_exports"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.ImplementationClassUID = '1.2.3.4'
    
    # Create the FileDataset
    timestamp = datetime.now()
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add patient info
    ds.PatientName = patient_name
    ds.PatientID = f"ID_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    # Add study info
    ds.StudyDate = timestamp.strftime('%Y%m%d')
    ds.StudyTime = timestamp.strftime('%H%M%S')
    ds.StudyDescription = 'AI Analysis Result'
    
    # Add image data
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = img.shape[1]
    ds.Rows = img.shape[0]
    ds.PixelData = img.tobytes()
    
    # Save the DICOM file
    output_path = os.path.join(output_dir, f"scan_{timestamp.strftime('%Y%m%d_%H%M%S')}.dcm")
    ds.save_as(output_path)
    
    return output_path 