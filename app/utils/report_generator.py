from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from datetime import datetime
import os
from app.utils.gemini_api import generate_ai_suggestion_for_report

def get_age_group_message(age):
    if age < 10:
        return "Early years of life. Tumors in this age group are rare but need close monitoring for development."
    elif 10 <= age < 20:
        return "During adolescence, rapid tumor growth can occur. Regular monitoring and timely interventions are essential."
    elif 20 <= age < 30:
        return "Young adulthood is a critical phase where tumors may develop more aggressively. Surgical treatment might be necessary."
    elif 30 <= age < 40:
        return "In your 30s, tumors can grow at a steady rate. Treatment options include surgery, radiation, or chemotherapy."
    elif 40 <= age < 50:
        return "In your 40s, tumor progression can vary. Regular screenings are important to track the tumor."
    elif 50 <= age < 60:
        return "At this age, tumors may have slower growth rates, but intervention could still be necessary."
    elif 60 <= age < 70:
        return "Older adults may face different tumor progression, often with slower growth but increased risk of complications."
    elif 70 <= age < 80:
        return "In your 70s, treatment should focus on managing growth and minimizing side effects."
    elif 80 <= age < 90:
        return "At this age, tumor progression is typically slower, with focus on comfort and quality of life."
    else:
        return "Focus is on symptom management and maintaining quality of life."

def get_gender_specific_message(gender, prediction):
    if prediction == "glioma":
        if gender == "Male":
            return "Gliomas are more common in males, especially in young adults. Early intervention can lead to better outcomes."
        else:
            return "Although gliomas are less common in females, they still require prompt treatment and careful monitoring."
    elif prediction == "meningioma":
        if gender == "Female":
            return "Meningiomas are more commonly found in females, especially in older age groups. Regular monitoring is advised."
        else:
            return "While less frequent in males, meningiomas can grow larger and may require surgical intervention."
    elif prediction == "pituitary":
        return "Pituitary tumors affect both genders similarly, but symptoms can vary based on hormone imbalances."
    else:  # no-tumor
        return "No tumor was detected. Regular check-ups are recommended for ongoing monitoring."

def get_tumor_recommendations(prediction):
    prediction = prediction.lower().strip()

    # Handle variations of "no tumor" prediction
    if any(x in prediction for x in ['no tumor', 'no-tumor', 'notumor', 'normal']):
        return "No tumor was detected. Regular interval check-ups are recommended for monitoring."

    # Handle variations of tumor types
    if 'glioma' in prediction:
        return "Glioma detected - requires immediate medical attention. Treatment typically involves a combination of surgery, radiation, and chemotherapy. Early intervention is crucial for better outcomes."
    elif 'meningioma' in prediction:
        return "Meningioma detected - requires regular monitoring. May need surgical intervention if symptoms develop or tumor shows growth. Follow-up MRI scans recommended every 6-12 months."
    elif 'pituitary' in prediction:
        return "Pituitary tumor detected - requires endocrine evaluation. Treatment may include medication for hormone regulation and/or surgery depending on tumor size and symptoms."
    elif 'aneurysm' in prediction:
        return "Aneurysm detected - requires immediate vascular neurosurgery consultation. Close monitoring and potential surgical intervention may be necessary."
    elif 'cancer' in prediction:
        return "Cancer detected - requires immediate oncological consultation. Treatment plan will be determined based on type, stage, and location."
    elif 'tumor' in prediction:
        return "Tumor detected - requires further evaluation and specialist consultation. Additional imaging and tests may be needed to determine specific treatment approach."
    else:
        # Default case for unknown predictions
        return f"{prediction} detected - requires medical evaluation. Please consult with a healthcare provider for proper assessment and treatment planning."

def generate_report(patient_name, patient_age, patient_gender, scan_type, prediction, confidence, image_path, output_dir="reports"):
    try:
        # Create reports directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/report_{patient_name}_{timestamp}.pdf"

        # Create PDF
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica", 20)
        c.drawString(1*inch, height-1*inch, "Medical Image Analysis Report")

        # Patient Information
        c.setFont("Helvetica", 12)
        y = height - 1.5*inch
        c.drawString(1*inch, y, f"Patient Name: {patient_name}")
        c.drawString(1*inch, y-20, f"Age: {patient_age} years")
        c.drawString(1*inch, y-40, f"Gender: {patient_gender}")
        c.drawString(1*inch, y-60, f"Scan Type: {scan_type}")
        c.drawString(1*inch, y-80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Analysis Results
        y -= 120
        c.setFont("Helvetica", 14)
        c.drawString(1*inch, y, "Analysis Results")
        c.setFont("Helvetica", 12)

        prediction_lower = prediction.lower()
        if any(x in prediction_lower for x in ['no tumor', 'no-tumor', 'notumor', 'normal']):
            # Simplified report for no-tumor case
            c.drawString(1*inch, y-30, get_tumor_recommendations(prediction))
        else:
            # Detailed report for tumor cases
            c.drawString(1*inch, y-30, f"Diagnosis: {prediction}")
            c.drawString(1*inch, y-50, f"Confidence: {confidence}")

            # Clinical Assessment for tumor cases
            y -= 90
            c.setFont("Helvetica", 14)
            c.drawString(1*inch, y, "Clinical Assessment")
            c.setFont("Helvetica", 12)

            # Get recommendations
            recommendations = get_tumor_recommendations(prediction)

            # Wrap text for recommendations
            words = recommendations.split()
            line = []
            y_offset = 30
            for word in words:
                line.append(word)
                if len(' '.join(line)) > 65:
                    c.drawString(1*inch, y-y_offset, ' '.join(line[:-1]))
                    line = [word]
                    y_offset += 20
            if line:
                c.drawString(1*inch, y-y_offset, ' '.join(line))

            y = y - y_offset - 20

            # Additional notes for tumor cases
            c.setFont("Helvetica", 12)
            c.drawString(1*inch, y, "Follow-up Actions:")

            # Define follow-up actions based on prediction
            if 'glioma' in prediction_lower:
                c.drawString(1*inch, y-20, "• Immediate neurosurgical consultation")
                c.drawString(1*inch, y-40, "• MRI with contrast every 2-3 months")
                c.drawString(1*inch, y-60, "• Regular neurological assessments")
            elif 'meningioma' in prediction_lower:
                c.drawString(1*inch, y-20, "• Follow-up MRI in 3-6 months")
                c.drawString(1*inch, y-40, "• Monitor for new or worsening symptoms")
                c.drawString(1*inch, y-60, "• Regular neurosurgical evaluation")
            elif 'pituitary' in prediction_lower:
                c.drawString(1*inch, y-20, "• Endocrine function tests")
                c.drawString(1*inch, y-40, "• Visual field examination")
                c.drawString(1*inch, y-60, "• Follow-up MRI in 3 months")
            elif 'aneurysm' in prediction_lower:
                c.drawString(1*inch, y-20, "• Immediate vascular neurosurgery consultation")
                c.drawString(1*inch, y-40, "• CT angiogram or MRA may be needed")
                c.drawString(1*inch, y-60, "• Blood pressure monitoring")
            elif 'cancer' in prediction_lower:
                c.drawString(1*inch, y-20, "• Urgent oncology consultation")
                c.drawString(1*inch, y-40, "• Additional imaging studies")
                c.drawString(1*inch, y-60, "• Treatment planning with specialist team")
            else:
                c.drawString(1*inch, y-20, "• Specialist consultation recommended")
                c.drawString(1*inch, y-40, "• Additional diagnostic imaging may be needed")
                c.drawString(1*inch, y-60, "• Follow-up as directed by specialist")

            # Add AI Suggestion section
            y -= 100
            c.setFont("Helvetica", 14)
            c.drawString(1*inch, y, "AI-Generated Suggestion")
            c.setFont("Helvetica", 12)

            # Get AI suggestion
            patient_info = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender
            }

            scan_result = {
                'scan_type': scan_type,
                'prediction': prediction,
                'confidence': confidence
            }

            ai_suggestion = generate_ai_suggestion_for_report(patient_info, scan_result)

            # Wrap text for AI suggestion
            words = ai_suggestion.split()
            line = []
            y_offset = 30
            for word in words:
                line.append(word)
                if len(' '.join(line)) > 65:
                    c.drawString(1*inch, y-y_offset, ' '.join(line[:-1]))
                    line = [word]
                    y_offset += 20
            if line:
                c.drawString(1*inch, y-y_offset, ' '.join(line))

            y = y - y_offset - 20

        # Add image if available
        if os.path.exists(image_path):
            try:
                # Calculate remaining space
                if any(x in prediction_lower for x in ['no tumor', 'no-tumor', 'notumor', 'normal']):
                    image_y = y - 100  # Higher position for no-tumor case
                else:
                    image_y = y - 150  # Lower position for tumor cases

                image_height = 3*inch
                c.drawImage(image_path, 1*inch, image_y-image_height, width=4*inch, height=image_height)
            except Exception as img_error:
                print(f"Warning: Could not add image to report: {str(img_error)}")

        # Footer
        c.setFont("Helvetica", 10)
        c.drawString(1*inch, 1*inch, "This is an AI-generated report and should be reviewed by a medical professional.")
        c.drawString(1*inch, 0.75*inch, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save the PDF
        c.save()
        return filename

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise Exception(f"Failed to generate report: {str(e)}")