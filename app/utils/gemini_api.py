import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize the Gemini API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_ai_suggestion(patient_info, scan_result):
    """
    Generate AI suggestions based on patient information and scan results

    Args:
        patient_info (dict): Patient information including name, age, gender
        scan_result (dict): Scan result information including prediction, confidence, scan_type

    Returns:
        str: AI-generated suggestion
    """
    try:
        # Create the prompt for Gemini
        prompt = f"""
        Generate a short, concise medical suggestion based on the following patient information and scan results.
        Keep the response under 200 words and focus on providing clear, actionable advice.

        Patient Information:
        - Name: {patient_info.get('name', 'Unknown')}
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}

        Scan Results:
        - Scan Type: {scan_result.get('scan_type', 'Unknown')}
        - Prediction: {scan_result.get('prediction', 'Unknown')}
        - Confidence: {scan_result.get('confidence', '0')}%

        Format your response with the following sections using markdown formatting:
        1. **Summary:** A brief summary of the findings (1-2 sentences)
        2. **Meaning:** What this might mean (1-2 sentences)
        3. **Recommended Next Steps:**
           * First recommendation
           * Second recommendation
           * Third recommendation if needed

        End with a **Disclaimer:** paragraph that this is AI-generated advice and not a substitute for professional medical consultation.

        Use proper markdown formatting with bold headers and bullet points.
        """

        # Configure the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Generate the response
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        print(f"Error generating AI suggestion: {str(e)}")
        return "Unable to generate AI suggestion at this time. Please consult with a healthcare professional for proper medical advice."

def generate_ai_suggestion_for_report(patient_info, scan_result):
    """
    Generate AI suggestions for PDF report

    Args:
        patient_info (dict): Patient information including name, age, gender
        scan_result (dict): Scan result information including prediction, confidence, scan_type

    Returns:
        str: AI-generated suggestion formatted for PDF report
    """
    try:
        # Create the prompt for Gemini
        prompt = f"""
        Generate a concise medical suggestion for a PDF report based on the following patient information and scan results.
        Keep the response under 150 words and focus on providing clear, actionable advice.

        Patient Information:
        - Name: {patient_info.get('name', 'Unknown')}
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}

        Scan Results:
        - Scan Type: {scan_result.get('scan_type', 'Unknown')}
        - Prediction: {scan_result.get('prediction', 'Unknown')}
        - Confidence: {scan_result.get('confidence', '0')}%

        Format your response with the following sections using markdown formatting:
        1. **AI Analysis Summary:** (1-2 sentences)
        2. **Recommended Next Steps:**
           * First recommendation
           * Second recommendation
           * Third recommendation if needed

        End with a **Disclaimer:** paragraph that this is AI-generated advice and not a substitute for professional medical consultation.

        Use proper markdown formatting with bold headers and bullet points.
        """

        # Configure the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Generate the response
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        print(f"Error generating AI suggestion for report: {str(e)}")
        return "AI suggestion unavailable. Please consult with a healthcare professional for proper medical advice."
