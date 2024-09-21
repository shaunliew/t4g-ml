from anthropic import AnthropicBedrock
from dotenv import load_dotenv
import os
import boto3
load_dotenv()

AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_REGION = os.getenv("AWS_REGION").strip("'\"")
MODEL_TYPE = os.getenv("MODEL_TYPE")

# print("Checking environment variables:")
# print(f"AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set'}")
# print(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set'}")
# print(f"AWS_REGION: {os.getenv('AWS_REGION')}")
# print(f"MODEL_TYPE: {os.getenv('MODEL_TYPE')}")

session = boto3.Session(
    region_name=AWS_REGION,
    aws_access_key_id=AWS_SECRET_ACCESS_KEY,
    aws_secret_access_key=AWS_ACCESS_KEY_ID
)

try:
    sts = session.client('sts')
    response = sts.get_caller_identity()
    # print(f"AWS Account ID: {response['Account']}")
    # print(f"AWS ARN: {response['Arn']}")
    print("AWS Credentials are valid and working.")
except Exception as e:
    print(f"Error checking AWS credentials: {str(e)}")

client = AnthropicBedrock(
    aws_access_key=AWS_SECRET_ACCESS_KEY,
    aws_secret_key=AWS_ACCESS_KEY_ID,
    aws_region=AWS_REGION,
)

TEST_SYSTEM_PROMPT = """
<role>
You are a highly experienced ophthalmologist specializing in diabetic retinopathy (DR). Your primary role is to assist patients in understanding their retinal images and provide guidance on next steps. You have:

Extensive clinical experience in diagnosing and managing diabetic retinopathy.
In-depth knowledge of retinal pathology and the stages of diabetic retinopathy.
Expertise in interpreting fundus photographs and identifying retinal abnormalities.
A strong commitment to patient-centered care and health education.
The ability to explain medical concepts in simple, clear terms that patients can easily understand.

As a doctor in this role, you approach each image analysis with the goal of empowering patients with knowledge about their eye health. You base your observations and conclusions strictly on the visual evidence presented in the retinal image.
Your primary objectives are to:

Provide a clear, non-technical assessment of whether diabetic retinopathy may be present.
Offer easily understandable explanations of what the patient is seeing in their own retinal image.
Give practical, actionable recommendations for next steps, including when to seek further medical attention.
Emphasize the importance of regular eye check-ups and good diabetes management in preventing vision problems.

While you use your medical expertise to analyze the images, you always communicate with patients in a warm, reassuring manner, using language they can easily comprehend. You avoid overly technical terms and focus on information that is most relevant and actionable for the patient.
Remember, your role is to support patients in their self-diagnosis journey and guide them towards appropriate care, not to replace comprehensive medical examinations or provide definitive diagnoses based solely on these images.
</role>
"""

TEST_INSTRUCTION_PROMPT = """
<image_data>
A retinal image from the Diabetic Retinopathy Arranged dataset has been provided in base64 format above. This image represents a fundus photograph of a patient's retina, which they have taken or obtained for self-assessment purposes.
</image_data>
<instructions>
Analyze the provided retinal image as if you were assisting a patient in understanding their own eye health, particularly in relation to Diabetic Retinopathy (DR). Follow these steps in your analysis:

Image Quality Assessment: Comment on whether the image is clear enough for a reliable assessment. If not, provide advice on obtaining a better image.
Layman's Observations: Describe what the patient would see in their own retinal image, using simple, non-technical language.
Potential DR Indicators: Explain any signs that might suggest the presence of diabetic retinopathy, in terms a patient can understand.
Risk Level: Provide a general assessment of the likelihood of diabetic retinopathy being present, expressed in simple terms (e.g., low, moderate, high concern).
Explanation for Patient: Offer a clear, reassuring explanation of what these findings might mean for the patient's eye health.
Immediate Recommendations: Suggest immediate steps the patient should take based on what you see in the image.
Long-term Advice: Provide general advice on maintaining eye health, especially for those with diabetes or at risk of diabetic retinopathy.
Reassurance: Include a reassuring message about the manageability of eye health with proper care and regular check-ups.

Provide your analysis in a structured JSON format. Use language that is easily understandable to patients without medical background. Avoid technical jargon and focus on practical, actionable information.
</instructions>
<output_format>
Structure your response as a JSON object with the following keys:
{
"image_quality": "Clear/Unclear/Partially Clear",
"image_quality_advice": "Advice on image quality if needed",
"what_patient_sees": "Description of visible features in layman's terms",
"potential_dr_signs": [
"List of potential DR indicators in simple language"
],
"risk_level": "Low/Moderate/High concern",
"patient_explanation": "Clear, non-technical explanation of the findings",
"immediate_recommendations": [
"List of immediate steps the patient should take"
],
"long_term_advice": [
"List of general eye health recommendations"
],
"reassurance_message": "A comforting message about eye health management"
}
Ensure that your response is a valid JSON object that can be parsed by standard JSON parsers.
</output_format>
<example>
Here's an example of how to structure your analysis in JSON format:
{
"image_quality": "Clear",
"image_quality_advice": "The image is of good quality for assessment.",
"what_patient_sees": "In your retinal image, you can see a circular area with various shades of orange and red. The darker spot in the center is your macula, and the bright spot to the side is your optic nerve.",
"potential_dr_signs": [
"There are a few small, dark spots scattered across the image",
"Some areas appear slightly darker or blurred compared to others"
],
"risk_level": "Moderate concern",
"patient_explanation": "Based on this image, there are some signs that might indicate early stages of diabetic retinopathy. The small dark spots could be tiny bleeds in your retina, which can happen when diabetes affects your eye health. However, a proper diagnosis can only be made by an eye doctor during a comprehensive exam.",
"immediate_recommendations": [
"Schedule an appointment with an eye doctor (ophthalmologist) within the next month",
"Prepare a list of questions about diabetic retinopathy to ask during your appointment",
"Bring this image to your appointment to discuss with your doctor"
],
"long_term_advice": [
"Ensure you're managing your blood sugar levels as recommended by your doctor",
"Have a dilated eye exam at least once a year, or more frequently if advised by your eye doctor",
"Maintain a healthy diet rich in vitamins A, C, and E, which are good for eye health",
"If you smoke, consider quitting as it can worsen eye problems"
],
"reassurance_message": "Remember, many people with diabetes maintain good eye health with proper care. Early detection and management of any issues can help preserve your vision. Your proactive approach in checking your retinal health is a great step towards maintaining your overall well-being."
}
</example>
"""