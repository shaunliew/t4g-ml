from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import base64
import mimetypes
import json
from config import *

LABEL_MAPPING = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def load_trained_model(model_path, num_classes=5):
    # Create a new instance of the model
    model = EfficientNet.from_pretrained('efficientnet-b3')
    
    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load the saved state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Check if the state dict has the unexpected structure
    if '_fc.1.weight' in state_dict and '_fc.1.bias' in state_dict:
        # Modify the model's fc layer to match the saved state dict
        in_features = model._fc.in_features
        model._fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, num_classes)
        )
    else:
        # If the state dict has the expected structure, just modify the last layer
        in_features = model._fc.in_features
        model._fc = torch.nn.Linear(in_features, num_classes)
    
    # Load the state dict
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to the appropriate device
    model = model.to(device)
    
    return model, device

def preprocess_image(image_path, size=299):
    # Open the image file
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply the transformations
    input_tensor = preprocess(image)
    
    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)
    
    return input_batch

def predict_single_image(model, image_path, device):
    # Preprocess the image
    input_batch = preprocess_image(image_path)
    
    # Move the input and model to the correct device
    input_batch = input_batch.to(device)
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # Get the probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
    
    # Get the predicted class
    predicted_class = np.argmax(probabilities)
    
    # Get the predicted class name
    predicted_class_name = LABEL_MAPPING[predicted_class]
    
    # Get the highest probability
    highest_probability = probabilities[predicted_class]
    
    return predicted_class_name, highest_probability

def create_image_message(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
    base64_encoded_data = base64.b64encode(binary_data)
    base64_string = base64_encoded_data.decode('utf-8')
    mime_type, _ = mimetypes.guess_type(image_path)
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_string
        }
    }
    return image_block

def generate_insight(image_path, predicted_label, highest_probability):
    image_block = create_image_message(image_path)
    
    SYSTEM_PROMPT = """
    <role>
    You are a highly experienced ophthalmologist specializing in diabetic retinopathy (DR). Your primary role is to assist patients in understanding their retinal images and provide guidance on next steps. You have:

    Extensive clinical experience in diagnosing and managing diabetic retinopathy.
    In-depth knowledge of retinal pathology and the stages of diabetic retinopathy.
    Expertise in interpreting fundus photographs and identifying retinal abnormalities.
    A strong commitment to patient-centered care and health education.
    The ability to explain medical concepts in simple, clear terms that patients can easily understand.

    As a doctor in this role, you approach each image analysis with the goal of empowering patients with knowledge about their eye health. You base your observations and conclusions strictly on the visual evidence presented in the retinal image and the provided classification result.

    Your primary objectives are to:
    1. Provide a clear, non-technical assessment of whether diabetic retinopathy may be present, considering both the image and the classification result.
    2. Offer easily understandable explanations of what the patient is seeing in their own retinal image.
    3. Give practical, actionable recommendations for next steps, including when to seek further medical attention.
    4. Emphasize the importance of regular eye check-ups and good diabetes management in preventing vision problems.

    While you use your medical expertise to analyze the images, you always communicate with patients in a warm, reassuring manner, using language they can easily comprehend. You avoid overly technical terms and focus on information that is most relevant and actionable for the patient.

    Remember, your role is to support patients in their self-diagnosis journey and guide them towards appropriate care, not to replace comprehensive medical examinations or provide definitive diagnoses based solely on these images and classification results.
    </role>
    """

    INSTRUCTION_PROMPT = f"""
    <image_data>
    A retinal image from the Diabetic Retinopathy Arranged dataset has been provided in base64 format above. This image represents a fundus photograph of a patient's retina, which they have taken or obtained for self-assessment purposes.

    The image has been analyzed by an AI model with the following results:
    Predicted Label: {predicted_label}
    Confidence: {highest_probability:.2f}
    </image_data>

    <instructions>
    Analyze the provided retinal image and AI classification results as if you were assisting a patient in understanding their own eye health, particularly in relation to Diabetic Retinopathy (DR). Follow these steps in your analysis:

    1. Image Quality Assessment: Comment on whether the image is clear enough for a reliable assessment. If not, provide advice on obtaining a better image.
    2. Layman's Observations: Describe what the patient would see in their own retinal image, using simple, non-technical language.
    3. AI Result Interpretation: Explain the AI classification result in simple terms, and how it relates to what's visible in the image.
    4. Potential DR Indicators: Explain any signs that might suggest the presence of diabetic retinopathy, in terms a patient can understand.
    5. Risk Level: Provide a general assessment of the likelihood of diabetic retinopathy being present, expressed in simple terms (e.g., low, moderate, high concern), considering both the image and the AI result.
    6. Explanation for Patient: Offer a clear, reassuring explanation of what these findings might mean for the patient's eye health.
    7. Immediate Recommendations: Suggest immediate steps the patient should take based on what you see in the image and the AI result.
    8. Long-term Advice: Provide general advice on maintaining eye health, especially for those with diabetes or at risk of diabetic retinopathy.
    9. Reassurance: Include a reassuring message about the manageability of eye health with proper care and regular check-ups.

    Provide your analysis in a structured JSON format. Use language that is easily understandable to patients without medical background. Avoid technical jargon and focus on practical, actionable information.
    </instructions>

    <output_format>
    Structure your response as a JSON object with the following keys:

    {{
    "image_quality": "Clear/Unclear/Partially Clear",
    "image_quality_advice": "Advice on image quality if needed",
    "what_patient_sees": "Description of visible features in layman's terms",
    "ai_result_interpretation": "Explanation of the AI classification result",
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
    }}

    Ensure that your response is a valid JSON object that can be parsed by standard JSON parsers.
    </output_format>
    """

    message_list = [
        {
            "role": "user",
            "content": [
                image_block,
                {
                    "type": "text",
                    "text": INSTRUCTION_PROMPT
                }
            ],
        }
    ]

    response = client.messages.create(
        model=MODEL_TYPE,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=message_list,
        temperature=0.05,
    )

    return json.loads(response.content[0].text)