from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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
    state_dict = torch.load(model_path, map_location=device)
    
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

# Define the label mapping
LABEL_MAPPING = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Load the model
model_path = 'weights/model.pth'
model, device = load_trained_model(model_path)

print("Model loaded successfully and moved to", device)

# Path to your local image
image_path = 'raw_data/0/84_left.jpeg'

# Make the prediction
predicted_label, highest_probability = predict_single_image(model, image_path, device)

# Print the results
print(f"Predicted label: {predicted_label}")
print(f"Highest probability: {highest_probability:.4f}")