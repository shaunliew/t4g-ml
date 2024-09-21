import json
from helpers import *

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

# Generate insight using Claude AI
insight = generate_insight(image_path, predicted_label, highest_probability)

# Print the generated insight
print(json.dumps(insight, indent=2))