import traceback
from fastapi import FastAPI, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from helpers import load_trained_model, predict_single_image, generate_insight
from config import *
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model at startup
model_path = 'weights/model.pth'
model, device = load_trained_model(model_path)
print(f"Model loaded successfully and moved to {device}")

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "success": True,
            "message": "Diabetic Retinopathy Analysis API is running",
            "content": {
                "api_version": "1.0",
                "model_type": "EfficientNet-B3",
                "device": str(device)
            }
        },
        status_code=status.HTTP_200_OK
    )
    
@app.post("/analyze_retina")
async def analyze_retina(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Save the image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)
        
        # Make the prediction
        predicted_label, highest_probability = predict_single_image(model, temp_image_path, device)
        
        # Generate insight using Claude AI
        insight = generate_insight(temp_image_path, predicted_label, highest_probability)
        
        # Prepare the content
        content = {
            "predicted_label": predicted_label,
            "highest_probability": float(highest_probability),
            "insight": insight
        }
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Image {file.filename} processed successfully.",
                "content": content
            },
            status_code=status.HTTP_200_OK
        )
    
    except Exception as e:
        # Get the full traceback
        error_traceback = traceback.format_exc()
        
        # Prepare a detailed error message
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_traceback
        }
        
        # Log the error (you might want to use a proper logging system in production)
        print(f"Error processing image: {error_details}")
        
        return JSONResponse(
            content={
                "success": False,
                "message": "Error processing image. Please check server logs for details.",
                "error_details": error_details
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@app.get("/check_claude_connection")
async def check_claude_connection():
    try:
        # Attempt to make a simple request to the Claude API
        response = client.messages.create(
            model=os.getenv("MODEL_TYPE"),
            max_tokens=10,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful' if you receive this message."
                }
            ]
        )

        if response.content[0].text.strip().lower() == "connection successful":
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Successfully connected to Claude API",
                    "content": {
                        "api_response": response.content[0].text
                    }
                },
                status_code=status.HTTP_200_OK
            )
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Connected to Claude API but received unexpected response",
                    "content": {
                        "api_response": response.content[0].text
                    }
                },
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": "Failed to connect to Claude API",
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )