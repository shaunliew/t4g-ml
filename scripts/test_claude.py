from config import *
import base64
import mimetypes

def create_image_message(image_path):
    # Open the image file in binary read mode
    with open(image_path, "rb") as image_file:
        # Read the binary data of the image
        binary_data = image_file.read()
    
    # Encode the binary data to base64
    base64_encoded_data = base64.b64encode(binary_data)
    
    # Convert the base64 bytes to a UTF-8 string
    base64_string = base64_encoded_data.decode('utf-8')
    
    # Guess the MIME type of the image based on its file extension
    mime_type, _ = mimetypes.guess_type(image_path)
    
    # Create the image block dictionary
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_string
        }
    }
    
    return image_block

image_path = "raw_data/0/84_left.jpeg"
image_block = create_image_message(image_path)

message_list = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": TEST_INSTRUCTION_PROMPT
                    }
                ],
            }
        ]

response = client.messages.create(
        model=MODEL_TYPE,
        max_tokens=1024,
        system = TEST_SYSTEM_PROMPT,
        messages=message_list,
        temperature=0.05,
    )

print(response.content[0].text)