import torch
from torchvision import transforms
from PIL import Image
import io
import config

def preprocess_image(image_bytes):
    """
    Preprocesses the input image bytes to match model requirements.
    
    Args:
        image_bytes (bytes): The image file bytes (e.g., from Streamlit uploader).

    Returns:
        torch.Tensor: The preprocessed image tensor ready for the model.
    """
 
    image = Image.open(io.BytesIO(image_bytes))
            
    image = image.convert("RGB")
            
    # Define the same transformations used during validation/testing
    transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
            ])
            
    image_tensor = transform(image)
            
    # Add batch dimension (model expects batch_size, channels, height, width)
    image_tensor = image_tensor.unsqueeze(0)
            
    return image_tensor        


if __name__ == '__main__':
    # Simple test with a dummy black image
    try:
        dummy_image = Image.new('RGB', (600, 400), color = 'black')
        buf = io.BytesIO()
        dummy_image.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        
        tensor = preprocess_image(image_bytes)
        if tensor is not None:
            print(f"Image preprocessing test successful. Output tensor shape: {tensor.shape}")
        else:
            print("Image preprocessing test failed.")
            
    except Exception as e:
        print(f"Error in preprocessing test: {e}")