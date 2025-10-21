import torch
import timm
import config

def load_model(model_path=config.MODEL_PATH, 
               model_name=config.MODEL_NAME, 
               num_classes=config.NUM_CLASSES, 
               device=config.DEVICE):
    """
    Loads the pre-trained model architecture and weights.
    
    Args:
        model_path (str): Path to the saved model state dictionary (.pth file).
        model_name (str): Name of the model architecture (e.g., 'efficientnet_b0').
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Use map_location to ensure it loads correctly even if trained on GPU and run on CPU
    state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    
    model.eval()
    
    model.to(device)
    return model

if __name__ == '__main__':
    # Simple test to ensure model loading works
    try:
        model = load_model()
        print("Model loading test successful.")
    except Exception as e:
        print(f"Error loading model: {e}")