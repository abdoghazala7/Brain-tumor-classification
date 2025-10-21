import torch
import torch.nn.functional as F
import config

def predict(model, image_tensor, device=config.DEVICE):
    """
    Performs inference on the preprocessed image tensor using the loaded model.

    Args:
        model (torch.nn.Module): The loaded PyTorch model.
        image_tensor (torch.Tensor): The preprocessed image tensor.
        device (torch.device): Device the model and tensor are on.

    Returns:
        tuple: A tuple containing:
            - predicted_class (str): The name of the predicted class.
            - probabilities (dict): A dictionary mapping class names to probabilities.
    """
    model.eval() 
    image_tensor = image_tensor.to(device) 
    
    with torch.no_grad(): 
        outputs = model(image_tensor)
        
        probabilities_tensor = F.softmax(outputs, dim=1)
        
        _, predicted_idx = torch.max(probabilities_tensor, 1)
        
        predicted_class_index = predicted_idx.item()
        predicted_class_name = config.CLASS_NAMES[predicted_class_index]
        
        probabilities_list = probabilities_tensor.squeeze().cpu().tolist()
        probabilities_dict = {name: prob for name, prob in zip(config.CLASS_NAMES, probabilities_list)}
        
        return predicted_class_name, probabilities_dict

if __name__ == '__main__':
    # Simple test (requires a dummy model and tensor)
    try:
        # Create a dummy model for testing structure
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, config.NUM_CLASSES)
            def forward(self, x):
                # Simulate output logits based on input shape
                batch_size = x.shape[0]
                return self.linear(torch.randn(batch_size, 1))

        dummy_model = DummyModel().to(config.DEVICE)
        dummy_tensor = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE) # Batch size 1

        pred_class, probs = predict(dummy_model, dummy_tensor)
        
        print("Prediction test successful.")
        print(f"Predicted Class: {pred_class}")
        print("Probabilities:")
        for name, prob in probs.items():
            print(f"- {name}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error in prediction test: {e}")