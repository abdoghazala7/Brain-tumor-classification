import torch

MODEL_NAME = 'efficientnet_b0' 
MODEL_PATH = 'efficientnet_finetuned_final.pth'
NUM_CLASSES = 4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary'] 

# --- Preprocessing (Must match validation transforms) ---
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]