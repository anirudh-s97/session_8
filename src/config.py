import torch 
import warnings
warnings.filterwarnings("ignore")

class Config:
    # Dataset parameters
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 55
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    NUM_CLASSES = 10
    
    # Paths
    DATA_DIR = "data/"
    MODEL_SAVE_DIR = "checkpoints/"
