import torch
import os

class Config:
    # GPU Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Configuration
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Fast and accurate
    # Alternative: 'sentence-transformers/paraphrase-distilroberta-base-v1'
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    
    # Dataset
    RESUME_DATASET = os.path.join(RAW_DATA_DIR, 'UpdatedResumeDataSet.csv')
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    MAX_LENGTH = 512
    
    # Scoring Weights
    SEMANTIC_WEIGHT = 0.45
    SKILL_WEIGHT = 0.30
    EXPERIENCE_WEIGHT = 0.15
    EDUCATION_WEIGHT = 0.10
    KEYWORD_WEIGHT = 0.00
    
    # Threshold
    SHORTLIST_THRESHOLD = 0.60
    
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
