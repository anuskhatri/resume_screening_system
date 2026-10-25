import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.training.data_preprocessing import DataPreprocessor
from src.training.feature_extraction import FeatureExtractor
from src.training.model_training import ModelTrainer
import torch

def main():
    print("="*80)
    print("INTELLIGENT RESUME SCREENING SYSTEM - TRAINING")
    print("="*80)
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Step 1: Load and preprocess data
    preprocessor = DataPreprocessor()
    
    # Check if processed data exists
    df_processed = preprocessor.load_processed_data()
    
    if df_processed is None:
        print("\nProcessed data not found. Processing raw data...")
        df = preprocessor.load_dataset(Config.RESUME_DATASET)
        df_processed = preprocessor.preprocess_resumes(df)
        preprocessor.save_processed_data(df_processed)
    else:
        print("Loaded existing processed data")
    
    # Step 2: Extract features using Sentence-BERT on GPU
    feature_extractor = FeatureExtractor()
    
    # Check if embeddings exist
    embeddings = feature_extractor.load_embeddings('resume_embeddings.npy')
    
    if embeddings is None:
        print("\nGenerating embeddings on GPU...")
        resume_texts = df_processed['processed_resume'].tolist()
        embeddings = feature_extractor.encode_texts(resume_texts)
        feature_extractor.save_embeddings(embeddings, 'resume_embeddings.npy')
    else:
        print("Loaded existing embeddings")
    
    # Step 3: Train classification model on GPU
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*80)
    
    labels = df_processed['Category'].tolist()
    trainer = ModelTrainer(embeddings, labels)
    
    X_train, X_test, y_train, y_test, num_classes = trainer.prepare_data()
    trainer.train_model(X_train, y_train, X_test, y_test, num_classes)
    trainer.save_model()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
