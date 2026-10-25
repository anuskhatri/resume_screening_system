import pandas as pd
import numpy as np
from src.inference.utils import TextCleaner
from src.config import Config
import os

class DataPreprocessor:
    def __init__(self):
        self.cleaner = TextCleaner()
    
    def load_dataset(self, filepath):
        """Load resume dataset"""
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {len(df)} resumes")
        return df
    
    def preprocess_resumes(self, df):
        """Preprocess resume data"""
        print("Preprocessing resumes...")
        
        # Make a copy
        df_processed = df.copy()
        
        # Clean resume text
        df_processed['cleaned_resume'] = df_processed['Resume'].apply(
            lambda x: self.cleaner.clean_text(str(x))
        )
        
        # Tokenize and lemmatize
        df_processed['processed_resume'] = df_processed['cleaned_resume'].apply(
            self.cleaner.tokenize_and_lemmatize
        )
        
        # Extract skills
        df_processed['extracted_skills'] = df_processed['Resume'].apply(
            self.cleaner.extract_skills
        )
        
        # Extract experience
        df_processed['experience_years'] = df_processed['Resume'].apply(
            self.cleaner.extract_experience_years
        )
        
        # Remove duplicates
        df_processed = df_processed.drop_duplicates(subset=['processed_resume'])
        
        print(f"Preprocessing complete: {len(df_processed)} resumes")
        return df_processed
    
    def save_processed_data(self, df, filename='processed_resumes.csv'):
        """Save processed data"""
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        filepath = os.path.join(Config.PROCESSED_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filename='processed_resumes.csv'):
        """Load processed data"""
        filepath = os.path.join(Config.PROCESSED_DATA_DIR, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
