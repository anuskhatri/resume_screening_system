import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.training.data_preprocessing import DataPreprocessor
from src.inference.resume_ranking import ResumeRanker
import pandas as pd

def main():
    print("="*80)
    print("INTELLIGENT RESUME SCREENING SYSTEM")
    print("="*80)
    
    # Load processed data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.load_processed_data()
    
    if df_processed is None:
        print("Error: No processed data found. Please run train.py first.")
        return
    
    print(f"\nLoaded {len(df_processed)} resumes")
    
    # Initialize ranker
    ranker = ResumeRanker()
    
    # Example job description
    job_description = """
    We are looking for a Python Developer with experience in Machine Learning and NLP.
    The candidate should have strong skills in Python, TensorFlow, PyTorch, and scikit-learn.
    Experience with web frameworks like Flask or Django is a plus.
    Minimum 2 years of experience required.
    Bachelor's degree in Computer Science or related field.
    """
    
    required_experience = 2
    required_education = "Bachelor's degree in Computer Science"
    
    # Rank candidates
    scores_df = ranker.rank_candidates(
        df_processed,
        job_description,
        required_experience,
        required_education
    )
    
    # Display top candidates
    ranker.display_top_candidates(scores_df, top_n=10)
    
    # Save results
    output_path = os.path.join(Config.PROCESSED_DATA_DIR, 'ranked_candidates.csv')
    scores_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total candidates: {len(scores_df)}")
    print(f"Shortlisted candidates: {scores_df['shortlisted'].sum()}")
    print(f"Average score: {scores_df['final_score'].mean():.4f}")
    print(f"Top score: {scores_df['final_score'].max():.4f}")
    print(f"Lowest score: {scores_df['final_score'].min():.4f}")

if __name__ == "__main__":
    main()
