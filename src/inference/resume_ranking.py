import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from src.training.feature_extraction import FeatureExtractor
from src.inference.utils import TextCleaner
from src.inference.jd_parser import JobDescriptionParser
from src.config import Config


class ResumeRanker:
    """
    Intelligent Candidate Screening and Ranking Engine.
    Combines Sentence-BERT semantic embeddings, ontology skill-gap analysis,
    experience alignment, and education matching.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.cleaner = TextCleaner()
        self.jd_parser = JobDescriptionParser()

    def calculate_keyword_match_score(self, resume_text: str, job_description: str) -> float:
        """Calculate token keyword overlap score."""
        if not resume_text or not job_description:
            return 0.0

        resume_keywords = set(resume_text.lower().split())
        job_keywords = set(job_description.lower().split())

        if len(job_keywords) == 0:
            return 0.0

        common_keywords = resume_keywords.intersection(job_keywords)
        score = len(common_keywords) / len(job_keywords)

        return min(float(score), 1.0)

    def calculate_experience_score(self, resume_experience: float, required_experience: float) -> float:
        """Calculate experience matching score with reasonable saturation."""
        if required_experience <= 0:
            return 1.0

        if resume_experience >= required_experience:
            # Full score if meets requirement; slight bonus capped at 1.0
            return 1.0
        else:
            return float(resume_experience / required_experience)

    def calculate_education_score(self, resume_text: str, required_education: str) -> float:
        """Calculate education matching score."""
        if not required_education:
            return 1.0

        resume_lower = resume_text.lower()
        required_lower = required_education.lower()

        degree_hierarchy = {
            "phd": 4,
            "doctorate": 4,
            "master": 3,
            "m.s": 3,
            "mba": 3,
            "m.tech": 3,
            "bachelor": 2,
            "b.s": 2,
            "b.tech": 2,
            "undergraduate": 2,
            "diploma": 1,
            "associate": 1
        }

        req_level = 2  # default bachelor
        for deg, lvl in degree_hierarchy.items():
            if deg in required_lower:
                req_level = max(req_level, lvl)

        cand_level = 0
        for deg, lvl in degree_hierarchy.items():
            if deg in resume_lower:
                cand_level = max(cand_level, lvl)

        if cand_level >= req_level:
            return 1.0
        elif cand_level > 0:
            return float(cand_level / req_level)
        return 0.5  # Neutral default if degree text unspecified

    def rank_candidates(
        self,
        resumes_df: pd.DataFrame,
        job_description: str,
        required_experience: int = 0,
        required_education: str = "",
        required_skills: Optional[List[str]] = None,
        anonymize: bool = False,
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Rank candidate resumes against the job description using multi-factor NLP.
        """
        print(f"\n[ResumeRanker] Ranking {len(resumes_df)} candidates...")

        # If requirements were not explicitly provided, parse them using NLP
        parsed_jd = None
        if not required_skills or required_experience == 0 or not required_education:
            parsed_jd = self.jd_parser.parse_jd(job_description)
            if not required_skills:
                required_skills = parsed_jd["all_skills"]
            if required_experience == 0:
                required_experience = parsed_jd["required_experience"]
            if not required_education:
                required_education = parsed_jd["required_education"]

        # Clean and vectorize job description
        cleaned_job_desc = self.cleaner.clean_text(job_description)
        processed_job_desc = self.cleaner.tokenize_and_lemmatize(cleaned_job_desc)

        # Generate embeddings for job description
        job_embedding = self.feature_extractor.encode_texts([processed_job_desc])

        # Generate embeddings for resumes
        if 'processed_resume' in resumes_df.columns:
            resume_texts = resumes_df['processed_resume'].fillna('').tolist()
        else:
            resume_texts = [
                self.cleaner.tokenize_and_lemmatize(self.cleaner.clean_text(str(r)))
                for r in resumes_df['Resume'].fillna('').tolist()
            ]

        resume_embeddings = self.feature_extractor.encode_texts(resume_texts)

        # Calculate semantic cosine similarity
        semantic_scores = self.feature_extractor.calculate_cosine_similarity(
            resume_embeddings, job_embedding
        ).flatten()

        scores_data = []
        weights = scoring_weights or {
            "semantic": Config.SEMANTIC_WEIGHT,
            "skill": Config.SKILL_WEIGHT,
            "experience": Config.EXPERIENCE_WEIGHT,
            "education": Config.EDUCATION_WEIGHT,
        }
        weight_total = sum(max(0.0, float(value)) for value in weights.values()) or 1.0
        weights = {key: max(0.0, float(value)) / weight_total for key, value in weights.items()}

        for position, (idx, row) in enumerate(resumes_df.iterrows()):
            semantic_score = float(semantic_scores[position])

            resume_text = str(row.get('Resume', ''))

            # Candidate skills
            cand_skills = row.get('extracted_skills')
            if not cand_skills or not isinstance(cand_skills, list):
                cand_skills = self.cleaner.extract_skills(resume_text)

            # Skill Gap Analysis
            gap_analysis = self.cleaner.calculate_skill_gap(cand_skills, required_skills or [])
            skill_score = float(gap_analysis["match_percentage"] / 100.0)

            # Experience score
            cand_exp = row.get('experience_years', 0)
            if cand_exp is None or pd.isna(cand_exp):
                cand_exp = self.cleaner.extract_experience_years(resume_text)
            experience_score = self.calculate_experience_score(float(cand_exp), float(required_experience))

            # Education score
            education_score = self.calculate_education_score(resume_text, required_education)

            # Keyword score
            keyword_score = self.calculate_keyword_match_score(resume_text, job_description)

            # Normalize weights so the score remains comparable when recruiters
            # adjust the relative importance of the four signals.
            final_score = (
                weights["semantic"] * semantic_score +
                weights["skill"] * skill_score +
                weights["experience"] * experience_score +
                weights["education"] * education_score
            )
            final_score = max(0.0, min(1.0, final_score))

            # Fit tier
            if final_score >= 0.70:
                match_tier = "Strong Fit"
            elif final_score >= 0.50:
                match_tier = "Moderate Fit"
            else:
                match_tier = "Low Fit"

            # Extractive highlights
            highlights = self.cleaner.extract_summary_highlights(resume_text, max_sentences=3)
            resume_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", resume_text) if s.strip()]
            experience_evidence = next((s[:260] for s in resume_sentences if re.search(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\b", s, re.I)), "")
            education_evidence = next((s[:260] for s in resume_sentences if re.search(r"\b(?:ph\.?d|doctorate|master'?s?|m\.?s\.?|m\.?tech|mba|bachelor'?s?|b\.?s\.?|b\.?tech|diploma|associate)\b", s, re.I)), "")

            # Filename / candidate title
            cand_name = row.get('filename', f"Candidate_{idx}")
            if anonymize:
                cand_name = f"Anonymous_Candidate_{position + 1}"

            scores_data.append({
                'resume_id': idx,
                'filename': cand_name,
                'category': row.get('Category', 'Applicant'),
                'final_score': round(final_score, 4),
                'semantic_score': round(semantic_score, 4),
                'skill_score': round(skill_score, 4),
                'keyword_score': round(keyword_score, 4),
                'experience_score': round(experience_score, 4),
                'education_score': round(education_score, 4),
                'skills': cand_skills,
                'matched_skills': gap_analysis["matched_skills"],
                'missing_skills': gap_analysis["missing_skills"],
                'bonus_skills': gap_analysis["bonus_skills"],
                'skill_match_pct': gap_analysis["match_percentage"],
                'experience_years': cand_exp,
                'match_tier': match_tier,
                'highlights': highlights,
                'skill_evidence': self.cleaner.find_skill_evidence(resume_text, gap_analysis["matched_skills"]),
                'nlp_edge_cases': self.cleaner.analyze_skill_edge_cases(resume_text),
                'requirements': {
                    'skills': {
                        'required': len(gap_analysis["matched_skills"]) + len(gap_analysis["missing_skills"]),
                        'matched': len(gap_analysis["matched_skills"]),
                        'meets': not gap_analysis["missing_skills"],
                    },
                    'experience': {
                        'required': float(required_experience),
                        'candidate': float(cand_exp),
                        'meets': float(cand_exp) >= float(required_experience),
                        'evidence': experience_evidence
                    },
                    'education': {
                        'required': required_education or "Not specified",
                        'candidate': self.cleaner.extract_education(resume_text),
                        'meets': education_score >= 1.0,
                        'evidence': education_evidence
                    }
                },
                'candidate_education': self.cleaner.extract_education(resume_text),
                'resume_preview': resume_text[:600]
            })

        scores_df = pd.DataFrame(scores_data)
        scores_df = scores_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        scores_df['rank'] = scores_df.index + 1

        threshold = getattr(Config, 'SHORTLIST_THRESHOLD', 0.60)
        scores_df['shortlisted'] = scores_df['final_score'] >= threshold

        print(f"[ResumeRanker] Ranking complete! {scores_df['shortlisted'].sum()} shortlisted.")
        return scores_df

    def display_top_candidates(self, scores_df: pd.DataFrame, top_n: int = 10):
        """Display top N candidates in terminal."""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} CANDIDATES")
        print(f"{'='*80}\n")

        for _, row in scores_df.head(top_n).iterrows():
            print(f"Rank #{row['rank']} | {row['filename']} ({row['match_tier']})")
            print(f"Final Score: {row['final_score'] * 100:.1f}% (Semantic: {row['semantic_score']*100:.1f}%, Skill: {row['skill_score']*100:.1f}%)")
            print(f"Matched Skills: {row['matched_skills']}")
            print(f"Missing Skills: {row['missing_skills']}")
            print(f"Experience: {row['experience_years']} yrs | Shortlisted: {'YES' if row['shortlisted'] else 'NO'}")
            print(f"{'-'*80}\n")
