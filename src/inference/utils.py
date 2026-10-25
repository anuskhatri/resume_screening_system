import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from typing import List, Dict, Any, Tuple

from src.inference.skill_ontology import ALL_SKILLS, SKILL_ALIASES, normalize_skill, get_skill_category

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove special characters and digits, keep spaces and clean hyphens
        text = re.sub(r'[^a-zA-Z\s\+\#]', ' ', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text for NLP vectorization."""
        if not text:
            return ""
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(tokens)

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract technical & soft skills using comprehensive ontology matching.
        Matches multi-word phrases and single tokens with word-boundary awareness.
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        extracted = set()

        skill_patterns = set(ALL_SKILLS) | set(SKILL_ALIASES)
        for skill in skill_patterns:
            # Match phrases and punctuation variants without matching substrings
            # (for example, "go" inside "good").
            pattern = rf"(?<!\w){re.escape(skill).replace(r'\ ', r'\s+')}(?!\w)"
            occurrences = list(re.finditer(pattern, text_lower))
            for occurrence in occurrences:
                left = max(text_lower.rfind(mark, 0, occurrence.start()) for mark in ".!?\n") + 1
                context = text_lower[left:occurrence.start()].strip()
                negated = re.search(
                    r"\b(?:no|not|never|without|lack(?:s|ing)?|unfamiliar with|"
                    r"not familiar with|not experienced in|not proficient in|"
                    r"limited exposure to|no experience (?:with|of)|no knowledge of|"
                    r"no background in|lack of experience with|never worked with)(?:\s+\w+){0,5}\s*$",
                    context,
                )
                if not negated:
                    extracted.add(normalize_skill(skill))
                    break

        return sorted(list(extracted))

    def find_skill_evidence(self, text: str, skills: List[str], limit: int = 2) -> Dict[str, List[str]]:
        """Return resume sentences supporting each detected skill."""
        if not text:
            return {}
        sentences = [s.strip(" \t•-*\r") for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        evidence = {}
        for skill in skills:
            canonical = normalize_skill(skill)
            hits = []
            variants = {skill, canonical}
            variants.update(alias for alias, target in SKILL_ALIASES.items() if normalize_skill(target) == canonical)
            pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in sorted(variants, key=len, reverse=True)) + r")(?!\w)", re.I)
            for sentence in sentences:
                match = pattern.search(sentence)
                if not match:
                    continue
                prefix = sentence[:match.start()].lower()
                if re.search(r"\b(?:no|not|never|without|lack(?:s|ing)?|unfamiliar with|not familiar with|not experienced in|not proficient in|limited exposure to|no experience (?:with|of)|no knowledge of|no background in|lack of experience with|never worked with)(?:\s+\w+){0,5}\s*$", prefix):
                    continue
                hits.append(sentence[:260])
                if len(hits) >= limit:
                    break
            if hits:
                evidence[canonical] = hits
        return evidence

    def analyze_skill_edge_cases(self, text: str) -> Dict[str, List[str]]:
        """Capture reviewable NLP edge cases for each resume and screening run."""
        if not text:
            return {"recognized_aliases": [], "negated_mentions": [], "project_section_skills": []}
        lowered = text.lower()
        aliases = sorted({alias for alias in SKILL_ALIASES if re.search(rf"(?<!\w){re.escape(alias).replace(r'\ ', r'\s+')}(?!\w)", lowered)})
        negated = set()
        negation = re.compile(r"\b(?:no|not|never|without|lack(?:s|ing)?|unfamiliar with|not familiar with|not experienced in|not proficient in|limited exposure to|no experience (?:with|of)|no knowledge of|no background in|lack of experience with|never worked with)(?:\s+\w+){0,5}\s*$", re.I)
        for skill in set(ALL_SKILLS) | set(SKILL_ALIASES):
            pattern = rf"(?<!\w){re.escape(skill).replace(r'\ ', r'\s+')}(?!\w)"
            for match in re.finditer(pattern, lowered):
                start = max(lowered.rfind(mark, 0, match.start()) for mark in ".!?\n") + 1
                if negation.search(lowered[start:match.start()].strip()):
                    negated.add(normalize_skill(skill))
                    break
        project_match = re.search(r"(?ims)^\s*(?:personal\s+)?projects?\b(.*?)(?=^\s*(?:experience|employment|education|skills|certifications|summary)\b|\Z)", text)
        project_skills = self.extract_skills(project_match.group(1)) if project_match else []
        return {
            "recognized_aliases": sorted(set(normalize_skill(alias) for alias in aliases)),
            "negated_mentions": sorted(negated),
            "project_section_skills": project_skills,
        }

    def extract_experience_years(self, text: str) -> int:
        """
        Context-aware experience extraction.
        Extracts candidate's total years of experience, avoiding false positives like
        graduation years (e.g. 2021) or phone numbers.
        """
        if not text or not isinstance(text, str):
            return 0

        text_lower = text.lower()

        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|hands-on|industry|work|professional))?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(?:total\s+experience|work\s+experience)\s*[:\-]?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+in\s+[a-z]+'
        ]

        detected = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                if isinstance(m, str) and m.isdigit():
                    val = int(m)
                    if 1 <= val <= 35:
                        detected.append(val)

        if detected:
            return max(detected)

        # Fallback regex with tighter bounds
        matches = re.findall(r'\b(\d{1,2})\+?\s*(?:years?|yrs?)\b', text_lower)
        valid = [int(y) for y in matches if 1 <= int(y) <= 30]
        return max(valid) if valid else 0

    def extract_education(self, text: str) -> str:
        """Extract candidate education level and discipline."""
        if not text:
            return "Not Specified"

        text_lower = text.lower()
        if re.search(r'\b(ph\.?d|doctorate)\b', text_lower):
            return "PhD"
        elif re.search(r'\b(master|m\.?s|m\.?tech|mba|m\.?sc)\b', text_lower):
            return "Master's Degree"
        elif re.search(r'\b(bachelor|b\.?s|b\.?tech|b\.?e|b\.?sc|undergraduate)\b', text_lower):
            return "Bachelor's Degree"
        elif re.search(r'\b(diploma|associate)\b', text_lower):
            return "Diploma / Associate"

        return "Not Specified"

    def calculate_skill_gap(self, candidate_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
        """
        Compute matched, missing, and bonus skills along with match percentage.
        """
        cand_set = set(normalize_skill(s) for s in candidate_skills)
        req_set = set(normalize_skill(s) for s in required_skills)

        matched = sorted(list(cand_set.intersection(req_set)))
        missing = sorted(list(req_set - cand_set))
        bonus = sorted(list(cand_set - req_set))

        if req_set:
            match_pct = round((len(matched) / len(req_set)) * 100, 1)
        else:
            match_pct = 100.0 if cand_set else 0.0

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "bonus_skills": bonus[:10],
            "match_percentage": match_pct,
            "total_required": len(req_set),
            "total_matched": len(matched)
        }

    def extract_summary_highlights(self, text: str, max_sentences: int = 3) -> List[str]:
        """
        Extract the top 2-3 most impactful highlight sentences from the resume
        using sentence tokenization and action-oriented scoring.
        """
        if not text or not isinstance(text, str):
            return []

        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 20]

        impact_keywords = [
            "developed", "led", "architected", "built", "designed", "optimized",
            "managed", "implemented", "achieved", "engineered", "increased",
            "reduced", "deployed", "spearheaded", "experience", "specialized", "created"
        ]

        scored_sentences = []
        for s in sentences:
            clean_s = s.strip().replace("\n", " ")
            word_count = len(clean_s.split())
            if 8 <= word_count <= 40:
                s_lower = clean_s.lower()
                # Score based on presence of action verbs and skills
                score = sum(1 for kw in impact_keywords if kw in s_lower)
                # Boost if sentence starts with bullet or capital action
                if any(clean_s.startswith(w) for w in ["•", "-", "*"]):
                    clean_s = clean_s.lstrip("•-* ").strip()
                if score > 0:
                    scored_sentences.append((score, clean_s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        highlights = [s[1] for s in scored_sentences[:max_sentences]]

        if not highlights and sentences:
            highlights = [s.strip() for s in sentences if 10 <= len(s.split()) <= 35][:2]

        return highlights

    def anonymize_text(self, text: str) -> str:
        """
        Redact Personally Identifiable Information (PII) for blind screening.
        Masks Names, Emails, Phone numbers, and addresses.
        """
        if not text:
            return ""

        # Redact email
        anonymized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL REDACTED]', text)
        # Redact phone
        anonymized = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE REDACTED]', anonymized)
        # Redact URLs/LinkedIn
        anonymized = re.sub(r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/\S+', '[LINKEDIN REDACTED]', anonymized)
        anonymized = re.sub(r'(?:https?:\/\/)?(?:www\.)?github\.com\/\S+', '[GITHUB REDACTED]', anonymized)

        # Use spaCy NER to redact PERSON names in the first 500 characters
        header_text = anonymized[:500]
        try:
            doc = nlp(header_text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text) > 2:
                    header_text = header_text.replace(ent.text, "[CANDIDATE NAME]")
            anonymized = header_text + anonymized[500:]
        except Exception:
            pass

        return anonymized
