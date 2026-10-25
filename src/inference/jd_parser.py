"""
Intelligent Job Description (JD) Parser using NLP (spaCy).
Automatically extracts:
- Job Title & Seniority Level
- Required Years of Experience
- Required Education & Majors
- Core vs Preferred Skills
- Key Responsibilities & Highlights
"""

import re
import spacy
from typing import Dict, List, Any
from src.inference.skill_ontology import ALL_SKILLS, normalize_skill, get_skill_category

try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = spacy.blank('en')


class JobDescriptionParser:
    """NLP-driven parser to extract structured hiring criteria from unstructured Job Descriptions."""

    def __init__(self):
        self.seniority_keywords = {
            "Lead / Principal": ["principal", "lead", "architect", "staff", "head of", "director"],
            "Senior": ["senior", "sr.", "sr ", "advanced", "expert"],
            "Mid-Level": ["mid-level", "mid level", "intermediate", "experienced"],
            "Junior / Entry": ["junior", "jr.", "jr ", "entry level", "associate", "graduate"],
            "Internship": ["intern", "internship", "trainee", "apprentice"]
        }

        self.degree_patterns = {
            "PhD": [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"],
            "Master's Degree": [r"\bmaster'?s?\b", r"\bm\.?s\.?\b", r"\bm\.?tech\b", r"\bmba\b", r"\bm\.?sc\b"],
            "Bachelor's Degree": [r"\bbachelor'?s?\b", r"\bb\.?s\.?\b", r"\bb\.?tech\b", r"\bb\.?e\.?\b", r"\bb\.?sc\b", r"\bundergraduate\b"],
            "Associate / Diploma": [r"\bdiploma\b", r"\bassociate'?s?\b"]
        }

        self.major_patterns = [
            "computer science", "information technology", "data science", "software engineering",
            "electrical engineering", "mathematics", "statistics", "artificial intelligence",
            "computational biology", "related quantitative field", "related field", "engineering"
        ]

    def parse_jd(self, jd_text: str) -> Dict[str, Any]:
        """Main method to parse raw or extracted JD text into structured attributes."""
        if not jd_text or not isinstance(jd_text, str):
            return {
                "job_title": "Software Professional",
                "seniority": "Mid-Level",
                "required_experience": 0,
                "required_education": "Bachelor's degree in Computer Science or related field",
                "core_skills": [],
                "preferred_skills": [],
                "all_skills": [],
                "skills_by_category": {},
                "responsibilities": [],
                "raw_text": ""
            }

        cleaned_text = jd_text.strip()
        doc = nlp(cleaned_text[:5000])

        title = self._extract_job_title(cleaned_text, doc)
        seniority = self._extract_seniority(cleaned_text)
        experience = self._extract_experience(cleaned_text)
        education = self._extract_education(cleaned_text)
        core_skills, preferred_skills, all_skills = self._extract_skills_from_jd(cleaned_text)
        responsibilities = self._extract_responsibilities(cleaned_text)

        # Categorize skills for nice UI visualization
        skills_by_cat = {}
        for s in all_skills:
            cat = get_skill_category(s)
            skills_by_cat.setdefault(cat, []).append(s)

        return {
            "job_title": title,
            "seniority": seniority,
            "required_experience": experience,
            "required_education": education,
            "core_skills": core_skills,
            "preferred_skills": preferred_skills,
            "all_skills": all_skills,
            "skills_by_category": skills_by_cat,
            "responsibilities": responsibilities[:6],
            "raw_text": cleaned_text
        }

    def _extract_job_title(self, text: str, doc) -> str:
        """Extract job title from the beginning of JD or via keyword matching."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            first_line = lines[0]
            # If the first line is concise and looks like a title
            if len(first_line.split()) <= 7 and not any(kw in first_line.lower() for kw in ["about", "company", "description", "welcome"]):
                # Clean punctuation
                return re.sub(r'[:\-–|].*$', '', first_line).strip().title()

        # Look for "We are looking for a / an [Title]"
        pattern = r"(?:looking for|seeking|hiring|role of|position of|as an?)\s+([A-Za-z0-9\s\-/]+?)(?:\.|\n|who|with|to|\,)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate_title = match.group(1).strip()
            if 2 <= len(candidate_title.split()) <= 6:
                return candidate_title.title()

        # Default fallback
        common_roles = [
            "Data Scientist", "Machine Learning Engineer", "Python Developer",
            "Full Stack Developer", "Backend Developer", "Frontend Developer",
            "DevOps Engineer", "Cloud Engineer", "Software Engineer", "AI Engineer"
        ]
        text_lower = text.lower()
        for role in common_roles:
            if role.lower() in text_lower:
                return role

        return "Software Professional"

    def _extract_seniority(self, text: str) -> str:
        """Extract seniority level from JD."""
        text_lower = text.lower()
        for level, keywords in self.seniority_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    return level
        return "Mid-Level"

    def _extract_experience(self, text: str) -> int:
        """Extract minimum years of experience using contextual regex and bounds."""
        text_lower = text.lower()

        # Patterns like: "3-5 years", "minimum 3 years", "at least 4 yrs", "5+ years of experience"
        patterns = [
            r'(?:min(?:imum)?|at least|\+)?\s*(\d+)\s*(?:-|to)\s*(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|hands-on|industry|work))?\s*experience',
            r'(?:min(?:imum)?|at least)\s*(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|hands-on|industry|work))?\s*experience',
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|hands-on|industry|work))?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+in\s+[a-z]+'
        ]

        detected_years = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                if isinstance(m, tuple):
                    for val in m:
                        if val.isdigit() and 1 <= int(val) <= 25:
                            detected_years.append(int(val))
                elif isinstance(m, str) and m.isdigit():
                    val = int(m)
                    if 1 <= val <= 25:
                        detected_years.append(val)

        if detected_years:
            # Usually the minimum required is the primary threshold
            return min(detected_years)

        # Fallback to general regex
        generic_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)', text_lower)
        if generic_match:
            val = int(generic_match.group(1))
            if 1 <= val <= 20:
                return val

        return 0

    def _extract_education(self, text: str) -> str:
        """Extract degree and major requirements from JD."""
        text_lower = text.lower()
        found_degree = None

        for degree_name, patterns in self.degree_patterns.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    found_degree = degree_name
                    break
            if found_degree:
                break

        found_major = None
        for major in self.major_patterns:
            if major in text_lower:
                found_major = major.title()
                break

        if found_degree and found_major:
            return f"{found_degree} in {found_major} or related discipline"
        elif found_degree:
            return f"{found_degree} in Computer Science or related quantitative field"
        elif found_major:
            return f"Bachelor's degree in {found_major}"

        return "Bachelor's degree in Computer Science or related field"

    def _extract_skills_from_jd(self, text: str):
        """Extract skills and partition them into core vs preferred if section headers exist."""
        text_lower = text.lower()

        # Split into sections if present
        core_section = ""
        preferred_section = ""

        # Identify sections by headings
        lines = text.split("\n")
        current_mode = "core"
        core_lines = []
        pref_lines = []

        for line in lines:
            line_low = line.lower()
            if any(h in line_low for h in ["preferred", "nice to have", "plus", "bonus", "good to have", "desired"]):
                current_mode = "preferred"
            elif any(h in line_low for h in ["required", "qualification", "must have", "requirements", "what you need", "skills"]):
                current_mode = "core"

            if current_mode == "preferred":
                pref_lines.append(line)
            else:
                core_lines.append(line)

        core_text = " ".join(core_lines).lower()
        pref_text = " ".join(pref_lines).lower()

        core_skills = set()
        preferred_skills = set()
        all_skills = set()

        for skill in ALL_SKILLS:
            # Word boundary regex for exact matching
            pattern = rf"(?:\b|\s){re.escape(skill)}(?:\b|\s)"
            if re.search(pattern, text_lower):
                normalized = normalize_skill(skill)
                all_skills.add(normalized)

                if pref_lines and re.search(pattern, pref_text):
                    preferred_skills.add(normalized)
                else:
                    core_skills.add(normalized)

        # If no split was cleanly made, all skills default to core
        if not core_skills and all_skills:
            core_skills = set(all_skills)

        return sorted(list(core_skills)), sorted(list(preferred_skills)), sorted(list(all_skills))

    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract top bullet points or action sentences representing responsibilities."""
        lines = text.split("\n")
        responsibilities = []
        capture = False

        for line in lines:
            trimmed = line.strip()
            trimmed_low = trimmed.lower()
            if any(h in trimmed_low for h in ["responsibilities", "what you'll do", "what you will do", "duties", "role overview"]):
                capture = True
                continue
            elif capture and any(h in trimmed_low for h in ["requirements", "qualifications", "skills", "benefits", "about us", "who you are"]):
                capture = False

            if capture and len(trimmed) > 15:
                # Remove bullet symbols
                clean_bullet = re.sub(r'^[\*\-\•\d+\.]\s*', '', trimmed)
                if len(clean_bullet) > 15 and len(clean_bullet.split()) >= 4:
                    responsibilities.append(clean_bullet)

        if not responsibilities:
            # Fallback: extract sentences with strong action verbs
            action_verbs = ["design", "develop", "build", "lead", "manage", "collaborate", "architect", "optimize", "implement", "deploy"]
            for line in lines:
                clean_line = re.sub(r'^[\*\-\•\d+\.]\s*', '', line.strip())
                if any(clean_line.lower().startswith(v) for v in action_verbs) and len(clean_line.split()) >= 4:
                    responsibilities.append(clean_line)

        return responsibilities
