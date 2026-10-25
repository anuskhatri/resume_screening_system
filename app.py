import os
import sys
import io
import csv
import json
import uuid
import random
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import PyPDF2
from docx import Document

from src.config import Config
from src.inference.resume_ranking import ResumeRanker
from src.inference.utils import TextCleaner
from src.inference.jd_parser import JobDescriptionParser

app = Flask(__name__)
app.secret_key = 'ai-resume-screener-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP components
ranker = ResumeRanker()
cleaner = TextCleaner()
jd_parser = JobDescriptionParser()
DEMO_DATASET_DIR = Path(app.root_path) / 'resume dataset' / 'data'
DEMO_RESUME_CACHE = None

# Cache latest results in memory for export and modal inspection
LATEST_SCREENING = {
    "results": [],
    "job_metadata": {},
    "timestamp": None
}
SCREENING_RUNS_DIR = Path(app.root_path) / 'results' / 'screening_runs'
SCREENING_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_scoring_weights(form):
    """Read recruiter-adjustable weights and normalize them to sum to one."""
    defaults = {'semantic': .45, 'skill': .30, 'experience': .15, 'education': .10}
    weights = {}
    for key, default in defaults.items():
        try:
            weights[key] = max(0.0, min(100.0, float(form.get(f'{key}_weight', default * 100))))
        except (TypeError, ValueError):
            weights[key] = default * 100
    total = sum(weights.values())
    if total <= 0:
        weights = {key: value * 100 for key, value in defaults.items()}
        total = sum(weights.values())
    return {key: round(value / total, 4) for key, value in weights.items()}


def persist_screening_run(results, metadata, mode):
    """Persist a review as a self-contained JSON record for later retrieval."""
    run_id = uuid.uuid4().hex[:12]
    saved_at = datetime.now().astimezone().isoformat(timespec='seconds')
    record = {
        'id': run_id,
        'saved_at': saved_at,
        'mode': mode,
        'model_version': Config.MODEL_NAME,
        'metadata': metadata,
        'results': results,
    }
    path = SCREENING_RUNS_DIR / f'{run_id}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return record


def prepare_results(scores_df):
    """Convert pandas/numpy scalars to regular JSON-safe Python values."""
    return json.loads(scores_df.to_json(orient='records', force_ascii=False))


def summarize_nlp_edge_cases(results):
    """Summarize alias, negation, and project-section detections for recruiter QA."""
    cases = [row.get('nlp_edge_cases', {}) for row in results]
    return {
        'alias_candidates': sum(bool(item.get('recognized_aliases')) for item in cases),
        'negated_candidates': sum(bool(item.get('negated_mentions')) for item in cases),
        'project_candidates': sum(bool(item.get('project_section_skills')) for item in cases),
        'recognized_aliases': sorted({term for item in cases for term in item.get('recognized_aliases', [])}),
        'negated_mentions': sorted({term for item in cases for term in item.get('negated_mentions', [])}),
        'project_section_skills': sorted({term for item in cases for term in item.get('project_section_skills', [])}),
    }
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extract text from PDF"""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return text


def extract_text_from_docx(filepath):
    """Extract text from DOCX"""
    try:
        doc = Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX {filepath}: {e}")
        return ""


def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif ext == 'docx':
        return extract_text_from_docx(filepath)
    elif ext == 'txt':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {filepath}: {e}")
            return ""
    return ""


def load_demo_pdf_resumes(sample_size=40):
    """Extract and cache a stable sample from the categorized resume PDF corpus."""
    global DEMO_RESUME_CACHE
    if DEMO_RESUME_CACHE is not None:
        return DEMO_RESUME_CACHE.copy()

    pdf_paths = sorted(DEMO_DATASET_DIR.rglob('*.pdf'))
    if not pdf_paths:
        return None

    rng = random.Random(42)
    selected_paths = sorted(rng.sample(pdf_paths, min(sample_size, len(pdf_paths))))
    resume_rows = []

    for pdf_path in selected_paths:
        resume_text = extract_text_from_pdf(str(pdf_path))
        if len(resume_text.strip()) <= 20:
            continue

        cleaned_text = cleaner.clean_text(resume_text)
        resume_rows.append({
            'filename': pdf_path.stem,
            'Resume': resume_text,
            'cleaned_resume': cleaned_text,
            'processed_resume': cleaner.tokenize_and_lemmatize(cleaned_text),
            'extracted_skills': cleaner.extract_skills(resume_text),
            'experience_years': cleaner.extract_experience_years(resume_text),
            'Category': pdf_path.parent.name.replace('-', ' ').title(),
        })

    if resume_rows:
        DEMO_RESUME_CACHE = pd.DataFrame(resume_rows)
        return DEMO_RESUME_CACHE.copy()
    return None


@app.route('/')
def index():
    """Modern Home Landing Page"""
    return render_template('index.html')


@app.route('/api/parse-jd', methods=['POST'])
def api_parse_jd():
    """
    API endpoint: Accepts uploaded JD file or raw text and uses NLP
    to automatically extract Role, Experience, Education, and Skills.
    """
    jd_text = ""

    # Check if a file was uploaded
    if 'jd_file' in request.files:
        file = request.files['jd_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"jd_{filename}")
            file.save(filepath)
            jd_text = extract_text_from_file(filepath)
            try:
                os.remove(filepath)
            except Exception:
                pass

    # Fallback to json or form text
    if not jd_text:
        if request.is_json:
            jd_text = request.json.get('job_description', '')
        else:
            jd_text = request.form.get('job_description', '')

    if not jd_text or len(jd_text.strip()) < 10:
        return jsonify({'error': 'Please provide a valid Job Description file or text.'}), 400

    parsed = jd_parser.parse_jd(jd_text)
    return jsonify({
        'status': 'success',
        'job_title': parsed['job_title'],
        'seniority': parsed['seniority'],
        'required_experience': parsed['required_experience'],
        'required_education': parsed['required_education'],
        'core_skills': parsed['core_skills'],
        'preferred_skills': parsed['preferred_skills'],
        'all_skills': parsed['all_skills'],
        'skills_by_category': parsed['skills_by_category'],
        'responsibilities': parsed['responsibilities'],
        'raw_text': parsed['raw_text']
    })


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload resumes and job description (with optional JD file upload)"""
    if request.method == 'POST':
        jd_text = request.form.get('job_description', '').strip()

        # Check if user uploaded a JD file
        if 'jd_file' in request.files and request.files['jd_file'].filename:
            jd_file = request.files['jd_file']
            if allowed_file(jd_file.filename):
                jd_filename = secure_filename(jd_file.filename)
                jd_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"jd_{jd_filename}")
                jd_file.save(jd_filepath)
                extracted_jd = extract_text_from_file(jd_filepath)
                try:
                    os.remove(jd_filepath)
                except Exception:
                    pass
                if extracted_jd:
                    jd_text = extracted_jd

        if not jd_text or len(jd_text.strip()) < 10:
            flash('Please upload a Job Description file or enter job requirements.', 'error')
            return redirect(url_for('upload'))

        # Parse JD using NLP
        parsed_jd = jd_parser.parse_jd(jd_text)

        # Get manual overrides if provided
        req_exp_str = request.form.get('required_experience')
        if req_exp_str and req_exp_str.strip().isdigit():
            required_experience = int(req_exp_str)
        else:
            required_experience = parsed_jd['required_experience']

        req_edu_str = request.form.get('required_education', '').strip()
        required_education = req_edu_str if req_edu_str else parsed_jd['required_education']

        anonymize = request.form.get('anonymize') == 'on'
        scoring_weights = get_scoring_weights(request.form)

        # Get uploaded resumes
        files = request.files.getlist('resumes')
        if not files or files[0].filename == '':
            flash('Please upload at least one candidate resume.', 'error')
            return redirect(url_for('upload'))

        resumes_data = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                resume_text = extract_text_from_file(filepath)
                try:
                    os.remove(filepath)
                except Exception:
                    pass

                if resume_text and len(resume_text.strip()) > 20:
                    if anonymize:
                        resume_text = cleaner.anonymize_text(resume_text)

                    cleaned_text = cleaner.clean_text(resume_text)
                    processed_text = cleaner.tokenize_and_lemmatize(cleaned_text)
                    skills = cleaner.extract_skills(resume_text)
                    experience = cleaner.extract_experience_years(resume_text)

                    resumes_data.append({
                        'filename': filename,
                        'Resume': resume_text,
                        'cleaned_resume': cleaned_text,
                        'processed_resume': processed_text,
                        'extracted_skills': skills,
                        'experience_years': experience,
                        'Category': 'Uploaded Candidate'
                    })

        if not resumes_data:
            flash('No valid resumes could be processed. Please check file formats.', 'error')
            return redirect(url_for('upload'))

        df_resumes = pd.DataFrame(resumes_data)

        # Rank candidates using enhanced multi-factor NLP
        scores_df = ranker.rank_candidates(
            df_resumes,
            jd_text,
            required_experience=required_experience,
            required_education=required_education,
            required_skills=parsed_jd['core_skills'] or parsed_jd['all_skills'],
            anonymize=anonymize,
            scoring_weights=scoring_weights
        )

        results = prepare_results(scores_df)
        nlp_diagnostics = summarize_nlp_edge_cases(results)

        # Cache results for export
        LATEST_SCREENING['results'] = results
        LATEST_SCREENING['job_metadata'] = {
            'job_title': parsed_jd['job_title'],
            'seniority': parsed_jd['seniority'],
            'required_experience': required_experience,
            'required_education': required_education,
            'required_skills': parsed_jd['core_skills'] or parsed_jd['all_skills'],
            'core_skills': parsed_jd['core_skills'],
            'preferred_skills': parsed_jd['preferred_skills'],
            'job_description': jd_text,
            'scoring_weights': scoring_weights,
            'model_version': Config.MODEL_NAME,
            'nlp_diagnostics': nlp_diagnostics
        }
        LATEST_SCREENING['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run = persist_screening_run(results, LATEST_SCREENING['job_metadata'], 'resume upload')

        shortlisted_count = int(scores_df['shortlisted'].sum())
        avg_score = round(float(scores_df['final_score'].mean() * 100), 1)

        return render_template(
            'results.html',
            results=results,
            job_description=jd_text,
            parsed_jd=parsed_jd,
            total_candidates=len(results),
            shortlisted_count=shortlisted_count,
            avg_score=avg_score,
            required_experience=required_experience,
            required_education=required_education,
            scoring_weights=scoring_weights,
            nlp_diagnostics=nlp_diagnostics,
            run_id=run['id'],
            run_saved_at=run['saved_at']
        )

    return render_template('upload.html')


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Interactive Demo Playground with rich sample jobs"""
    sample_jobs = {
        'AI / Machine Learning Engineer': (
            "We are looking for an AI/Machine Learning Engineer to build and deploy deep learning and NLP models. "
            "The ideal candidate must have strong experience with Python, PyTorch, TensorFlow, Hugging Face Transformers, "
            "Large Language Models (LLMs), RAG pipelines, LangChain, and vector databases like Chroma or Pinecone. "
            "Experience with Docker, Kubernetes, and AWS MLOps is highly desirable. "
            "Minimum 3 years of hands-on experience in machine learning. "
            "Bachelor's or Master's degree in Computer Science, Data Science, or related quantitative field."
        ),
        'Senior Full Stack Developer': (
            "Seeking a Senior Full Stack Developer to lead development of scalable cloud applications. "
            "Required skills: Python, Django, FastAPI, JavaScript, TypeScript, React, Next.js, Redux, HTML5, CSS3, and REST APIs. "
            "Strong database knowledge in PostgreSQL, Redis, and MongoDB. Familiarity with Docker, Git, CI/CD, and AWS. "
            "Minimum 4 years of professional web development experience. "
            "Bachelor's degree in Computer Science or Software Engineering required."
        ),
        'Data Scientist & Analytics Lead': (
            "We are hiring a Data Scientist with expertise in predictive modeling, statistical analysis, and machine learning. "
            "Must be proficient in Python, R, SQL, Pandas, NumPy, Scikit-learn, Tableau, and data visualization. "
            "Experience building classification, regression, and clustering algorithms is required. "
            "Minimum 2 years of relevant experience. "
            "Master's degree in Data Science, Statistics, Mathematics, or Computer Science."
        ),
        'Cloud & DevOps Engineer': (
            "Looking for a Cloud DevOps Engineer to maintain our high-availability cloud infrastructure. "
            "Must have solid experience with Linux, AWS, Docker, Kubernetes, Terraform, Ansible, CI/CD with GitHub Actions, "
            "Prometheus, Grafana, and Bash scripting. Python programming is a plus. "
            "Minimum 3 years of experience in DevOps or Cloud Infrastructure engineering. "
            "Bachelor's degree in Information Technology or Computer Science."
        )
    }

    if request.method == 'POST':
        df_sample = load_demo_pdf_resumes(sample_size=40)
        if df_sample is None or df_sample.empty:
            flash('No readable resume PDFs were found in resume dataset/data.', 'error')
            return redirect(url_for('demo'))

        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            flash('Please select or provide a job description.', 'error')
            return redirect(url_for('demo'))

        # Parse JD using NLP
        parsed_jd = jd_parser.parse_jd(job_description)

        req_exp_str = request.form.get('required_experience')
        if req_exp_str and req_exp_str.strip().isdigit():
            required_experience = int(req_exp_str)
        else:
            required_experience = parsed_jd['required_experience']

        req_edu_str = request.form.get('required_education', '').strip()
        required_education = req_edu_str if req_edu_str else parsed_jd['required_education']

        scoring_weights = get_scoring_weights(request.form)
        scores_df = ranker.rank_candidates(
            df_sample,
            job_description,
            required_experience=required_experience,
            required_education=required_education,
            required_skills=parsed_jd['core_skills'] or parsed_jd['all_skills'],
            scoring_weights=scoring_weights
        )

        results = prepare_results(scores_df)
        nlp_diagnostics = summarize_nlp_edge_cases(results)

        # Cache results for export
        LATEST_SCREENING['results'] = results
        LATEST_SCREENING['job_metadata'] = {
            'job_title': parsed_jd['job_title'],
            'seniority': parsed_jd['seniority'],
            'required_experience': required_experience,
            'required_education': required_education,
            'required_skills': parsed_jd['core_skills'] or parsed_jd['all_skills'],
            'core_skills': parsed_jd['core_skills'],
            'preferred_skills': parsed_jd['preferred_skills'],
            'job_description': job_description,
            'scoring_weights': scoring_weights,
            'model_version': Config.MODEL_NAME,
            'nlp_diagnostics': nlp_diagnostics
        }
        LATEST_SCREENING['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run = persist_screening_run(results, LATEST_SCREENING['job_metadata'], 'sample dataset')

        shortlisted_count = int(scores_df['shortlisted'].sum())
        avg_score = round(float(scores_df['final_score'].mean() * 100), 1)

        return render_template(
            'results.html',
            results=results,
            job_description=job_description,
            parsed_jd=parsed_jd,
            total_candidates=len(results),
            shortlisted_count=shortlisted_count,
            avg_score=avg_score,
            required_experience=required_experience,
            required_education=required_education,
            scoring_weights=scoring_weights,
            nlp_diagnostics=nlp_diagnostics,
            run_id=run['id'],
            run_saved_at=run['saved_at']
        )

    return render_template('demo.html', sample_jobs=sample_jobs)


@app.route('/api/export-csv')
def export_csv():
    """Download the latest candidate ranking results as a formatted CSV file."""
    results = LATEST_SCREENING.get('results', [])
    if not results:
        return Response("No screening results available to export.", status=400, mimetype="text/plain")

    si = io.StringIO()
    cw = csv.writer(si)

    # Header
    cw.writerow([
        'Rank', 'Candidate / File', 'Category', 'Match Tier', 'Final Score (%)',
        'Semantic Score (%)', 'Skill Match (%)', 'Experience (Years)',
        'Matched Skills', 'Missing Skills', 'Bonus Skills', 'Shortlisted'
    ])

    for r in results:
        cw.writerow([
            r.get('rank', ''),
            r.get('filename', ''),
            r.get('category', ''),
            r.get('match_tier', ''),
            round(r.get('final_score', 0) * 100, 1),
            round(r.get('semantic_score', 0) * 100, 1),
            r.get('skill_match_pct', 0),
            r.get('experience_years', 0),
            ", ".join(r.get('matched_skills', [])),
            ", ".join(r.get('missing_skills', [])),
            ", ".join(r.get('bonus_skills', [])),
            'YES' if r.get('shortlisted') else 'NO'
        ])

    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=ranked_candidates_report.csv"}
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
