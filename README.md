# Fieldnote — Resume Screening System

Fieldnote is a Flask-based NLP application for reviewing resumes against a job description. It extracts text from PDF, DOCX, and TXT files, identifies role and candidate attributes, computes semantic and skills-based fit signals, ranks candidates, and presents evidence for a recruiter to review.

The project includes a web interface, a sample-data demo, a command-line ranking example, and a separate training pipeline for resume-category classification.

> **Decision-support tool:** Scores and extracted attributes are estimates. Review the source resume and supporting evidence before making hiring decisions. The system does not make or recommend a final hiring decision.

## Contents

- [What the application does](#what-the-application-does)
- [How the NLP pipeline works](#how-the-nlp-pipeline-works)
- [Ranking and score interpretation](#ranking-and-score-interpretation)
- [Requirements](#requirements)
- [Install and run locally](#install-and-run-locally)
- [Use the web application](#use-the-web-application)
- [Routes and API](#routes-and-api)
- [Saved screening runs](#saved-screening-runs)
- [Training and evaluation](#training-and-evaluation)
- [Command-line example](#command-line-example)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Privacy and deployment notes](#privacy-and-deployment-notes)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What the application does

- Accepts job descriptions and batches of resumes in PDF, DOCX, and TXT formats.
- Parses job descriptions for a role title, seniority, minimum experience, education, skills, and responsibilities.
- Splits detected skills into core and preferred groups when the job description uses recognizable section headings.
- Extracts candidate skills, experience, education, and selected resume highlights.
- Compares the job description and resumes with Sentence-BERT embeddings and cosine similarity.
- Combines semantic fit, core-skill coverage, experience alignment, and education alignment into a configurable score.
- Shows ranked candidates, matched and missing skills, requirement checks, score components, and resume evidence.
- Flags alias and abbreviation matches, negated skill mentions, and skills found in project sections for recruiter review.
- Saves successful screening runs as local JSON files and exports the latest in-process result set as CSV.
- Offers a demo that extracts and evaluates a reproducible sample of up to 40 actual PDF resumes from `resume dataset/data/`.

The landing-page name is **Fieldnote**. The Python package and repository are referred to as the Resume Screening System in code and artifacts.

## How the NLP pipeline works

### 1. Document extraction

The Flask application extracts text with PyPDF2 for PDF, python-docx for DOCX, and Python's text reader for TXT. The upload limit is 32 MB for the complete request. Temporary uploaded documents are removed after extraction.

Scanned PDFs without an embedded text layer may produce no usable text; OCR is not currently included.

### 2. Text preparation and skill extraction

The TextCleaner in src/inference/utils.py performs these tasks:

- Cleans text used for embeddings by lowercasing it, removing URLs and email addresses, removing digits and most punctuation, and collapsing whitespace.
- Uses NLTK tokenization, English stop-word removal, and WordNet lemmatization for embedding input.
- Matches skills from the taxonomy in src/inference/skill_ontology.py with word-boundary-aware patterns and canonicalizes known aliases.
- Extracts experience years with regular expressions and identifies an education level using degree keywords.
- Selects a small set of career-highlight sentences using sentence tokenization and action-word heuristics.
- Optionally redacts email addresses, phone numbers, selected profile URLs, and PERSON entities found in the first 500 characters of the resume.

The raw resume text is still used for extraction, evidence, and previews. The cleaned and lemmatized text is used for semantic embeddings.

### 3. Job-description parsing

JobDescriptionParser in src/inference/jd_parser.py uses regular expressions and heading heuristics to estimate:

- Role title and seniority
- Minimum years of experience
- Education level and a possible field of study
- Skills, partitioned into core and preferred where section headings can be recognized
- Responsibilities from labeled sections or action-oriented lines

If no clear preferred-skills section is detected, skills generally default to the core group. Parsing results should be checked and can be edited in the upload form.

### 4. Semantic similarity

FeatureExtractor loads sentence-transformers/all-MiniLM-L6-v2, which produces 384-dimensional embeddings. The job description and each resume are encoded, then cosine similarity is calculated. PyTorch uses CUDA when available and otherwise falls back to CPU.

The embedding model is downloaded from Hugging Face on first use unless it is already cached locally. Internet access is therefore needed for initial setup when the model is not present in the cache.

### 5. Candidate ranking and evidence

ResumeRanker in src/inference/resume_ranking.py computes four score components and sorts candidates by their weighted total. It also returns skill gaps, highlights, qualification checks, and short evidence excerpts for matched skills, experience, and education where text can be found.

The score is a prioritization aid, not a probability that a person will succeed in a role. The final decision should account for context that the parser cannot reliably infer.

## Ranking and score interpretation

The default weights are:

| Signal | Default weight | What it represents |
|---|---:|---|
| Role context | 45% | Cosine similarity between job-description and resume embeddings |
| Core-skill coverage | 30% | Matched core skills divided by detected core skills |
| Experience | 15% | Candidate years relative to the stated minimum, capped at a full score when the minimum is met |
| Education | 10% | Heuristic comparison of detected degree levels |

The upload and demo forms let the recruiter adjust these four weights. The server normalizes the selected values so they sum to one before ranking. Setting every slider to zero restores the defaults.

Conceptually, the score is:

    final score = semantic weight × semantic score
                + skill weight × core-skill coverage
                + experience weight × experience score
                + education weight × education score

The separate requirement indicators show whether core skills, minimum experience, and education appear to be met. They do not automatically exclude a candidate. Shortlisting is currently based on the configured composite-score threshold, not a strict hard-requirement gate.

Other score details:

- Semantic similarity is based on the pre-trained embedding model, not the resume-category classifier trained in this repository.
- Missing degree text receives a neutral education score in the composite, while the detail view marks the requirement for review when there is no degree evidence.
- Experience is extracted heuristically. If no required experience is supplied, the experience component receives a full score.
- The keyword-overlap score is calculated for analysis but is not part of the composite ranking formula.
- Fit labels are Strong Fit at 70% or above, Moderate Fit from 50% to below 70%, and Low Fit below 50%.
- The shortlist threshold is 60% by default, as defined in src/config.py.

### NLP extraction checks

The results page summarizes three kinds of edge cases across the current run:

- **Aliases and abbreviations:** recognized spellings are mapped to canonical skills where defined in the ontology.
- **Negated mentions:** common phrases such as “no experience with Python” are excluded from positive skill matches and listed for review.
- **Project-section mentions:** skills found under a recognizable Projects heading are identified so reviewers can consider their context.

These checks are heuristic. They do not determine how recently, deeply, or independently a candidate used a skill. Project mentions still count as detected candidate skills; the interface surfaces the context rather than treating project work as equivalent to employment experience.

## Requirements

- Python 3.10 or later is recommended. The included Dockerfile uses Python 3.10; the local development environment has also been run with Python 3.12.
- pip and a virtual environment.
- Internet access on first setup to install dependencies, download NLTK/spaCy resources, and fetch the Sentence-BERT model if it is not cached.
- GPU is optional. CUDA-capable PyTorch and a compatible NVIDIA setup can accelerate embedding generation; CPU inference is supported but slower.

Dependencies are listed in requirements.txt. Training plots additionally use Matplotlib and Seaborn, which are included there.

## Install and run locally

Run commands from the project root directory (the directory containing app.py and requirements.txt).

### Windows PowerShell

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
    python app.py

If the Python launcher does not have 3.12 installed, use an installed compatible version with python -m venv .venv.

### macOS or Linux

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
    python app.py

Open http://localhost:5000 after the model has finished loading. The first startup may take longer because the Sentence-BERT model is loaded into memory.

The port can be changed with the PORT environment variable. For example, in PowerShell:

    $env:PORT = "5001"
    python app.py

The development server currently binds to all network interfaces. See [Privacy and deployment notes](#privacy-and-deployment-notes) before exposing it beyond a trusted local environment.

## Use the web application

### Screen uploaded resumes

1. Open **Review candidates** or go to /upload.
2. Add a job description as a PDF, DOCX, or TXT file, or paste the text in the form.
3. Review or adjust the detected minimum experience, education, and ranking weights.
4. Upload one or more candidate resumes. Supported formats are PDF, DOCX, and TXT; the complete request must be under 32 MB.
5. Optionally enable the anonymization pass. Review the results and their evidence; correct parser assumptions using the original resume as needed.
6. Export the current result set as CSV if required.

### Run the sample-data demo

1. Open **Sample review** or go to /demo.
2. Choose one of the sample roles or edit the job-description text.
3. Adjust required experience, education, and ranking weights.
4. Submit to screen the same reproducibly selected sample of up to 40 readable PDF resumes from `resume dataset/data/`.

The first demo run extracts and preprocesses the sample, then caches it in memory for later runs. PDF files are grouped into category folders, and each folder name is used as the candidate category. The demo requires this PDF corpus; the processed CSV remains part of the separate training and command-line workflows.

### Review a candidate

The results list includes rank, composite score, skill coverage, experience, and shortlist status. **View NLP Breakdown** opens component scores, role-requirement checks, matched and missing skills, evidence excerpts, highlights, and a resume-text preview. Evidence is extracted from the resume; it is not an independent verification of the claim.

## Routes and API

The application currently implements these routes:

| Method | Path | Purpose |
|---|---|---|
| GET | / | Fieldnote landing page |
| GET, POST | /upload | Upload a job description and resume batch, then render ranked results |
| GET, POST | /demo | Choose a sample role and rank a stable sample of categorized PDF resumes |
| POST | /api/parse-jd | Parse job-description text or a JD document and return extracted fields as JSON |
| GET | /api/export-csv | Download the most recent screening result set held in the current app process |

There is no general-purpose JSON endpoint for ranking resumes and no candidate-by-ID API in the current implementation.

### Parse a job description with JSON

Request:

    curl -X POST http://localhost:5000/api/parse-jd \
      -H "Content-Type: application/json" \
      -d '{"job_description":"Senior Data Scientist. Required skills: Python, SQL, and scikit-learn. Minimum 4 years of experience. Preferred: Tableau."}'

The JSON response contains status, job_title, seniority, required_experience, required_education, core_skills, preferred_skills, all_skills, skills_by_category, responsibilities, and raw_text.

A document can be parsed with multipart form data:

    curl -X POST http://localhost:5000/api/parse-jd -F "jd_file=@role-description.pdf"

Accepted JD formats are PDF, DOCX, and TXT.

### Export the latest results

    curl -OJ http://localhost:5000/api/export-csv

The CSV includes rank, candidate/file, category, fit tier, final and component scores, experience, matched/missing/bonus skills, and shortlist status. The endpoint uses in-memory state: it only exports the latest completed run in this process and returns an error if no run is available.

## Saved screening runs

After a successful upload screening or demo screening, the app writes a JSON record to results/screening_runs/. The record includes:

- Unique run ID, local timestamp, and run mode
- Job title, seniority, full job-description text, and requirement settings
- Normalized scoring weights and Sentence-BERT model identifier
- Candidate rankings, score components, skill analysis, evidence, and resume previews
- Aggregated NLP edge-case diagnostics

These files are local records; there is no browser-based history or retrieval route. The CSV endpoint still uses the latest in-memory result set, so it does not retrieve an older JSON run after an app restart. The folder is listed in .gitignore because the records can contain personal candidate data. Protect the folder with the same care as the original resumes and remove records according to your retention requirements.

## Training and evaluation

The project has a separate category-classification training path:

1. DataPreprocessor reads data/raw/UpdatedResumeDataSet.csv, whose expected columns are Category and Resume.
2. It creates cleaned and lemmatized text, detected skills, and extracted experience, then writes data/processed/processed_resumes.csv.
3. FeatureExtractor creates Sentence-BERT embeddings and caches them at models/saved_models/resume_embeddings.npy.
4. ModelTrainer trains a feed-forward classifier to predict the dataset's resume category and saves its weights and label encoder under models/saved_models/.

Run training from the project root:

    python -m src.training.train

Training reuses an existing processed CSV and cached embeddings when present. If the source data or preprocessing changes, ensure those cached files are regenerated before interpreting a new training run.

The saved classifier is an auxiliary resume-category classifier. The Flask ranking flow does **not** load classifier_model.pth; it ranks directly with the pre-trained Sentence-BERT embeddings and the heuristic signals described above.

### Existing evaluation snapshot

The checked-in training_metrics.json and training_report.txt currently report the category classifier's held-out accuracy as **5.88%**, weighted precision as **0.0035**, weighted recall as **5.88%**, and weighted F1 as **0.0065** across 25 categories. The snapshot represents a small dataset and one split; these results are weak and should not be presented as evidence that the candidate-ranking flow is accurate. The ranking formula itself does not currently have a labeled ranking benchmark or a reported Precision@k / NDCG evaluation.

Regenerate training metrics before using them in a report. Do not use older example numbers in generated documentation as current measured performance unless the artifacts support them.

Training outputs include:

- models/saved_models/classifier_model.pth
- models/saved_models/label_encoder.npy
- models/saved_models/resume_embeddings.npy
- models/saved_models/training_metrics.json
- models/saved_models/training_report.txt
- models/saved_models/plots/

To regenerate the supplementary results Markdown from the saved metrics and plots, run:

    python generate_results_documentation.py

That script writes results/PROJECT_RESULTS.md. Check the output against the current metric files before citing it; it may contain illustrative sample ranking text and is not a ranking evaluation.

## Command-line example

main.py ranks the processed dataset against a built-in example job description, prints the top candidates, and writes data/processed/ranked_candidates.csv.

    python main.py

The ranking class can also be called from a Python script. The DataFrame must contain a Resume text column; optional columns include processed_resume, extracted_skills, experience_years, filename, and Category.

    import pandas as pd
    from src.inference.resume_ranking import ResumeRanker

    candidates = pd.DataFrame([
        {
            "filename": "candidate-a.txt",
            "Resume": "Python developer with 5 years of experience. Built NLP services with PyTorch.",
            "experience_years": 5,
            "extracted_skills": ["python", "nlp", "pytorch"],
        }
    ])

    ranker = ResumeRanker()
    results = ranker.rank_candidates(
        candidates,
        "Python engineer with NLP and PyTorch experience. Minimum 3 years.",
        required_experience=3,
        required_education="Bachelor's degree",
        required_skills=["python", "nlp", "pytorch"],
    )
    print(results[["rank", "filename", "final_score", "matched_skills"]])

Instantiating ResumeRanker loads the Sentence-BERT model, so the first call may download model files and use significant memory.

## Configuration

Runtime and training settings are defined in src/config.py:

| Setting | Default | Purpose |
|---|---:|---|
| MODEL_NAME | sentence-transformers/all-MiniLM-L6-v2 | Sentence embedding model |
| DEVICE | CUDA when available, otherwise CPU | PyTorch device |
| BATCH_SIZE | 32 | Embedding batch size |
| SEMANTIC_WEIGHT | 0.45 | Default semantic score weight |
| SKILL_WEIGHT | 0.30 | Default core skill score weight |
| EXPERIENCE_WEIGHT | 0.15 | Default experience score weight |
| EDUCATION_WEIGHT | 0.10 | Default education score weight |
| SHORTLIST_THRESHOLD | 0.60 | Minimum composite score to mark shortlisted |
| LEARNING_RATE | 0.00002 | Classifier training learning rate |
| EPOCHS | 10 | Classifier training epochs |

The web form supplies per-run weights; Config values are the fallback defaults for code that calls ResumeRanker directly.

## Project structure

    app.py                              Flask routes, upload processing, CSV export, local run persistence
    main.py                             CLI sample ranking entry point
    requirements.txt                    Python dependencies
    Dockerfile                          Container build and Gunicorn startup
    gunicorn.conf.py                    Separate Gunicorn configuration file
    src/
      config.py                         Runtime, paths, model, and scoring defaults
      inference/
        jd_parser.py                    Job-description parser
        resume_ranking.py               Candidate scoring and ranking
        skill_ontology.py               Skill taxonomy, aliases, normalization
        utils.py                        Text cleaning, skill/experience/education extraction, evidence
      training/
        data_preprocessing.py           Raw resume dataset preprocessing
        feature_extraction.py           Sentence-BERT embeddings and cosine similarity
        model_training.py               Resume-category neural classifier and metrics
        train.py                        Preprocess, embed, train, and save pipeline
    templates/                          Flask/Jinja pages: home, upload, demo, results, base
    static/css/style.css                Fieldnote theme and responsive design system
    static/js/main.js                   Upload interactions, parsing UI, weights, result controls
    data/raw/                            Original resume dataset
    resume dataset/data/<CATEGORY>/      Categorized source PDF resumes used by the web demo
    data/processed/                      Processed resumes and sample ranking CSV
    models/saved_models/                 Classifier, encoder, embeddings, metrics, plots
    results/                             Generated project report and local screening-run JSON files
    uploads/                             Temporary upload workspace

## Privacy and deployment notes

This application processes personal resume information. The current code is intended for local development or a carefully controlled demonstration, not an unreviewed public deployment.

- The Flask development server binds to 0.0.0.0. Use it only on a trusted network; a firewall or localhost-only binding should be configured before exposing it elsewhere.
- Flask currently uses a hard-coded secret key. Set a secure deployment secret outside source control before using sessions in a shared environment.
- The app has no user login, role-based access, CSRF protection, or per-user data separation.
- Saved run JSON contains job descriptions, candidate attributes, evidence, and resume excerpts. Keep results/screening_runs/ private and apply a deliberate retention policy.
- The anonymization option is partial. It does not guarantee de-identification, may miss names or other personal attributes, and does not remove every possible proxy for identity.
- Uploaded resumes are removed after text extraction, but screening results and excerpts are retained in memory and in local JSON records after a successful run.
- The CSV export route is not authenticated. Do not expose the app or its exports to untrusted users.
- The included Dockerfile launches Gunicorn with two workers. Each worker may load its own model copy; verify RAM/VRAM capacity before using that configuration with a GPU.
- For production use, configure authentication, authorization, CSRF protection, secure secret management, upload validation, logging, data retention, and an appropriate WSGI deployment before making the app reachable to external users.

## Known limitations

- JD parsing, candidate extraction, and evidence selection use rules and keyword patterns; they are not a validated hiring assessment.
- The skill ontology has finite coverage. Unlisted technologies, unconventional wording, and context-dependent meanings can be missed or misclassified.
- Negation handling covers selected common phrases, not all sentence structures. Check the source text.
- Skills found in a projects section are surfaced as contextual mentions; the system does not infer project quality, ownership, depth, or recency.
- Experience extraction can mistake unrelated year counts or miss nonstandard resume formats.
- Education scoring compares degree keywords and does not reliably verify the field of study, accreditation, completion, or equivalency.
- Semantic similarity can reward broad topical overlap and should not be read as a calibrated probability or a measure of candidate quality.
- Shortlist status is a threshold on the composite score, not a minimum-requirements gate.
- Scanned/image-only PDFs need OCR, which is not provided.
- Ranking evaluation data and recruiter feedback loops are not currently included.

## Troubleshooting

### spaCy model errors

Install the English model in the active environment and restart the app:

    python -m spacy download en_core_web_sm

The JD parser can fall back to a blank spaCy pipeline, but the resume cleaner attempts to install the English model when it is unavailable. Installing it explicitly is more predictable.

### NLTK resource errors

Download the tokenizer and corpora into the active Python environment:

    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

### First startup or model download is slow

The model is loaded during application initialization and may download from Hugging Face on first use. Allow the initial load to finish. If running offline, pre-populate the Hugging Face model cache first.

### CUDA is unavailable

The application selects CPU when PyTorch cannot access CUDA. It should still run, but embedding generation may be slower. Confirm that the installed PyTorch build matches the local CUDA driver and GPU setup if acceleration is expected.

### No candidates appear in the demo

The demo looks for readable PDF files under `resume dataset/data/`, including its category subfolders. Check that the dataset is present and that the PDFs contain selectable text. Scanned image-only PDFs need OCR, which is not currently built in. The processed CSV is used by the separate training and command-line workflows, not by the web demo.

### A PDF produces no resume text

The PDF may be image-based or scanned. Convert it to a text-searchable PDF or provide a DOCX/TXT resume; OCR is not currently built in.

## License

No LICENSE file is currently included in the project directory. Confirm the intended license before redistributing this project or its dataset.
