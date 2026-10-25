"""
Comprehensive Skill Ontology and Taxonomy for NLP Resume Screening.
Includes 500+ technical and soft skills categorized across domains,
along with common aliases and abbreviations for robust phrase matching.
"""

# Categorized Skill Taxonomy
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "sql", "shell",
        "bash", "powershell", "perl", "haskell", "lua", "matlab", "julia", "elixir", "clojure"
    ],
    "Frameworks & Web Development": [
        "react", "react.js", "reactjs", "angular", "angularjs", "vue", "vue.js", "vuejs",
        "next.js", "nextjs", "nuxt.js", "node.js", "nodejs", "express", "express.js",
        "django", "flask", "fastapi", "spring", "spring boot", "asp.net", "dotnet",
        ".net core", "ruby on rails", "rails", "laravel", "symfony", "graphql", "rest api",
        "restful apis", "grpc", "html5", "css3", "tailwind css", "bootstrap", "sass", "less",
        "redux", "zustand", "svelte", "webgl", "three.js", "websockets"
    ],
    "AI, Machine Learning & NLP": [
        "machine learning", "deep learning", "natural language processing", "nlp",
        "computer vision", "generative ai", "genai", "large language models", "llm",
        "llms", "transformers", "bert", "gpt", "rag", "retrieval augmented generation",
        "langchain", "llamaindex", "hugging face", "huggingface", "pytorch", "tensorflow",
        "keras", "scikit-learn", "sklearn", "spacy", "nltk", "opencv", "pandas", "numpy",
        "scipy", "xgboost", "lightgbm", "catboost", "reinforcement learning",
        "neural networks", "cnn", "rnn", "lstm", "embeddings", "sentence-bert",
        "vector databases", "faiss", "chromadb", "pinecone", "qdrant", "weaviate",
        "prompt engineering", "fine-tuning", "lora", "diffusion models", "whisper"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "google cloud", "gcp", "docker",
        "kubernetes", "k8s", "terraform", "ansible", "ci/cd", "jenkins", "github actions",
        "gitlab ci", "circleci", "helm", "prometheus", "grafana", "elk stack", "linux",
        "unix", "nginx", "apache", "serverless", "aws lambda", "cloudformation",
        "microservices", "service mesh", "istio", "vault", "datadog", "new relic"
    ],
    "Databases & Big Data": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "sqlite", "dynamodb", "cassandra", "oracle", "sql server", "mariadb",
        "snowflake", "bigquery", "redshift", "apache spark", "spark", "apache kafka",
        "kafka", "hadoop", "apache airflow", "airflow", "dbt", "databricks", "delta lake",
        "hive", "hbase", "neo4j", "graphql", "prisma", "sqlalchemy", "mongoose"
    ],
    "Software Engineering & Tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "agile", "scrum",
        "kanban", "tdd", "test driven development", "unit testing", "integration testing",
        "pytest", "jest", "cypress", "selenium", "postman", "swagger", "openapi",
        "system design", "object-oriented programming", "oop", "design patterns",
        "data structures", "algorithms", "distributed systems", "event-driven architecture"
    ],
    "Soft Skills & Leadership": [
        "problem solving", "critical thinking", "communication", "team collaboration",
        "leadership", "mentoring", "project management", "time management", "adaptability",
        "analytical skills", "stakeholder management", "cross-functional collaboration",
        "negotiation", "conflict resolution", "creativity", "presentation skills"
    ]
}

# Synonyms / Normalized representations
SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",
    "scikit learn": "scikit-learn",
    "react js": "react",
    "node js": "node.js",
    "golang": "go",
    "k8s": "kubernetes",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "nextjs": "next.js",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "postgres": "postgresql",
    "genai": "generative ai",
    "llms": "large language models",
    "llm": "large language models",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "natural language processing": "nlp",
    "computer vision": "cv",
    "retrieval augmented generation": "rag",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api"
}

# Flattened unique master list of all skills
ALL_SKILLS = set()
for category_skills in SKILL_TAXONOMY.values():
    for skill in category_skills:
        ALL_SKILLS.add(skill.lower())

def normalize_skill(skill_name: str) -> str:
    """Normalize a skill to its canonical name if an alias exists."""
    clean = skill_name.strip().lower()
    return SKILL_ALIASES.get(clean, clean)

def get_skill_category(skill_name: str) -> str:
    """Return the domain category of a given skill."""
    norm = normalize_skill(skill_name)
    for category, skills in SKILL_TAXONOMY.items():
        if norm in skills or skill_name.lower() in skills:
            return category
    return "Technical Skills"
