# =========================================================
# Smart Career AI - Full Flask App
# =========================================================

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import joblib
import pdfplumber
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)

# =========================================================
# Upload Folder
# =========================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs("static/charts", exist_ok=True)

# =========================================================
# Load Models
# =========================================================

salary_model = joblib.load(
    "models/salary_model.pkl"
)

resume_model = joblib.load(
    "models/resume_classifier.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# =========================================================
# Load Dataset
# =========================================================

jobs_df = pd.read_csv(
    "datasets/dice_com-job_us_sample.csv"
)

jobs_df = jobs_df[[
    "jobtitle",
    "company",
    "skills"
]]

jobs_df = jobs_df.dropna()

jobs_df["combined_features"] = (
    jobs_df["jobtitle"].astype(str)
    + " "
    + jobs_df["skills"].astype(str)
)

# =========================================================
# TF-IDF Recommendation
# =========================================================

job_tfidf = TfidfVectorizer(
    stop_words="english"
)

job_tfidf_matrix = job_tfidf.fit_transform(
    jobs_df["combined_features"]
)

# =========================================================
# Skills List
# =========================================================

skills_list = [

    "python",
    "machine learning",
    "data science",
    "sql",
    "flask",
    "django",
    "tensorflow",
    "pytorch",
    "react",
    "javascript",
    "html",
    "css",
    "java",
    "c++",
    "deep learning",
    "nlp",
    "power bi",
    "tableau",
    "excel",
    "mongodb",
    "mysql"
]

# =========================================================
# Dashboard Analytics Storage
# =========================================================

all_resume_scores = []

all_categories = []

all_skills = []

# =========================================================
# Job Recommendation Function
# =========================================================

def recommend_jobs(resume_text):

    input_vector = job_tfidf.transform(
        [resume_text]
    )

    similarity = cosine_similarity(
        input_vector,
        job_tfidf_matrix
    )

    similar_jobs = similarity.argsort()[0][-5:]

    recommendations = jobs_df.iloc[
        similar_jobs
    ][[
        "jobtitle",
        "company"
    ]]

    return recommendations

# =========================================================
# Skill Extraction
# =========================================================

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_list:

        if skill in text:

            found_skills.append(skill)

    return found_skills

# =========================================================
# Resume Score
# =========================================================

def calculate_resume_score(skills):

    score = len(skills) * 5

    if score > 100:

        score = 100

    return score

# =========================================================
# Home Route
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")

# =========================================================
# Salary Prediction
# =========================================================

@app.route(
    "/predict_salary_form",
    methods=["POST"]
)
def predict_salary_form():

    experience = int(
        request.form["experience"]
    )

    employment = int(
        request.form["employment"]
    )

    remote = int(
        request.form["remote"]
    )

    company = int(
        request.form["company"]
    )

    prediction = salary_model.predict([
        [
            experience,
            employment,
            remote,
            company
        ]
    ])

    return f"""

    <html>

    <head>

    <link href=
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
    rel="stylesheet">

    </head>

    <body class='bg-dark text-white'>

    <div class='container mt-5'>

        <div class='card p-5 shadow'>

            <h1 class='text-center text-success'>
                Salary Prediction
            </h1>

            <hr>

            <h2 class='text-center'>
                Predicted Salary:
            </h2>

            <h1 class='text-center text-primary'>
                ${prediction[0]:,.2f}
            </h1>

            <div class='text-center mt-4'>

                <a href='/'
                class='btn btn-dark'>
                    Back Home
                </a>

            </div>

        </div>

    </div>

    </body>

    </html>

    """

# =========================================================
# Resume Upload
# =========================================================

@app.route(
    "/upload_resume",
    methods=["POST"]
)
def upload_resume():

    file = request.files["resume_file"]

    if file.filename == "":

        return """
        <h2>No File Selected</h2>
        """

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    text = ""

    # =====================================================
    # Read PDF
    # =====================================================

    try:

        with pdfplumber.open(filepath) as pdf:

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:

                    text += extracted

    except Exception as e:

        return f"""
        <h2>Error Reading PDF</h2>
        <p>{str(e)}</p>
        """

    # =====================================================
    # Resume Classification
    # =====================================================

    transformed = tfidf.transform([text])

    prediction = resume_model.predict(
        transformed
    )

    # =====================================================
    # Job Recommendation
    # =====================================================

    recommended_jobs = recommend_jobs(text)

    # =====================================================
    # Skill Extraction
    # =====================================================

    extracted_skills = extract_skills(text)

    # =====================================================
    # Resume Score
    # =====================================================

    resume_score = calculate_resume_score(
        extracted_skills
    )

    # =====================================================
    # Store Analytics
    # =====================================================

    all_resume_scores.append(resume_score)

    all_categories.append(prediction[0])

    all_skills.extend(extracted_skills)

    # =====================================================
    # Skills HTML
    # =====================================================

    skills_html = ""

    for skill in extracted_skills:

        skills_html += f"""
        <span class='badge bg-primary m-1 p-2'>
            {skill}
        </span>
        """

    # =====================================================
    # Jobs HTML
    # =====================================================

    jobs_html = ""

    for _, row in recommended_jobs.iterrows():

        jobs_html += f"""

        <li class='list-group-item'>

            <strong>{row['jobtitle']}</strong>

            <br>

            {row['company']}

        </li>

        """

    return f"""

    <html>

    <head>

    <link href=
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
    rel="stylesheet">

    </head>

    <body class='bg-light'>

    <div class='container mt-5'>

        <div class='card shadow-lg p-5'>

            <h1 class='text-center text-success'>
                Resume Analysis Result
            </h1>

            <hr>

            <h3>
                Resume Category:
            </h3>

            <h2 class='text-primary'>
                {prediction[0]}
            </h2>

            <hr>

            <h3>
                Resume Score:
            </h3>

            <div class='progress mb-4'>

                <div class='progress-bar progress-bar-striped
                progress-bar-animated bg-success'

                style='width:{resume_score}%'>
                    {resume_score}%
                </div>

            </div>

            <hr>

            <h3>Extracted Skills</h3>

            {skills_html}

            <hr>

            <h3>Recommended Jobs</h3>

            <ul class='list-group'>
                {jobs_html}
            </ul>

            <div class='text-center mt-5'>

                <a href='/dashboard'
                class='btn btn-dark btn-lg'>

                    View Dashboard

                </a>

            </div>

        </div>

    </div>

    </body>

    </html>

    """

# =========================================================
# Dashboard
# =========================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    total_resumes = len(all_resume_scores)

    avg_score = 0

    if total_resumes > 0:

        avg_score = sum(all_resume_scores) / total_resumes

    # =====================================================
    # Most Common Category
    # =====================================================

    top_category = "No Data"

    if len(all_categories) > 0:

        top_category = max(
            set(all_categories),
            key=all_categories.count
        )

    # =====================================================
    # Skill Frequency
    # =====================================================

    skill_counts = {}

    for skill in all_skills:

        if skill in skill_counts:

            skill_counts[skill] += 1

        else:

            skill_counts[skill] = 1

    # =====================================================
    # Generate Chart
    # =====================================================

    if len(skill_counts) > 0:

        plt.figure(figsize=(10, 5))

        plt.bar(
            skill_counts.keys(),
            skill_counts.values()
        )

        plt.xticks(rotation=45)

        plt.title("Top Skills")

        plt.tight_layout()

        chart_path = "static/charts/skills_chart.png"

        plt.savefig(chart_path)

        plt.close()

    return render_template(
        "dashboard.html",

        total_resumes=total_resumes,

        avg_score=round(avg_score, 2),

        top_category=top_category
    )

# =========================================================
# Run Flask App
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )