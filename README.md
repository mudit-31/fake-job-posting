🧠 Fake Job Posting Detection

A machine learning web app that detects fake or fraudulent job postings based on job description text.
This project uses Natural Language Processing (NLP) and a Logistic Regression classifier to identify scams and protect job seekers.

🚀 Features

Predicts whether a job posting is Real or Fake

Interactive web interface built with Streamlit

Uses TF-IDF Vectorization for text processing

Real-time predictions with confidence levels

Reported test accuracy: ~96%

🧰 Technologies

Python 3.8+

Streamlit

scikit-learn

pandas

numpy

regex (for text cleaning)

⚙️ Installation & Usage

Prerequisites: Python 3.8 or later, pip

Clone the repository

git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector


Install dependencies

pip install -r requirements.txt


Place the dataset CSV (Fake Job Posting Prediction) in the project root (same folder as app.py) or update the path in the app.

Run the Streamlit app

streamlit run app.py


Open the displayed local URL (usually http://localhost:8501
) in your browser.

📂 Dataset

Dataset used: Fake Job Posting Prediction (Kaggle)
Link: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-job-posting-dataset

Make sure the CSV filename matches what app.py expects (e.g., fake_job_postings.csv) or update the code accordingly.

The dataset contains ~18,000 job postings, each labeled as real (0) or fake (1).

🧠 Model Details

Algorithm: Logistic Regression

Feature Extraction: TF-IDF Vectorizer

Train/Test Split: 80% / 20%

Balancing: Real postings are downsampled to match fake postings

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

Typical Accuracy: ~96% (on test split)

The model learns from textual cues like word patterns, tone, and structure in job descriptions to distinguish real postings from fraudulent ones.

🧹 Data Preprocessing

Filled missing text fields with empty strings

Merged text features (title, company_profile, description) into a single field

Removed special characters, digits, and URLs

Applied lowercase normalization and token cleaning

Transformed text using TF-IDF Vectorization

Encoded target label (fraudulent) as 0/1

Balanced the dataset to prevent bias

✅ Suggestions / Improvements

Use advanced models (e.g., BERT, RoBERTa) for deeper contextual understanding

Incorporate additional features such as company reputation or domain name

Use SMOTE or weighted loss for more robust class balancing

Deploy the app publicly on Streamlit Cloud or Hugging Face Spaces

Add visual analytics (word clouds, confusion matrices, etc.)

👨‍💻 Author

Mudit Sharma (Roll No: 16014223055)
Machine Learning Mini Project

📧 mudit.sharma@somaiya.edu
