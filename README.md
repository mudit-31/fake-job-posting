# 🧠 Fake Job Posting Detection

A machine learning web app that detects **fake or fraudulent job postings** based on job description text.  
This project uses **Natural Language Processing (NLP)** and a **Naive Bayes classifier** to identify scams and protect job seekers.

---

## 🚀 Features
- Predicts whether a job posting is **Real** or **Fake**
- Interactive web interface built with **Streamlit**
- Uses **TF-IDF Vectorization** for text processing
- Reported test accuracy: ~95%

---

## 🧰 Technologies
- Python 3.8+  
- Streamlit  
- scikit-learn  
- pandas  
- numpy  

---

## ⚙️ Installation & Usage

Prerequisites: Python 3.8 or later, pip

1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Place the dataset CSV (Fake Job Posting Prediction) in the project root (same folder as `app.py`) or update the path in the app.

4. Run the Streamlit app
```bash
streamlit run app.py
```

Open the displayed local URL (usually http://localhost:8501) in your browser.

---

## 📂 Dataset

Dataset used: Fake Job Posting Prediction (Kaggle)  
Link: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-job-posting-dataset (or your dataset source).  
Make sure the CSV filename matches what `app.py` expects (e.g., `fake_job_postings.csv`) or update the code.

---

## 🧠 Model Details

- Algorithm: Multinomial Naive Bayes  
- Feature Extraction: TF-IDF Vectorizer  
- Train/Test Split: 80% / 20%  
- Typical Accuracy: ~95% (on the test split used)

If you save the trained model (e.g., `model.pkl` and `tfidf.pkl`), include those files in the repo or describe how to recreate them via a training script.

---

## 🧹 Data Preprocessing

- Fill missing values with empty strings  
- Combine text fields (title, company_profile, description, requirements, benefits) into one feature  
- Apply TF-IDF vectorization  
- Encode target label (fraudulent) as 0/1  
- Train/test split and train MultinomialNB

---

## 🔧 Example

Start app and paste a job description in the web UI. The app will show a prediction (Real / Fake) and the model confidence.

If you have a CLI or script for predictions, show sample usage here.

---

## ✅ Suggestions / Improvements

- Use transformer models (BERT, RoBERTa) for better contextual understanding  
- Add more metadata features (company info, logo, remote flag)  
- Add unit tests for preprocessing and model pipeline  
- Deploy publicly (Streamlit Cloud, Heroku, Vercel)

---

## 📄 License
Add a license (e.g., MIT) or specify project license.

---

## 👨‍💻 Author
Mudit Sharma (Roll No: 16014223055)  
Machine Learning Mini Project – College Assignment

---

## ✉️ Contact
Include an email or GitHub profile for questions or contributions.
