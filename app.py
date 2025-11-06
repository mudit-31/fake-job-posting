import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# ==================================
# 🎛️ Streamlit Page Configuration
# ==================================
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Custom CSS for a cleaner UI
st.markdown("""
    <style>
    body { background-color: #f8f9fa; }
    .title { text-align:center; font-size:40px; font-weight:700; color:#2b2b2b; }git 
    .subtitle { text-align:center; font-size:18px; color:#5a5a5a; margin-bottom:30px; }
    .stTextInput, .stTextArea { border-radius: 10px !important; }
    .result-box {
        border-radius: 12px;
        padding: 20px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    .fake { background-color: #ffe6e6; color: #b30000; }
    .real { background-color: #e6ffe6; color: #006600; }
    </style>
""", unsafe_allow_html=True)

# ==================================
# 🧠 Title Section
# ==================================
st.markdown('<div class="title">🧠 Fake Job Posting Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a job post is Real or Fake using Machine Learning</div>', unsafe_allow_html=True)

st.sidebar.image("logo2.jpeg", use_column_width=True)
st.sidebar.header("📊 Project Summary")
st.sidebar.info("""
This ML model detects **fake job postings** by analyzing job titles and descriptions.  
It uses **TF-IDF + Logistic Regression** trained on a balanced dataset.
""")

# ==================================
# 🧩 Data Preparation
# ==================================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("fake_job_postings.csv")

    df_fake = df[df['fraudulent'] == 1]
    df_real = df[df['fraudulent'] == 0].sample(n=len(df_fake) * 2, random_state=42)
    df_bal = pd.concat([df_fake, df_real])

    def clean_text(t):
        t = re.sub(r"http\S+|www\S+|@\S+|\d+", "", str(t))
        return re.sub(r"[^a-zA-Z ]", " ", t.lower())

    df_bal['text'] = (df_bal['title'].fillna('') + ' ' + df_bal['description'].fillna('')).apply(clean_text)
    return df_bal

# ==================================
# ⚙️ Model Training
# ==================================
@st.cache_resource
def train_model():
    df_bal = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df_bal['text'], df_bal['fraudulent'], test_size=0.2, stratify=df_bal['fraudulent'], random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)

    # Sidebar metrics
    st.sidebar.subheader("📈 Model Performance")
    st.sidebar.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
    st.sidebar.metric("Fake Precision", f"{report['Fake']['precision']*100:.2f}%")
    st.sidebar.metric("Fake Recall", f"{report['Fake']['recall']*100:.2f}%")

    return model, vectorizer

model, vectorizer = train_model()

# ==================================
# 🧾 User Input Section
# ==================================
st.header("🔍 Enter Job Posting Details")

col1, col2 = st.columns([1, 1])
with col1:
    title = st.text_input("Job Title", "Software Developer Intern")
with col2:
    company_profile = st.text_input("Company Name or Profile", "Global Tech Solutions Pvt. Ltd.")

description = st.text_area("Job Description", "Work from home opportunity, no experience needed, apply now!")

# ==================================
# 🔮 Prediction Logic
# ==================================
def clean_text(t):
    t = re.sub(r"http\S+|www\S+|@\S+|\d+", "", str(t))
    return re.sub(r"[^a-zA-Z ]", " ", t.lower())

if st.button("🚀 Predict", use_container_width=True):
    input_text = f"{title} {description} {company_profile}"
    input_clean = [clean_text(input_text)]
    input_tfidf = vectorizer.transform(input_clean)

    with st.spinner("Analyzing job posting..."):
        time.sleep(2)
        prediction = model.predict(input_tfidf)[0]
        prob = model.predict_proba(input_tfidf)[0]

    st.markdown("### 🧩 Prediction Result")
    if prediction == 1:
        st.markdown(f'<div class="result-box fake">⚠️ This job posting seems <b>FAKE</b> ({prob[1]*100:.2f}% confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box real">✅ This job posting seems <b>REAL</b> ({prob[0]*100:.2f}% confidence)</div>', unsafe_allow_html=True)

    # Confidence breakdown
    st.progress(int(prob[1]*100) if prediction == 1 else int(prob[0]*100))
    st.write(f"**Fake Probability:** {prob[1]*100:.2f}% | **Real Probability:** {prob[0]*100:.2f}%")

# ==================================
# 👣 Footer
# ==================================
st.markdown("---")
st.markdown("""
**👨‍💻 Developed by:** Mudit Sharma  
📧 [mudit.sharma@somaiya.edu](mailto:mudit.sharma@somaiya.edu)  
🎓 Roll No: 16014223055  
💡 Machine Learning Mini Project
""")
import streamlit as st
import os

with st.sidebar:
    if os.path.exists("logo2.jpeg"):
        st.image("logo2.jpeg", use_container_width=True)
    else:
        st.warning("Logo not found — please add 'logo2.jpeg' to the project folder.")
    st.title("🧠 Fake Job Posting Detector")
    st.write("Detect fraudulent job postings using Machine Learning")
