import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Load and prepare dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("fake_job_postings.csv")
    df = df.fillna('')
    df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']
    X = df['text']
    y = df['fraudulent']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return vectorizer, model, accuracy

vectorizer, model, accuracy = load_data()

# --- App UI ---
st.title("🧠 Fake Job Posting Detection App")
st.write("Enter a job description and find out if it's **real or fake!**")

user_input = st.text_area("Paste job description here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        text_tfidf = vectorizer.transform([user_input])
        pred = model.predict(text_tfidf)[0]
        if pred == 1:
            st.error("🚨 This looks like a **Fake Job Posting**!")
        else:
            st.success("✅ This seems like a **Real Job Posting**!")

st.sidebar.header("Model Info")
st.sidebar.write(f"Model Accuracy: **{accuracy*100:.2f}%**")
st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Feature Extraction: TF-IDF Vectorizer")
