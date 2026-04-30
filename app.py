import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load models
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Clean text
def clean_resume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Prediction
def predict_category(resume_text):
    cleaned = clean_resume(resume_text)
    vector = tfidf.transform([cleaned])
    prediction = svc_model.predict(vector)

    try:
        confidence = max(svc_model.decision_function(vector)[0])
    except:
        confidence = None

    return le.inverse_transform(prediction)[0], confidence

# UI
st.set_page_config(page_title="Resume Classifier", layout="centered")

st.title("Resume Classification System")
st.write("Upload a resume (PDF, DOCX, or TXT) to predict its job category.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    try:
        st.write("File:", uploaded_file.name)

        # Extract text
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)
        else:
            text = extract_text_from_txt(uploaded_file)

        if not text.strip():
            st.error("Could not extract text from the file.")
            st.stop()

        st.success("Text extracted successfully.")

        if st.checkbox("Show Resume Text"):
            st.text_area("Resume Content", text, height=300)

        category, confidence = predict_category(text)

        st.subheader("Prediction Result")
        st.write("Category:", category)

        if confidence is not None:
            st.write("Confidence Score:", round(confidence, 2))

    except Exception as e:
        st.error(f"Error: {e}")