# Resume Screening App

This project is a simple resume screening web application built using Python and Streamlit.

It allows users to upload a resume (PDF, DOCX, or TXT), extracts the text, and predicts the job category using a trained machine learning model.
## LIVE LINK!
https://ml-resume-classifier-bgtr57ptfkw8cx8pifzqox.streamlit.app/
## Features
- Upload resume files
- Extract text from PDF, DOCX, and TXT
- Clean and process resume content
- Predict job category using ML model

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   python -m streamlit run app.py

3. Open in browser:
   http://localhost:8501

## Project Structure
- app.py → main application
- clf.pkl → trained model
- tfidf.pkl → vectorizer
- encoder.pkl → label encoder
- resume_model_training.ipynb → training notebook

## Note
Make sure all `.pkl` files are in the same folder as `app.py` before running.

---

This project was created as part of a learning / final-year project.
