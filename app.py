
# app.py
import os
import subprocess
import sys
 
import joblib
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")
 
 
# Rebuild the data and model once each time the app starts, so old saved
# files from an earlier run are never reused.
@st.cache_resource
def prepare_files():
    subprocess.run([sys.executable, "train_model.py"], check=True)
    return True
 
 
with st.spinner("Preparing data and training the model (first load takes a minute)..."):
    prepare_files()
 
 
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")
 
 
model = load_model()
data = pd.read_csv("student_data.csv")
 
st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")
st.warning(
    "Predictions are not diagnoses and must not be the sole basis "
    "for real student decisions."
)
 
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))
 
st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
 
st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")
 
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)
 
uploaded = st.file_uploader("Choose student CSV", type=["csv"])
 
if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]
 
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]
        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )
