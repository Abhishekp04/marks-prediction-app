import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Marks Predictor", page_icon="🎓", layout="centered")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.title {
    text-align: center;
    color: #00ffcc;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 18px;
    margin-bottom: 30px;
}

.stNumberInput input {
    border-radius: 10px;
    border: 2px solid #000;
}

.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.result {
    text-align: center;
    font-size: 28px;
    color: #00ffcc;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("student_model.pkl")

# Title
st.markdown('<div class="title">🎓 Student Marks Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter student details to predict marks</div>', unsafe_allow_html=True)

# Inputs
study = st.number_input("📘 Study Hours", min_value=0.0, max_value=10.0, step=0.5)
attendance = st.number_input("📅 Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
sleep = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, step=0.5)

# Button
if st.button("🚀 Predict Marks"):
    inp = pd.DataFrame([[study, attendance, sleep]],
                       columns=["study_hours","attendance","sleep_hours"])
    
    result = model.predict(inp)
    
    st.markdown(f'<div class="result">🎯 Expected Marks: {round(result[0],2)} / 100</div>', unsafe_allow_html=True)
