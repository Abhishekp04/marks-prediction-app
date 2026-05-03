import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("student_model.pkl")

st.title("🎓 Student Marks Predictor")

# Inputs
study = st.number_input("Enter study hours", min_value=0.0)
attendance = st.number_input("Enter attendance", min_value=0.0)
sleep = st.number_input("Enter sleep hours", min_value=0.0)

# Button
if st.button("Predict"):
    inp = pd.DataFrame([[study, attendance, sleep]],
                       columns=["study_hours","attendance","sleep_hours"])
    
    result = model.predict(inp)
    
    st.success(f"Predicted Marks: {round(result[0],2)}")