import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model
model = joblib.load("aqi_model.pkl")

# Load feature columns
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

st.title("🌫️ Air Quality Prediction (AQI) App")
st.write("Predict next-day AQI using your trained LightGBM model")

# Input fields
st.sidebar.header("Enter pollutant values")

inputs = {}
for col in feature_columns:
    inputs[col] = st.sidebar.number_input(col, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Predict
if st.button("Predict AQI"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted AQI: **{prediction:.2f}**")

    # AQI Bucket
    if prediction <= 50:
        bucket = "Good"
        color = "green"
    elif prediction <= 100:
        bucket = "Satisfactory"
        color = "yellow"
    elif prediction <= 200:
        bucket = "Moderate"
        color = "orange"
    elif prediction <= 300:
        bucket = "Poor"
        color = "red"
    elif prediction <= 400:
        bucket = "Very Poor"
        color = "purple"
    else:
        bucket = "Severe"
        color = "maroon"

    st.markdown(f"<h3 style='color:{color}'>AQI Bucket: {bucket}</h3>", unsafe_allow_html=True)

