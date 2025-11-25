import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import timedelta

# ---------------------------
# CONFIG
# ---------------------------
API_URL = "https://api.waqi.info/feed/here/?token=bcc6181d5c0d142a0a51edbda26df33e17490739"

MODEL = joblib.load("aqi_model.pkl")
FEATURES = joblib.load("features.pkl")

LAGS = [1,3,6,12,24,48,72]
ROLLS = [3,12,24,72]

# ---------------------------
# API Fetcher
# ---------------------------
def fetch_live_aqi():
    try:
        r = requests.get(API_URL).json()
        if r["status"] != "ok":
            return None
        aqi = r["data"]["aqi"]
        dt  = pd.to_datetime(r["data"]["time"]["iso"])
        return pd.DataFrame([{"date": dt, "aqi": aqi}])
    except:
        return None

# ---------------------------
# Feature builder
# ---------------------------
def build_features(df):
    df = df.copy()
    df["aqi_clean"] = df["aqi"].astype(float).clip(10,350)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["aqi_clean"].shift(lag)

    for r in ROLLS:
        df[f"roll_{r}"] = df["aqi_clean"].rolling(r).mean()

    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    return df.dropna()

# ---------------------------
# FORECAST LOOP
# ---------------------------
def forecast(df, hrs):
    future = df.copy()

    for _ in range(hrs):
        X = future.iloc[-1][FEATURES].values.reshape(1, -1)
        pred = MODEL.predict(X)[0]

        new_row = future.iloc[-1:].copy()
        new_row["date"] = new_row["date"] + timedelta(hours=1)
        new_row["aqi_clean"] = pred

        for lag in LAGS:
            new_row[f"lag_{lag}"] = future["aqi_clean"].iloc[-lag]

        for r in ROLLS:
            new_row[f"roll_{r}"] = future["aqi_clean"].iloc[-r:].mean()

        future = pd.concat([future, new_row])

    return future.tail(hrs)[["date","aqi_clean"]]

# ---------------------------
# UI
# ---------------------------
st.title("🌎 Live AQI Forecasting")

predict_hours = st.slider("Predict next hours", 6, 72, 24)

# keep cached recent data
if "hist" not in st.session_state:
    st.session_state.hist = pd.DataFrame(columns=["date","aqi"])

# FETCH NEW
new_data = fetch_live_aqi()
if new_data is not None:
    st.session_state.hist = pd.concat([st.session_state.hist, new_data]).drop_duplicates("date")

hist = st.session_state.hist.sort_values("date")

st.subheader("Recent AQI Data (Live)")
st.dataframe(hist.tail(20))

# build features
df_feat = build_features(hist)

if len(df_feat) < 100:
    st.warning("Collect more live data to generate model features.")
else:
    result = forecast(df_feat, predict_hours)
    st.subheader(f"Predicted AQI Next {predict_hours} hours")
    st.dataframe(result)

    st.line_chart(result.set_index("date"))
