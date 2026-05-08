# `weather_app.py` (Realistic Forecast Version)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
import datetime
import os

from tensorflow.keras.models import load_model

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Forecasting System")
st.write("Predict the next 10 days weather using RNN and LSTM models.")

# ============================================
# SETTINGS
# ============================================
API_KEY = "YOUR_OPENWEATHER_API_KEY"

FEATURES = [
    'Temperature (C)',
    'Humidity',
    'Pressure (millibars)',
    'Wind Speed (km/h)',
    'Visibility (km)'
]

# ============================================
# FILE CHECK
# ============================================
required_files = [
    "weatherHistory.csv",
    "weather_rnn_model.h5",
    "weather_lstm_model.h5",
    "scaler.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file not found: {file}")
        st.stop()

# ============================================
# LOAD RESOURCES
# ============================================
@st.cache_resource
def load_models():
    rnn_model = load_model("weather_rnn_model.h5", compile=False)
    lstm_model = load_model("weather_lstm_model.h5", compile=False)
    return rnn_model, lstm_model


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler


@st.cache_data
def load_history():
    df = pd.read_csv("weatherHistory.csv")
    return df[FEATURES].dropna()


rnn_model, lstm_model = load_models()
scaler = load_scaler()
history_df = load_history()

# ============================================
# LIVE WEATHER FUNCTION
# ============================================
def get_live_weather(city):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={API_KEY}&units=metric"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("cod") != 200:
            return None, data.get("message", "Unknown error")

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]

        # OpenWeather wind speed = m/s, convert to km/h
        wind_speed = data["wind"]["speed"] * 3.6

        # Visibility meter -> km
        visibility = data.get("visibility", 10000) / 1000

        features = [
            temperature,
            humidity,
            pressure,
            wind_speed,
            visibility
        ]

        return features, data

    except Exception as e:
        return None, str(e)

# ============================================
# CLIP FORECAST TO REALISTIC RANGE
# ============================================
def clip_forecast(forecast):
    forecast = forecast.copy()

    # Temperature (°C)
    forecast[:, 0] = np.clip(forecast[:, 0], -10, 50)

    # Humidity (%)
    forecast[:, 1] = np.clip(forecast[:, 1], 0, 100)

    # Pressure (hPa)
    forecast[:, 2] = np.clip(forecast[:, 2], 950, 1050)

    # Wind Speed (km/h)
    forecast[:, 3] = np.clip(forecast[:, 3], 0, 150)

    # Visibility (km)
    forecast[:, 4] = np.clip(forecast[:, 4], 0, 20)

    return forecast

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Settings")
city = st.sidebar.text_input("Enter City Name", "Delhi")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Both", "RNN", "LSTM"]
)

# ============================================
# MAIN BUTTON
# ============================================
if st.button("🚀 Generate 10-Day Forecast"):

    if API_KEY == "YOUR_OPENWEATHER_API_KEY":
        st.error("❌ Please add your OpenWeather API key in API_KEY variable.")
        st.stop()

    # ========================================
    # GET LIVE WEATHER
    # ========================================
    live_features, raw_data = get_live_weather(city)

    if live_features is None:
        st.error(f"❌ API Error: {raw_data}")
        st.stop()

    # ========================================
    # SHOW CURRENT WEATHER
    # ========================================
    st.subheader(f"🌍 Current Weather in {city.title()}")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("🌡️ Temp", f"{live_features[0]:.2f} °C")
    c2.metric("💧 Humidity", f"{live_features[1]:.2f} %")
    c3.metric("🌍 Pressure", f"{live_features[2]:.2f} hPa")
    c4.metric("🌬️ Wind", f"{live_features[3]:.2f} km/h")
    c5.metric("👀 Visibility", f"{live_features[4]:.2f} km")

    # ========================================
    # BUILD REALISTIC INPUT SEQUENCE
    # last 29 historical days + today's live data
    # ========================================
    scaled_history = scaler.transform(history_df)
    last_29_days = scaled_history[-29:]

    live_scaled = scaler.transform([live_features])

    final_seq = np.vstack([last_29_days, live_scaled])
    final_seq = final_seq.reshape(1, 30, 5)

    # ========================================
    # PREDICTIONS
    # ========================================
    rnn_pred = rnn_model.predict(final_seq, verbose=0)
    lstm_pred = lstm_model.predict(final_seq, verbose=0)

    rnn_pred = rnn_pred.reshape(10, 5)
    lstm_pred = lstm_pred.reshape(10, 5)

    # Inverse transform
    rnn_forecast = scaler.inverse_transform(rnn_pred)
    lstm_forecast = scaler.inverse_transform(lstm_pred)

    # Clip to realistic ranges
    rnn_forecast = clip_forecast(rnn_forecast)
    lstm_forecast = clip_forecast(lstm_forecast)

    # ========================================
    # DATE LIST
    # ========================================
    today = datetime.date.today()
    dates = [
        (today + datetime.timedelta(days=i)).strftime("%d-%b-%Y")
        for i in range(1, 11)
    ]

    # ========================================
    # CREATE DATAFRAME
    # ========================================
    if model_choice == "RNN":
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Temperature (°C)": rnn_forecast[:, 0],
            "Humidity (%)": rnn_forecast[:, 1],
            "Pressure (hPa)": rnn_forecast[:, 2],
            "Wind Speed (km/h)": rnn_forecast[:, 3],
            "Visibility (km)": rnn_forecast[:, 4]
        })

    elif model_choice == "LSTM":
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Temperature (°C)": lstm_forecast[:, 0],
            "Humidity (%)": lstm_forecast[:, 1],
            "Pressure (hPa)": lstm_forecast[:, 2],
            "Wind Speed (km/h)": lstm_forecast[:, 3],
            "Visibility (km)": lstm_forecast[:, 4]
        })

    else:
        forecast_df = pd.DataFrame({
            "Date": dates,
            "RNN Temp": rnn_forecast[:, 0],
            "LSTM Temp": lstm_forecast[:, 0],
            "RNN Humidity": rnn_forecast[:, 1],
            "LSTM Humidity": lstm_forecast[:, 1],
            "RNN Pressure": rnn_forecast[:, 2],
            "LSTM Pressure": lstm_forecast[:, 2],
            "RNN Wind": rnn_forecast[:, 3],
            "LSTM Wind": lstm_forecast[:, 3],
            "RNN Visibility": rnn_forecast[:, 4],
            "LSTM Visibility": lstm_forecast[:, 4]
        })

    # ========================================
    # SHOW TABLE
    # ========================================
    st.subheader("📅 10-Day Forecast")
    st.dataframe(forecast_df.round(2), use_container_width=True)

    # ========================================
    # TEMPERATURE GRAPH
    # ========================================
    st.subheader("📈 Temperature Forecast")

    fig, ax = plt.subplots(figsize=(12, 5))

    if model_choice in ["Both", "RNN"]:
        ax.plot(
            dates,
            rnn_forecast[:, 0],
            marker="o",
            linewidth=2,
            label="RNN"
        )

    if model_choice in ["Both", "LSTM"]:
        ax.plot(
            dates,
            lstm_forecast[:, 0],
            marker="o",
            linewidth=2,
            label="LSTM"
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"10-Day Temperature Forecast for {city.title()}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ========================================
    # DOWNLOAD CSV
    # ========================================
    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name=f"{city.lower()}_forecast.csv",
        mime="text/csv"
    )

# ============================================
# SIDEBAR ABOUT
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📌 About Project")
st.sidebar.write(
    """
    This project compares RNN and LSTM models for
    10-day multivariate weather forecasting.

    Predicted features:
    - Temperature
    - Humidity
    - Pressure
    - Wind Speed
    - Visibility

    Input used for prediction:
    - Last 29 historical days
    - Current live weather from OpenWeather API
    """
)

## Required Files in GitHub Repo

weather_app.py
requirements.txt
weatherHistory.csv
weather_rnn_model.h5
weather_lstm_model.h5
scaler.pkl

## `requirements.txt`


streamlit
pandas
numpy
matplotlib
requests
scikit-learn
tensorflow
h5py
