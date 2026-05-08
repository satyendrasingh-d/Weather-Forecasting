# ============================================
# weather_app.py
# Streamlit UI for Weather Forecasting Project
# RNN vs LSTM (10-Day Multi-Feature Forecast)
# ============================================

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
st.markdown(
    """
    Predict the next **10 days weather forecast**
    using **RNN** and **LSTM** models.
    """
)

# ============================================
# LOAD MODELS AND SCALER
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


# Check files
required_files = [
    "weather_rnn_model.h5",
    "weather_lstm_model.h5",
    "scaler.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file not found: {file}")
        st.stop()

# Load resources
rnn_model, lstm_model = load_models()
scaler = load_scaler()

# ============================================
# OPENWEATHER API
# ============================================

API_KEY = "f530270436ed7cd9a06324c89c953281"


def get_live_weather(city):
    """
    Returns:
    [temperature, humidity, pressure, wind_speed, visibility]
    """
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={API_KEY}&units=metric"
        )

        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            return None, data.get("message", "Unknown error")

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]

        # Visibility in meters -> kilometers
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
# SIDEBAR
# ============================================

st.sidebar.header("⚙️ Settings")

city = st.sidebar.text_input("Enter City Name", "Delhi")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Both", "RNN", "LSTM"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📌 Features Predicted
    - 🌡️ Temperature
    - 💧 Humidity
    - 🌍 Pressure
    - 🌬️ Wind Speed
    - 👀 Visibility
    """
)

# ============================================
# GET FORECAST BUTTON
# ============================================

if st.button("🚀 Generate 10-Day Forecast"):

    # ---------------- LIVE WEATHER ----------------
    live_features, raw_data = get_live_weather(city)

    if live_features is None:
        st.error(f"❌ API Error: {raw_data}")
        st.stop()

    # ---------------- CURRENT WEATHER ----------------
    st.subheader(f"🌍 Current Weather in {city.title()}")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🌡️ Temp", f"{live_features[0]:.2f} °C")
    col2.metric("💧 Humidity", f"{live_features[1]:.2f} %")
    col3.metric("🌍 Pressure", f"{live_features[2]:.2f} hPa")
    col4.metric("🌬️ Wind", f"{live_features[3]:.2f} m/s")
    col5.metric("👀 Visibility", f"{live_features[4]:.2f} km")

    # ---------------- CREATE INPUT SEQUENCE ----------------
    # Repeat current live data 30 times
    # (Simple approach for live forecasting)
    dummy_sequence = np.tile(live_features, (30, 1))

    scaled_seq = scaler.transform(dummy_sequence)
    scaled_seq = scaled_seq.reshape(1, 30, 5)

    # ---------------- PREDICTIONS ----------------
    rnn_pred = rnn_model.predict(scaled_seq, verbose=0)
    lstm_pred = lstm_model.predict(scaled_seq, verbose=0)

    # Reshape to (10, 5)
    rnn_pred = rnn_pred.reshape(10, 5)
    lstm_pred = lstm_pred.reshape(10, 5)

    # Inverse scaling
    rnn_forecast = scaler.inverse_transform(rnn_pred)
    lstm_forecast = scaler.inverse_transform(lstm_pred)

    # ---------------- DATE COLUMN ----------------
    today = datetime.date.today()

    dates = [
        (today + datetime.timedelta(days=i)).strftime("%d-%b-%Y")
        for i in range(1, 11)
    ]

    # ---------------- DATAFRAME ----------------
    if model_choice == "RNN":
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Temperature (°C)": rnn_forecast[:, 0],
            "Humidity (%)": rnn_forecast[:, 1],
            "Pressure (hPa)": rnn_forecast[:, 2],
            "Wind Speed": rnn_forecast[:, 3],
            "Visibility (km)": rnn_forecast[:, 4]
        })

    elif model_choice == "LSTM":
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Temperature (°C)": lstm_forecast[:, 0],
            "Humidity (%)": lstm_forecast[:, 1],
            "Pressure (hPa)": lstm_forecast[:, 2],
            "Wind Speed": lstm_forecast[:, 3],
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

    # ---------------- SHOW TABLE ----------------
    st.subheader("📅 10-Day Forecast")
    st.dataframe(forecast_df.round(2), use_container_width=True)

    # ---------------- TEMPERATURE GRAPH ----------------
    st.subheader("📈 Temperature Forecast Comparison")

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

    # ---------------- DOWNLOAD CSV ----------------
    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name=f"{city.lower()}_weather_forecast.csv",
        mime="text/csv"
    )

# ============================================
# SIDEBAR ABOUT
# ============================================

st.sidebar.markdown("---")
st.sidebar.header("📌 About Project")
st.sidebar.write(
    """
    This project compares **RNN** and **LSTM**
    models for multivariate weather forecasting.

    **Input:** Past 30 days weather data  
    **Output:** Next 10 days forecast

    **Predicted Features:**
    - Temperature
    - Humidity
    - Pressure
    - Wind Speed
    - Visibility
    """
)
