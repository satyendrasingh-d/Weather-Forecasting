
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
FEATURES = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed'
]

# ============================================
# CHECK REQUIRED FILES
# ============================================
required_files = [
    "delhi_weather_3years.csv",
    "weather_rnn_model.h5",
    "weather_lstm_model.h5",
    "scaler.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file not found: {file}")
        st.stop()

# ============================================
# LOAD MODELS
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
    df = pd.read_csv("delhi_weather_3years.csv")
    return df[FEATURES].dropna().reset_index(drop=True)


rnn_model, lstm_model = load_models()
scaler = load_scaler()
history_df = load_history()

# ============================================
# CITY COORDINATES
# ============================================
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Noida": (28.5355, 77.3910),
    "Gurgaon": (28.4595, 77.0266)
}

# ============================================
# OPEN-METEO LIVE WEATHER FUNCTION
# ============================================
def get_live_weather(city):
    if city not in CITY_COORDS:
        return None, f"Coordinates not available for {city}"

    lat, lon = CITY_COORDS[city]

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "surface_pressure,"
        "wind_speed_10m"
        "&timezone=Asia/Kolkata"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return None, data

        current = data["current"]

        features = [
            current["temperature_2m"],
            current["relative_humidity_2m"],
            current["surface_pressure"],
            current["wind_speed_10m"]
        ]

        return features, current

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

    return forecast

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Settings")
city = st.sidebar.selectbox("Select City", list(CITY_COORDS.keys()))

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Both", "RNN", "LSTM"]
)

# ============================================
# MAIN BUTTON
# ============================================
if st.button("🚀 Generate 10-Day Forecast"):

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
    st.subheader(f"🌍 Current Weather in {city}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🌡️ Temp", f"{live_features[0]:.2f} °C")
    c2.metric("💧 Humidity", f"{live_features[1]:.2f} %")
    c3.metric("🌍 Pressure", f"{live_features[2]:.2f} hPa")
    c4.metric("🌬️ Wind", f"{live_features[3]:.2f} km/h")

    # ========================================
    # BUILD INPUT SEQUENCE
    # last 29 historical days + current live weather
    # ========================================

    history_values = history_df[FEATURES].values
    scaled_history = scaler.transform(history_values)

    last_29_days = scaled_history[-29:]

    live_df = pd.DataFrame([live_features], columns=FEATURES)
    live_scaled = scaler.transform(live_df.values)

    final_seq = np.vstack([last_29_days, live_scaled])
    final_seq = final_seq.reshape(1, 30, 4)
    # ========================================
    # PREDICTIONS
    # ========================================
    rnn_pred = rnn_model.predict(final_seq, verbose=0)
    lstm_pred = lstm_model.predict(final_seq, verbose=0)

    rnn_pred = rnn_pred.reshape(10, 4)
    lstm_pred = lstm_pred.reshape(10, 4)

    # Inverse transform
    rnn_forecast = scaler.inverse_transform(rnn_pred)
    lstm_forecast = scaler.inverse_transform(lstm_pred)

    # Clip realistic values
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
            "Wind Speed (km/h)": rnn_forecast[:, 3]
        })

    elif model_choice == "LSTM":
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Temperature (°C)": lstm_forecast[:, 0],
            "Humidity (%)": lstm_forecast[:, 1],
            "Pressure (hPa)": lstm_forecast[:, 2],
            "Wind Speed (km/h)": lstm_forecast[:, 3]
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
            "LSTM Wind": lstm_forecast[:, 3]
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
        ax.plot(dates, rnn_forecast[:, 0], marker='o', linewidth=2, label='RNN')

    if model_choice in ["Both", "LSTM"]:
        ax.plot(dates, lstm_forecast[:, 0], marker='o', linewidth=2, label='LSTM')

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"10-Day Temperature Forecast for {city}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ========================================
    # DOWNLOAD CSV
    # ========================================
    csv = forecast_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name=f"{city.lower()}_forecast.csv",
        mime='text/csv'
    )

# ============================================
# SIDEBAR ABOUT
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📌 About Project")
st.sidebar.write(
    """
    This project compares RNN and LSTM models for
    10-day weather forecasting.

    Predicted features:
    - Temperature
    - Humidity
    - Pressure
    - Wind Speed

    Live weather source:
    - Open-Meteo API (No API Key Required)
    """
)
