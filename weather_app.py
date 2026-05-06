import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle
import os

from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Weather Forecast (RNN vs LSTM)", layout="centered")

st.title("🌦️ Weather Forecast (RNN vs LSTM)")
st.write("Compare predictions using RNN and LSTM with Live Data")

# ---------------- API ----------------
API_KEY = "f530270436ed7cd9a06324c89c953281"   # 🔴 apni API key daalo

def get_live_weather(city="Delhi"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        rain = res.get('rain', {}).get('1h', 0)

        return [
            res["main"]["temp"],
            res["main"]["humidity"],
            res["main"]["pressure"],
            res["wind"]["speed"],
            rain
        ]
    except:
        return None

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_models():
    if not os.path.exists("weatherHistory_rnn_model.h5") or not os.path.exists("weatherHistory_lstm_model.h5"):
        return None, None
    return load_model("weatherHistory_rnn_model.h5", compile=False), \
       load_model("weatherHistory_lstm_model.h5", compile=False)

@st.cache_resource
def load_scaler():
    if not os.path.exists("scaler.pkl"):
        return None
    return pickle.load(open("scaler.pkl", "rb"))

rnn_model, lstm_model = load_models()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

if rnn_model is None or lstm_model is None:
    st.error("❌ Model files missing!")
    st.stop()

if scaler is None:
    st.error("❌ Scaler file missing!")
    st.stop()

# ---------------- INPUT ----------------
city = st.text_input("Enter City Name", "Delhi")

# ---------------- BUTTON ----------------
if st.button("🚀 Get Prediction"):

    live_data = get_live_weather(city)

    if live_data is None:
        st.error("❌ API Error or Invalid City")
        st.stop()

    # ---------------- SHOW LIVE DATA ----------------
    st.subheader(f"🌍 Current Weather in {city}")

    st.write(f"🌡️ Temp: {live_data[0]} °C")
    st.write(f"💧 Humidity: {live_data[1]} %")
    st.write(f"🌍 Pressure: {live_data[2]} hPa")
    st.write(f"🌬️ Wind: {live_data[3]} m/s")
    st.write(f"🌧️ Rainfall: {live_data[4]} mm")

    # ---------------- CREATE INPUT SEQUENCE ----------------
    # dummy sequence (replace with real historical data if available)
    dummy = np.array([live_data] * 30)
    
if not hasattr(scaler, "data_min_"):
    # ---------------- LIVE DATA ----------------
    live_data = get_live_weather(city)

if live_data is None:
    st.error("❌ API Error")
    st.stop()

# ---------------- DUMMY SEQUENCE ----------------
dummy = np.tile(live_data, (30, 1))

# ---------------- SCALING ----------------
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    scaled_seq = scaler.transform(dummy)
except:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(dummy)
    scaled_seq = scaler.transform(dummy)

scaled_seq = scaled_seq.reshape(1, 30, 5)

    # ---------------- SCALING ----------------
if not hasattr(scaler, "data_min_"):
    scaler.fit(dummy)

scaled_seq = scaler.transform(dummy)
scaled_seq = scaled_seq.reshape(1, 30, 5)

# ---------------- PREDICTION ----------------
rnn_pred = rnn_model.predict(scaled_seq).reshape(10, 1)
lstm_pred = lstm_model.predict(scaled_seq).reshape(10, 1)

# ---------------- INVERSE ----------------
rnn_full = scaler.inverse_transform(
    np.hstack([rnn_pred, np.zeros((10,4))])
)[:, 0]

lstm_full = scaler.inverse_transform(
    np.hstack([lstm_pred, np.zeros((10,4))])
)[:, 0]

    # ---------------- DATAFRAME ----------------
# ---------------- DAYS ----------------
days = [f"Day {i+1}" for i in range(10)]

# ---------------- DATAFRAME ----------------
df_pred = pd.DataFrame({
    "Day": days,
    "RNN Temp (°C)": rnn_full,
    "LSTM Temp (°C)": lstm_full
})

# ---------------- UI ----------------
st.subheader("📅 10-Day Temperature Forecast")
st.dataframe(df_pred)

st.subheader("📈 Comparison Graph")

fig, ax = plt.subplots()
ax.plot(days, rnn_full, label="RNN", marker='o')
ax.plot(days, lstm_full, label="LSTM", marker='o')

ax.set_xlabel("Days")
ax.set_ylabel("Temperature")
ax.legend()

st.pyplot(fig)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 About")
st.sidebar.write("""
This app compares:
- RNN vs LSTM models  
- Uses Live Weather API  
- Predicts next 10 days temperature  
""")
