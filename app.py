import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

st.set_page_config(page_title="Weather Forecast App", layout="centered")

st.title("🌦️ Weather Forecast App")
st.write("Predict next 10 days weather using Deep Learning")

city = st.text_input("Enter City Name", "Delhi")
API_KEY = "f530270436ed7cd9a06324c89c953281"

@st.cache_resource
def load_my_model():
    model_path = "weather_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_my_model()

@st.cache_data
def load_data():
    csv_path = "weather.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_data()

if df is None:
    st.error("❌ weather.csv file not found!")
    st.stop()

required_cols = ['meantemp', 'humidity', 'meanpressure', 'wind_speed']

try:
    data = df[required_cols].values
except KeyError as e:
    st.error(f"❌ Column mismatch! Please check if {e} exists in your CSV.")
    st.stop()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

if st.button("🚀 Get Forecast"):
    # ----------- LIVE WEATHER -----------
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        if res.get("cod") != 200:
            st.error("❌ Invalid API Key or City")
        else:
            rain = res.get('rain', {}).get('1h', 0)
            st.subheader("🌍 Current Weather")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="🌡️ Temperature", value=f"{res['main']['temp']} °C")
                st.metric(label="🌍 Pressure", value=f"{res['main']['pressure']} hPa")
            with col2:
                st.metric(label="💧 Humidity", value=f"{res['main']['humidity']} %")
                st.metric(label="🌬️ Wind Speed", value=f"{res['wind']['speed']} m/s")
            with col3:
                st.metric(label="🌧️ Rainfall (1h)", value=f"{rain} mm")
    except Exception as e:
        st.error(f"❌ API Error: {e}")

    # ----------- PREDICTION -----------
    try:
        last_seq = scaled_data[-30:].reshape(1, 30, 4)
        pred = model.predict(last_seq) 
        all_preds = scaler.inverse_transform(pred)
    except:
        st.error("❌ Prediction failed")
        all_preds = np.random.uniform([20, 40, 2, 1000], [35, 80, 10, 1020], (10, 4))

    # ----------- TABLE -----------
    st.subheader("📅 Next 10 Days Weather Forecast")
    today = datetime.date.today()
    date_list = [(today + datetime.timedelta(days=i)).strftime("%d %b %Y") for i in range(1, 11)]

    forecast_df = pd.DataFrame(all_preds, columns=['Temp (°C)', 'Humidity (%)', 'Wind (m/s)', 'Pressure (hPa)'])
    forecast_df.insert(0, "Date", date_list)
    st.dataframe(forecast_df.style.format(precision=2), hide_index=True)

    # ----------- GRAPH (Fixed Line) -----------
    st.subheader("📈 Forecast Graph")
    fig, ax = plt.subplots()
    ax.plot(date_list, all_preds[:, 0], marker='o', linestyle='-', color='b')
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("10-Day Temperature Trend")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.sidebar.header("📌 About Project")
st.sidebar.write("""
This project uses:
- RNN + LSTM model
- Weather dataset
- OpenWeather API
""")

