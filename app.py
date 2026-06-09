import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction Using GRU")

@st.cache_resource
def load_artifacts():
    model = load_model("gru_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

ticker = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):

    data = yf.download(ticker, period="2y")

    if data.empty:
        st.error("Invalid stock symbol.")
        st.stop()

    close_prices = data[['Close']]

    scaled_data = scaler.transform(close_prices)

    sequence_length = 60

    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])

    X = np.array(X)

    predictions = model.predict(X, verbose=0)

    predictions = scaler.inverse_transform(predictions)

    actual = close_prices.iloc[sequence_length:].values

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(actual, label="Actual Price")
    ax.plot(predictions, label="GRU Prediction")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.legend()

    st.pyplot(fig)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.6f}")
    col2.metric("RMSE", f"{rmse:.6f}")
    col3.metric("R²", f"{r2:.6f}")

    st.success("GRU model prediction completed successfully.")
