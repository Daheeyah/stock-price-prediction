import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load GRU Model (NOT LSTM)
model = load_model("gru_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("📈 NIFTY Stock Price Prediction (GRU)")
st.markdown("""
This app uses a **Gated Recurrent Unit (GRU)** model trained on NIFTY 1-minute data.
Enter the last **60 closing prices** (comma separated).
""")

# ... (rest of the code I gave you earlier)
