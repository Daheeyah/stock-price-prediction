import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("gru_model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("📈 NIFTY Stock Price Prediction (GRU)")
st.markdown("""
**How to use:** Paste the last **60 closing prices** (comma-separated).
The model will predict the next minute's closing price.
""")

user_input = st.text_area(
    "Paste 60 closing prices here:",
    placeholder="8250.50, 8251.20, 8249.80, ...",
    height=150
)

if user_input:
    try:
        past_prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        st.write(f"✅ Received {len(past_prices)} prices")
        
        if len(past_prices) != 60:
            st.warning(f"⚠️ Please enter exactly 60 prices. You entered {len(past_prices)}.")
        
        if st.button("🔮 Predict Next Price", type="primary") and len(past_prices) == 60:
            with st.spinner("Predicting..."):
                prices_array = np.array(past_prices).reshape(-1, 1)
                prices_scaled = scaler.transform(prices_array)
                sequence = prices_scaled.reshape(1, 60, 1).astype('float32')
                
                pred_scaled = model.predict(sequence, verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                
                st.success(f"### Predicted Next Closing Price: **₹{pred_price:,.2f}**")
                st.line_chart(past_prices)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
