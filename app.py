import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("gru_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("📈 NIFTY Stock Price Prediction (GRU)")
st.markdown("**Paste the last 60 closing prices (comma-separated)**")

user_input = st.text_area(
    "60 Closing Prices:",
    placeholder="8250.50, 8251.20, 8249.80, ...",
    height=140
)

if user_input:
    try:
        prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        
        st.write(f"✅ Received {len(prices)} prices")
        
        if len(prices) != 60:
            st.warning(f"⚠️ Please enter exactly 60 prices (you gave {len(prices)})")
        
        if st.button("🔮 Predict Next Price", type="primary") and len(prices) == 60:
            with st.spinner("Predicting..."):
                array = np.array(prices).reshape(-1, 1)
                scaled = scaler.transform(array)
                sequence = scaled.reshape(1, 60, 1).astype('float32')
                
                prediction = model.predict(sequence, verbose=0)
                result = scaler.inverse_transform(prediction)[0][0]
                
                st.success(f"### Predicted Next Price: **₹{result:,.2f}**")
                st.line_chart(prices)
                
    except Exception as e:
        st.error(f"Error: {e}")
