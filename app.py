import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle

# Define the model architecture (must match training)
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Load model and scaler
@st.cache_resource
def load_assets():
    model = GRUModel()
    model.load_state_dict(torch.load("gru_model.pth", weights_only=True, map_location='cpu'))
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("📈 NIFTY Stock Price Prediction (GRU)")
st.markdown("**Enter the last 60 closing prices (comma-separated)**")

user_input = st.text_area(
    "Paste 60 prices here:",
    placeholder="8250.50, 8251.20, 8249.80, ...",
    height=150
)

if user_input:
    try:
        prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        st.write(f"✅ Received {len(prices)} prices")
        
        if len(prices) != 60:
            st.warning(f"⚠️ Please enter exactly 60 prices. You entered {len(prices)}.")
        else:
            if st.button("🔮 Predict Next Price", type="primary"):
                with st.spinner("Predicting..."):
                    array = np.array(prices).reshape(-1, 1)
                    scaled = scaler.transform(array)
                    sequence = torch.FloatTensor(scaled).reshape(1, 60, 1)
                    
                    with torch.no_grad():
                        pred = model(sequence)
                    price = scaler.inverse_transform(pred.numpy())[0][0]
                    
                    st.success(f"### Predicted Next Closing Price: **₹{price:,.2f}**")
                    st.line_chart(prices)
    except Exception as e:
        st.error(f"Error: {e}")
