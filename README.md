# stock-price-prediction
NIFTY Stock Price Prediction using GRU
# NIFTY Stock Price Prediction using GRU

A web application that predicts the next 1-minute closing price of NIFTY using a Gated Recurrent Unit (GRU) deep learning model.

## Model Performance
- **MAE**: 0.001058
- **RMSE**: 0.001460  
- **R² Score**: 0.999764

## How to Use
1. Enter the last **60 closing prices** (comma separated)
2. Click "Predict Next Price"
3. Get the predicted next minute price

## Live App
[Your Streamlit URL will appear here after deployment]

## Technologies Used
- Python, TensorFlow/Keras
- GRU (Gated Recurrent Unit)
- Streamlit (for web interface)
- MinMaxScaler for data normalization
