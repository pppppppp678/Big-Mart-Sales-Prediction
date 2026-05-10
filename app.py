import streamlit as st
import joblib
import numpy as np
import pandas as pd

# १. मोडेल लोड गर्ने
model = joblib.load('big_mart_model.pkl')

# २. पेज सेटिङ
st.set_page_config(page_title="Big Mart Sales Predictor", layout="centered")
st.title("🛒 Big Mart Sales Prediction App")
st.markdown("Enter the product and outlet details to predict the sales.")

# ३. इनपुट फर्म
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        mrp = st.number_input("Item MRP", min_value=0.0, value=150.0)
        weight = st.number_input("Item Weight", min_value=0.0, value=12.0)
        visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.05)
        
    with col2:
        outlet_age = st.number_input("Outlet Age (Years)", min_value=0, max_value=50, value=10)
        fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])

    submit = st.form_submit_button("Predict Sales")

# ४. भविष्यवाणी गर्ने प्रक्रिया
if submit:
    # Encoding (तालिम दिँदाकै जस्तो हुनुपर्छ)
    fat_val = 0 if fat_content == "Low Fat" else 1
    size_val = {"Small": 0, "Medium": 1, "High": 2}[outlet_size]
    
    # फिचर एरे (क्रम: Weight, Fat, Visibility, MRP, Size, Age)
    features = np.array([[weight, fat_val, visibility, mrp, size_val, outlet_age]])
    
    prediction = model.predict(features)
    
    st.success(f"💰 Estimated Sales: रु. {prediction[0]:,.2f}")
