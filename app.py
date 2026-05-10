import gradio as gr
import numpy as np
import joblib
import os

# मोडेल लोड गर्ने सुरक्षित तरिका
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'big_mart_model.pkl')

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Model Load Error: {e}")

def predict_sales(weight, fat, visibility, item_type_idx, mrp, year, size, loc_type, out_type, years):
    if model is None:
        return "Error: मोडेल फाइल भेटिएन वा लोड हुन सकेन।"
    
    try:
        # क्याटेगोरिकल इन्कोडिङ
        fat_val = 0 if fat == "Low Fat" else 1
        size_dict = {"Small": 0, "Medium": 1, "High": 2}
        size_val = size_dict.get(size, 0)
        
        # १० वटा फिचरको सही क्रम (Order)
        features = np.array([[
            float(weight), 
            fat_val, 
            float(visibility), 
            int(item_type_idx), 
            float(mrp), 
            int(year), 
            size_val, 
            int(loc_type), 
            int(out_type), 
            int(years)
        ]])
        
        prediction = model.predict(features)
        return f"अनुमानित बिक्री: रु. {prediction[0]:,.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Item Weight (तौल)"),
        gr.Dropdown(["Low Fat", "Regular"], label="Fat Content"),
        gr.Slider(0, 1, step=0.01, label="Visibility (दृश्यता)"),
        gr.Dropdown(choices=list(range(16)), label="Item Type (ID: 0-15)"),
        gr.Number(label="Item MRP (मूल्य)"),
        gr.Number(label="Establishment Year (स्थापना वर्ष)"),
        gr.Dropdown(["Small", "Medium", "High"], label="Outlet Size"),
        gr.Dropdown(choices=[0, 1, 2], label="Location Type (Tier 0-2)"),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Outlet Type (ID 0-3)"),
        gr.Number(label="Outlet Age (पसलको उमेर)")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Big Mart Sales Prediction (Final)",
    description="१० वटै विवरणहरू भर्नुहोस् र सेल्सको भविष्यवाणी हेर्नुहोस्।"
)

if __name__ == "__main__":
    interface.launch()
