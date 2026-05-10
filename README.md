Big Mart Sales Prediction
An End-to-End Machine Learning Regression Project
This project predicts the sales of various products across different Big Mart outlets. By leveraging historical sales data, we identify key factors that drive product demand, enabling the business to optimize inventory and marketing strategies.

📌 Project Overview

The dataset contains information on 1,559 products across 10 different outlets. The objective is to predict the Item_Outlet_Sales (Target Variable) based on product attributes (weight, visibility, price) and store attributes (location, size, type).

🛠️ Technical Stack

Language: Python 3.12

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (Random Forest Regressor)

Deployment: Gradio / Streamlit

📊 Data Pipeline

1. Data Preprocessing & Cleaning
Handling Missing Values: Imputed missing Item_Weight using the mean and Outlet_Size using the mode.

Categorical Encoding: Standardized Item_Fat_Content (mapping 'LF', 'low fat', and 'reg' to 'Low Fat' and 'Regular').

Feature Engineering: Derived a new feature Outlet_Age from the establishment year to capture the store's maturity.

2. Model Development
3. 
Converted categorical text data into numerical format using Label Encoding.

Implemented a Random Forest Regressor to handle non-linear relationships and provide robust predictions.

Performed a Train-Test Split (80/20) to validate model performance on unseen data.

3. Deployment
Developed an interactive web interface using Gradio, allowing users to input product/store details and receive real-time sales predictions.

🚀 Key Insights
Item MRP is the strongest predictor of sales.

Outlet Type (Supermarket vs. Grocery Store) significantly impacts the volume of sales.

Store maturity (Outlet Age) correlates with customer loyalty and higher sales consistency.

📂 Repository Structure

Plaintext
├── data/                   # Train and Test CSV files
├── notebooks/              # Jupyter/Colab notebooks with EDA & Modeling
├── app.py                  # Gradio/Streamlit application code
├── big_mart_model.pkl      # Saved trained model
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
