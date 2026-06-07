Real-Time Credit Card Fraud Detection System

A machine learning-based fraud detection system that identifies fraudulent credit card transactions using XGBoost. The project includes data preprocessing, model training, explainable AI with SHAP, API deployment using FastAPI, and an interactive Streamlit dashboard.

Features
Fraud detection using XGBoost
SHAP explainability for prediction insights
FastAPI-based REST API
Streamlit dashboard for real-time analysis
Power BI dashboard for fraud visualization
Tech Stack

Python, Pandas, XGBoost, SHAP, FastAPI, Streamlit, Power BI

How to Run
pip install -r requirements.txt
python train_model.py
uvicorn api:app --reload
streamlit run dashboard.py
