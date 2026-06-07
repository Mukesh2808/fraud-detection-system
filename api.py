from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import shap
import pandas as pd
import numpy as np

app = FastAPI(title='Fraud Detection API')

# Load model once when the server starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

# These are the column names the model expects
FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


class Transaction(BaseModel):
    features: list  # 30 numbers in order: Time, V1-V28, Amount


@app.get('/')
def root():
    return {'message': 'Fraud Detection API is running'}


@app.post('/predict')
def predict(transaction: Transaction):
    # Convert to DataFrame
    X = pd.DataFrame([transaction.features], columns=FEATURE_NAMES)

    # Get fraud probability
    prob = float(model.predict_proba(X)[0][1])

    # Get SHAP explanation
    shap_vals = explainer.shap_values(X)[0].tolist()

    # Find top 5 factors
    factors = sorted(
        zip(FEATURE_NAMES, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        'fraud_probability': round(prob, 4),
        'is_fraud': prob > 0.5,
        'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW',
        'top_factors': [
            {'feature': f, 'impact': round(v, 4)}
            for f, v in factors
        ]
    }
