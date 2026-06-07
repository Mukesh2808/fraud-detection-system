import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

results = X_test.copy()
results['actual_fraud'] = y_test.values
results['predicted_fraud'] = preds
results['fraud_probability'] = probs
results['risk_level'] = pd.cut(
    probs,
    bins=[0, 0.3, 0.7, 1.0],
    labels=['LOW', 'MEDIUM', 'HIGH']
)
results['correct'] = (results['actual_fraud'] == results['predicted_fraud'])

results.to_csv('fraud_results.csv', index=False)
print(f'Saved {len(results)} rows to fraud_results.csv')
