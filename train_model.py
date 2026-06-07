import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Step 1: Load the data
print('Loading data...')
df = pd.read_csv('creditcard.csv')
print(f'Loaded {len(df)} transactions')
print(f'Fraud cases: {df["Class"].sum()} out of {len(df)}')

# Step 2: Split into features (X) and target (y)
X = df.drop('Class', axis=1)   # everything except the Class column
y = df['Class']                # just the Class column

# Step 3: Split into train and test sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train the model
# scale_pos_weight=577 handles the imbalance (non-fraud / fraud ratio)
print('Training model... (this takes 1-2 minutes)')
model = XGBClassifier(
    scale_pos_weight=577,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
print('\nModel Results:')
print(classification_report(y_test, model.predict(X_test)))

# Step 6: Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('\nModel saved as model.pkl')
