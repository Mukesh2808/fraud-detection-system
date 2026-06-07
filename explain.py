import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)

# Take a sample of 200 rows to explain (full dataset is slow)
X_sample = X.sample(200, random_state=42)

# Create the SHAP explainer
print('Calculating SHAP values... (takes ~1 minute)')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Plot 1: Which features matter most overall?
print('Saving summary plot...')
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print('Saved shap_summary.png')
print('SHAP done!')
