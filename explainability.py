import joblib
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

shap.initjs()

# Load saved model and features
model = joblib.load('outputs/fraud_model.pkl')
features = joblib.load('outputs/features.pkl')

transaction = pd.read_csv('data/train_transaction.csv')
X = transaction[features].dropna().sample(1000, random_state=42)

# encode categorical columns
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype not in ['float64', 'int64']:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# build explainer
explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")
shap_values = explainer(X_encoded)

shap_array = shap_values.values
shap_fraud = shap_array[:, :, 1]

# summary plot only
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_fraud,
    X_encoded,
    feature_names=features,
    show=False
)
plt.title('SHAP Feature Importance — FraudWatch Model')
plt.tight_layout()
plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to outputs/shap_summary.png")