import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

#Setting experiment name
mlflow.set_experiment("fraudwatch-training")

#Loading data
transaction = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')

#Merging data
df = transaction.merge(identity, how='left', on='TransactionID')
print(f"Merged data shape: {df.shape}")

#Feature Selection
#Keep columns with less than 50% missing + key columns
threshold = len(df) * 0.5

final_cols = []
for col in df.columns:
    if df[col].isnull().sum() < threshold:
        final_cols.append(col)

df = df[final_cols]

print("Available columns after threshold filter:")

# Select key features for model
features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'D1', 'D2', 'D3', 'D4', 'M1', 'M2', 'M3', 'M4', 'M6']
target = 'isFraud'

# Keep only features that exist after the threshold filter
df = df[features + [target]].copy()
print(f"Final feature set: {df.shape}")
print(f"Fraud rate: {df[target].mean():.4f}")

# Encode categoricals
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

# Fill missing
df = df.fillna(-999)

print("\nData ready. Sample:")
print(df.head())
print("\nFeatures:", features)

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


with mlflow.start_run():
    # Train model 
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # handle the 3.5% fraud imbalance
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)

    # Evaluate 
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    #Parameters — what settings were used
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("fraud_rate", round(y.mean(), 4))

    #Metrics — what results were extracted from classification report
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))

    #Mode artifact: log the trained model
    mlflow.sklearn.log_model(model, "random_forest_model")

    #Feature importance artifact — save and log as a chart
    feat_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(feat_importance['feature'], feat_importance['importance'])
    plt.title('Feature Importance — FraudWatch')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    mlflow.log_artifact('outputs/feature_importance.png')

    # 10. Save model 
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/fraud_model.pkl')
    joblib.dump(features, 'outputs/features.pkl')
    print("\nModel saved to outputs/fraud_model.pkl")