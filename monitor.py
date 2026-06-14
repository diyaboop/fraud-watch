import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split

# Load saved model and features
model = joblib.load('outputs/fraud_model.pkl')
features = joblib.load('outputs/features.pkl')

# Reload and prepare test data — same as audit.py
transaction = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')

df = transaction.merge(identity, how='left', on='TransactionID')

threshold = len(df) * 0.5
final_cols = [col for col in df.columns if df[col].isnull().sum() < threshold]
df = df[final_cols]

target = 'isFraud'
df = df[features + [target]].copy()

cat_cols = df.select_dtypes(include=['str']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

df = df.fillna(-999)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get predictions with same threshold as audit.py
fraud_proba = model.predict_proba(X_test)[:, 1]
threshold_pred = 0.3
y_pred = (fraud_proba >= threshold_pred).astype(int)

print(f"Test set loaded: {X_test.shape}")
print(f"Fraud predictions: {y_pred.sum()}")

# Build monitoring DataFrame 
monitor_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'fraud_proba': fraud_proba
})

# Assign week numbers — split into 12 weekly batches
monitor_df['week'] = pd.qcut(range(len(monitor_df)), q=12, labels=False)

print(f"Transactions per week:")
print(monitor_df['week'].value_counts().sort_index())

# Calculate metrics per week 
weekly_metrics = []

for week in sorted(monitor_df['week'].unique()):
    week_data = monitor_df[monitor_df['week'] == week]
    
    actual = week_data['actual']
    predicted = week_data['predicted']
    
    # precision — of flagged transactions, how many are real fraud
    tp = ((actual == 1) & (predicted == 1)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()
    fn = ((actual == 1) & (predicted == 0)).sum()
    tn = ((actual == 0) & (predicted == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    weekly_metrics.append({
        'week': int(week) + 1,
        'total_transactions': len(week_data),
        'fraud_cases': actual.sum(),
        'flagged_transactions': predicted.sum(),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'fpr': round(fpr, 4)
    })

metrics_df = pd.DataFrame(weekly_metrics)
print("\nWeekly Metrics:")
print(metrics_df)

# Drift Detection
RECALL_THRESHOLD = 0.85
FPR_THRESHOLD = 0.55

metrics_df['recall_alert'] = metrics_df['recall'] < RECALL_THRESHOLD
metrics_df['fpr_alert'] = metrics_df['fpr'] > FPR_THRESHOLD
metrics_df['drift_flag'] = metrics_df['recall_alert'] | metrics_df['fpr_alert']

print("\nDrift Alerts:")
alerts = metrics_df[metrics_df['drift_flag']]
if alerts.empty:
    print("No drift detected — model performing stably across all weeks")
else:
    print(alerts[['week', 'recall', 'fpr', 'recall_alert', 'fpr_alert']])

# Visualize 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('FraudWatch — Model Performance Monitor (12 Weeks)', 
             fontsize=14, fontweight='bold')

# Recall over time
axes[0].plot(metrics_df['week'], metrics_df['recall'], 
             color='#1D9E75', marker='o', linewidth=2)
axes[0].axhline(RECALL_THRESHOLD, color='red', linestyle='--', label=f'Alert threshold ({RECALL_THRESHOLD})')
axes[0].set_title('Recall Over Time')
axes[0].set_xlabel('Week')
axes[0].set_ylabel('Recall')
axes[0].set_ylim(0, 1)
axes[0].legend()

# Precision over time
axes[1].plot(metrics_df['week'], metrics_df['precision'],
             color='#534AB7', marker='o', linewidth=2)
axes[1].set_title('Precision Over Time')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Precision')
axes[1].set_ylim(0, 0.5)

# FPR over time
axes[2].plot(metrics_df['week'], metrics_df['fpr'],
             color='#D85A30', marker='o', linewidth=2)
axes[2].axhline(FPR_THRESHOLD, color='red', linestyle='--', label=f'Alert threshold ({FPR_THRESHOLD})')
axes[2].set_title('False Positive Rate Over Time')
axes[2].set_xlabel('Week')
axes[2].set_ylabel('FPR')
axes[2].set_ylim(0, 1)
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/drift_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Drift chart saved to outputs/drift_chart.png")

# Save to SQLite 
conn = sqlite3.connect('outputs/fraudwatch.db')
metrics_df.to_sql('weekly_metrics', conn, if_exists='replace', index=False)
conn.close()
print("Weekly metrics saved to fraudwatch.db")