import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

#Loading the saved model and features from train.py
model = joblib.load('outputs/fraud_model.pkl')
features = joblib.load('outputs/features.pkl')

#Reload and preparing the test data
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

#Making predictions on the test set
y_pred = model.predict(X_test)
fraud_prob_scores = model.predict_proba(X_test)[:,1]
threshold = 0.3
y_pred = (fraud_prob_scores >= threshold).astype(int)
print(f"Fraud probability scores: \n{fraud_prob_scores}")
print(f"y_test: \n {y_test}")
print(f"y_pred: \n {y_pred}")

#Building the audit dataframe
audit_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'fraud_prob_scores': fraud_prob_scores,
    'card4': X_test['card4'].values,
    'card6': X_test['card6'].values,
    'ProductCD': X_test['ProductCD'].values,
    'addr2': X_test['addr2'].values
})

def calculate_group_metrics(group):
    actual = group['actual']
    predicted = group['predicted']
    
    # legitimate transactions only
    legitimate = group[actual == 0]
    fraud = group[actual == 1]
    
    # false positive rate — wrongly flagged legitimate
    fpr = (predicted[actual == 0] == 1).sum() / len(legitimate) if len(legitimate) > 0 else 0
    
    # false negative rate — missed fraud
    fnr = (predicted[actual == 1] == 0).sum() / len(fraud) if len(fraud) > 0 else 0
    
    # recall — fraud caught
    recall = (predicted[actual == 1] == 1).sum() / len(fraud) if len(fraud) > 0 else 0
    
    return pd.Series({
        'total_transactions': len(group),
        'legitimate_transactions': len(legitimate),
        'fraud_transactions': len(fraud),
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'recall': round(recall, 4)
    })

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Calculate disparity per segment
for segment in ['card4', 'card6', 'ProductCD', 'addr2']:
    if segment == 'addr2':
        valid_groups = audit_df['addr2'].value_counts()
        valid_groups = valid_groups[valid_groups >= 100].index
        segment_df = audit_df[audit_df['addr2'].isin(valid_groups)]
    else:
        segment_df = audit_df
    result = audit_df.groupby(segment).apply(calculate_group_metrics)

# Overall false positive rate
overall_fpr = ((audit_df['actual']==0) & (audit_df['predicted']==1)).sum() / (audit_df['actual']==0).sum()
print(f"Overall FPR: {overall_fpr:.4f}")

# Compare credit (1) vs debit (2)
credit = audit_df[audit_df['card6']==1]
debit = audit_df[audit_df['card6']==2]

credit_fp = ((credit['actual']==0) & (credit['predicted']==1)).sum()
credit_leg = (credit['actual']==0).sum()

debit_fp = ((debit['actual']==0) & (debit['predicted']==1)).sum()
debit_leg = (debit['actual']==0).sum()

z_stat, p_value = proportions_ztest(
    [credit_fp, debit_fp],
    [credit_leg, debit_leg]
)

cat0 = audit_df[audit_df['ProductCD']==0]
cat4 = audit_df[audit_df['ProductCD']==4]

cat0_fp = ((cat0['actual']==0) & (cat0['predicted']==1)).sum()
cat0_leg = (cat0['actual']==0).sum()

cat4_fp = ((cat4['actual']==0) & (cat4['predicted']==1)).sum()
cat4_leg = (cat4['actual']==0).sum()

z_stat, p_value = proportions_ztest(
    [cat0_fp, cat4_fp],
    [cat0_leg, cat4_leg]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('FraudWatch — Bias Audit: False Positive Rate by Segment', 
             fontsize=14, fontweight='bold')

# Chart 1 — card6
card6_fpr = audit_df.groupby('card6').apply(
    lambda g: ((g['actual']==0) & (g['predicted']==1)).sum() / (g['actual']==0).sum()
).reset_index()
card6_fpr.columns = ['card6', 'fpr']
card6_fpr = card6_fpr[card6_fpr['card6'].isin([1, 2])]
card6_fpr['label'] = card6_fpr['card6'].map({1: 'Credit', 2: 'Debit'})

axes[0].bar(card6_fpr['label'], card6_fpr['fpr'], 
            color=['#D85A30', '#1D9E75'])
axes[0].axhline(overall_fpr, color='black', linestyle='--', label='Overall FPR')
axes[0].set_title('Card Type\n(Z=66.6, p<0.001)', fontsize=11)
axes[0].set_ylabel('False Positive Rate')
axes[0].set_ylim(0, 1)
axes[0].legend()

# Chart 2 — ProductCD
product_fpr = audit_df.groupby('ProductCD').apply(
    lambda g: ((g['actual']==0) & (g['predicted']==1)).sum() / (g['actual']==0).sum()
).reset_index()
product_fpr.columns = ['ProductCD', 'fpr']

colors = ['#D85A30' if fpr > overall_fpr * 1.5 else '#1D9E75' 
          for fpr in product_fpr['fpr']]
axes[1].bar(product_fpr['ProductCD'].astype(str), product_fpr['fpr'], color=colors)
axes[1].axhline(overall_fpr, color='black', linestyle='--', label='Overall FPR')
axes[1].set_title('Product Category\n(Z=109.1, p<0.001)', fontsize=11)
axes[1].set_ylabel('False Positive Rate')
axes[1].set_ylim(0, 1)
axes[1].legend()

# Chart 3 — addr2 filtered
addr2_audit = audit_df[audit_df['addr2'].isin([-999.0, 60.0, 87.0, 96.0])]
addr2_fpr = addr2_audit.groupby('addr2').apply(
    lambda g: ((g['actual']==0) & (g['predicted']==1)).sum() / (g['actual']==0).sum()
).reset_index()
addr2_fpr.columns = ['addr2', 'fpr']
addr2_fpr['label'] = addr2_fpr['addr2'].map({
    -999.0: 'Unknown', 60.0: 'Region 60', 87.0: 'US', 96.0: 'Region 96'
})

colors = ['#D85A30' if fpr > overall_fpr * 1.5 else '#1D9E75' 
          for fpr in addr2_fpr['fpr']]
axes[2].bar(addr2_fpr['label'], addr2_fpr['fpr'], color=colors)
axes[2].axhline(overall_fpr, color='black', linestyle='--', label='Overall FPR')
axes[2].set_title('Billing Region\n(US vs International)', fontsize=11)
axes[2].set_ylabel('False Positive Rate')
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=15)
axes[2].legend()

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/disparity_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved to outputs/disparity_chart.png")

# Save audit results to CSV
os.makedirs('outputs', exist_ok=True)

# Save each segment result
for segment in ['card4', 'card6', 'ProductCD']:
    result = audit_df.groupby(segment).apply(calculate_group_metrics).reset_index()
    result.to_csv(f'outputs/audit_{segment}.csv', index=False)

# Save to SQLite
conn = sqlite3.connect('outputs/fraudwatch.db')
audit_df.to_sql('audit_results', conn, if_exists='replace', index=False)

# Save key findings as a summary table
summary = pd.DataFrame([
    {'segment': 'card6', 'group_a': 'Credit', 'group_b': 'Debit',
     'fpr_a': 0.5682, 'fpr_b': 0.3431, 'z_stat': 66.6, 'p_value': 0.0, 'significant': True},
    {'segment': 'ProductCD', 'group_a': 'Cat_0', 'group_b': 'Cat_4',
     'fpr_a': 0.8226, 'fpr_b': 0.3081, 'z_stat': 109.1, 'p_value': 0.0, 'significant': True},
])
summary.to_sql('bias_summary', conn, if_exists='replace', index=False)
summary.to_csv('outputs/bias_summary.csv', index=False)

# Run a SQL query surfacing highest risk segments
query = """
SELECT 'card6' as segment, card6 as group_value,
    COUNT(*) as total,
    SUM(CASE WHEN actual=0 AND predicted=1 THEN 1 ELSE 0 END) as false_positives,
    ROUND(SUM(CASE WHEN actual=0 AND predicted=1 THEN 1.0 ELSE 0 END) / 
          SUM(CASE WHEN actual=0 THEN 1.0 ELSE 0 END), 4) as fpr
FROM audit_results
WHERE actual = 0
GROUP BY card6
ORDER BY fpr DESC
"""
sql_result = pd.read_sql(query, conn)
print("\nSQL Query — Highest Risk Segments by card6:")
print(sql_result)

conn.close()
print("\nResults saved to outputs/fraudwatch.db and outputs/")