# FraudWatch — Fraud Detection Bias Audit & Drift Monitor

 **Live Dashboard:** [fraudwatch.onrender.com](https://fraud-watch.onrender.com)  

> End-to-end ML pipeline that trains a fraud detection classifier, audits it for 
> demographic bias, and monitors performance drift — surfacing that credit card 
> users face a 65% higher wrongful block rate than debit users (Z=66.6, p<0.001).

## What it does
End-to-end ML pipeline that trains a fraud detection model on real transaction data, 
audits it for demographic bias, and monitors performance drift over time.

## Key Findings
- **Credit card holders wrongly flagged 65% more often than debit card holders** 
  (FPR 56.8% vs 34.3%, Z=66.6, p<0.001)
- **Product category disparity of 2.7x** — ProductCD=0 flagged at 82.3% FPR vs 
  ProductCD=4 at 30.8% (Z=109.1, p<0.001)
- **No performance drift detected** — model stable across 12 simulated weeks 
  (recall consistently above 0.92)

## Pipeline
- `train.py` — trains Random Forest on IEEE-CIS fraud data (590,540 transactions, 3.5% fraud rate)
- `audit.py` — bias audit across card type, product category, and billing region
- `monitor.py` — weekly performance drift monitoring with alert thresholds

## Tech Stack
Python, Pandas, Scikit-learn, SciPy, Matplotlib, SQLite, Docker

## Data
Download the IEEE-CIS Fraud Detection dataset from Kaggle:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place `train_transaction.csv` and `train_identity.csv` in the `data/` folder.

## How to Run
```bash
pip install -r requirements.txt

# Train model
python train.py

# Run bias audit
python audit.py

# Run drift monitor
python monitor.py
```

## Outputs
- `outputs/fraud_model.pkl` — trained Random Forest model
- `outputs/disparity_chart.png` — bias audit visualization
- `outputs/drift_chart.png` — weekly performance monitor
- `outputs/fraudwatch.db` — SQLite database with audit and monitoring results
- `outputs/bias_summary.csv` — bias findings summary

## Relevance
Built to demonstrate responsible AI monitoring for financial services — 
directly applicable to fraud detection systems at fintechs and 
bias auditing workflows at AI governance companies.
