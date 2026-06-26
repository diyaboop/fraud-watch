# FraudWatch — Fraud Detection Bias Audit & Drift Monitor

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-fraud--watch.onrender.com-D85A30?style=flat-square)](https://fraud-watch.onrender.com)

End-to-end ML pipeline that trains a fraud detection classifier on the IEEE-CIS dataset, audits it for demographic bias, monitors performance drift, scores individual transactions in real-time, and explains each prediction using LIME — surfacing that credit card users face a **65% higher wrongful block rate** than debit users (Z=66.6, p<0.001).

---

## Tech Stack

Python · FastAPI · scikit-learn · SHAP · LIME · SQLite · Chart.js · Docker · Render

---

## Features

- **Bias Audit** — per-group false positive rate disparity across card type, product category, and billing region; Z-tests with p-values; regulatory mapping (ECOA, EU AI Act, CFPB, SR 11-7)
- **Drift Monitor** — weekly performance tracking across 12 simulated periods with statistical alert thresholds for recall and FPR
- **Real-time Scoring** — score any transaction via REST API and get instant fraud probability with risk level (LOW / MEDIUM / HIGH)
- **LIME Explainability** — per-prediction explanation chart showing which features drove the decision for that specific transaction
- **SHAP Summary** — global feature importance across 1,000 transactions showing which signals matter most across the full model

---

## Key Findings

> **Credit card users face a 65% higher wrongful block rate than debit users** (FPR 56.8% vs 34.3%, Z=66.6, p<0.001).
>
> **Time-delta features (D1–D3)** are the strongest fraud signals globally — unusually short or missing time gaps between transactions strongly predict fraud.
>
> **card6 (card type) is the 5th most important feature globally** per SHAP analysis, directly confirming the bias audit finding that card type drives disproportionate false positive rates.

---

## How to Run

```bash
pip install -r requirements.txt

# Train model (requires data/ — see Dataset section)
python train.py

# Run bias audit
python audit.py

# Run drift monitor
python monitor.py

# Start API + dashboard
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the live dashboard.

**Docker:**
```bash
docker build -t fraudwatch .
docker run -p 8000:8000 fraudwatch
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Aggregate KPIs — recall, precision, FPR, totals |
| GET | `/audit` | Bias audit data + chart PNG |
| GET | `/monitor` | Weekly drift metrics + chart PNG |
| POST | `/predict` | Score a single transaction (30 features) |
| POST | `/explain` | LIME explanation for a single transaction |
| GET | `/shap-summary` | SHAP summary plot (global feature importance) |
| GET | `/shap-waterfall` | SHAP waterfall plot (single transaction) |

---

## Dataset

IEEE-CIS Fraud Detection dataset from Kaggle:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place `train_transaction.csv` and `train_identity.csv` in the `data/` folder before running the training pipeline. The `data/` directory is gitignored; pre-computed model outputs in `outputs/` are committed and shipped with the repo so the dashboard runs without retraining.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/fraud_model.pkl` | Trained Random Forest classifier |
| `outputs/features.pkl` | Ordered feature list used at train time |
| `outputs/fraudwatch.db` | SQLite DB — audit results, bias summary, weekly metrics |
| `outputs/disparity_chart.png` | Bias audit FPR chart |
| `outputs/drift_chart.png` | Weekly performance monitor chart |
| `outputs/shap_summary.png` | SHAP beeswarm summary plot |
| `outputs/shap_waterfall.png` | SHAP waterfall for a representative transaction |

---

## Author

Shrijani (Diya) Manna · Duke MEM '26 · [github.com/diyaboop/fraud-watch](https://github.com/diyaboop/fraud-watch)
