import sqlite3
import io
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="FraudWatch API", version="1.0")

DB_PATH = Path("outputs/fraudwatch.db")
MODEL_PATH = Path("outputs/fraud_model.pkl")
FEATURES_PATH = Path("outputs/features.pkl")
STATIC_PATH = Path("static")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _model():
    return joblib.load(MODEL_PATH)


def _features():
    return joblib.load(FEATURES_PATH)


def _chart_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# /metrics  — overall model performance numbers
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM weekly_metrics ORDER BY week").fetchall()
    if not rows:
        raise HTTPException(404, "No metrics found")

    df = pd.DataFrame([dict(r) for r in rows])
    return {
        "avg_precision": round(df["precision"].mean(), 4),
        "avg_recall": round(df["recall"].mean(), 4),
        "avg_fpr": round(df["fpr"].mean(), 4),
        "total_transactions": int(df["total_transactions"].sum()),
        "total_fraud_cases": int(df["fraud_cases"].sum()),
        "total_flagged": int(df["flagged_transactions"].sum()),
        "drift_weeks": int(df["drift_flag"].sum()),
        "weeks": len(df),
    }


# ---------------------------------------------------------------------------
# /audit  — bias audit data + chart
# ---------------------------------------------------------------------------

@app.get("/audit")
def audit():
    with get_conn() as conn:
        summary = conn.execute("SELECT * FROM bias_summary").fetchall()
        audit_rows = conn.execute("SELECT * FROM audit_results").fetchall()

    summary_list = [dict(r) for r in summary]
    df = pd.DataFrame([dict(r) for r in audit_rows])

    overall_fpr = ((df["actual"] == 0) & (df["predicted"] == 1)).sum() / (df["actual"] == 0).sum()

    # --- card6 FPR ---
    card6_fpr = (
        df.groupby("card6")
        .apply(lambda g: ((g["actual"] == 0) & (g["predicted"] == 1)).sum() / max((g["actual"] == 0).sum(), 1))
        .reset_index()
    )
    card6_fpr.columns = ["card6", "fpr"]
    card6_fpr = card6_fpr[card6_fpr["card6"].isin([1, 2])]
    card6_fpr["label"] = card6_fpr["card6"].map({1: "Credit", 2: "Debit"})

    # --- ProductCD FPR ---
    product_fpr = (
        df.groupby("ProductCD")
        .apply(lambda g: ((g["actual"] == 0) & (g["predicted"] == 1)).sum() / max((g["actual"] == 0).sum(), 1))
        .reset_index()
    )
    product_fpr.columns = ["ProductCD", "fpr"]

    # --- addr2 FPR ---
    addr2_audit = df[df["addr2"].isin([-999.0, 60.0, 87.0, 96.0])]
    addr2_fpr = (
        addr2_audit.groupby("addr2")
        .apply(lambda g: ((g["actual"] == 0) & (g["predicted"] == 1)).sum() / max((g["actual"] == 0).sum(), 1))
        .reset_index()
    )
    addr2_fpr.columns = ["addr2", "fpr"]
    addr2_fpr["label"] = addr2_fpr["addr2"].map(
        {-999.0: "Unknown", 60.0: "Region 60", 87.0: "US", 96.0: "Region 96"}
    )

    # --- build chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FraudWatch — Bias Audit: False Positive Rate by Segment", fontsize=14, fontweight="bold")

    axes[0].bar(card6_fpr["label"], card6_fpr["fpr"], color=["#D85A30", "#1D9E75"])
    axes[0].axhline(overall_fpr, color="black", linestyle="--", label="Overall FPR")
    axes[0].set_title("Card Type\n(Z=66.6, p<0.001)", fontsize=11)
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    colors = ["#D85A30" if fpr > overall_fpr * 1.5 else "#1D9E75" for fpr in product_fpr["fpr"]]
    axes[1].bar(product_fpr["ProductCD"].astype(str), product_fpr["fpr"], color=colors)
    axes[1].axhline(overall_fpr, color="black", linestyle="--", label="Overall FPR")
    axes[1].set_title("Product Category\n(Z=109.1, p<0.001)", fontsize=11)
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    colors2 = ["#D85A30" if fpr > overall_fpr * 1.5 else "#1D9E75" for fpr in addr2_fpr["fpr"]]
    axes[2].bar(addr2_fpr["label"], addr2_fpr["fpr"], color=colors2)
    axes[2].axhline(overall_fpr, color="black", linestyle="--", label="Overall FPR")
    axes[2].set_title("Billing Region\n(US vs International)", fontsize=11)
    axes[2].set_ylabel("False Positive Rate")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].legend()

    plt.tight_layout()
    chart_b64 = _chart_to_b64(fig)

    return {
        "overall_fpr": round(float(overall_fpr), 4),
        "bias_summary": summary_list,
        "card6": card6_fpr[["label", "fpr"]].rename(columns={"label": "group"}).to_dict(orient="records"),
        "product": product_fpr.to_dict(orient="records"),
        "region": addr2_fpr[["label", "fpr"]].rename(columns={"label": "group"}).to_dict(orient="records"),
        "chart_png_b64": chart_b64,
    }


# ---------------------------------------------------------------------------
# /monitor  — weekly drift monitoring data + chart
# ---------------------------------------------------------------------------

@app.get("/monitor")
def monitor():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM weekly_metrics ORDER BY week").fetchall()
    if not rows:
        raise HTTPException(404, "No monitor data found")

    df = pd.DataFrame([dict(r) for r in rows])

    RECALL_THRESHOLD = 0.85
    FPR_THRESHOLD = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FraudWatch — Model Performance Monitor (12 Weeks)", fontsize=14, fontweight="bold")

    axes[0].plot(df["week"], df["recall"], color="#1D9E75", marker="o", linewidth=2)
    axes[0].axhline(RECALL_THRESHOLD, color="red", linestyle="--", label=f"Alert ({RECALL_THRESHOLD})")
    axes[0].set_title("Recall Over Time")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(df["week"], df["precision"], color="#534AB7", marker="o", linewidth=2)
    axes[1].set_title("Precision Over Time")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 0.5)

    axes[2].plot(df["week"], df["fpr"], color="#D85A30", marker="o", linewidth=2)
    axes[2].axhline(FPR_THRESHOLD, color="red", linestyle="--", label=f"Alert ({FPR_THRESHOLD})")
    axes[2].set_title("False Positive Rate Over Time")
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("FPR")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    chart_b64 = _chart_to_b64(fig)

    drift_weeks = df[df["drift_flag"] == 1]["week"].tolist()

    return {
        "weekly": df[["week", "precision", "recall", "fpr", "drift_flag", "fraud_cases", "flagged_transactions"]].to_dict(orient="records"),
        "drift_weeks": drift_weeks,
        "recall_threshold": RECALL_THRESHOLD,
        "fpr_threshold": FPR_THRESHOLD,
        "chart_png_b64": chart_b64,
    }


# ---------------------------------------------------------------------------
# /predict  — score a single transaction
# ---------------------------------------------------------------------------

FEATURE_LIST = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
    "card5", "card6", "addr1", "addr2", "P_emaildomain", "C1", "C2",
    "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "D1", "D2",
    "D3", "D4", "M1", "M2", "M3", "M4", "M6",
]


class Transaction(BaseModel):
    TransactionAmt: float = 100.0
    ProductCD: float = 0.0
    card1: float = 0.0
    card2: float = 0.0
    card3: float = 150.0
    card4: float = 0.0
    card5: float = 226.0
    card6: float = 2.0
    addr1: float = 0.0
    addr2: float = 87.0
    P_emaildomain: float = 0.0
    C1: float = 1.0
    C2: float = 1.0
    C3: float = 0.0
    C4: float = 0.0
    C5: float = 0.0
    C6: float = 1.0
    C7: float = 0.0
    C8: float = 0.0
    C9: float = 1.0
    C10: float = 0.0
    D1: float = 0.0
    D2: float = -999.0
    D3: float = -999.0
    D4: float = -999.0
    M1: float = -1.0
    M2: float = -1.0
    M3: float = -1.0
    M4: float = -1.0
    M6: float = -1.0


@app.post("/predict")
def predict(tx: Transaction):
    model = _model()
    data = pd.DataFrame([tx.model_dump()])[FEATURE_LIST]
    proba = model.predict_proba(data)[0][1]
    flagged = bool(proba >= 0.3)
    return {
        "fraud_probability": round(float(proba), 4),
        "flagged": flagged,
        "risk_level": "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.3 else "LOW",
    }


# ---------------------------------------------------------------------------
# Serve SPA
# ---------------------------------------------------------------------------

STATIC_PATH.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
