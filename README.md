# 📊 Churn Prediction XAI Dashboard

A Streamlit dashboard for batch customer churn prediction with Explainable AI (SHAP) and PDF report export.

## Features
- Upload CSV, XLSX, JSON, or SQL files
- Auto-detects churn column
- Full ML pipeline: preprocessing → encoding → SMOTE → 5 models → SHAP
- Simple Mode / Expert Mode toggle
- Individual customer risk gauge
- PDF report download (technical + non-technical)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages
| Page | What it does |
|------|-------------|
| 📁 Upload & Configure | Upload data, pick target column, run pipeline |
| 📊 Overview | KPI cards, churn rate, plain English summary |
| 🏆 Model Results | Compare 5 models, ROC curves, confusion matrix |
| 🔍 XAI Explanations | SHAP global importance + beeswarm (Expert Mode) |
| 👤 Customer Lookup | Per-customer churn probability + SHAP waterfall |
| 📥 Download Report | One-click PDF report generation |

## Requirements
Python 3.9+
