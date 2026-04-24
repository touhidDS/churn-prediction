# 📊 Churn Prediction XAI — Streamlit App

## What it does
Upload a **CSV or SQL** file → the app automatically runs your full churn-prediction pipeline and produces a **downloadable PDF report** containing:

- Dataset summary
- 4-model comparison table (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Confusion matrices
- ROC curves
- Metric comparison charts
- SHAP global feature importance
- SHAP beeswarm plot
- SHAP waterfall for a sample churned customer
- Top churn drivers ranked by SHAP impact

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **New app**.
3. Point it to your repo and `app.py`.
4. Click **Deploy** — done!

## File structure
```
churn_app/
├── app.py            ← main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← this file
```

## Sidebar settings you can tune
| Setting | Default | Description |
|---|---|---|
| Test Split Size | 0.20 | Fraction of data held out for evaluation |
| Correlation Threshold | 0.90 | Remove features correlated above this value |
| Missing-Value Drop Threshold | 0.50 | Drop columns with more than this fraction missing |
