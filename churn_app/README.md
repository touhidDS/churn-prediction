# Churn Prediction XAI — Streamlit App

A web app that uploads a CSV or SQL file, trains churn models, explains predictions via SHAP, and generates a downloadable PDF report.

## Quick Start

```bash
cd churn_app
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **File upload**: CSV or SQL dump
- **Auto-preprocessing**: numeric conversion, date handling, missing values, encoding
- **SMOTE** for class imbalance
- **4 models**: RandomForest, GradientBoosting, XGBoost, LightGBM
- **SHAP explainability**: global feature importance + summary beeswarm
- **PDF report**: model comparison, metrics, top churn drivers

## Deploy to Cloud

### Streamlit Cloud
1. Push to GitHub
2. Connect repo at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Set `app.py` as the main file

### Render
1. Create `render.yaml` with Python runtime
2. Add start command: `streamlit run app.py --server.port $PORT`

## Project Structure

```
churn_app/
├── app.py
├── requirements.txt
├── src/
│   ├── file_loader.py   # CSV + SQL loading
│   ├── preprocessor.py  # Numeric/date/cat processing
│   ├── trainer.py       # Feature eng + training
│   ├── explainer.py      # SHAP analysis
│   └── pdf_report.py    # Text-only PDF generation
```