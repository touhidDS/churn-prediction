# Customer Churn Prediction Dashboard

A Streamlit app that takes a raw customer dataset (CSV, XLSX, or a SQL dump), runs it through a full churn-prediction pipeline, and explains *why* the model thinks each customer is at risk — using SHAP.

**Live demo:** https://churn-prediction-bjjwugurckebsr6pwtmbj6.streamlit.app/

## What it does

Upload a dataset, point it at your churn/target column (or let it auto-detect one), and the app will:

1. Clean and encode the data — handles missing values, categorical encoding (label encoding for low/high-cardinality columns, one-hot for the middle range), and basic feature engineering (ratios, threshold flags, top-correlation interaction terms).
2. Balance the training set with SMOTE, since churn datasets are almost always skewed toward "stayed."
3. Train and compare five models — Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM — and pick the best one by F1-score.
4. Explain the predictions with SHAP: global feature importance, a beeswarm plot, and per-customer waterfall breakdowns.
5. Generate a PDF report with an executive summary, risk segmentation, and recommended actions — written in plain language, not just model metrics.

## Pages

| Page | What it's for |
|---|---|
| Upload & Configure | Upload a file, pick the target column, run the pipeline |
| Overview | Churn rate, KPIs, model comparison at a glance |
| XAI Explanations | Global SHAP importance + beeswarm plot |
| Customer Lookup | Individual churn probability with a SHAP waterfall explanation |
| Download Report | One-click PDF export |

## Running it locally

```bash
git clone <your-repo-url>
cd churn-prediction-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Requires Python 3.9+.

## How it works under the hood

- **Data loading**: CSV and XLSX go through pandas directly. SQL dumps get run against an in-memory SQLite database — MySQL-specific syntax (`AUTO_INCREMENT`, `ENGINE=`, charset declarations, etc.) is stripped out first so the dump is portable.
- **Target detection**: if you don't specify a column, it searches column names for common churn-related keywords (`churn`, `attrition`, `exited`, `target`, ...) in priority order.
- **Feature selection**: after scaling, features with pairwise correlation ≥ 0.90 get pruned (keeping whichever one correlates more strongly with the target), then a Random Forest importance ranking keeps the top ~70%.
- **SHAP**: uses `TreeExplainer` for tree-based models and `LinearExplainer` for logistic regression, computed once per session and reused across the importance chart, beeswarm, and waterfall views.

## Known limitations

- SHAP computation can be slow on large datasets (1000+ rows) since it's computed for the full test set on pipeline run.
- The SQL cleanup targets MySQL-style dumps specifically; other dialects may need manual tweaks.
- Everything runs in-memory and resets on refresh — there's no persistence layer, which is fine for a demo/analysis tool but not for production use.

## Tech stack

Streamlit · pandas / numpy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · LightGBM · SHAP · ReportLab (PDF generation)
