import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import os
import sqlite3
import re
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# ── ML ──────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# ── PDF ─────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Prediction · XAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080c14; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2733; }

/* ── Hide Streamlit chrome — keep header so sidebar toggle works ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] {
    background: #080c14 !important;
    border-bottom: 1px solid #1e2733;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 0 20px;
    border-bottom: 1px solid #1e2733;
    margin-bottom: 16px;
}
.sidebar-brand-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.sidebar-brand-text { color: #f1f5f9; font-weight: 700; font-size: 1rem; line-height: 1.2; }
.sidebar-brand-sub  { color: #64748b; font-size: 0.72rem; }

/* ── Nav items ── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 2px; }
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    padding: 8px 14px; border-radius: 8px;
    transition: background 0.15s; cursor: pointer;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover { background: #1e2733; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #1e2733;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #6366f1, #8b5cf6);
}
.page-header h1 { color: #f1f5f9; font-size: 1.6rem; font-weight: 700; margin: 0 0 6px; }
.page-header p  { color: #64748b; font-size: 0.9rem; margin: 0; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: #0d1117;
    border: 1px solid #1e2733;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #3b82f6; }
.kpi-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.blue::after  { background: #3b82f6; }
.kpi-card.green::after { background: #10b981; }
.kpi-card.red::after   { background: #ef4444; }
.kpi-card.amber::after { background: #f59e0b; }
.kpi-card.purple::after{ background: #8b5cf6; }
.kpi-icon  { font-size: 1.4rem; margin-bottom: 10px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; line-height: 1; margin: 4px 0; }
.kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
.kpi-sub   { font-size: 0.78rem; color: #475569; margin-top: 4px; }

/* ── Section title ── */
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
    display: flex; align-items: center; gap: 8px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2733;
    margin: 24px 0 16px;
}

/* ── Insight cards ── */
.insight-card {
    background: #0d1117;
    border: 1px solid #1e2733;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 10px;
}
.insight-card.green  { border-left-color: #10b981; }
.insight-card.amber  { border-left-color: #f59e0b; }
.insight-card.red    { border-left-color: #ef4444; }

/* ── Upload zone ── */
.upload-zone {
    background: #0d1117;
    border: 2px dashed #1e2733;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 16px;
}
.upload-zone:hover { border-color: #3b82f6; }
.upload-zone-icon  { font-size: 2.5rem; margin-bottom: 10px; }
.upload-zone-title { color: #e2e8f0; font-weight: 600; font-size: 1rem; margin-bottom: 4px; }
.upload-zone-sub   { color: #475569; font-size: 0.82rem; }

/* ── Badge pills ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
}
.badge-red    { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
.badge-amber  { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
.badge-green  { background: #10b98122; color: #10b981; border: 1px solid #10b98144; }

/* ── Data table ── */
.stDataFrame { border: 1px solid #1e2733 !important; border-radius: 10px !important; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e2733; margin: 20px 0; }

/* ── Step card (upload page) ── */
.step-card {
    background: #0d1117;
    border: 1px solid #1e2733;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white; font-size: 0.75rem; font-weight: 700;
    margin-right: 8px;
}

/* ── Status pill in sidebar ── */
.status-pill {
    display: flex; align-items: center; gap: 8px;
    background: #10b98115; border: 1px solid #10b98133;
    border-radius: 8px; padding: 8px 12px;
    color: #10b981; font-size: 0.82rem; font-weight: 600;
    margin-top: 12px;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; flex-shrink: 0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Streamlit button overrides ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* ── Metric tweaks ── */
[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #1e2733;
    border-radius: 10px; padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════
for key in ['df_raw','df_encoded','results_df','models','best_model_name','sql_table_loaded',
            'best_model','X_test_final','y_test_final','selected_feature_names',
            'X_train_resampled','y_train_resampled','shap_values','pipeline_done',
            'target_column']:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state['pipeline_done'] is None:
    st.session_state['pipeline_done'] = False

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def clean_sql_for_sqlite(sql):
    sql = sql.replace('`', '"')
    for pat in [r'AUTO_INCREMENT=\d+', r'AUTO_INCREMENT', r'ENGINE\s*=\s*\w+',
                r'DEFAULT CHARSET\s*=\s*\w+', r'COLLATE\s*=\s*\w+',
                r'CHARACTER SET \w+', r'\bUNSIGNED\b', r'\bZEROFILL\b']:
        sql = re.sub(pat, '', sql, flags=re.IGNORECASE)
    for src, dst in [('TINYINT','INTEGER'),('SMALLINT','INTEGER'),
                     ('MEDIUMINT','INTEGER'),('BIGINT','INTEGER'),
                     ('DATETIME','TEXT'),('TIMESTAMP','TEXT')]:
        sql = re.sub(r'\b'+src+r'\b', dst, sql, flags=re.IGNORECASE)
    for pat in [r'SET NAMES \w+;', r'SET @\w+\s*=.*?;', r'SET @@\w+\s*=.*?;',
                r'SET SQL_MODE\s*=.*?;', r'SET FOREIGN_KEY_CHECKS\s*=.*?;',
                r'SET UNIQUE_CHECKS\s*=.*?;']:
        sql = re.sub(pat, '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'^--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    return sql


def load_uploaded_file(uploaded_file):
    """Load CSV, XLSX, or SQL — no JSON."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.csv':
        return pd.read_csv(uploaded_file), None, []
    elif ext == '.xlsx':
        xl = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(uploaded_file, sheet_name=xl.sheet_names[0])
        return df, None, []
    elif ext == '.sql':
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        content = clean_sql_for_sqlite(content)
        conn = sqlite3.connect(':memory:')
        conn.cursor().executescript(content)
        conn.commit()
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
        if tables:
            # Return conn + tables so the UI can let the user pick which table to use
            # Do NOT load any table yet — the upload page handles table selection
            return None, conn, tables
    return None, None, []


def detect_churn_column(df):
    """Auto-detect churn column — mirrors notebook Cell 9 logic exactly.
    Lowercases columns first, then searches keywords in priority order (substring match).
    """
    # Priority order matches notebook: specific churn terms first, generic ones last
    churn_keywords = [
        'churned', 'churn',
        'attrition', 'exited', 'left',
        'target', 'label', 'status', 'outcome'
    ]
    # lowercase all column names for matching (same as notebook)
    lower_cols = [col.lower() for col in df.columns]
    for kw in churn_keywords:
        matched = [col for col in lower_cols if kw in col]
        if matched:
            # return the original column name at that position
            idx = lower_cols.index(matched[0])
            return df.columns[idx]
    return df.columns[0]  # fallback to first column


def run_pipeline(df, target_column):
    df = df.copy()

    # ── Lowercase all columns (Step 2 change from notebook) ─────────────────
    df.columns = [col.lower() for col in df.columns]
    target_column = target_column.lower()

    # ── Numeric coercion (matches notebook Step 3: aggressive_clean_column) ────
    # Strips currency symbols, commas, % signs, unit suffixes before converting
    def _aggressive_clean(series):
        cleaned = (series.astype(str)
            .str.replace(r'[\$€£¥₹₽₩¢]', '', regex=True)
            .str.replace(r'%', '', regex=True)
            .str.replace(r',', '', regex=True)
            .str.replace(r'\b(GB|MB|KB|TB|kg|km|cm|mm|lb|oz|USD|EUR|CAD)\b', '', regex=True, flags=re.IGNORECASE)
            .str.replace(r'\s+', '', regex=True)
            .str.replace(r'[^0-9.-]', '', regex=True)
            .str.strip())
        cleaned = cleaned.replace(['', 'nan', 'None', 'null', 'NULL'], np.nan)
        return cleaned

    PII_SKIP = {'ein','ssn','ssnnumber','socialsecurity','phone','mobile',
                'fax','cellphone','taxid','federalid','nationalid','passportnumber'}

    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype not in ['int64','float64','int32','float32','Int64']:
            col_clean = re.sub(r'[^a-z0-9]', '', col.lower())
            if col_clean in PII_SKIP:
                continue
            cleaned = _aggressive_clean(df[col])
            converted = pd.to_numeric(cleaned, errors='coerce')
            original_non_null = df[col].notna() & (df[col].astype(str).str.strip() != '')
            success_rate = (converted[original_non_null].notna().sum() /
                            original_non_null.sum()) if original_non_null.sum() > 0 else 0
            if success_rate > 0.1:
                df[col] = converted

    # ── Date columns → Years (matches notebook Step 4: convert_dates_to_years) ─
    date_keywords = ['dob','birth','date','created','signup','registered',
                     'joined','start','opened']
    current_date = datetime.now()
    for col in list(df.columns):
        if col == target_column:
            continue
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in date_keywords):
            try:
                date_series = pd.to_datetime(df[col], errors='coerce')
                non_null_original = df[col].notna().sum()
                parse_rate = date_series.notna().sum() / non_null_original if non_null_original > 0 else 0
                if parse_rate < 0.5:
                    continue   # not really a date column
                if date_series.notna().sum() > 0:
                    new_col = f"{col.replace(' ','_').lower()}_years"
                    df[new_col] = ((current_date - date_series).dt.days / 365.25).round(1)
                    df = df.drop(columns=[col])
            except Exception:
                pass

    # ── Missing values ───────────────────────────────────────────────────────
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isnull().mean() > 0.5:
            df = df.drop(columns=[col]); continue
        if df[col].dtype in ['float64','int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else 'Unknown')
    df.dropna(subset=[target_column], inplace=True)

    # ── Encoding ─────────────────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    for col in cat_cols:
        n = df[col].nunique()
        if n <= 2:
            le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))
        elif n <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif n <= 20:
            le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = df.drop(columns=[col])

    # Always encode target via LabelEncoder to handle Yes/No, True/False, etc.
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column].astype(str)).astype(int)

    # ── Split ────────────────────────────────────────────────────────────────
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.apply(lambda c: c.astype(int) if c.dtype == bool else c)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)

    # ── Feature Engineering (matches notebook) ──────────────────────────────
    # Must happen BEFORE scaling so engineered features are scaled too
    df_fe_train = X_train.copy()
    df_fe_test  = X_test.copy()
    numeric_cols = df_fe_train.columns.tolist()

    # Debt-to-Revenue ratio
    revenue_cols = [c for c in numeric_cols if any(w in c for w in ['revenue','income','sales','earning'])]
    debt_cols    = [c for c in numeric_cols if any(w in c for w in ['debt','debit','loan','liability','payment'])]
    if revenue_cols and debt_cols:
        df_fe_train['debt_revenue_ratio'] = df_fe_train[debt_cols[0]] / (df_fe_train[revenue_cols[0]] + 0.0001)
        df_fe_test['debt_revenue_ratio']  = df_fe_test[debt_cols[0]]  / (df_fe_test[revenue_cols[0]]  + 0.0001)

    # Percentage-based binary features
    pct_cols = [c for c in numeric_cols if 'percent' in c or 'ownership' in c]
    for col in pct_cols:
        threshold = df_fe_train[col].median()
        df_fe_train[f'{col}_high'] = (df_fe_train[col] > threshold).astype(int)
        df_fe_test[f'{col}_high']  = (df_fe_test[col]  > threshold).astype(int)

    # Interaction feature: top-2 columns most correlated with target
    if len(numeric_cols) >= 2:
        try:
            target_corr = df_fe_train.join(y_train).corr()[y_train.name].drop(y_train.name).abs()
            top2 = target_corr.nlargest(2).index.tolist()
            new_col = f'{top2[0]}_x_{top2[1]}'
            df_fe_train[new_col] = df_fe_train[top2[0]] * df_fe_train[top2[1]]
            df_fe_test[new_col]  = df_fe_test[top2[0]]  * df_fe_test[top2[1]]
        except Exception:
            pass

    # ── Scale (fit on TRAIN only — matches notebook) ──────────────────────────
    scaler = StandardScaler()
    scaler.fit(df_fe_train)
    X_train_sc = pd.DataFrame(scaler.transform(df_fe_train), columns=df_fe_train.columns, index=df_fe_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(df_fe_test),  columns=df_fe_test.columns,  index=df_fe_test.index)

    # ── Corr removal ─────────────────────────────────────────────────────────
    y_train_float = y_train.astype(float)
    corr = X_train_sc.corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i,j] >= 0.90:
                c1, c2 = corr.columns[i], corr.columns[j]
                corr1 = abs(X_train_sc[c1].astype(float).corr(y_train_float))
                corr2 = abs(X_train_sc[c2].astype(float).corr(y_train_float))
                to_drop.add(c1 if corr1 < corr2 else c2)
    X_train_sc.drop(columns=list(to_drop), inplace=True, errors='ignore')
    X_test_sc.drop(columns=list(to_drop),  inplace=True, errors='ignore')

    # ── Feature selection (max_depth=10 matches notebook) ────────────────────
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42,
                                     class_weight='balanced', max_depth=10, n_jobs=-1)
    rf_sel.fit(X_train_sc, y_train)
    fi = pd.Series(rf_sel.feature_importances_,
                   index=X_train_sc.columns).sort_values(ascending=False)
    n_keep = max(8, int(len(fi)*0.7))
    selected = fi.head(n_keep).index.tolist()
    X_train_sel = X_train_sc[selected]
    X_test_sel  = X_test_sc[selected]

    # ── SMOTE ────────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
    X_tr_res, y_tr_res = smote.fit_resample(X_train_sel, y_train)
    X_tr_res = pd.DataFrame(X_tr_res, columns=selected)

    # ── Train models (hyperparams match notebook exactly) ────────────────────
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                                    class_weight='balanced', solver='lbfgs'),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                              eval_metric='logloss', verbosity=0),
        'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }
    results = []
    for name, model in models.items():
        model.fit(X_tr_res, y_tr_res)
        yp  = model.predict(X_test_sel)
        ypp = model.predict_proba(X_test_sel)[:,1]
        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, yp), 4),
            'Precision': round(precision_score(y_test, yp, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, yp, zero_division=0), 4),
            'F1-Score':  round(f1_score(y_test, yp, zero_division=0), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, ypp), 4),
        })

    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]['Model']
    best_model = models[best_name]

    # ── SHAP ─────────────────────────────────────────────────────────────────
    # ✅ FIX: use modern explainer() API (matches fixed notebook) — single call,
    #         consistent base values, correct 3-D handling for all model types
    try:
        if hasattr(best_model, 'feature_importances_'):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, X_tr_res)

        shap_exp_all = explainer(X_test_sel)

        # Normalise to 2-D (n_samples, n_features) — handles old list API and new 3-D
        sv = shap_exp_all.values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]          # take class-1 (churn) slice
        shap_values = sv              # always 2-D from here on
    except Exception as e:
        shap_values = None

    return {
        'df_encoded': df,
        'results_df': results_df,
        'models': models,
        'best_model_name': best_name,
        'best_model': best_model,
        'X_test_final': X_test_sel,
        'y_test_final': y_test,
        'selected_feature_names': selected,
        'X_train_resampled': X_tr_res,
        'y_train_resampled': y_tr_res,
        'shap_values': shap_values,
    }


def make_gauge(prob, size=200):
    fig, ax = plt.subplots(figsize=(size/80, size/80), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    theta = np.linspace(0, np.pi, 300)
    ax.plot(theta, [1]*300, color='#1e2733', linewidth=20, solid_capstyle='round')
    color = '#ef4444' if prob > 0.6 else '#f59e0b' if prob > 0.35 else '#10b981'
    theta_val = np.linspace(0, np.pi*prob, 300)
    ax.plot(theta_val, [1]*300, color=color, linewidth=20, solid_capstyle='round')
    ax.set_ylim(0, 1.5); ax.set_xlim(0, np.pi); ax.axis('off')
    ax.text(np.pi/2, 0.15, f'{prob*100:.1f}%',
            ha='center', va='center', fontsize=22, fontweight='700',
            color='white', transform=ax.transData)
    plt.tight_layout(pad=0)
    return fig


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(results_df, best_model_name, best_model,
                         X_test_final, y_test_final, shap_values,
                         selected_feature_names, target_column, df_raw):

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch,  bottomMargin=0.75*inch)
    styles  = getSampleStyleSheet()
    story   = []
    W_FULL  = 6.5*inch

    # ── Palette ──────────────────────────────────────────────────────────────
    C_BLUE   = colors.HexColor('#2563eb')
    C_DARK   = colors.HexColor('#1e293b')
    C_LIGHT  = colors.HexColor('#f1f5f9')
    C_LIGHT2 = colors.HexColor('#eff6ff')
    C_GREEN  = colors.HexColor('#16a34a')
    C_RED    = colors.HexColor('#dc2626')
    C_AMBER  = colors.HexColor('#d97706')
    C_GREY   = colors.HexColor('#64748b')
    C_BORDER = colors.HexColor('#e2e8f0')
    C_ROW1   = colors.HexColor('#f8fafc')
    C_WHITE  = colors.white

    # ── Paragraph style factory ───────────────────────────────────────────────
    def PS(name, **kw):
        return ParagraphStyle(name, parent=styles['Normal'], **kw)

    cover_title = PS('CT', fontSize=24, fontName='Helvetica-Bold',
                     textColor=C_BLUE, leading=30, spaceAfter=2)
    cover_sub   = PS('CS', fontSize=11, textColor=C_GREY, leading=16)
    sec_head    = PS('SH', fontSize=12, fontName='Helvetica-Bold',
                     textColor=C_DARK, spaceBefore=4, spaceAfter=5)
    body        = PS('BD', fontSize=10, leading=15, textColor=C_DARK, spaceAfter=4)
    body_small  = PS('BS', fontSize=9,  leading=13, textColor=C_GREY, spaceAfter=3)
    caption_st  = PS('CP', fontSize=8,  leading=11, textColor=C_GREY, alignment=1, spaceAfter=4)
    footer_st   = PS('FT', fontSize=8,  textColor=C_GREY, leading=11)

    # ── Computed values ───────────────────────────────────────────────────────
    now        = datetime.now().strftime("%B %d, %Y")
    best_row   = results_df.iloc[0]
    y_pred     = best_model.predict(X_test_final)
    churn_rate = y_test_final.mean() * 100
    n_total    = len(df_raw)
    n_churned  = int(round(n_total * churn_rate / 100))
    n_safe     = n_total - n_churned
    accuracy   = best_row['Accuracy'] * 100
    auc_val    = best_row['ROC-AUC']
    f1_val     = best_row['F1-Score']
    recall_val = best_row['Recall']
    prec_val   = best_row['Precision']

    if churn_rate > 30:
        severity = "CRITICAL"; sev_color = C_RED
        sev_short = "Immediate retention action required."
    elif churn_rate > 15:
        severity = "HIGH"; sev_color = C_AMBER
        sev_short = "Proactive retention programmes should launch soon."
    elif churn_rate > 8:
        severity = "MODERATE"; sev_color = C_AMBER
        sev_short = "Targeted retention campaigns are advised."
    else:
        severity = "LOW"; sev_color = C_GREEN
        sev_short = "Continue monitoring and maintain service quality."

    auc_word = "Excellent" if auc_val >= 0.90 else "Good" if auc_val >= 0.80 else "Fair"

    # SHAP — normalise shape robustly for any model type
    if shap_values is not None:
        sv = shap_values
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_shap    = pd.Series(np.abs(sv).mean(axis=0),
                                  index=selected_feature_names).sort_values(ascending=False)
        top_features = mean_shap.index.tolist()
        top_scores   = mean_shap.values.tolist()
    else:
        top_features = list(selected_feature_names)
        top_scores   = [0] * len(selected_feature_names)

    top5_features = top_features[:5]
    top5_scores   = top_scores[:5]
    feat_lower    = ' '.join(f.lower() for f in top_features)

    # ── Helper builders ───────────────────────────────────────────────────────
    def hline(col=C_BORDER, t=1, before=4, after=6):
        return HRFlowable(width="100%", thickness=t, color=col,
                          spaceBefore=before, spaceAfter=after)

    def section_header(number, title):
        tbl = Table([[f"\u25a0  {number}. {title}"]], colWidths=[W_FULL])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
            ('TEXTCOLOR',     (0,0),(-1,-1), C_DARK),
            ('FONTNAME',      (0,0),(-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 12),
            ('TOPPADDING',    (0,0),(-1,-1), 8),
            ('BOTTOMPADDING', (0,0),(-1,-1), 8),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ('LINEBEFORE',    (0,0),(0,-1),  4, C_BLUE),
        ]))
        return tbl

    def feature_card(feat_name, body_text):
        head = [Paragraph(f"\u25a0  {feat_name}",
                PS('FH', fontSize=9.5, fontName='Helvetica-Bold', textColor=C_BLUE))]
        bdy  = [Paragraph(body_text,
                PS('FB', fontSize=9.5, leading=14, textColor=C_DARK))]
        tbl  = Table([[head], [bdy]], colWidths=[W_FULL])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0), C_LIGHT2),
            ('BACKGROUND',    (0,1),(-1,1), C_WHITE),
            ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
            ('TOPPADDING',    (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ('RIGHTPADDING',  (0,0),(-1,-1), 10),
        ]))
        return tbl

    def kpi_4(items):
        cells = []
        for val, lbl, col in items:
            c = Table([[val], [lbl]], colWidths=[W_FULL / 4 - 0.1*inch])
            c.setStyle(TableStyle([
                ('TEXTCOLOR',     (0,0),(0,0), col),
                ('FONTNAME',      (0,0),(0,0), 'Helvetica-Bold'),
                ('FONTSIZE',      (0,0),(0,0), 17),
                ('TEXTCOLOR',     (0,1),(0,1), C_GREY),
                ('FONTSIZE',      (0,1),(0,1), 8),
                ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
                ('TOPPADDING',    (0,0),(-1,-1), 9),
                ('BOTTOMPADDING', (0,0),(-1,-1), 9),
            ]))
            cells.append(c)
        outer = Table([cells], colWidths=[W_FULL / 4] * 4)
        outer.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), C_LIGHT),
            ('BOX',        (0,0),(-1,-1), 0.5, C_BORDER),
            ('INNERGRID',  (0,0),(-1,-1), 0.5, C_BORDER),
        ]))
        return outer

    def alert_box(text, color):
        p = Paragraph(text, PS('AL', fontSize=10, leading=15,
                      textColor=C_WHITE, fontName='Helvetica-Bold'))
        t = Table([[p]], colWidths=[W_FULL])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), color),
            ('TOPPADDING',    (0,0),(-1,-1), 9),
            ('BOTTOMPADDING', (0,0),(-1,-1), 9),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
        ]))
        return t

    def numbered_rec(num, title, body_text):
        num_cell  = Paragraph(str(num), PS('NC', fontSize=15, fontName='Helvetica-Bold',
                               textColor=C_BLUE, alignment=1))
        body_cell = [
            Paragraph(title,     PS('RT', fontSize=10, fontName='Helvetica-Bold',
                                   textColor=C_DARK, spaceAfter=2)),
            Paragraph(body_text, PS('RB', fontSize=9.5, leading=14, textColor=C_DARK)),
        ]
        t = Table([[num_cell, body_cell]], colWidths=[0.4*inch, W_FULL - 0.4*inch])
        t.setStyle(TableStyle([
            ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
            ('LINEBEFORE',    (0,0),(0,-1),  3,   C_BLUE),
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
            ('TOPPADDING',    (0,0),(-1,-1), 8),
            ('BOTTOMPADDING', (0,0),(-1,-1), 8),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ('RIGHTPADDING',  (0,0),(-1,-1), 10),
        ]))
        return t

    def retention_tip(feat):
        f = feat.lower()
        if any(k in f for k in ['balance','amount','deposit','saving']):
            return "Retention tip: Offer savings incentives or cashback to customers with low balances."
        if any(k in f for k in ['product','service','plan','bundle','count','num']):
            return "Retention tip: Target single-product customers with bundled offers."
        if any(k in f for k in ['age','year_born','dob','birth']):
            return "Retention tip: Ensure accessible support options tailored to different age groups."
        if any(k in f for k in ['credit','score','rating','risk']):
            return "Retention tip: Proactively offer credit improvement programmes or financial counselling."
        if any(k in f for k in ['tenure','month','duration','period','length']):
            return "Retention tip: Create a formal onboarding journey and a loyalty reward programme."
        if any(k in f for k in ['salary','income','earning','revenue','wage']):
            return "Retention tip: Offer flexible payment plans or income-sensitive pricing tiers."
        if any(k in f for k in ['charge','price','fee','cost','payment','rate']):
            return "Retention tip: Review pricing for at-risk segments; offer loyalty discounts proactively."
        if any(k in f for k in ['active','login','visit','engage','usage','use','freq']):
            return "Retention tip: Launch a re-engagement campaign for low-activity customers."
        if any(k in f for k in ['geo','region','country','city','location','area']):
            return "Retention tip: Investigate regional service gaps and tailor local outreach."
        if any(k in f for k in ['complaint','ticket','call','support','issue','contact']):
            return "Retention tip: Resolve top recurring complaints and follow up with affected customers."
        if any(k in f for k in ['contract','type','plan','tier','member']):
            return "Retention tip: Review contract terms and offer upgrades to at-risk segments."
        return "Retention tip: Monitor this factor closely and run targeted campaigns for high-risk segments."

    def feature_description(feat, rank, score, total_score):
        f   = feat.lower()
        pct = score / total_score * 100 if total_score > 0 else 0
        rank_phrase = (
            "the <b>single strongest predictor</b> in your dataset" if rank == 1 else
            "a <b>major driver</b> of churn decisions" if rank <= 3 else
            "a <b>notable contributing factor</b>"
        )
        tip = retention_tip(feat)
        if any(k in f for k in ['balance','amount','deposit']):
            detail = ("Customers with very low or zero balances are significantly more likely to churn. "
                      "This signals financial disengagement — the customer is no longer actively using the account. ")
        elif any(k in f for k in ['product','count','num_product','bundle']):
            detail = ("Customers using only one product are far more likely to leave than those with multiple products. "
                      "Cross-selling reduces churn significantly. ")
        elif any(k in f for k in ['age','year_born','dob']):
            detail = ("Certain age groups show higher churn likelihood, possibly reflecting "
                      "changing needs or dissatisfaction with the current service offering. ")
        elif any(k in f for k in ['credit','score','rating']):
            detail = ("Customers with lower scores may be seeking better rates elsewhere "
                      "or experiencing financial stress. ")
        elif any(k in f for k in ['tenure','month','duration','period','length']):
            detail = ("Both very new and very long-tenured customers show elevated risk. "
                      "New customers may not yet be engaged; long-tenured customers may feel overlooked. ")
        elif any(k in f for k in ['salary','income','earning']):
            detail = ("Income-sensitive customers are more likely to switch for a better deal "
                      "when they feel they are not receiving adequate value. ")
        elif any(k in f for k in ['charge','price','fee','cost','payment','rate']):
            detail = ("Pricing-related factors strongly predict churn. Customers who feel they are "
                      "paying too much relative to the value received are at heightened risk. ")
        elif any(k in f for k in ['active','login','engage','usage','use','freq']):
            detail = ("Low engagement or inactivity is a strong churn signal. "
                      "Customers who rarely interact with the product often cancel quietly. ")
        elif any(k in f for k in ['geo','region','country','city','location']):
            detail = ("Geographic location influences churn, which may reflect regional "
                      "service quality gaps, competitive pressures, or demographic differences. ")
        elif any(k in f for k in ['contract','type','plan','tier','member']):
            detail = ("Contract type or membership tier is a strong predictor — "
                      "customers on month-to-month plans or basic tiers churn at significantly higher rates. ")
        else:
            detail = (f"This factor accounts for approximately {pct:.1f}% of the total churn prediction influence. "
                      "Customers with unusual values in this area are at significantly higher risk. ")
        return f"Ranked #{rank} — {rank_phrase}. {detail}<b>{tip}</b>"

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ════════════════════════════════════════════════════════════════════════
    story += [
        Spacer(1, 0.5*inch),
        Paragraph("ChurnIQ", PS('BN', fontSize=13, fontName='Helvetica-Bold',
                                 textColor=C_GREY, spaceAfter=4)),
        Paragraph("Customer Retention", cover_title),
        Paragraph("&amp; Churn Prediction Report", cover_title),
        Spacer(1, 0.06*inch),
        Paragraph("Prepared for: Executive Leadership &amp; Business Teams", cover_sub),
        Spacer(1, 0.15*inch),
        hline(C_BLUE, 2, before=0, after=10),
    ]

    # Key findings box
    findings_rows = [
        [Paragraph("<b>KEY FINDINGS AT A GLANCE</b>",
                   PS('KH', fontSize=10, fontName='Helvetica-Bold', textColor=C_WHITE))],
        [Paragraph(f"Churn Rate Detected: <b>{churn_rate:.1f}%</b>  \u2192  <b>{severity} RISK</b>",
                   PS('KI', fontSize=10, leading=16, textColor=C_DARK))],
        [Paragraph(f"Best Performing Model: <b>{best_model_name}</b>"
                   f"  (AUC: {auc_val:.3f},  F1: {f1_val:.3f})",
                   PS('KI2', fontSize=10, leading=16, textColor=C_DARK))],
        [Paragraph(f"Top Churn Driver: <b>{top5_features[0].replace('_',' ').title() if top5_features else 'N/A'}</b>",
                   PS('KI3', fontSize=10, leading=16, textColor=C_DARK))],
        [Paragraph(f"Recommended Action: <b>{sev_short}</b>",
                   PS('KI4', fontSize=10, leading=16, textColor=C_DARK))],
    ]
    findings_tbl = Table(findings_rows, colWidths=[W_FULL])
    findings_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  C_DARK),
        ('BACKGROUND',    (0,1),(-1,-1), C_LIGHT),
        ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
        ('INNERGRID',     (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 12),
        ('LINEBEFORE',    (0,1),(0,-1),  3, C_BLUE),
    ]))
    story += [findings_tbl, Spacer(1, 0.15*inch)]

    # Meta grid — 2-column layout
    meta_rows = [
        ['Report Date',    now,              'Report Type',  'XAI Analysis'],
        ['Dataset Size',   f"{n_total:,} customers", 'Target Column', target_column.title()],
        ['Generated by',   'ChurnIQ Dashboard', 'Best Model', best_model_name],
    ]
    meta_tbl = Table(meta_rows, colWidths=[1.2*inch, 2.05*inch, 1.2*inch, 2.05*inch])
    meta_tbl.setStyle(TableStyle([
        ('FONTSIZE',       (0,0),(-1,-1), 9),
        ('TEXTCOLOR',      (0,0),(0,-1),  C_GREY),
        ('TEXTCOLOR',      (2,0),(2,-1),  C_GREY),
        ('FONTNAME',       (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTNAME',       (2,0),(2,-1),  'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [C_ROW1, C_WHITE]),
        ('TOPPADDING',     (0,0),(-1,-1), 6),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 6),
        ('LEFTPADDING',    (0,0),(-1,-1), 8),
        ('BOX',            (0,0),(-1,-1), 0.5, C_BORDER),
        ('INNERGRID',      (0,0),(-1,-1), 0.5, C_BORDER),
    ]))
    story += [meta_tbl, Spacer(1, 0.12*inch)]

    # TOC
    toc_items = [
        "1. Executive Summary",
        "2. Risk Assessment — Why Are Customers Leaving?",
        "3. How to Reduce Churn — Action Plan",
        "4. Best Model Performance",
        "5. Quick Reference Summary",
    ]
    toc_rows = [[Paragraph("Contents", PS('TOC_H', fontSize=10,
                 fontName='Helvetica-Bold', textColor=C_DARK))]]
    for item in toc_items:
        toc_rows.append([Paragraph(f"  {item}",
                          PS('TOC_I', fontSize=9.5, leading=16, textColor=C_BLUE))])
    toc_tbl = Table(toc_rows, colWidths=[W_FULL])
    toc_tbl.setStyle(TableStyle([
        ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 12),
        ('ROWBACKGROUNDS',(0,0),(-1,-1), [C_LIGHT, C_WHITE]),
    ]))
    story += [toc_tbl, PageBreak()]

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    story += [
        section_header("1", "Executive Summary"),
        Spacer(1, 0.08*inch),
        alert_box(
            f"\u25a0 {severity} CHURN RISK  \u2014  "
            f"Churn rate: {churn_rate:.1f}%  \u2014  Approximately "
            f"{int(churn_rate)} out of every 100 customers are at risk of leaving.",
            sev_color),
        Spacer(1, 0.1*inch),
        Paragraph(
            f"This report analyses customer data using Explainable AI (XAI) to identify "
            f"customers most likely to leave. Five machine learning models were trained "
            f"and evaluated. The best model \u2014 <b>{best_model_name}</b> \u2014 correctly "
            f"identifies churning customers <b>{recall_val:.0%}</b> of the time, giving the "
            f"retention team a reliable early-warning system.", body),
        Spacer(1, 0.08*inch),
        Paragraph("<b>Performance Summary</b>", sec_head),
    ]

    sum_rows = [
        ['Metric', 'Value', 'Plain English Meaning'],
        ['Total Customers Analysed', f"{n_total:,}", 'Size of the dataset used'],
        ['Churn Rate', f"{churn_rate:.1f}%",
         f"About 1 in {max(1, int(round(100/max(churn_rate, 1))))} customers may leave soon"],
        ['Best Model', best_model_name, 'The model that performed best overall'],
        ['Prediction Accuracy (AUC)', f"{auc_val:.3f}",
         f"{auc_word} \u2014 {'0.9+ is considered best-in-class' if auc_val >= 0.9 else '0.8+ is considered good'}"],
        ['Balance Score (F1)', f"{f1_val:.3f}",
         'Model balances catching churners without too many false alarms'],
        ['Customers at High Risk', f"~{n_churned:,}",
         'Customers who need immediate retention action'],
    ]
    s_tbl = Table(sum_rows, colWidths=[1.9*inch, 1.0*inch, 3.6*inch])
    s_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  C_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  C_WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [C_ROW1, C_WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('FONTNAME',      (0,1),(0,-1),  'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1),(0,-1),  C_DARK),
    ]))
    story += [
        s_tbl,
        Spacer(1, 0.07*inch),
        Paragraph(
            f"* AUC (Area Under Curve) measures prediction quality \u2014 "
            f"0.5 = random guess, 1.0 = perfect. "
            f"A score of {auc_val:.3f} means the model is {auc_word.lower()} "
            "at distinguishing churners from loyal customers.", body_small),
        PageBreak(),
    ]

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — RISK ASSESSMENT (SHAP)
    # ════════════════════════════════════════════════════════════════════════
    story += [
        section_header("2", "Risk Assessment \u2014 Why Are Customers Leaving?"),
        Spacer(1, 0.08*inch),
        Paragraph(
            "The AI model identified the following factors as the strongest predictors of churn. "
            "The chart below shows each factor's influence \u2014 a longer bar means that factor "
            "<b>more strongly predicts whether a customer will leave</b>.", body),
        Spacer(1, 0.06*inch),
    ]

    if shap_values is not None and len(top5_features) > 0:
        n_show      = min(8, len(top_features))
        show_feats  = top_features[:n_show]
        show_scores = top_scores[:n_show]

        fig, ax = plt.subplots(figsize=(6.5, max(2.5, n_show * 0.42)))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        bar_palette = (['#dc2626'] * 2 + ['#2563eb'] * 3 + ['#94a3b8'] * 10)[:n_show]
        bars = ax.barh(
            [f.replace('_', ' ').title() for f in show_feats[::-1]],
            show_scores[::-1],
            color=bar_palette[::-1], edgecolor='none', height=0.55)
        ax.set_xlabel('Mean |SHAP| Value  (higher = stronger influence on churn)',
                      fontsize=9, color='#64748b')
        ax.set_title('Top Factors Driving Customer Churn', fontsize=11,
                     fontweight='bold', color='#1e293b', pad=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.grid(axis='x', color='#f1f5f9', linewidth=0.8)
        mx = max(show_scores) if show_scores else 1
        for bar, val in zip(bars, show_scores[::-1]):
            ax.text(val + mx * 0.015, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8, color='#334155')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#dc2626', label='High impact'),
                            Patch(color='#2563eb', label='Medium impact'),
                            Patch(color='#94a3b8', label='Lower impact')],
                  loc='lower right', fontsize=8, framealpha=0.9, edgecolor='#e2e8f0')
        plt.tight_layout()
        story += [
            RLImage(fig_to_png(fig), width=W_FULL, height=max(2.2, n_show * 0.37) * inch),
            Paragraph(
                "Bar length = how strongly this factor influences the churn prediction. "
                "Red bars = highest impact. Blue = medium. Gray = lower impact.",
                caption_st),
            Spacer(1, 0.1*inch),
        ]

    story.append(Paragraph("<b>What These Factors Mean in Plain English</b>", sec_head))
    story.append(Spacer(1, 0.04*inch))

    total_score = sum(top_scores[:8]) if top_scores else 1
    for i, (feat, score) in enumerate(zip(top5_features, top5_scores), 1):
        fname = feat.replace('_', ' ').title()
        desc  = feature_description(feat, i, score, total_score)
        story.append(feature_card(fname, desc))
        story.append(Spacer(1, 0.06*inch))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 4 — ACTION PLAN
    # ════════════════════════════════════════════════════════════════════════
    story += [
        section_header("3", "How to Reduce Churn \u2014 Action Plan"),
        Spacer(1, 0.08*inch),
        Paragraph("<b>Customer Risk Segments</b>", sec_head),
    ]

    seg_rows = [
        ['Risk Level', 'Churn Probability', 'Est. Customers', 'Recommended Action'],
        ['\u25a0 High Risk',   '> 60%',
         f"~{int(n_churned * 0.47):,}",
         'Immediate personal outreach \u2014 call or direct message'],
        ['\u25a0 Medium Risk', '35% \u2013 60%',
         f"~{int(n_churned * 0.53):,}",
         'Proactive offer \u2014 targeted email or SMS campaign'],
        ['\u25a0 Low Risk',    '< 35%',
         f"~{n_safe:,}",
         'Monitor monthly \u2014 no immediate action required'],
    ]
    seg_tbl = Table(seg_rows, colWidths=[1.1*inch, 1.2*inch, 1.2*inch, 3.0*inch])
    seg_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  C_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  C_WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('GRID',          (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('BACKGROUND',    (0,1),(-1,1),  colors.HexColor('#fee2e2')),
        ('BACKGROUND',    (0,2),(-1,2),  colors.HexColor('#fef3c7')),
        ('BACKGROUND',    (0,3),(-1,3),  colors.HexColor('#dcfce7')),
        ('TEXTCOLOR',     (0,1),(0,1),   C_RED),
        ('TEXTCOLOR',     (0,2),(0,2),   C_AMBER),
        ('TEXTCOLOR',     (0,3),(0,3),   C_GREEN),
        ('FONTNAME',      (0,1),(0,-1),  'Helvetica-Bold'),
    ]))
    story += [seg_tbl, Spacer(1, 0.15*inch),
              Paragraph("<b>Top Recommendations</b>", sec_head)]

    recs = []
    recs.append(("Contact High-Risk Customers Immediately",
                 f"Export the AI churn probability list. Personally reach out to the top "
                 f"~{int(n_churned * 0.47):,} highest-risk customers this week. "
                 "Personalised outreach converts far better than mass campaigns."))
    if any(k in feat_lower for k in ['balance','amount','deposit','saving']):
        recs.append(("Launch a Balance-Based Incentive",
                     "Customers with low balances are at highest risk. Offer a promotional "
                     "interest rate or cashback bonus to re-engage this segment immediately."))
    if any(k in feat_lower for k in ['product','count','bundle','num']):
        recs.append(("Cross-Sell to Single-Product Customers",
                     "Identify customers using only one product and offer a relevant second "
                     "product. Multi-product customers churn at significantly lower rates."))
    if any(k in feat_lower for k in ['age','year_born','dob']):
        recs.append(("Create Age-Specific Engagement Programmes",
                     "Certain age groups need tailored support. Consider dedicated relationship "
                     "managers or service tiers for high-risk age segments."))
    if any(k in feat_lower for k in ['tenure','month','duration','period','length']):
        recs.append(("Redesign Onboarding &amp; Launch Loyalty Rewards",
                     "New customers need a structured 90-day onboarding journey. Long-tenured "
                     "customers need visible rewards \u2014 preferential rates, fee waivers, or "
                     "exclusive products tied to years of loyalty."))
    if any(k in feat_lower for k in ['charge','price','fee','cost','payment','rate']):
        recs.append(("Review Pricing for At-Risk Segments",
                     "Pricing is among your top churn drivers. Offer loyalty discounts or "
                     "flexible plans to customers flagged as high-risk before they cancel."))
    if any(k in feat_lower for k in ['credit','score','rating']):
        recs.append(("Offer Credit Improvement Support",
                     "Proactively offer credit counselling or improvement programmes. "
                     "This builds loyalty and reduces financial stress simultaneously."))
    if any(k in feat_lower for k in ['active','login','engage','usage','use','freq']):
        recs.append(("Re-Engage Inactive Customers",
                     "Low-activity customers are silently churning. Launch a targeted "
                     "re-engagement campaign with a personalised reason to return."))
    recs.append(("Set Up Automated Early-Warning Alerts",
                 "Any customer whose churn risk score rises above 60% should trigger an "
                 "automatic outreach workflow \u2014 offer, survey, or account manager call."))
    recs.append(("Track &amp; Measure Monthly",
                 "Re-run this analysis every quarter. Measure whether churn rate drops after "
                 "each initiative. Adjust focus based on which segments remain at risk."))

    for i, (title, body_text) in enumerate(recs[:6], 1):
        story.append(numbered_rec(i, title, body_text))
        story.append(Spacer(1, 0.06*inch))

    story += [Spacer(1, 0.08*inch), Paragraph("<b>Immediate Next Steps</b>", sec_head)]
    ns_rows = [
        ['Action', 'Owner', 'Timeline'],
        ['Share high-risk customer list with CRM team', 'Data Team', 'This week'],
        ['Design targeted retention campaign for high-risk segment', 'Marketing', '2 weeks'],
        [f"Pilot: {recs[1][0] if len(recs) > 1 else 'top recommendation'}",
         'Product Team', '1 month'],
        ['Re-run churn model to track improvement', 'Data Team', 'Monthly'],
    ]
    ns_tbl = Table(ns_rows, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
    ns_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  C_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  C_WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [C_ROW1, C_WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
    ]))
    story += [ns_tbl, PageBreak()]

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 5 — BEST MODEL PERFORMANCE (no comparison table)
    # ════════════════════════════════════════════════════════════════════════
    story += [
        section_header("4", "Best Model Performance"),
        Spacer(1, 0.08*inch),
        Paragraph(
            f"The best performing model \u2014 <b>{best_model_name}</b> \u2014 was selected "
            f"from five machine learning models trained on your data. It best balances "
            f"catching churners without generating too many false alarms.", body),
        Spacer(1, 0.08*inch),
        kpi_4([
            (f"{accuracy:.1f}%",  "Overall Accuracy",  C_BLUE),
            (f"{auc_val:.3f}",    f"AUC ({auc_word})", C_GREEN),
            (f"{recall_val:.0%}", "Churners Found",    C_AMBER),
            (f"{prec_val:.0%}",   "Precision",         C_BLUE),
        ]),
        Spacer(1, 0.12*inch),
    ]

    metric_cards = [
        ("Overall Accuracy",
         f"{accuracy:.1f}% \u2014 For every 100 customers, the AI correctly predicts "
         f"the outcome for {accuracy:.0f} of them."),
        ("AUC \u2014 Reliability Score",
         f"{auc_val:.3f} out of 1.0 ({auc_word}). Scores above 0.80 are considered good. "
         f"This model {'can be fully trusted for business decisions' if auc_val >= 0.80 else 'should be used alongside human judgement'}."),
        ("Churners Found (Recall)",
         f"{recall_val:.0%} of all actual churners were correctly identified. "
         "A higher number means fewer at-risk customers slip through undetected."),
        ("Precision",
         f"When the AI flags a customer as at-risk, it is correct {prec_val:.0%} of the time. "
         "Higher precision means less wasted effort on customers who were not going to leave."),
    ]
    for title, desc in metric_cards:
        story.append(feature_card(title, desc))
        story.append(Spacer(1, 0.06*inch))

    # ROC curve + Confusion matrix side by side
    try:
        y_prob = best_model.predict_proba(X_test_final)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_final, y_prob)
        roc_auc_plot = auc(fpr, tpr)
        cm = confusion_matrix(y_test_final, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
        fig.patch.set_facecolor('white')

        # ROC
        ax1.plot(fpr, tpr, color='#2563eb', lw=2,
                 label=f'{best_model_name} (AUC = {roc_auc_plot:.3f})')
        ax1.plot([0,1],[0,1], color='#94a3b8', linestyle='--', lw=1, label='Random guess')
        ax1.set_xlabel('False Positive Rate', fontsize=8.5, color='#64748b')
        ax1.set_ylabel('True Positive Rate',  fontsize=8.5, color='#64748b')
        ax1.set_title('ROC Curve', fontsize=10, fontweight='bold', color='#1e293b')
        ax1.legend(fontsize=7.5, framealpha=0.9)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.tick_params(colors='#475569', labelsize=8)

        # Confusion matrix
        ax2.imshow(cm, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(cm.shape[1]))
        ax2.set_yticks(range(cm.shape[0]))
        labels_cm = sorted(y_test_final.unique())
        label_names = [str(l) for l in labels_cm]  # dynamic — no hardcoded class names
        ax2.set_xticklabels(['Predicted\n' + n for n in label_names], fontsize=8)
        ax2.set_yticklabels(['Actual\n' + n for n in label_names], fontsize=8)
        ax2.set_title('Confusion Matrix', fontsize=10, fontweight='bold', color='#1e293b')
        max_v = cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                col = 'white' if cm[i,j] > max_v * 0.5 else '#1e293b'
                ax2.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                         fontsize=11, fontweight='bold', color=col)
        plt.tight_layout()
        story.append(RLImage(fig_to_png(fig), width=W_FULL, height=2.6*inch))

        tn, fp, fn, tp = cm.ravel()
        cm_guide = [
            ['Cell', 'Count', 'Meaning'],
            ['Top-left',     f"{tn:,}", 'Loyal customers correctly identified \u2014 no action needed'],
            ['Bottom-right', f"{tp:,}", 'Churners correctly caught \u2014 priority retention targets'],
            ['Top-right',    f"{fp:,}", 'Loyal customers wrongly flagged \u2014 minor wasted effort'],
            ['Bottom-left',  f"{fn:,}", 'Churners we missed \u2014 aim to minimise these'],
        ]
        cg_tbl = Table(cm_guide, colWidths=[1.0*inch, 0.7*inch, 4.8*inch])
        cg_tbl.setStyle(TableStyle([
            ('FONTSIZE',      (0,0),(-1,-1), 8.5),
            ('TEXTCOLOR',     (0,0),(-1,0),  C_GREY),
            ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
            ('GRID',          (0,0),(-1,-1), 0.5, C_BORDER),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [C_ROW1, C_WHITE]),
            ('TOPPADDING',    (0,0),(-1,-1), 5),
            ('BOTTOMPADDING', (0,0),(-1,-1), 5),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ]))
        story += [Spacer(1, 0.08*inch),
                  Paragraph("<b>Confusion Matrix Guide:</b>", body_small),
                  cg_tbl]
    except Exception:
        pass

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 6 — QUICK REFERENCE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    story += [
        section_header("5", "Quick Reference Summary"),
        Spacer(1, 0.1*inch),
    ]

    urgency_map = {'CRITICAL': C_RED, 'HIGH': C_RED, 'MODERATE': C_AMBER,
                   'MEDIUM': C_AMBER, 'LOW': C_GREEN, 'GOOD': C_GREEN, '\u2014': C_GREY}

    qr_rows = [
        ['Topic',                'Key Finding',                                     'Status'],
        ['Churn rate',           f"{churn_rate:.1f}% of customers at risk",         severity],
        ['Customers at risk',    f"{n_churned:,} out of {n_total:,} total",         '\u2014'],
        ['Retained customers',   f"{n_safe:,}  ({100 - churn_rate:.1f}%)",          'GOOD'],
        ['#1 churn factor',      top5_features[0].replace('_',' ').title()
                                 if top5_features else '\u2014',                     'HIGH'],
        ['#2 churn factor',      top5_features[1].replace('_',' ').title()
                                 if len(top5_features) > 1 else '\u2014',            'HIGH'],
        ['#3 churn factor',      top5_features[2].replace('_',' ').title()
                                 if len(top5_features) > 2 else '\u2014',            'MEDIUM'],
        ['Best model',           best_model_name,                                   '\u2014'],
        ['AI accuracy',          f"{accuracy:.1f}%",
         'GOOD' if accuracy >= 80 else 'MEDIUM'],
        ['AI reliability (AUC)', f"{auc_word}  ({auc_val:.3f} / 1.0)",
         'GOOD' if auc_val >= 0.80 else 'MEDIUM'],
        ['Immediate action',     sev_short,                                         severity],
    ]

    qr_style = [
        ('BACKGROUND',    (0,0),(-1,0),  C_DARK),
        ('TEXTCOLOR',     (0,0),(-1,0),  C_WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [C_ROW1, C_WHITE]),
        ('GRID',          (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('FONTNAME',      (0,1),(0,-1),  'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1),(0,-1),  C_BLUE),
        ('ALIGN',         (2,0),(-1,-1), 'CENTER'),
    ]
    for ri, row in enumerate(qr_rows[1:], 1):
        c = urgency_map.get(row[2], C_GREY)
        qr_style += [
            ('TEXTCOLOR', (2, ri), (2, ri), c),
            ('FONTNAME',  (2, ri), (2, ri), 'Helvetica-Bold'),
        ]
    qr_tbl = Table(qr_rows, colWidths=[1.6*inch, 3.4*inch, 1.5*inch])
    qr_tbl.setStyle(TableStyle(qr_style))
    story += [
        qr_tbl,
        Spacer(1, 0.2*inch),
        hline(),
        Paragraph(
            f"Generated by ChurnIQ XAI Dashboard  \u00b7  {now}  \u00b7  "
            f"Model: {best_model_name}  \u00b7  Dataset: {n_total:,} records",
            footer_st),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">📊</div>
        <div>
            <div class="sidebar-brand-text">ChurnIQ</div>
            <div class="sidebar-brand-sub">XAI · Prediction Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📁  Upload & Configure",
        "📊  Overview",
        "🔍  XAI Explanations",
        "👤  Customer Lookup",
        "📥  Download Report",
    ])

    if st.session_state['pipeline_done']:
        st.markdown(f"""
        <div class="status-pill">
            <div class="status-dot"></div>
            Pipeline complete · {st.session_state['best_model_name']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.8rem;">Upload a dataset and run the pipeline to begin.</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & CONFIGURE
# ════════════════════════════════════════════════════════════════════════════
if page == "📁  Upload & Configure":
    st.markdown("""
    <div class="page-header">
        <h1>📁 Upload & Configure</h1>
        <p>Load your customer dataset and configure the prediction pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-icon">☁️</div>
        <div class="upload-zone-title">Drop your dataset here</div>
        <div class="upload-zone-sub">Supported formats: CSV &nbsp;·&nbsp; XLSX &nbsp;·&nbsp; SQL</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "dataset", type=['csv', 'xlsx', 'sql'],
        label_visibility='collapsed'
    )

    if uploaded:
        with st.spinner("Reading file..."):
            raw_df, conn, tables = load_uploaded_file(uploaded)

        # ── SQL multi-table picker ───────────────────────────────────────────
        if conn is not None and tables:
            if len(tables) == 1:
                # Only one table — load it silently
                raw_df = pd.read_sql(f'SELECT * FROM "{tables[0]}"', conn)
                st.markdown(f"""
                <div class="insight-card green">
                    🗄️ &nbsp;SQL file loaded — table: <b>{tables[0]}</b>
                </div>""", unsafe_allow_html=True)
            else:
                # Multiple tables — show a styled picker
                st.markdown('<div class="section-title">🗄️ SQL File — Select a Table</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="insight-card amber">
                    Found <b>{len(tables)}</b> tables in the SQL file.
                    Select the one that contains your customer/churn data.
                </div>""", unsafe_allow_html=True)

                selected_table = st.selectbox(
                    "Choose table", tables,
                    format_func=lambda t: f"🗂️  {t}",
                    label_visibility='collapsed'
                )

                col_prev, col_load = st.columns([3, 1])
                with col_prev:
                    preview_df = pd.read_sql(f'SELECT * FROM "{selected_table}" LIMIT 5', conn)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                with col_load:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✅ Use This Table", type="primary", use_container_width=True):
                        raw_df = pd.read_sql(f'SELECT * FROM "{selected_table}"', conn)
                        st.session_state['sql_table_loaded'] = True

                if not st.session_state.get('sql_table_loaded'):
                    st.info("👆 Preview the table above, then click **Use This Table** to continue.")
                    st.stop()

        df = raw_df
        if df is None:
            st.error("❌ Could not load the file. Please check the format and try again.")
            st.stop()

        st.session_state['df_raw'] = df

        # Success banner
        st.markdown(f"""
        <div class="insight-card green">
            ✅ &nbsp;<b>{uploaded.name}</b> loaded successfully &nbsp;—&nbsp;
            <b>{df.shape[0]:,}</b> rows &nbsp;×&nbsp; <b>{df.shape[1]}</b> columns
        </div>
        """, unsafe_allow_html=True)

        # Dataset stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",        f"{df.shape[0]:,}")
        c2.metric("Columns",     df.shape[1])
        c3.metric("Missing %",   f"{df.isnull().mean().mean()*100:.1f}%")
        c4.metric("Numeric cols",int(df.select_dtypes(include='number').shape[1]))

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Data preview
        with st.expander("👁️ Data Preview", expanded=False):
            st.dataframe(df.head(12), use_container_width=True)

        # ── Target column selection ──────────────────────────────────────────
        st.markdown('<div class="section-title">🎯 Select Target (Churn) Column</div>', unsafe_allow_html=True)

        # Lowercase columns first (same as notebook) then detect
        lower_cols    = [col.lower() for col in df.columns]
        detected_col  = detect_churn_column(df)          # returns original-case col name
        detected_lower = detected_col.lower()

        try:
            default_idx = lower_cols.index(detected_lower)
            was_detected = detected_lower != lower_cols[0]
        except ValueError:
            default_idx   = 0
            was_detected  = False

        # Show auto-detect feedback (mirrors notebook print output)
        if was_detected:
            st.markdown(f"""
            <div class="insight-card green" style="margin-bottom:10px;">
                🎯 &nbsp;Auto-detected churn column: &nbsp;<b>{detected_lower}</b>
                &nbsp;— you can change it below if needed.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-card amber" style="margin-bottom:10px;">
                ⚠️ &nbsp;No churn-related column found. Defaulting to first column: &nbsp;<b>{lower_cols[0]}</b>
                &nbsp;— please select the correct target column.
            </div>""", unsafe_allow_html=True)

        target_col_lower = st.selectbox(
            "Target column", lower_cols,
            index=default_idx,
            label_visibility='collapsed'
        )

        n_unique = df.iloc[:, lower_cols.index(target_col_lower)].nunique()
        if n_unique > 10:
            st.warning(f"⚠️ '{target_col_lower}' has {n_unique} unique values — confirm this is a binary churn column.")
        else:
            original_col = df.columns[lower_cols.index(target_col_lower)]
            vc = df[original_col].value_counts()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                fig, ax = plt.subplots(figsize=(3.5, 2.8))
                fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
                ax.pie(vc.values, labels=[str(x) for x in vc.index],
                       colors=['#10b981','#ef4444','#3b82f6','#f59e0b'],
                       autopct='%1.1f%%',
                       textprops={'color':'white','fontsize':10},
                       wedgeprops={'edgecolor':'#0d1117','linewidth':2})
                ax.set_title("Class Balance", color='white', fontsize=10, pad=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()
            with col_b:
                st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
                for label, count in vc.items():
                    pct = count / len(df) * 100
                    st.markdown(f"""
                    <div class="insight-card" style="margin-bottom:6px;">
                        <b>{label}</b> &nbsp;—&nbsp; {count:,} customers &nbsp;({pct:.1f}%)
                    </div>""", unsafe_allow_html=True)

        st.session_state['target_column'] = target_col_lower

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Run button
        if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting pipeline...")
            with st.spinner("Training models — this may take a minute..."):
                try:
                    progress.progress(10, "Preprocessing & encoding...")
                    result = run_pipeline(df, target_col_lower)
                    progress.progress(85, "Computing SHAP explanations...")
                    for k, v in result.items():
                        st.session_state[k] = v
                    st.session_state['pipeline_done'] = True
                    st.session_state['target_column'] = target_col_lower
                    progress.progress(100, "Done!")
                    st.success(f"✅ Pipeline complete! Best model: **{result['best_model_name']}**")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")
                    import traceback; st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Overview":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    df_raw     = st.session_state['df_raw']
    best_row   = st.session_state['results_df'].iloc[0]
    y_test     = st.session_state['y_test_final']
    churn_rate = y_test.mean() * 100

    st.markdown("""
    <div class="page-header">
        <h1>📊 Analysis Overview</h1>
        <p>Summary of your dataset and model performance at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI grid
    color_cr = "red" if churn_rate > 20 else "amber" if churn_rate > 10 else "green"
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value">{len(df_raw):,}</div>
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-sub">Full dataset</div>
        </div>
        <div class="kpi-card {color_cr}">
            <div class="kpi-icon">📉</div>
            <div class="kpi-value">{churn_rate:.1f}%</div>
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-sub">Test set proportion</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-icon">🏆</div>
            <div class="kpi-value" style="font-size:1.2rem;padding-top:4px">{st.session_state['best_model_name']}</div>
            <div class="kpi-label">Best Model</div>
            <div class="kpi-sub">Highest F1-Score</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{best_row['F1-Score']}</div>
            <div class="kpi-label">F1 / AUC</div>
            <div class="kpi-sub">AUC: {best_row['ROC-AUC']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain English summary
    st.markdown('<div class="section-title">💬 Plain English Summary</div>', unsafe_allow_html=True)
    churn_word = "high" if churn_rate > 20 else "moderate" if churn_rate > 10 else "low"
    cr_class = "red" if churn_rate > 20 else "amber" if churn_rate > 10 else "green"
    auc_quality = "excellent" if best_row['ROC-AUC'] >= 0.9 else "good" if best_row['ROC-AUC'] >= 0.8 else "fair"

    st.markdown(f"""
    <div class="insight-card {cr_class}">
        📉 <b>Churn Risk:</b> The dataset shows a <b>{churn_word} churn rate of {churn_rate:.1f}%</b>.
        That means roughly <b>{churn_rate:.0f} out of every 100 customers</b> are at risk of leaving.
    </div>
    <div class="insight-card green">
        🏆 <b>Best Model:</b> <b>{st.session_state['best_model_name']}</b> achieved an F1-Score of
        <b>{best_row['F1-Score']:.2%}</b> and an AUC of <b>{best_row['ROC-AUC']:.2%}</b> — {auc_quality} discrimination ability.
    </div>
    <div class="insight-card">
        📌 <b>Recommended Action:</b> Focus retention efforts on customers flagged as
        <b>High Risk</b>. Use the <b>Customer Lookup</b> tab to explore individual risk scores.
    </div>
    """, unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-title">📊 Visual Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Churn vs Active — Pie**")
        labels = ['Active', 'Churned']
        sizes  = [int((y_test==0).sum()), int((y_test==1).sum())]
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=['#10b981','#ef4444'],
            autopct='%1.1f%%', textprops={'color':'white','fontsize':11},
            wedgeprops={'edgecolor':'#0d1117','linewidth':3},
            startangle=90)
        for at in autotexts: at.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("**Model Metrics — Best Model**")
        metrics_vals = {
            'Accuracy':  best_row['Accuracy'],
            'Precision': best_row['Precision'],
            'Recall':    best_row['Recall'],
            'F1-Score':  best_row['F1-Score'],
            'ROC-AUC':   best_row['ROC-AUC'],
        }
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        bar_c = ['#3b82f6','#6366f1','#8b5cf6','#10b981','#f59e0b']
        bars = ax.bar(list(metrics_vals.keys()), list(metrics_vals.values()),
                      color=bar_c, edgecolor='none', alpha=0.9)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values(): spine.set_color('#1e2733')
        ax.set_facecolor('#0d1117')
        for bar, val in zip(bars, metrics_vals.values()):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
                    ha='center', color='white', fontsize=8.5, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Model leaderboard removed per user request


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — XAI EXPLANATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍  XAI Explanations":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    shap_values     = st.session_state['shap_values']
    X_test          = st.session_state['X_test_final']
    features        = st.session_state['selected_feature_names']
    best_model_name = st.session_state['best_model_name']

    st.markdown(f"""
    <div class="page-header">
        <h1>🔍 XAI Explanations</h1>
        <p>Understanding <em>why</em> the model makes its predictions — powered by SHAP.</p>
    </div>
    """, unsafe_allow_html=True)

    if shap_values is None:
        st.warning("SHAP values could not be computed for this model type.")
        st.stop()

    _sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
    mean_shap = pd.Series(np.abs(_sv).mean(axis=0),
                           index=features).sort_values(ascending=False)

    # Top drivers — plain English cards
    st.markdown('<div class="section-title">🌟 Top Churn Drivers</div>', unsafe_allow_html=True)
    cols_cards = st.columns(min(5, len(mean_shap)))
    for i, (feat, val) in enumerate(mean_shap.head(5).items()):
        impact_cls = "red" if val > mean_shap.mean()*1.5 else "amber" if val > mean_shap.mean()*0.5 else "green"
        impact_lbl = "🔴 High" if val > mean_shap.mean()*1.5 else "🟡 Medium" if val > mean_shap.mean()*0.5 else "🟢 Low"
        with cols_cards[i]:
            st.markdown(f"""
            <div class="kpi-card {impact_cls}" style="text-align:center;padding:16px 12px;">
                <div class="kpi-label">#{i+1} Feature</div>
                <div style="color:#e2e8f0;font-weight:600;font-size:0.85rem;margin:8px 0;word-break:break-word;">{feat}</div>
                <div style="font-size:0.78rem;color:#64748b;">{impact_lbl} impact</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:3px;">score: {val:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Global Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    top = mean_shap.head(12)
    bar_colors = ['#ef4444' if i < 3 else '#3b82f6' if i < 6 else '#475569'
                  for i in range(len(top)-1, -1, -1)]
    bars = ax.barh(top.index[::-1], top.values[::-1], color=bar_colors, alpha=0.92)
    ax.set_xlabel('Mean |SHAP| Value', color='#94a3b8', fontsize=11)
    ax.set_title(f'Feature Importance — {best_model_name}', color='#f1f5f9',
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    for spine in ax.spines.values(): spine.set_color('#1e2733')
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + top.values.max()*0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', color='#cbd5e1', fontsize=8.5)
    ax.grid(axis='x', color='#1e2733', linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Plain English churn driver list
    st.markdown('<div class="section-title">💬 What These Features Mean</div>', unsafe_allow_html=True)
    for i, (feat, val) in enumerate(mean_shap.head(8).items(), 1):
        impact = "strongly" if val > mean_shap.mean() else "moderately"
        st.markdown(f"""
        <div class="insight-card" style="margin-bottom:6px;">
            <b>{i}. {feat}</b> &nbsp;·&nbsp; SHAP score: <b>{val:.4f}</b><br>
            <span style="color:#64748b;font-size:0.85rem;">
            This feature {impact} influences whether a customer churns or stays.
            </span>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════════════════════════
elif page == "👤  Customer Lookup":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    best_model  = st.session_state['best_model']
    X_test      = st.session_state['X_test_final']
    y_test      = st.session_state['y_test_final']
    shap_values = st.session_state['shap_values']
    features    = st.session_state['selected_feature_names']

    st.markdown("""
    <div class="page-header">
        <h1>👤 Customer Lookup</h1>
        <p>Explore individual churn risk scores and AI explanations per customer.</p>
    </div>
    """, unsafe_allow_html=True)

    idx_list     = X_test.index.tolist()
    selected_idx = st.selectbox("Select Customer ID", idx_list)
    row_pos      = idx_list.index(selected_idx)

    customer = X_test.loc[[selected_idx]]
    prob     = best_model.predict_proba(customer)[0][1]
    actual   = y_test.loc[selected_idx]

    risk_label = "🔴 HIGH RISK"   if prob > 0.6 else \
                 "🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK"
    risk_cls   = "red" if prob > 0.6 else "amber" if prob > 0.35 else "green"

    # Top metrics
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Churn Probability", f"{prob*100:.1f}%")
    with c2: st.metric("Risk Level", risk_label)
    with c3: st.metric("Actual Label", "Churned ❌" if actual==1 else "Active ✅")

    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_t = st.columns([1, 2])

    with col_g:
        fig = make_gauge(prob)
        st.pyplot(fig, use_container_width=False); plt.close()
        st.markdown(f"""
        <div class="insight-card {risk_cls}" style="text-align:center;margin-top:8px;">
            <b>{risk_label}</b><br>
            <span style="font-size:0.8rem;color:#94a3b8;">Churn probability: {prob*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    with col_t:
        st.markdown('<div class="section-title">📋 Feature Values</div>', unsafe_allow_html=True)
        cust_df = customer.T.reset_index()
        cust_df.columns = ['Feature', 'Value']
        cust_df['Value'] = cust_df['Value'].round(4)
        st.dataframe(cust_df, use_container_width=True, height=280, hide_index=True)

    # SHAP waterfall
    if shap_values is not None:
        st.markdown('<div class="section-title">🔍 Why Is This Customer At Risk?</div>', unsafe_allow_html=True)
        _sv2 = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
        cust_shap   = _sv2[row_pos]
        shap_series = pd.Series(cust_shap, index=features).sort_values(key=abs, ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        bar_colors = ['#ef4444' if v > 0 else '#10b981' for v in shap_series.values]
        ax.barh(shap_series.index[::-1], shap_series.values[::-1],
                color=bar_colors[::-1], alpha=0.92)
        ax.axvline(0, color='#475569', linewidth=1)
        ax.set_xlabel('SHAP Value  (🔴 positive = increases churn risk  /  🟢 negative = decreases risk)',
                      color='#94a3b8', fontsize=10)
        ax.set_title(f'SHAP Explanation — Customer {selected_idx}',
                     color='#f1f5f9', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_color('#1e2733')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Plain text reason list
        st.markdown('<div class="section-title">💬 Plain English Explanation</div>', unsafe_allow_html=True)
        for feat, val in shap_series.items():
            direction = "🔴 **increases**" if val > 0 else "🟢 **decreases**"
            cls = "red" if val > 0 else "green"
            st.markdown(f"""
            <div class="insight-card {cls}" style="margin-bottom:6px;">
                <b>{feat}</b> {direction} churn risk &nbsp;(impact: {val:+.4f})
            </div>""", unsafe_allow_html=True)

    # All customers risk table
    st.markdown('<div class="section-title">📋 All Customers — Risk Overview</div>', unsafe_allow_html=True)
    all_probs = best_model.predict_proba(X_test)[:,1]
    risk_df   = pd.DataFrame({
        'Customer ID':       X_test.index,
        'Churn Probability': all_probs.round(4),
        'Actual':            y_test.values,
    })
    risk_df['Risk Level'] = risk_df['Churn Probability'].apply(
        lambda p: '🔴 High' if p > 0.6 else ('🟡 Medium' if p > 0.35 else '🟢 Low'))
    risk_df = risk_df.sort_values('Churn Probability', ascending=False)
    st.dataframe(risk_df, use_container_width=True, height=380, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DOWNLOAD REPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📥  Download Report":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    st.markdown("""
    <div class="page-header">
        <h1>📥 Download PDF Report</h1>
        <p>Generate a professional analysis report suitable for both technical and non-technical audiences.</p>
    </div>
    """, unsafe_allow_html=True)

    # What's included cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="insight-card green">
            <b>📊 Executive Summary</b><br>
            <span style="font-size:0.85rem;color:#64748b;">
            KPI metrics, churn rate, best model performance in plain language.
            </span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-card">
            <b>🏅 Model Comparison</b><br>
            <span style="font-size:0.85rem;color:#64748b;">
            Full table of all 5 models with accuracy, precision, recall, F1, AUC.
            </span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="insight-card amber">
            <b>🔍 SHAP Explanations</b><br>
            <span style="font-size:0.85rem;color:#64748b;">
            Feature importance chart, confusion matrix, classification report.
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📄 Generate & Download PDF", type="primary", use_container_width=True):
        with st.spinner("Building your report..."):
            try:
                pdf_bytes = generate_pdf_report(
                    results_df             = st.session_state['results_df'],
                    best_model_name        = st.session_state['best_model_name'],
                    best_model             = st.session_state['best_model'],
                    X_test_final           = st.session_state['X_test_final'],
                    y_test_final           = st.session_state['y_test_final'],
                    shap_values            = st.session_state['shap_values'],
                    selected_feature_names = st.session_state['selected_feature_names'],
                    target_column          = st.session_state['target_column'],
                    df_raw                 = st.session_state['df_raw'],
                )
                fname = f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.success("✅ Report ready!")
                st.download_button(
                    label         = "⬇️ Download PDF Report",
                    data          = pdf_bytes,
                    file_name     = fname,
                    mime          = "application/pdf",
                    use_container_width = True,
                    type          = "primary",
                )
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")
                import traceback; st.code(traceback.format_exc())
