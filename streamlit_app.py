"""
Churn Prediction XAI — Streamlit Dashboard
Supports CSV, SQL, XLSX uploads
Pages: Overview | Model Performance | Explainability (SHAP+LIME) | Customer Risk Table
PDF report download included
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import io
import math
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap
import lime
import lime.lime_tabular
from fpdf import FPDF

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction XAI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark professional theme */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a3040;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2d3654;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    line-height: 1.1;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8892a4;
}
.kpi-red    { color: #ff6b6b; }
.kpi-green  { color: #51cf66; }
.kpi-blue   { color: #74c0fc; }
.kpi-yellow { color: #ffd43b; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e8eaf0;
    border-left: 4px solid #4dabf7;
    padding-left: 14px;
    margin: 32px 0 16px 0;
}

/* Risk badges */
.risk-high   { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b55; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.risk-medium { background:#ffd43b22; color:#ffd43b; border:1px solid #ffd43b55; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.risk-low    { background:#51cf6622; color:#51cf66; border:1px solid #51cf6655; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* Model winner badge */
.best-badge {
    background: linear-gradient(90deg, #4dabf7, #748ffc);
    color: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b27;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8892a4;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #2d3654 !important;
    color: #e8eaf0 !important;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #4dabf7, #748ffc); }

/* Metric */
[data-testid="metric-container"] {
    background: #1a2035;
    border: 1px solid #2d3654;
    border-radius: 10px;
    padding: 12px 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4dabf7, #748ffc);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 20px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Info boxes */
.info-box {
    background: #1a2035;
    border: 1px solid #2d3654;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Plain-english churn reason */
.reason-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #2a3040;
    font-size: 0.92rem;
}

.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a2035",
    "axes.facecolor":    "#1a2035",
    "axes.edgecolor":    "#2d3654",
    "axes.labelcolor":   "#e8eaf0",
    "xtick.color":       "#8892a4",
    "ytick.color":       "#8892a4",
    "text.color":        "#e8eaf0",
    "grid.color":        "#2d3654",
    "grid.alpha":        0.4,
    "figure.dpi":        110,
    "font.family":       "DejaVu Sans",
})
PALETTE = ["#4dabf7", "#748ffc", "#51cf66", "#ffd43b", "#ff6b6b", "#f783ac"]


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (ported from notebook, adapted for Streamlit)
# ═══════════════════════════════════════════════════════════════════════

def clean_sql_for_sqlite(sql_content):
    sql_content = sql_content.replace('`', '"')
    sql_content = re.sub(r'AUTO_INCREMENT=\d+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'AUTO_INCREMENT', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'ENGINE\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'DEFAULT CHARSET\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'COLLATE\s*=\s*\w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'CHARACTER SET \w+', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bUNSIGNED\b', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bZEROFILL\b', '', sql_content, flags=re.IGNORECASE)
    for t in ['TINYINT','SMALLINT','MEDIUMINT','BIGINT']:
        sql_content = re.sub(rf'\b{t}\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bDATETIME\b', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bTIMESTAMP\b', 'TEXT', sql_content, flags=re.IGNORECASE)
    for pat in [r'SET NAMES \w+;', r'SET @\w+\s*=.*?;', r'SET @@\w+\s*=.*?;',
                r'SET SQL_MODE\s*=.*?;', r'SET FOREIGN_KEY_CHECKS\s*=.*?;',
                r'SET UNIQUE_CHECKS\s*=.*?;']:
        sql_content = re.sub(pat, '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^--.*$', '', sql_content, flags=re.MULTILINE)
    sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)
    return sql_content


def load_sql_bytes(file_bytes):
    sql_content = file_bytes.decode('utf-8', errors='ignore')
    sql_content = clean_sql_for_sqlite(sql_content)
    conn = sqlite3.connect(':memory:')
    try:
        conn.cursor().executescript(sql_content)
        conn.commit()
    except Exception as e:
        return None, [], str(e)
    tables = [r[0] for r in conn.cursor().execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()]
    return conn, tables, None


def get_numeric_patterns():
    return {
        'revenue','income','salary','wage','profit','loss','balance','amount','price',
        'cost','fee','charge','payment','debt','debit','credit','interest','rate','apr',
        'loan','mortgage','investment','id','code','number','no','num','account',
        'transaction','invoice','order','ticket','reference','ref','ein','ssn','tax',
        'zip','zipcode','postal','pincode','postcode','percent','percentage','ownership',
        'share','ratio','count','quantity','qty','total','sum','employees','staff',
        'headcount','volume','units','duration','minutes','seconds','calls','sms',
        'data_usage','bandwidth','speed','mbps','gb','mb','download','upload','sales',
        'orders','transactions','customers','users','visitors','views','clicks',
        'impressions','conversions','leads','age','weight','height','distance','length',
        'width','size','score','rating','rank','grade','level','tier','year','month',
        'day','quarter','week','hour','minute','patient_id','diagnosis_code','dosage',
        'temperature','pressure','heartrate','glucose','product_id','sku','barcode',
        'upc','stock','inventory','discount','shipping','value','metric','measure',
        'figure','stat','index'
    }

def is_numeric_col(col):
    col_clean = re.sub(r'[^a-z0-9]', '', col.lower().strip())
    return any(p in col_clean for p in get_numeric_patterns())

def aggressive_clean(series):
    c = series.astype(str)
    c = (c.str.replace(r'[\$€£¥₹₽₩¢]','',regex=True)
          .str.replace(r'%','',regex=True)
          .str.replace(r',','',regex=True)
          .str.replace(r'\b(GB|MB|KB|TB|kg|km|cm|lb|oz|USD|EUR|CAD)\b','',regex=True,flags=re.IGNORECASE)
          .str.replace(r'[^0-9.]','',regex=True)
          .str.strip())
    return c.replace(['','nan','None','null','NULL'], np.nan)

def smart_numeric_convert(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Cast nullable Int64 → float64 so sklearn/corr() never choke on them
            if str(df[col].dtype) in ['Int8','Int16','Int32','Int64','UInt8','UInt16','UInt32','UInt64']:
                df[col] = df[col].astype('float64')
            continue
        if is_numeric_col(col):
            cleaned = aggressive_clean(df[col])
            converted = pd.to_numeric(cleaned, errors='coerce')
            orig_nn = df[col].notna() & (df[col].astype(str).str.strip() != '')
            rate = converted[orig_nn].notna().sum() / orig_nn.sum() if orig_nn.sum() > 0 else 0
            if rate > 0.1:
                df[col] = converted
    return df

def convert_dates(df):
    df = df.copy()
    kws = ['dob','birth','date','created','signup','registered','joined','start','opened']
    for col in df.columns:
        if any(k in col.lower() for k in kws):
            try:
                ds = pd.to_datetime(df[col], errors='coerce')
                nn = df[col].notna().sum()
                if nn > 0 and ds.notna().sum()/nn >= 0.5:
                    df[f"{col.replace(' ','_').lower()}_years"] = ((datetime.now()-ds).dt.days/365.25).round(1)
                    df = df.drop(columns=[col])
            except Exception:
                pass
    return df

def handle_missing(df, target_col):
    df = df.copy()
    drop = [c for c in df.columns if c != target_col and df[c].isnull().mean() > 0.5]
    if drop:
        df = df.drop(columns=drop)
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    for col in df.select_dtypes(include='number').columns:
        if col != target_col and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        if col != target_col and df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else 'Unknown')
    return df

def encode_categoricals(df, target_col):
    df = df.copy()
    pos_vals = {'yes','true','active','paid','y','1','on','enabled'}
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != target_col]
    for col in cat_cols:
        u = df[col].nunique()
        if u == 2:
            vals = df[col].unique()
            if any(str(v).lower() in pos_vals for v in vals):
                mapping = {v: (1 if str(v).lower() in pos_vals else 0) for v in vals}
            else:
                mapping = {vals[0]: 0, vals[1]: 1}
            df[col] = df[col].map(mapping)
        elif 3 <= u <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif 11 <= u <= 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = df.drop(columns=[col])
    if target_col in df.columns and df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    return df


@st.cache_data(show_spinner=False)
def run_full_pipeline(file_bytes, file_ext, table_name, target_col):
    """Full ML pipeline — cached so it doesn't re-run on every widget interaction."""

    # ── Load dataframe ──────────────────────────────────────
    if file_ext == '.csv':
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif file_ext == '.xlsx':
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=table_name or 0)
    elif file_ext == '.sql':
        conn, tables, err = load_sql_bytes(file_bytes)
        if err:
            return None, f"SQL error: {err}"
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    else:
        return None, "Unsupported file type"

    # ── Kill all Arrow/nullable dtypes at source ──────────
    # pandas 2.x with pyarrow backend can produce ArrowDtype columns
    # from read_csv/read_excel; convert everything to standard numpy dtypes now.
    for _col in df.columns:
        try:
            if hasattr(df[_col].dtype, "pyarrow_dtype") or str(df[_col].dtype).startswith("arrow"):
                df[_col] = df[_col].astype(object)
        except Exception:
            pass
    df = df.convert_dtypes(convert_string=False, convert_integer=False,
                           convert_boolean=False, convert_floating=False)
    # Convert any remaining pandas nullable types to numpy equivalents
    for _col in df.select_dtypes(include="number").columns:
        if str(df[_col].dtype) not in ["float64","float32","int64","int32","int16","int8"]:
            df[_col] = df[_col].astype("float64")

    # ── Preprocessing ───────────────────────────────────────
    df = smart_numeric_convert(df)
    df = convert_dates(df)
    df = handle_missing(df, target_col)
    df = encode_categoricals(df, target_col)

    # ── Split ────────────────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── Feature engineering ──────────────────────────────────
    df_tr = X_train.copy(); df_te = X_test.copy()
    num_cols = df_tr.columns.tolist()
    rev = [c for c in num_cols if any(w in c.lower() for w in ['revenue','income','sales','earning'])]
    dbt = [c for c in num_cols if any(w in c.lower() for w in ['debt','debit','loan','liability','payment'])]
    if rev and dbt:
        df_tr['Debt_Revenue_Ratio'] = df_tr[dbt[0]] / (df_tr[rev[0]] + 1e-4)
        df_te['Debt_Revenue_Ratio'] = df_te[dbt[0]] / (df_te[rev[0]] + 1e-4)
    if len(num_cols) >= 2:
        try:
            # Reset index so join aligns correctly; cast everything to float64
            # so pandas .corr() never hits nullable-Int64 / ArrowDtype issues
            df_tr_f = df_tr[num_cols].copy().astype('float64').reset_index(drop=True)
            y_tr_f  = y_train.reset_index(drop=True).rename('__target__').astype('float64')
            corr_s  = df_tr_f.join(y_tr_f).corr()['__target__'].drop('__target__').abs().dropna()
            if len(corr_s) >= 2:
                top2 = corr_s.nlargest(2).index.tolist()
                new_feat = f"{top2[0]}_x_{top2[1]}"
                df_tr[new_feat] = df_tr[top2[0]].astype('float64') * df_tr[top2[1]].astype('float64')
                df_te[new_feat] = df_te[top2[0]].astype('float64') * df_te[top2[1]].astype('float64')
        except Exception:
            pass  # skip interaction feature silently if correlation fails

    # ── Scale ────────────────────────────────────────────────
    # Cast everything to float64 before scaling — eliminates nullable Int64 / ArrowDtype errors
    df_tr = df_tr.astype('float64')
    df_te = df_te.astype('float64')
    scaler = StandardScaler()
    # Force pure numpy float64 array — prevents ArrowDtype surviving into DataFrame
    X_tr_sc = pd.DataFrame(
        np.array(scaler.fit_transform(df_tr), dtype=np.float64),
        columns=df_tr.columns, index=df_tr.index)
    X_te_sc = pd.DataFrame(
        np.array(scaler.transform(df_te), dtype=np.float64),
        columns=df_te.columns, index=df_te.index)

    # ── Remove correlated features ───────────────────────────
    corr_m = pd.DataFrame(
        np.corrcoef(X_tr_sc.to_numpy(dtype=np.float64).T),
        index=X_tr_sc.columns, columns=X_tr_sc.columns)
    to_rm = set()
    for i in range(len(corr_m.columns)):
        for j in range(i+1, len(corr_m.columns)):
            if abs(corr_m.iloc[i,j]) >= 0.90:
                c1, c2 = corr_m.columns[i], corr_m.columns[j]
                _y = y_train.to_numpy(dtype=np.float64)
                cr1 = abs(np.corrcoef(X_tr_sc[c1].to_numpy(dtype=np.float64), _y)[0,1])
                cr2 = abs(np.corrcoef(X_tr_sc[c2].to_numpy(dtype=np.float64), _y)[0,1])
                to_rm.add(c1 if cr1 < cr2 else c2)
    X_tr_f = X_tr_sc.drop(columns=list(to_rm))
    X_te_f = X_te_sc.drop(columns=list(to_rm))

    # ── Feature selection ────────────────────────────────────
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42,
                                    class_weight='balanced', max_depth=10, n_jobs=-1)
    rf_sel.fit(X_tr_f, y_train)
    fi = pd.DataFrame({'Feature': X_tr_f.columns,
                       'Importance': rf_sel.feature_importances_}).sort_values('Importance', ascending=False)
    n_keep = max(8, int(X_tr_f.shape[1] * 0.7))
    sel_feats = fi.head(n_keep)['Feature'].tolist()
    X_tr_sel = X_tr_f[sel_feats]
    X_te_sel = X_te_f[sel_feats]

    # ── SMOTE ────────────────────────────────────────────────
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_sel, y_train)
    X_tr_res = pd.DataFrame(X_tr_res, columns=sel_feats)

    # ── Train 4 models ───────────────────────────────────────
    models = {
        'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':           XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM':          LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }
    results = []
    for name, model in models.items():
        model.fit(X_tr_res, y_tr_res)
        yp     = model.predict(X_te_sel)
        yp_pr  = model.predict_proba(X_te_sel)[:, 1]
        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, yp),  4),
            'Precision': round(precision_score(y_test, yp), 4),
            'Recall':    round(recall_score(y_test, yp),    4),
            'F1-Score':  round(f1_score(y_test, yp),        4),
            'ROC-AUC':   round(roc_auc_score(y_test, yp_pr),4),
        })

    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]['Model']
    best_model = models[best_name]

    # ── Predictions table ────────────────────────────────────
    all_proba = best_model.predict_proba(X_te_sel)[:, 1]
    pred_df   = X_te_sel.copy()
    pred_df['Churn_Probability'] = (all_proba * 100).round(1)
    pred_df['Predicted']         = best_model.predict(X_te_sel)
    pred_df['Actual']            = y_test.values
    pred_df['Risk_Level']        = pd.cut(all_proba,
        bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])

    # ── SHAP ─────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(best_model)
    shap_vals   = explainer.shap_values(X_te_sel)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_shap = pd.DataFrame({
        'Feature': sel_feats,
        'Impact':  np.abs(shap_vals).mean(axis=0)
    }).sort_values('Impact', ascending=False)

    # ── LIME explainer (fit only, explain on demand) ─────────
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data   = X_tr_res.values,
        feature_names   = sel_feats,
        class_names     = ['Active', 'Churned'],
        mode            = 'classification',
        random_state    = 42
    )

    return {
        'df_raw':        df,
        'X_test':        X_te_sel,
        'y_test':        y_test,
        'models':        models,
        'results_df':    results_df,
        'best_name':     best_name,
        'best_model':    best_model,
        'pred_df':       pred_df,
        'sel_feats':     sel_feats,
        'shap_vals':     shap_vals,
        'mean_shap':     mean_shap,
        'explainer':     explainer,
        'lime_explainer':lime_explainer,
        'X_tr_res':      X_tr_res,
        'class_labels':  [str(c) for c in sorted(y_test.unique())],
    }, None


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
      <div style='font-family:"DM Serif Display",serif; font-size:1.5rem; color:#e8eaf0;'>📉 ChurnXAI</div>
      <div style='font-size:0.75rem; color:#8892a4; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px;'>Explainable Churn Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV, XLSX or SQL",
        type=['csv', 'xlsx', 'sql'],
        label_visibility="collapsed"
    )

    df_preview = None
    file_bytes = file_ext = table_name = None

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_ext   = '.' + uploaded_file.name.rsplit('.', 1)[-1].lower()

        # Load preview for target selection
        try:
            if file_ext == '.csv':
                df_preview = pd.read_csv(io.BytesIO(file_bytes))
                table_name = None
            elif file_ext == '.xlsx':
                xl = pd.ExcelFile(io.BytesIO(file_bytes))
                sheets = xl.sheet_names
                if len(sheets) > 1:
                    table_name = st.selectbox("Select sheet", sheets)
                else:
                    table_name = sheets[0]
                df_preview = pd.read_excel(io.BytesIO(file_bytes), sheet_name=table_name)
            elif file_ext == '.sql':
                conn, tables, err = load_sql_bytes(file_bytes)
                if err:
                    st.error(f"SQL error: {err}")
                elif len(tables) == 0:
                    st.error("No tables found in SQL file.")
                else:
                    table_name = tables[0] if len(tables)==1 else st.selectbox("Select table", tables)
                    df_preview = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        except Exception as e:
            st.error(f"Could not preview file: {e}")

    if df_preview is not None:
        st.markdown("### 🎯 Target Column")
        churn_kws = ['churn','attrition','target','label','exited','left','churned','status','outcome']
        default   = next((c for c in df_preview.columns if any(k in c.lower() for k in churn_kws)),
                         df_preview.columns[0])
        target_col = st.selectbox("Select the churn/target column",
                                  df_preview.columns,
                                  index=list(df_preview.columns).index(default))
        n_uniq = df_preview[target_col].nunique()
        if n_uniq > 10:
            st.warning(f"⚠️ '{target_col}' has {n_uniq} unique values — this looks like a feature, not a binary churn label.")
        elif n_uniq == 1:
            st.error("⛔ Column has only 1 unique value — can't be used as target.")
        else:
            st.success(f"✅ {n_uniq} classes detected — looks good.")

        st.markdown("---")
        run_btn = st.button("🚀 Run Pipeline", use_container_width=True)
    else:
        run_btn = False
        target_col = None

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#555e6e; text-align:center;'>4 models · SHAP · LIME · PDF export</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════

if not uploaded_file:
    # Landing
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px 40px 20px;'>
      <div style='font-family:"DM Serif Display",serif; font-size:3rem; color:#e8eaf0; line-height:1.2;'>
        Predict & Explain<br><em style='color:#4dabf7;'>Customer Churn</em>
      </div>
      <div style='color:#8892a4; font-size:1.05rem; margin-top:16px; max-width:560px; margin-left:auto; margin-right:auto; line-height:1.7;'>
        Upload your dataset (CSV, XLSX, or SQL), select the churn column,<br>
        and get a full ML pipeline with explainable AI in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📊", "Model Comparison",   "4 ML models evaluated on Accuracy, F1, Precision, Recall & AUC"),
        (c2, "🔍", "Explainable AI",     "SHAP global & local explanations + LIME per-customer breakdown"),
        (c3, "📥", "PDF Report",         "One-click PDF with KPIs, charts, top churn drivers & risk table"),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
          <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
          <div style='font-weight:600; font-size:1rem; margin-bottom:6px; color:#e8eaf0;'>{title}</div>
          <div style='font-size:0.82rem; color:#8892a4; line-height:1.5;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

if run_btn and target_col and n_uniq <= 10 and n_uniq > 1:
    with st.spinner("⚙️ Running full ML pipeline — this takes ~30–60 seconds..."):
        res, err = run_full_pipeline(file_bytes, file_ext, table_name, target_col)
    if err:
        st.error(f"Pipeline error: {err}")
        st.stop()
    st.session_state.results = res
    st.success("✅ Pipeline complete! Explore the tabs below.")

if st.session_state.results is None:
    st.info("👈 Upload a file and click **Run Pipeline** to get started.")
    st.stop()

R = st.session_state.results  # shorthand


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview",
    "📈 Model Performance",
    "🔍 Explainability",
    "🎯 Customer Risk Table",
])


# ───────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ───────────────────────────────────────────────────────────────────────
with tab1:
    pred_df   = R['pred_df']
    total     = len(pred_df)
    n_churn   = int(pred_df['Predicted'].sum())
    churn_pct = n_churn / total * 100
    best_f1   = R['results_df'].iloc[0]['F1-Score']
    best_auc  = R['results_df'].iloc[0]['ROC-AUC']

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, cls in [
        (k1, f"{total:,}",           "Total Customers",  "kpi-blue"),
        (k2, f"{n_churn:,}",         "At-Risk Customers","kpi-red"),
        (k3, f"{churn_pct:.1f}%",    "Churn Rate",       "kpi-yellow"),
        (k4, f"{best_f1:.3f}",       "Best Model F1",    "kpi-green"),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-value {cls}'>{val}</div>
          <div class='kpi-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown("<div class='section-header'>Top Reasons Customers Churn</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#8892a4; margin-bottom:12px;'>Based on SHAP feature importance from the best model</div>", unsafe_allow_html=True)
        top5 = R['mean_shap'].head(5)
        icons = ["🔴","🟠","🟡","🔵","🟢"]
        for i, row in top5.iterrows():
            feat  = row['Feature'].replace('_',' ').title()
            impact = row['Impact']
            bar_w  = int(impact / top5['Impact'].max() * 100)
            rank   = list(top5.index).index(i)
            st.markdown(f"""
            <div class='info-box' style='margin-bottom:6px;'>
              <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span>{icons[rank]} <strong>{feat}</strong></span>
                <span style='color:#8892a4; font-size:0.8rem;'>impact {impact:.3f}</span>
              </div>
              <div style='background:#2d3654; border-radius:4px; height:5px;'>
                <div style='background:linear-gradient(90deg,#4dabf7,#748ffc); width:{bar_w}%; height:5px; border-radius:4px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)
        risk_counts = pred_df['Risk_Level'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        colors_pie = ['#ff6b6b','#ffd43b','#51cf66']
        wedge_props = {'linewidth': 2, 'edgecolor': '#1a2035'}
        ax.pie(
            [risk_counts.get('High',0), risk_counts.get('Medium',0), risk_counts.get('Low',0)],
            labels=['High Risk','Medium Risk','Low Risk'],
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=140,
            wedgeprops=wedge_props,
            textprops={'color':'#e8eaf0','fontsize':11}
        )
        fig.patch.set_facecolor('#1a2035')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Best model callout
    st.markdown(f"""
    <div class='info-box' style='border-left: 3px solid #4dabf7; margin-top:16px;'>
      🏆 <strong>Best Model:</strong> {R['best_name']}
      &nbsp;·&nbsp; F1-Score: <strong>{best_f1}</strong>
      &nbsp;·&nbsp; ROC-AUC: <strong>{best_auc}</strong>
      &nbsp;·&nbsp; Evaluated on held-out 20% test set
    </div>""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ───────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

    # Styled results table
    res_df = R['results_df'].copy()
    res_df.insert(0, 'Rank', range(1, len(res_df)+1))
    res_df[''] = res_df['Model'].apply(lambda m: '🏆 Best' if m == R['best_name'] else '')
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Metric bar charts
    st.markdown("<div class='section-header'>Metrics Breakdown</div>", unsafe_allow_html=True)
    metrics = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, metric, color in zip(axes, metrics, PALETTE):
        data = R['results_df'].sort_values(metric, ascending=True)
        bars = ax.barh(data['Model'], data[metric], color=color, alpha=0.85)
        ax.set_xlim([max(0, data[metric].min()-0.05), 1.0])
        ax.set_title(metric, fontsize=10, fontweight='bold', color='#e8eaf0')
        ax.tick_params(labelsize=8)
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_width()+0.004, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7.5, color='#e8eaf0')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ROC Curves
    st.markdown("<div class='section-header'>ROC Curves</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, model), color in zip(R['models'].items(), PALETTE):
        yp_pr = model.predict_proba(R['X_test'])[:,1]
        fpr, tpr, _ = roc_curve(R['y_test'], yp_pr)
        roc_a = auc(fpr, tpr)
        lw = 2.5 if name == R['best_name'] else 1.5
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_a:.3f})", linewidth=lw, color=color)
    ax.plot([0,1],[0,1],'--', color='#555e6e', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Confusion matrices
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    n_m = len(R['models'])
    n_cols_cm = 2
    n_rows_cm = math.ceil(n_m / n_cols_cm)
    fig, axes = plt.subplots(n_rows_cm, n_cols_cm, figsize=(13, 5*n_rows_cm))
    axes = axes.flatten()
    cl = R['class_labels']
    for idx, (name, model) in enumerate(R['models'].items()):
        cm = confusion_matrix(R['y_test'], model.predict(R['X_test']))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cl, yticklabels=cl,
                    ax=axes[idx], cbar=False,
                    annot_kws={'size':13,'weight':'bold'})
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold', color='#e8eaf0')
        axes[idx].set_xlabel('Predicted', color='#8892a4')
        axes[idx].set_ylabel('Actual', color='#8892a4')
        axes[idx].tick_params(colors='#8892a4')
    for idx in range(n_m, len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# TAB 3 — EXPLAINABILITY
# ───────────────────────────────────────────────────────────────────────
with tab3:
    xai_tab1, xai_tab2 = st.tabs(["🌐 SHAP Global", "👤 Customer Lookup"])

    with xai_tab1:
        st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        top_n = R['mean_shap'].head(12)
        bars = ax.barh(top_n['Feature'][::-1], top_n['Impact'][::-1], color=PALETTE[0], alpha=0.85)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Global Feature Impact — {R["best_name"]}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-header'>SHAP Summary Plot</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(R['shap_vals'], R['X_test'], show=False, plot_size=None, color_bar=True)
        fig = plt.gcf()
        fig.patch.set_facecolor('#1a2035')
        st.pyplot(fig, use_container_width=True)
        plt.close('all')

    with xai_tab2:
        st.markdown("<div class='section-header'>Individual Customer Explanation</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#8892a4; font-size:0.88rem; margin-bottom:16px;'>Select a customer index from the test set to see their churn probability and why the model predicted it.</div>", unsafe_allow_html=True)

        test_indices = list(R['X_test'].index)
        chosen_idx   = st.selectbox("Customer index", test_indices)
        row          = R['X_test'].loc[[chosen_idx]]
        prob         = R['best_model'].predict_proba(row)[0][1]
        actual       = int(R['y_test'].loc[chosen_idx])

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Predicted",  "🔴 Churn" if prob >= 0.5 else "🟢 Stay")
        c3.metric("Actual",     "Churned" if actual == 1 else "Active")

        # SHAP waterfall
        shap_exp = R['explainer'](row)
        if shap_exp.values.ndim == 3:
            shap_exp = shap_exp[:,:,1]
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_exp[0], show=False, max_display=12)
        fig = plt.gcf()
        fig.patch.set_facecolor('#1a2035')
        plt.title(f'Why Customer {chosen_idx} {"Churns" if prob>=0.5 else "Stays"}', fontsize=11, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close('all')

        # LIME
        st.markdown("<div class='section-header'>LIME Explanation</div>", unsafe_allow_html=True)
        lime_exp = R['lime_explainer'].explain_instance(
            row.values[0], R['best_model'].predict_proba, num_features=10)
        lime_list = lime_exp.as_list()
        lime_df   = pd.DataFrame(lime_list, columns=['Feature Condition','Weight'])
        lime_df['Direction'] = lime_df['Weight'].apply(lambda w: '🔴 Increases churn risk' if w > 0 else '🟢 Decreases churn risk')
        lime_df['Weight']    = lime_df['Weight'].round(4)
        st.dataframe(lime_df[['Feature Condition','Direction','Weight']], use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 4 — CUSTOMER RISK TABLE
# ───────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Customer Risk Table</div>", unsafe_allow_html=True)

    pred_display = R['pred_df'][['Churn_Probability','Predicted','Actual','Risk_Level']].copy()
    pred_display.index.name = 'Customer_ID'
    pred_display = pred_display.reset_index()

    # Filters
    fc1, fc2 = st.columns([1,2])
    with fc1:
        risk_filter = st.multiselect("Filter by Risk Level",
                                     ['High','Medium','Low'],
                                     default=['High','Medium','Low'])
    with fc2:
        prob_range = st.slider("Churn Probability Range (%)", 0, 100, (0, 100))

    filtered = pred_display[
        pred_display['Risk_Level'].isin(risk_filter) &
        pred_display['Churn_Probability'].between(prob_range[0], prob_range[1])
    ]

    st.markdown(f"<div style='color:#8892a4; font-size:0.85rem; margin-bottom:8px;'>Showing {len(filtered):,} of {len(pred_display):,} customers</div>", unsafe_allow_html=True)
    st.dataframe(filtered.sort_values('Churn_Probability', ascending=False),
                 use_container_width=True, hide_index=True)

    # CSV download
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button("📥 Download Risk Table (CSV)",
                       data=csv_bytes,
                       file_name="churn_risk_table.csv",
                       mime="text/csv",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PDF REPORT (sidebar button shown after results)
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf(R, target_col):
    pred_df  = R['pred_df']
    total    = len(pred_df)
    n_churn  = int(pred_df['Predicted'].sum())
    best_row = R['results_df'].iloc[0]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(30, 50, 90)
    pdf.cell(0, 12, 'Churn Prediction XAI Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 110, 130)
    pdf.cell(0, 7, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  Target column: {target_col}', ln=True, align='C')
    pdf.ln(6)

    # Divider
    pdf.set_draw_color(60, 100, 200)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # KPI section
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 50, 90)
    pdf.cell(0, 9, '1. Dataset & Churn Summary', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 60, 80)
    for label, val in [
        ('Total Customers in Test Set', f"{total:,}"),
        ('Predicted to Churn',          f"{n_churn:,}  ({n_churn/total*100:.1f}%)"),
        ('High Risk',   str(int((pred_df['Risk_Level']=='High').sum()))),
        ('Medium Risk', str(int((pred_df['Risk_Level']=='Medium').sum()))),
        ('Low Risk',    str(int((pred_df['Risk_Level']=='Low').sum()))),
    ]:
        pdf.cell(90, 7, label + ':', border=0)
        pdf.cell(0, 7, val, ln=True)
    pdf.ln(4)

    # Model performance
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 50, 90)
    pdf.cell(0, 9, '2. Model Performance', ln=True)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(40, 70, 160)
    for h, w in [('Model',44),('Accuracy',28),('Precision',28),('Recall',28),('F1-Score',28),('ROC-AUC',28)]:
        pdf.cell(w, 7, h, border=0, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(40, 50, 70)
    for i, row in R['results_df'].iterrows():
        fill = (i == 0)
        pdf.set_fill_color(230, 240, 255) if fill else pdf.set_fill_color(248, 249, 252)
        for val, w in [(row['Model'],44),(row['Accuracy'],28),(row['Precision'],28),
                       (row['Recall'],28),(row['F1-Score'],28),(row['ROC-AUC'],28)]:
            pdf.cell(w, 7, str(val), border='B', fill=True, align='C')
        pdf.ln()
    pdf.ln(4)

    # Top churn drivers
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 50, 90)
    pdf.cell(0, 9, '3. Top Churn Drivers (SHAP)', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 60, 80)
    for rank, (_, row) in enumerate(R['mean_shap'].head(8).iterrows(), 1):
        pdf.cell(0, 7, f"  {rank}. {row['Feature'].replace('_',' ').title()}  —  mean |SHAP| = {row['Impact']:.4f}", ln=True)
    pdf.ln(4)

    # High risk customers sample
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 50, 90)
    pdf.cell(0, 9, '4. Sample High-Risk Customers (Top 10)', ln=True)
    high_risk = pred_df[pred_df['Risk_Level']=='High'].sort_values(
        'Churn_Probability', ascending=False).head(10)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(40, 70, 160)
    for h, w in [('Customer ID', 50),('Churn Prob %', 50),('Risk Level', 50),('Predicted',40)]:
        pdf.cell(w, 7, h, border=0, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(40, 50, 70)
    for i, (idx, row) in enumerate(high_risk.iterrows()):
        pdf.set_fill_color(255, 240, 240) if i % 2 == 0 else pdf.set_fill_color(248, 249, 252)
        for val, w in [(str(idx),50),(f"{row['Churn_Probability']:.1f}%",50),
                       (str(row['Risk_Level']),50),('Churn' if row['Predicted']==1 else 'Stay',40)]:
            pdf.cell(w, 7, val, border='B', fill=True, align='C')
        pdf.ln()

    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 160, 180)
    pdf.cell(0, 6, 'Generated by Churn Prediction XAI Dashboard  —  github.com/your-repo', ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')


with st.sidebar:
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("### 📥 Download Report")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf(R, target_col)
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
