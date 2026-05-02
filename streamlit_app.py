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
for key in ['df_raw','df_encoded','results_df','models','best_model_name',
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
            df = pd.read_sql(f'SELECT * FROM "{tables[0]}"', conn)
            return df, conn, tables
    return None, None, []


def detect_churn_column(df):
    """Auto-detect churn column from lowercased column names."""
    keywords = ['churned','churn','attrition','exited','left','target','label','status','outcome']
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return df.columns[0]


def run_pipeline(df, target_column):
    df = df.copy()

    # ── Lowercase all columns (Step 2 change from notebook) ─────────────────
    df.columns = [col.lower() for col in df.columns]
    target_column = target_column.lower()

    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == 'object':
            converted = pd.to_numeric(
                df[col].astype(str).str.replace('[,$%]', '', regex=True), errors='coerce')
            if converted.notna().sum() / len(df) > 0.7:
                df[col] = converted

    # ── Missing values ───────────────────────────────────────────────────────
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isnull().mean() > 0.5:
            df.drop(columns=[col], inplace=True); continue
        if df[col].dtype in ['float64','int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else 'Unknown', inplace=True)
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
            df.drop(columns=[col], inplace=True)

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
    try:
        if hasattr(best_model, 'feature_importances_'):
            explainer  = shap.TreeExplainer(best_model)
            sv = explainer.shap_values(X_test_sel)
            shap_values = sv[1] if isinstance(sv, list) else sv
        else:
            explainer   = shap.LinearExplainer(best_model, X_tr_res)
            shap_values = explainer.shap_values(X_test_sel)
    except Exception:
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
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=0.75*inch, rightMargin=0.75*inch,
                              topMargin=0.75*inch,  bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    accent = colors.HexColor('#3b82f6')
    dark   = colors.HexColor('#0f172a')

    title_st = ParagraphStyle('T', parent=styles['Title'],
                               fontSize=22, textColor=accent, spaceAfter=4)
    h1_st    = ParagraphStyle('H1', parent=styles['Heading1'],
                               fontSize=13, textColor=dark, spaceBefore=14, spaceAfter=6)
    body_st  = ParagraphStyle('B', parent=styles['Normal'],
                               fontSize=10, leading=15, textColor=colors.HexColor('#334155'))
    small_st = ParagraphStyle('S', parent=styles['Normal'],
                               fontSize=8, leading=12, textColor=colors.HexColor('#64748b'))
    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4*inch),
        Paragraph("Churn Prediction · XAI Report", title_st),
        Paragraph("Explainable AI Analysis", styles['Heading2']),
        Spacer(1, 0.1*inch),
        HRFlowable(width="100%", thickness=2, color=accent),
        Spacer(1, 0.1*inch),
    ]
    meta_tbl = Table([
        ['Generated', now],
        ['Dataset rows', f"{len(df_raw):,}"],
        ['Target column', target_column],
        ['Best model', best_model_name],
    ], colWidths=[1.5*inch, 4.5*inch])
    meta_tbl.setStyle(TableStyle([
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('TEXTCOLOR',  (0,0),(0,-1), accent),
        ('FONTNAME',   (0,0),(0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story += [meta_tbl, Spacer(1, 0.3*inch)]

    # ── Executive summary ────────────────────────────────────────────────────
    churn_rate = y_test_final.mean() * 100
    best_row   = results_df.iloc[0]
    churn_word = "high" if churn_rate > 20 else "moderate" if churn_rate > 10 else "low"

    story.append(Paragraph("Executive Summary", h1_st))
    kpis = [
        ['Metric','Value','Interpretation'],
        ['Total Customers (test)', str(len(y_test_final)), 'Customers in test set'],
        ['Churn Rate', f'{churn_rate:.1f}%', 'Customers who churned in test set'],
        ['Best Model', best_model_name, 'Highest F1-Score model'],
        ['F1-Score', f"{best_row['F1-Score']:.4f}", 'Balance of precision & recall'],
        ['ROC-AUC',  f"{best_row['ROC-AUC']:.4f}",  'Discrimination ability (1.0 = perfect)'],
        ['Accuracy', f"{best_row['Accuracy']:.4f}",  'Overall prediction correctness'],
    ]
    kpi_tbl = Table(kpis, colWidths=[2.1*inch, 1.5*inch, 2.9*inch])
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), accent),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('TOPPADDING',   (0,0),(-1,-1), 7),
    ]))
    story += [kpi_tbl, Spacer(1, 0.2*inch)]

    # ── Non-technical summary ────────────────────────────────────────────────
    story.append(Paragraph("Plain English Interpretation", h1_st))
    story.append(Paragraph(
        f"The analysis found a <b>{churn_word} churn rate of {churn_rate:.1f}%</b>. "
        f"<b>{best_model_name}</b> was the best performing model with an F1-Score of "
        f"<b>{best_row['F1-Score']:.2%}</b> and AUC of <b>{best_row['ROC-AUC']:.2%}</b>. "
        f"An AUC above 0.80 is considered good — this model reliably distinguishes "
        f"customers who will churn from those who will stay.", body_st))
    story.append(Spacer(1, 0.2*inch))

    # ── Model comparison ─────────────────────────────────────────────────────
    story.append(Paragraph("Model Comparison", h1_st))
    cols_order = ['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    tbl_data   = [cols_order] + results_df[cols_order].values.tolist()
    mdl_tbl    = Table(tbl_data, colWidths=[1.9*inch,1*inch,1*inch,1*inch,1*inch,1*inch])
    mdl_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), dark),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('BACKGROUND', (0,1),(-1,1), colors.HexColor('#dcfce7')),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN',      (1,0),(-1,-1), 'CENTER'),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ('TOPPADDING',   (0,0),(-1,-1), 6),
    ]))
    story += [mdl_tbl, Paragraph("* Green row = best performing model", small_st),
              Spacer(1, 0.2*inch)]

    # ── SHAP chart ───────────────────────────────────────────────────────────
    story.append(Paragraph("Top Features Driving Churn", h1_st))
    if shap_values is not None:
        mean_shap = pd.Series(np.abs(shap_values).mean(axis=0),
                               index=selected_feature_names).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
        ax.barh(mean_shap.index[::-1], mean_shap.values[::-1],
                color=['#3b82f6' if i < 3 else '#94a3b8' for i in range(len(mean_shap)-1,-1,-1)])
        ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
        ax.set_title(f'Top Features — {best_model_name}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.35); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        img_buf = fig_to_png(fig)
        story += [RLImage(img_buf, width=6.5*inch, height=3.5*inch), Spacer(1,0.1*inch)]

        story.append(Paragraph("Top 5 Churn Drivers:", h1_st))
        for i, (feat, val) in enumerate(mean_shap.head(5).items(), 1):
            story.append(Paragraph(
                f"<b>{i}. {feat}</b> — SHAP impact: {val:.4f}. "
                f"{'Strongly' if val > mean_shap.mean() else 'Moderately'} influences churn decisions.",
                body_st))
        story.append(Spacer(1, 0.1*inch))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Model Performance Details", h1_st))
    y_pred = best_model.predict(X_test_final)
    cm     = confusion_matrix(y_test_final, y_pred)
    labels = [str(c) for c in sorted(y_test_final.unique())]

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    cm_buf = fig_to_png(fig)
    story += [RLImage(cm_buf, width=3.5*inch, height=2.5*inch), Spacer(1,0.15*inch)]

    story.append(Paragraph("Classification Report:", h1_st))
    cr     = classification_report(y_test_final, y_pred, target_names=labels, output_dict=True)
    cr_rows = [['Class','Precision','Recall','F1','Support']]
    for cls in labels:
        r = cr[cls]
        cr_rows.append([cls, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                         f"{r['f1-score']:.3f}", str(int(r['support']))])
    cr_tbl = Table(cr_rows, colWidths=[1.5*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
    cr_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), dark),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN',      (1,0),(-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8fafc'), colors.white]),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
    ]))
    story.append(cr_tbl)

    # ── Footer ───────────────────────────────────────────────────────────────
    story += [Spacer(1, 0.5*inch),
              HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1')),
              Paragraph(f"Generated by Churn Prediction XAI Dashboard · {now}", small_st)]

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
            df, conn, tables = load_uploaded_file(uploaded)

        if df is not None:
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
            c4.metric("Numeric cols",int((df.select_dtypes(include='number').shape[1])))

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Data preview
            with st.expander("👁️ Data Preview", expanded=False):
                st.dataframe(df.head(12), use_container_width=True)

            # ── Target column selection ──────────────────────────────────────
            st.markdown('<div class="section-title">🎯 Select Target (Churn) Column</div>', unsafe_allow_html=True)
            st.caption("Columns are lowercased automatically before the pipeline runs.")

            # Lowercase column names for the dropdown
            lower_cols   = [col.lower() for col in df.columns]
            default_col  = detect_churn_column(df)
            default_lower = default_col.lower()
            try:
                default_idx = lower_cols.index(default_lower)
            except ValueError:
                default_idx = 0

            target_col_lower = st.selectbox(
                "Target column (lowercased)",
                lower_cols,
                index=default_idx,
                label_visibility='collapsed'
            )

            n_unique = df[df.columns[lower_cols.index(target_col_lower)]].nunique()
            if n_unique > 10:
                st.warning(f"⚠️ '{target_col_lower}' has {n_unique} unique values — confirm this is the churn column.")
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

    # All-models leaderboard (compact table only — no full model results page)
    st.markdown('<div class="section-title">🏅 Model Leaderboard</div>', unsafe_allow_html=True)
    results_df = st.session_state['results_df']
    best_name  = st.session_state['best_model_name']

    def highlight_best(row):
        return ['background-color:#0a2218;color:#10b981;font-weight:600']*len(row) \
               if row['Model'] == best_name else ['']*len(row)

    st.dataframe(
        results_df.style.apply(highlight_best, axis=1).format(
            {'Accuracy':':.4f','Precision':':.4f','Recall':':.4f','F1-Score':':.4f','ROC-AUC':':.4f'}),
        use_container_width=True, hide_index=True)


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

    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0),
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
        cust_shap   = shap_values[row_pos]
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
