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
import math
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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Prediction XAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0f1117; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #4f8ef7;
    margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.kpi-card.green  { border-left-color: #2ecc71; }
.kpi-card.red    { border-left-color: #e74c3c; }
.kpi-card.yellow { border-left-color: #f39c12; }
.kpi-card.blue   { border-left-color: #4f8ef7; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0; }
.kpi-label { font-size: 0.85rem; color: #8899aa; margin: 0; text-transform: uppercase; letter-spacing: 1px; }

/* Risk badges */
.risk-high   { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c55; border-radius:6px; padding:4px 12px; font-weight:600; }
.risk-medium { background:#f39c1222; color:#f39c12; border:1px solid #f39c1255; border-radius:6px; padding:4px 12px; font-weight:600; }
.risk-low    { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc7155; border-radius:6px; padding:4px 12px; font-weight:600; }

/* Section headers */
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #ffffff;
    border-bottom: 2px solid #4f8ef7; padding-bottom: 8px; margin-bottom: 16px;
}
.plain-insight {
    background: #1e2130; border-radius: 10px; padding: 16px 20px;
    color: #cdd6e0; font-size: 0.95rem; line-height: 1.7;
    border-left: 3px solid #4f8ef7; margin-bottom: 8px;
}
.sidebar-logo { text-align:center; padding: 10px 0 20px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════
for key in ['df_raw','df_encoded','results_df','models','best_model_name',
            'best_model','X_test_final','y_test_final','selected_feature_names',
            'X_train_resampled','y_train_resampled','shap_values','pipeline_done',
            'target_column','expert_mode']:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state['pipeline_done'] is None:
    st.session_state['pipeline_done'] = False
if st.session_state['expert_mode'] is None:
    st.session_state['expert_mode'] = False

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
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.csv':
        return pd.read_csv(uploaded_file), None, []
    elif ext == '.xlsx':
        xl = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(uploaded_file, sheet_name=xl.sheet_names[0])
        return df, None, []
    elif ext == '.json':
        try:
            df = pd.read_json(uploaded_file)
        except ValueError:
            df = pd.read_json(uploaded_file, orient='records')
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
    keywords = ['churn','attrition','target','label','exited','left','churned','status','outcome']
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return df.columns[0]


def run_pipeline(df, target_column):
    """Full ML pipeline — returns dict of artifacts."""
    df = df.copy()

    # ── Numeric coercion ────────────────────────────────────────────────────
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col].astype(str).str.replace('[,$%]','',regex=True), errors='coerce')
            if converted.notna().sum() / len(df) > 0.7:
                df[col] = converted

    # ── Missing values ──────────────────────────────────────────────────────
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isnull().mean() > 0.5:
            df.drop(columns=[col], inplace=True)
            continue
        if df[col].dtype in ['float64','int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else 'Unknown', inplace=True)
    df.dropna(subset=[target_column], inplace=True)

    # ── Encoding ────────────────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    for col in cat_cols:
        n = df[col].nunique()
        if n <= 2:
            le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))
        elif n <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif n <= 20:
            le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)

    if df[target_column].dtype == 'object':
        le = LabelEncoder(); df[target_column] = le.fit_transform(df[target_column])

    # ── Split ───────────────────────────────────────────────────────────────
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── Scale ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns, index=X_test.index)

    # ── Corr removal ────────────────────────────────────────────────────────
    corr = X_train_sc.corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i,j] >= 0.90:
                c1,c2 = corr.columns[i], corr.columns[j]
                to_drop.add(c1 if abs(X_train_sc[c1].corr(y_train)) < abs(X_train_sc[c2].corr(y_train)) else c2)
    X_train_sc.drop(columns=list(to_drop), inplace=True, errors='ignore')
    X_test_sc.drop(columns=list(to_drop), inplace=True, errors='ignore')

    # ── Feature selection ───────────────────────────────────────────────────
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_sel.fit(X_train_sc, y_train)
    fi = pd.Series(rf_sel.feature_importances_, index=X_train_sc.columns).sort_values(ascending=False)
    n_keep = max(8, int(len(fi)*0.7))
    selected = fi.head(n_keep).index.tolist()
    X_train_sel = X_train_sc[selected]
    X_test_sel  = X_test_sc[selected]

    # ── SMOTE ───────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
    X_tr_res, y_tr_res = smote.fit_resample(X_train_sel, y_train)
    X_tr_res = pd.DataFrame(X_tr_res, columns=selected)

    # ── Train models ────────────────────────────────────────────────────────
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }
    results = []
    for name, model in models.items():
        model.fit(X_tr_res, y_tr_res)
        yp = model.predict(X_test_sel)
        ypp = model.predict_proba(X_test_sel)[:,1]
        results.append({'Model':name,
                        'Accuracy':  round(accuracy_score(y_test, yp),4),
                        'Precision': round(precision_score(y_test, yp, zero_division=0),4),
                        'Recall':    round(recall_score(y_test, yp, zero_division=0),4),
                        'F1-Score':  round(f1_score(y_test, yp, zero_division=0),4),
                        'ROC-AUC':   round(roc_auc_score(y_test, ypp),4)})

    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]['Model']
    best_model = models[best_name]

    # ── SHAP ────────────────────────────────────────────────────────────────
    try:
        if hasattr(best_model, 'feature_importances_'):
            explainer   = shap.TreeExplainer(best_model)
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
    """Return a matplotlib figure gauge chart."""
    fig, ax = plt.subplots(figsize=(size/80, size/80), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    # Background arc
    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [1]*200, color='#2a2f45', linewidth=18, solid_capstyle='round')

    # Value arc
    color = '#e74c3c' if prob > 0.6 else '#f39c12' if prob > 0.35 else '#2ecc71'
    theta_val = np.linspace(0, np.pi*prob, 200)
    ax.plot(theta_val, [1]*200, color=color, linewidth=18, solid_capstyle='round')

    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, np.pi)
    ax.axis('off')
    ax.text(np.pi/2, 0.2, f'{prob*100:.1f}%',
            ha='center', va='center', fontsize=20, fontweight='bold', color='white',
            transform=ax.transData)
    plt.tight_layout(pad=0)
    return fig


def generate_pdf_report(results_df, best_model_name, best_model,
                        X_test_final, y_test_final, shap_values,
                        selected_feature_names, target_column, df_raw):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ───────────────────────────────────────────────────────
    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                  fontSize=22, textColor=colors.HexColor('#4f8ef7'),
                                  spaceAfter=6)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                         fontSize=14, textColor=colors.HexColor('#2c3e50'),
                         spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle('Body2', parent=styles['Normal'],
                          fontSize=10, leading=14, textColor=colors.HexColor('#333333'))
    small = ParagraphStyle('Small', parent=styles['Normal'],
                           fontSize=9, leading=12, textColor=colors.HexColor('#666666'))

    # ── Cover ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("📊 Churn Prediction Report", title_style))
    story.append(Paragraph("Explainable AI Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#4f8ef7')))
    story.append(Spacer(1, 0.1*inch))

    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    meta = [
        ['Generated', now],
        ['Dataset rows', str(len(df_raw))],
        ['Target column', target_column],
        ['Best model', best_model_name],
    ]
    meta_tbl = Table(meta, colWidths=[1.5*inch, 4*inch])
    meta_tbl.setStyle(TableStyle([
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('TEXTCOLOR',  (0,0),(0,-1), colors.HexColor('#4f8ef7')),
        ('FONTNAME',   (0,0),(0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.3*inch))

    # ── KPI summary ─────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h1))
    churn_rate = y_test_final.mean() * 100
    best_row   = results_df.iloc[0]

    kpis = [['Metric','Value','Interpretation'],
            ['Total Customers (test)', str(len(y_test_final)), 'Customers in test set'],
            ['Churn Rate', f'{churn_rate:.1f}%', 'Customers who churned'],
            ['Best Model', best_model_name, 'Top performing model'],
            ['F1-Score', f"{best_row['F1-Score']:.4f}", 'Balance of precision & recall'],
            ['ROC-AUC', f"{best_row['ROC-AUC']:.4f}", 'Discrimination ability (1.0 = perfect)'],
            ['Accuracy', f"{best_row['Accuracy']:.4f}", 'Overall correctness'],
    ]
    kpi_tbl = Table(kpis, colWidths=[2*inch, 1.5*inch, 3*inch])
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#4f8ef7')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('TOPPADDING',   (0,0),(-1,-1), 7),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.2*inch))

    # ── Plain English ───────────────────────────────────────────────────────
    story.append(Paragraph("What This Means (Non-Technical Summary)", h1))
    churn_word = "high" if churn_rate > 20 else "moderate" if churn_rate > 10 else "low"
    story.append(Paragraph(
        f"The analysis found a <b>{churn_word} churn rate of {churn_rate:.1f}%</b> in the test data. "
        f"The best model, <b>{best_model_name}</b>, correctly identifies churners with an F1-Score of "
        f"<b>{best_row['F1-Score']:.2%}</b> and an AUC of <b>{best_row['ROC-AUC']:.2%}</b>. "
        f"An AUC above 0.80 is considered good — this model can reliably distinguish customers "
        f"who will churn from those who will stay.", body))
    story.append(Spacer(1, 0.2*inch))

    # ── Model comparison ────────────────────────────────────────────────────
    story.append(Paragraph("Model Comparison", h1))
    cols_order = ['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    tbl_data = [cols_order] + results_df[cols_order].values.tolist()
    mdl_tbl = Table(tbl_data, colWidths=[1.8*inch,1*inch,1*inch,1*inch,1*inch,1*inch])
    mdl_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('BACKGROUND', (0,1),(-1,1), colors.HexColor('#d5f5e3')),  # best model row
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN',      (1,0),(-1,-1), 'CENTER'),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ('TOPPADDING',   (0,0),(-1,-1), 6),
    ]))
    story.append(mdl_tbl)
    story.append(Paragraph("* Green row = best model", small))
    story.append(Spacer(1, 0.2*inch))

    # ── Feature importance chart ─────────────────────────────────────────────
    story.append(Paragraph("Top Features Driving Churn", h1))

    if shap_values is not None:
        mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=selected_feature_names).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        bars = ax.barh(mean_shap.index[::-1], mean_shap.values[::-1],
                       color=['#4f8ef7' if i < 3 else '#95a5a6' for i in range(len(mean_shap)-1,-1,-1)])
        ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
        ax.set_title(f'Top Features — {best_model_name}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.4)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buf.seek(0)
        story.append(RLImage(img_buf, width=6.5*inch, height=3.5*inch))
        story.append(Spacer(1, 0.1*inch))

        # Top 5 in plain text
        story.append(Paragraph("Top 5 Churn Drivers (Plain English):", h1))
        for i, (feat, val) in enumerate(mean_shap.head(5).items(), 1):
            story.append(Paragraph(
                f"<b>{i}. {feat}</b> — SHAP impact score: {val:.4f}. "
                f"This feature {'strongly' if val > mean_shap.mean() else 'moderately'} influences churn decisions.", body))
        story.append(Spacer(1, 0.1*inch))

    # ── Confusion matrix ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Model Performance Details", h1))
    y_pred = best_model.predict(X_test_final)
    cm = confusion_matrix(y_test_final, y_pred)
    labels = [str(c) for c in sorted(y_test_final.unique())]

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    cm_buf = io.BytesIO()
    plt.savefig(cm_buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(); cm_buf.seek(0)
    story.append(RLImage(cm_buf, width=3.5*inch, height=2.5*inch))
    story.append(Spacer(1, 0.15*inch))

    # Classification report
    story.append(Paragraph("Classification Report:", h1))
    cr = classification_report(y_test_final, y_pred, target_names=labels, output_dict=True)
    cr_rows = [['Class','Precision','Recall','F1','Support']]
    for cls in labels:
        r = cr[cls]
        cr_rows.append([cls, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", str(int(r['support']))])
    cr_tbl = Table(cr_rows, colWidths=[1.5*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
    cr_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ALIGN',      (1,0),(-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'), colors.white]),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
    ]))
    story.append(cr_tbl)

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7')))
    story.append(Paragraph(f"Generated by Churn Prediction XAI Dashboard · {now}", small))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><h2 style="color:#4f8ef7">📊 Churn XAI</h2></div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "📁 Upload & Configure",
        "📊 Overview",
        "🏆 Model Results",
        "🔍 XAI Explanations",
        "👤 Customer Lookup",
        "📥 Download Report",
    ])

    st.markdown("---")
    expert_mode = st.toggle("🔬 Expert Mode", value=st.session_state['expert_mode'])
    st.session_state['expert_mode'] = expert_mode

    if st.session_state['pipeline_done']:
        st.success("✅ Pipeline complete")
        st.info(f"Best: **{st.session_state['best_model_name']}**")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & CONFIGURE
# ════════════════════════════════════════════════════════════════════════════
if page == "📁 Upload & Configure":
    st.markdown('<div class="section-title">📁 Upload Your Dataset</div>', unsafe_allow_html=True)
    st.caption("Supported formats: CSV · XLSX · JSON · SQL")

    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Choose a file", type=['csv','xlsx','json','sql'],
                                     label_visibility='collapsed')

    if uploaded:
        with st.spinner("Loading file..."):
            df, conn, tables = load_uploaded_file(uploaded)

        if df is not None:
            st.session_state['df_raw'] = df
            st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

            # Preview
            with st.expander("👁️ Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Rows", f"{df.shape[0]:,}")
                c2.metric("Columns", df.shape[1])
                c3.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")

            # Target column
            st.markdown('<div class="section-title">🎯 Select Target (Churn) Column</div>', unsafe_allow_html=True)
            default_col = detect_churn_column(df)
            target_col = st.selectbox("Target column", df.columns.tolist(),
                                       index=df.columns.tolist().index(default_col))

            n_unique = df[target_col].nunique()
            if n_unique > 10:
                st.warning(f"⚠️ '{target_col}' has {n_unique} unique values — might not be binary. Please verify.")
            else:
                vc = df[target_col].value_counts()
                st.markdown("**Class distribution:**")
                fig, ax = plt.subplots(figsize=(4,2.5))
                fig.patch.set_facecolor('#1e2130')
                ax.set_facecolor('#1e2130')
                colors_pie = ['#2ecc71','#e74c3c','#4f8ef7','#f39c12']
                ax.pie(vc.values, labels=[str(x) for x in vc.index],
                       colors=colors_pie[:len(vc)], autopct='%1.1f%%',
                       textprops={'color':'white','fontsize':11})
                ax.set_title("Class Balance", color='white', fontsize=11)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            st.session_state['target_column'] = target_col

            st.markdown("---")
            if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
                progress = st.progress(0, text="Preprocessing data...")
                with st.spinner("Running ML pipeline — this may take a minute..."):
                    try:
                        progress.progress(20, "Encoding & cleaning...")
                        result = run_pipeline(df, target_col)
                        progress.progress(80, "Computing SHAP explanations...")
                        for k, v in result.items():
                            st.session_state[k] = v
                        st.session_state['pipeline_done'] = True
                        progress.progress(100, "Done!")
                        st.success("✅ Pipeline complete! Use the sidebar to explore results.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
                        import traceback; st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Overview":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    df_raw = st.session_state['df_raw']
    best_row = st.session_state['results_df'].iloc[0]
    y_test = st.session_state['y_test_final']
    churn_rate = y_test.mean()*100

    st.markdown('<div class="section-title">📊 Analysis Overview</div>', unsafe_allow_html=True)

    # KPI cards
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card blue"><p class="kpi-label">Total Customers</p><p class="kpi-value">{len(df_raw):,}</p></div>', unsafe_allow_html=True)
    with c2:
        color = "red" if churn_rate > 20 else "yellow" if churn_rate > 10 else "green"
        st.markdown(f'<div class="kpi-card {color}"><p class="kpi-label">Churn Rate</p><p class="kpi-value">{churn_rate:.1f}%</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card green"><p class="kpi-label">Best Model</p><p class="kpi-value" style="font-size:1.2rem">{st.session_state["best_model_name"]}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card yellow"><p class="kpi-label">Best F1 / AUC</p><p class="kpi-value">{best_row["F1-Score"]} / {best_row["ROC-AUC"]}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Plain English summary
    st.markdown('<div class="section-title">💬 Plain English Summary</div>', unsafe_allow_html=True)
    churn_word = "high" if churn_rate > 20 else "moderate" if churn_rate > 10 else "low"
    st.markdown(f'''
    <div class="plain-insight">🔍 <b>Churn Risk:</b> The dataset shows a <b>{churn_word} churn rate of {churn_rate:.1f}%</b>. That means roughly {churn_rate:.0f} out of every 100 customers are predicted to leave.</div>
    <div class="plain-insight">🏆 <b>Best Model:</b> <b>{st.session_state["best_model_name"]}</b> achieved the best F1-Score of <b>{best_row["F1-Score"]:.2%}</b> and AUC of <b>{best_row["ROC-AUC"]:.2%}</b>. An AUC above 0.80 means the model is good at identifying churners.</div>
    <div class="plain-insight">📌 <b>What to do:</b> Focus retention efforts on customers flagged as High Risk. Use the <b>Customer Lookup</b> tab to explore individual predictions.</div>
    ''', unsafe_allow_html=True)

    # Class distribution chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Test Set Class Distribution**")
        vc = y_test.value_counts()
        fig, ax = plt.subplots(figsize=(4,3))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        ax.bar([str(x) for x in vc.index], vc.values, color=['#2ecc71','#e74c3c'], alpha=0.85, edgecolor='none')
        ax.set_title('Actual vs Churn', color='white', fontsize=11)
        for spine in ax.spines.values(): spine.set_color('#2a2f45')
        ax.tick_params(colors='white'); ax.yaxis.label.set_color('white')
        ax.set_facecolor('#1e2130')
        for bar, val in zip(ax.patches, vc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
                    ha='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        if st.session_state['expert_mode']:
            st.markdown("**Dataset Info**")
            df_raw = st.session_state['df_raw']
            info = pd.DataFrame({'dtype': df_raw.dtypes.astype(str),
                                  'missing%': (df_raw.isnull().mean()*100).round(1)})
            st.dataframe(info, use_container_width=True, height=250)
        else:
            st.markdown("**Churn vs Active Split**")
            labels = ['Active (0)', 'Churned (1)']
            sizes  = [int((y_test==0).sum()), int((y_test==1).sum())]
            fig, ax = plt.subplots(figsize=(4,3))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.pie(sizes, labels=labels, colors=['#2ecc71','#e74c3c'],
                   autopct='%1.1f%%', textprops={'color':'white','fontsize':11},
                   wedgeprops={'edgecolor':'#1e2130','linewidth':2})
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Results":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    st.markdown('<div class="section-title">🏆 Model Comparison</div>', unsafe_allow_html=True)
    results_df      = st.session_state['results_df']
    best_model_name = st.session_state['best_model_name']
    best_model      = st.session_state['best_model']
    X_test          = st.session_state['X_test_final']
    y_test          = st.session_state['y_test_final']
    models          = st.session_state['models']

    # Styled table
    def highlight_best(row):
        return ['background-color: #1a3a2a; color: #2ecc71; font-weight:bold']*len(row) \
               if row['Model'] == best_model_name else ['']*len(row)

    st.dataframe(results_df.style.apply(highlight_best, axis=1).format({
        'Accuracy':':.4f','Precision':':.4f','Recall':':.4f','F1-Score':':.4f','ROC-AUC':':.4f'}),
        use_container_width=True, hide_index=True)

    # Bar charts
    st.markdown('<div class="section-title">📊 Metric Comparison Charts</div>', unsafe_allow_html=True)
    metrics = ['Accuracy','Precision','Recall','F1-Score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor('#0f1117')
    pal = ['#4f8ef7','#e74c3c','#2ecc71','#f39c12']
    for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
        ax.set_facecolor('#1e2130')
        data = results_df.sort_values(metric)
        bar_colors = [('#f39c12' if m==best_model_name else pal[idx]) for m in data['Model']]
        bars = ax.barh(data['Model'], data[metric], color=bar_colors, alpha=0.9)
        ax.set_xlabel(metric, color='white', fontsize=10)
        ax.set_title(f'{metric}', color='white', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#2a2f45')
        for bar, val in zip(bars, data[metric]):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                    va='center', color='white', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ROC curves
    st.markdown('<div class="section-title">📈 ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
    roc_colors = ['#4f8ef7','#e74c3c','#2ecc71','#f39c12','#9b59b6']
    for (name, model), color in zip(models.items(), roc_colors):
        ypp = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, ypp)
        roc_auc_val = auc(fpr, tpr)
        lw = 3 if name == best_model_name else 1.5
        ax.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC={roc_auc_val:.3f})')
    ax.plot([0,1],[0,1],'--', color='#5d6d7e', lw=1)
    ax.set_xlabel('False Positive Rate', color='white'); ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curves — All Models', color='white', fontsize=13, fontweight='bold')
    ax.legend(facecolor='#1e2130', edgecolor='#2a2f45', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#2a2f45')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Confusion matrix
    st.markdown('<div class="section-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(c) for c in sorted(y_test.unique())]
    fig, ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False,
                annot_kws={'size':14, 'weight':'bold'})
    ax.set_title(f'Confusion Matrix — {best_model_name}', color='white', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual', color='white'); ax.set_xlabel('Predicted', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    col1, col2 = st.columns([1,1])
    with col1:
        st.pyplot(fig, use_container_width=True); plt.close()
    with col2:
        tn,fp,fn,tp = cm.ravel()
        st.markdown(f"""
        <div class="plain-insight">
        ✅ <b>True Positives:</b> {tp} churners correctly identified<br>
        ✅ <b>True Negatives:</b> {tn} active customers correctly identified<br>
        ⚠️ <b>False Positives:</b> {fp} active customers wrongly flagged as churners<br>
        ❌ <b>False Negatives:</b> {fn} actual churners missed by the model
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — XAI EXPLANATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 XAI Explanations":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    shap_values    = st.session_state['shap_values']
    X_test         = st.session_state['X_test_final']
    features       = st.session_state['selected_feature_names']
    best_model_name = st.session_state['best_model_name']

    st.markdown('<div class="section-title">🔍 Explainable AI — SHAP Analysis</div>', unsafe_allow_html=True)

    if shap_values is None:
        st.warning("SHAP values could not be computed for this model type.")
        st.stop()

    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)

    # Simple mode
    if not st.session_state['expert_mode']:
        st.markdown('<div class="section-title">🌟 Top Churn Drivers (Simple View)</div>', unsafe_allow_html=True)
        for i, (feat, val) in enumerate(mean_shap.head(5).items(), 1):
            impact = "🔴 High" if val > mean_shap.mean()*1.5 else "🟡 Medium" if val > mean_shap.mean()*0.5 else "🟢 Low"
            st.markdown(f'<div class="plain-insight"><b>{i}. {feat}</b> &nbsp;·&nbsp; {impact} influence on churn (score: {val:.4f})</div>', unsafe_allow_html=True)
        st.markdown("")

    # Feature importance bar
    st.markdown("**Global Feature Importance (Mean |SHAP|)**")
    fig, ax = plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
    top = mean_shap.head(12)
    bar_colors = ['#e74c3c' if i < 3 else '#4f8ef7' if i < 6 else '#5d6d7e' for i in range(len(top))]
    ax.barh(top.index[::-1], top.values[::-1], color=bar_colors[::-1], alpha=0.9)
    ax.set_xlabel('Mean |SHAP| Value', color='white', fontsize=11)
    ax.set_title(f'Top Feature Importances — {best_model_name}', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#2a2f45')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    if st.session_state['expert_mode']:
        # SHAP beeswarm
        st.markdown("**SHAP Beeswarm (Expert View)**")
        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor('#0f1117')
        shap.summary_plot(shap_values, X_test, show=False, plot_size=(9,5))
        fig = plt.gcf(); fig.patch.set_facecolor('#0f1117')
        st.pyplot(fig, use_container_width=True); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════════════════════════
elif page == "👤 Customer Lookup":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    best_model = st.session_state['best_model']
    X_test     = st.session_state['X_test_final']
    y_test     = st.session_state['y_test_final']
    shap_values= st.session_state['shap_values']
    features   = st.session_state['selected_feature_names']

    st.markdown('<div class="section-title">👤 Individual Customer Risk Explorer</div>', unsafe_allow_html=True)

    idx_list = X_test.index.tolist()
    selected_idx = st.selectbox("Select Customer ID", idx_list)
    row_pos = idx_list.index(selected_idx)

    customer = X_test.loc[[selected_idx]]
    prob = best_model.predict_proba(customer)[0][1]
    actual = y_test.loc[selected_idx]

    risk_label = "🔴 HIGH RISK"   if prob > 0.6 else \
                 "🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK"
    risk_class = "risk-high" if prob > 0.6 else "risk-medium" if prob > 0.35 else "risk-low"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
    with col2:
        st.metric("Risk Level", risk_label)
    with col3:
        st.metric("Actual Label", "Churned ❌" if actual==1 else "Active ✅")

    # Gauge
    col_g, col_t = st.columns([1,2])
    with col_g:
        fig = make_gauge(prob)
        st.pyplot(fig, use_container_width=False); plt.close()

    with col_t:
        st.markdown("**Customer Feature Values**")
        cust_df = customer.T.reset_index()
        cust_df.columns = ['Feature','Value']
        cust_df['Value'] = cust_df['Value'].round(4)
        st.dataframe(cust_df, use_container_width=True, height=250, hide_index=True)

    # SHAP waterfall for this customer
    if shap_values is not None:
        st.markdown("---")
        st.markdown("**Why is this customer at risk? (Top SHAP drivers)**")
        cust_shap = shap_values[row_pos]
        shap_series = pd.Series(cust_shap, index=features).sort_values(key=abs, ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_series.values]
        ax.barh(shap_series.index[::-1], shap_series.values[::-1], color=bar_colors[::-1], alpha=0.9)
        ax.axvline(0, color='white', linewidth=0.8)
        ax.set_xlabel('SHAP Value (positive = increases churn risk)', color='white', fontsize=10)
        ax.set_title(f'SHAP Explanation — Customer {selected_idx}', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#2a2f45')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        if not st.session_state['expert_mode']:
            st.markdown("**Plain English Explanation:**")
            for feat, val in shap_series.items():
                direction = "🔴 **increases**" if val > 0 else "🟢 **decreases**"
                st.markdown(f'<div class="plain-insight">• <b>{feat}</b> {direction} churn risk (impact: {val:+.4f})</div>', unsafe_allow_html=True)

    # High-risk customer list
    st.markdown("---")
    st.markdown('<div class="section-title">📋 All Customers Risk Overview</div>', unsafe_allow_html=True)
    all_probs = best_model.predict_proba(X_test)[:,1]
    risk_df = pd.DataFrame({'Customer ID': X_test.index, 'Churn Probability': all_probs.round(4),
                             'Actual': y_test.values})
    risk_df['Risk Level'] = risk_df['Churn Probability'].apply(
        lambda p: '🔴 High' if p>0.6 else ('🟡 Medium' if p>0.35 else '🟢 Low'))
    risk_df = risk_df.sort_values('Churn Probability', ascending=False)
    st.dataframe(risk_df, use_container_width=True, height=350, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DOWNLOAD REPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📥 Download Report":
    if not st.session_state['pipeline_done']:
        st.warning("⚠️ Please upload data and run the pipeline first.")
        st.stop()

    st.markdown('<div class="section-title">📥 Download PDF Report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="plain-insight">
    The PDF report includes:<br>
    ✅ Executive summary with KPIs<br>
    ✅ Plain English interpretation (non-technical friendly)<br>
    ✅ Model comparison table<br>
    ✅ Confusion matrix & classification report<br>
    ✅ Top SHAP feature importance chart<br>
    ✅ Churn driver explanations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_report(
                    results_df=st.session_state['results_df'],
                    best_model_name=st.session_state['best_model_name'],
                    best_model=st.session_state['best_model'],
                    X_test_final=st.session_state['X_test_final'],
                    y_test_final=st.session_state['y_test_final'],
                    shap_values=st.session_state['shap_values'],
                    selected_feature_names=st.session_state['selected_feature_names'],
                    target_column=st.session_state['target_column'],
                    df_raw=st.session_state['df_raw'],
                )
                st.success("✅ PDF ready!")
                fname = f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")
                import traceback; st.code(traceback.format_exc())
