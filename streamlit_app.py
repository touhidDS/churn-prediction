import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import os
import io
import base64
import tempfile
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction XAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }

    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #64ffda; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #8892b0; font-size: 0.85rem; margin: 0; }

    .step-badge {
        background: #e94560;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: bold; font-size: 0.85rem;
        margin-right: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
    }
    .stButton > button:hover { opacity: 0.85; }
    .success-box {
        background: #0d2137;
        border-left: 4px solid #64ffda;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    div[data-testid="stExpander"] { border: 1px solid #0f3460; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#   HELPER FUNCTIONS  (ported from notebook)
# ════════════════════════════════════════════════════════════════════

def clean_sql_for_sqlite(sql_content):
    sql_content = sql_content.replace('`', '"')
    for pat in [r'AUTO_INCREMENT=\d+', r'AUTO_INCREMENT', r'ENGINE\s*=\s*\w+',
                r'DEFAULT CHARSET\s*=\s*\w+', r'COLLATE\s*=\s*\w+', r'CHARACTER SET \w+',
                r'\bUNSIGNED\b', r'\bZEROFILL\b']:
        sql_content = re.sub(pat, '', sql_content, flags=re.IGNORECASE)
    for old, new in [('\bTINYINT\b','INTEGER'),('\bSMALLINT\b','INTEGER'),
                     ('\bMEDIUMINT\b','INTEGER'),('\bBIGINT\b','INTEGER'),
                     ('\bDATETIME\b','TEXT'),('\bTIMESTAMP\b','TEXT')]:
        sql_content = re.sub(old, new, sql_content, flags=re.IGNORECASE)
    for pat in [r'SET NAMES \w+;', r'SET @\w+\s*=.*?;', r'SET @@\w+\s*=.*?;',
                r'SET SQL_MODE\s*=.*?;', r'SET FOREIGN_KEY_CHECKS\s*=.*?;',
                r'SET UNIQUE_CHECKS\s*=.*?;']:
        sql_content = re.sub(pat, '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^--.*$', '', sql_content, flags=re.MULTILINE)
    sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)
    return sql_content


def load_sql_bytes(content_bytes):
    sql_content = content_bytes.decode('utf-8', errors='ignore')
    sql_content = clean_sql_for_sqlite(sql_content)
    conn = sqlite3.connect(':memory:')
    try:
        conn.cursor().executescript(sql_content)
        conn.commit()
    except Exception as e:
        conn.close()
        return None, [], str(e)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cursor.fetchall()]
    return conn, tables, None


def get_numeric_patterns():
    return {
        'revenue','income','salary','wage','profit','loss','balance','amount','price',
        'cost','fee','charge','payment','debt','debit','credit','interest','rate','apr',
        'loan','mortgage','id','code','number','no','num','account','customer_id',
        'user_id','transaction','invoice','order','ticket','reference','ref','zip',
        'zipcode','postal','pincode','percent','percentage','ownership','share','ratio',
        'count','quantity','qty','total','sum','duration','minutes','seconds','calls',
        'sms','bandwidth','speed','mbps','gb','mb','sales','orders','transactions',
        'age','weight','height','distance','score','rating','rank','year','month',
        'day','quarter','week','hour','minute','value','metric','measure'
    }

def is_numeric_col(col_name):
    col_lower = re.sub(r'[^a-z0-9]', '', col_name.lower().strip())
    return any(p in col_lower for p in get_numeric_patterns())

def aggressive_clean(series):
    cleaned = (series.astype(str)
        .str.replace(r'[\$€£¥₹₽₩¢]','',regex=True)
        .str.replace(r'%','',regex=True)
        .str.replace(r',','',regex=True)
        .str.replace(r'-','',regex=True)
        .str.replace(r'\b(GB|MB|KB|TB|kg|km|cm|mm|lb|oz|USD|EUR|CAD)\b','',regex=True,flags=re.IGNORECASE)
        .str.replace(r'\s+','',regex=True)
        .str.replace(r'[(){}[\]]','',regex=True)
        .str.replace(r'[^0-9.]','',regex=True)
        .str.strip()
    )
    return cleaned.replace(['','nan','None','null','NULL'], np.nan)

def smart_numeric_convert(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ['int64','float64','int32','float32','Int64']:
            continue
        if is_numeric_col(col):
            cleaned = aggressive_clean(df[col])
            converted = pd.to_numeric(cleaned, errors='coerce')
            non_null = df[col].notna() & (df[col].astype(str).str.strip() != '')
            sr = converted[non_null].notna().sum() / non_null.sum() if non_null.sum() > 0 else 0
            if sr > 0.1:
                df[col] = converted
                nn = df[col].dropna()
                if len(nn) > 0 and (nn % 1 == 0).all():
                    df[col] = df[col].astype('Int64')
    return df

def convert_dates_to_years(df):
    df = df.copy()
    now = datetime.now()
    keywords = ['dob','birth','date','created','signup','registered','joined',
                'start','opened','last','updated','time']
    for col in list(df.columns):
        if any(k in col.lower() for k in keywords):
            try:
                ds = pd.to_datetime(df[col], errors='coerce')
                if ds.notna().sum() > 0:
                    new = f"{col.replace(' ','_').lower()}_years"
                    df[new] = ((now - ds).dt.days / 365.25).round(1)
                    df = df.drop(columns=[col])
            except:
                pass
    return df

def handle_missing(df, target_col, thresh=0.50):
    df = df.copy()
    drop_cols = [c for c in df.columns
                 if c != target_col and df[c].isnull().sum()/len(df) > thresh]
    df = df.drop(columns=drop_cols)
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    for col in df.select_dtypes(include=['number']).columns:
        if col != target_col and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col and df[col].isnull().sum() > 0:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'Unknown')
    return df

def smart_encode(df, target_col):
    df = df.copy()
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target_col]
    for col in cat_cols:
        u = df[col].nunique()
        if u == 2:
            vals = df[col].unique()
            pos = ['yes','true','active','paid','y','1','on','enabled']
            mapping = {v:(1 if str(v).lower() in pos else 0) for v in vals}
            df[col] = df[col].map(mapping)
        elif 3 <= u <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
        elif 11 <= u <= 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = df.drop(columns=[col])
    if target_col in df.columns and df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    return df


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════
#   PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════

def generate_pdf_report(results_df, best_model_name, report_data: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
        fontSize=24, textColor=colors.HexColor('#e94560'),
        spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#555555'),
        spaceAfter=4, alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=15, textColor=colors.HexColor('#0f3460'),
        spaceBefore=14, spaceAfter=6,
        borderPad=4)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=12, textColor=colors.HexColor('#16213e'),
        spaceBefore=10, spaceAfter=4)
    body_style  = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor('#333333'))
    label_style = ParagraphStyle('Label', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#666666'))

    story = []

    # ── Cover ──
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("📊 Customer Churn Prediction Report", title_style))
    story.append(Paragraph("Explainable AI Analysis", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", label_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e94560'),
                             spaceAfter=16))

    # ── Dataset Summary ──
    story.append(Paragraph("1. Dataset Summary", h1_style))
    ds = report_data.get('dataset_summary', {})
    summary_data = [
        ['Metric', 'Value'],
        ['Total Records',     str(ds.get('total_rows', 'N/A'))],
        ['Total Features',    str(ds.get('total_cols', 'N/A'))],
        ['Target Column',     str(ds.get('target_col', 'N/A'))],
        ['Churn Rate',        ds.get('churn_rate', 'N/A')],
        ['File Type',         ds.get('file_type', 'N/A')],
        ['Missing Values',    str(ds.get('missing', 'N/A'))],
    ]
    t = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f7f9fc'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # ── Model Results ──
    story.append(Paragraph("2. Model Comparison Results", h1_style))
    story.append(Paragraph(
        f"Four machine-learning classifiers were trained and evaluated. "
        f"The best-performing model was <b>{best_model_name}</b>.",
        body_style))
    story.append(Spacer(1, 0.1*inch))

    tbl_header = ['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    tbl_rows   = [tbl_header]
    for _, row in results_df.iterrows():
        tbl_rows.append([
            str(row['Model']),
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['ROC-AUC']:.4f}",
        ])
    t2 = Table(tbl_rows, colWidths=[1.6*inch,0.95*inch,0.95*inch,0.85*inch,0.95*inch,0.95*inch])
    best_idx = list(results_df['Model']).index(best_model_name) + 1
    t2_style = [
        ('BACKGROUND', (0,0),  (-1,0),         colors.HexColor('#0f3460')),
        ('TEXTCOLOR',  (0,0),  (-1,0),         colors.white),
        ('FONTNAME',   (0,0),  (-1,0),         'Helvetica-Bold'),
        ('FONTNAME',   (0,best_idx), (-1,best_idx), 'Helvetica-Bold'),
        ('BACKGROUND', (0,best_idx), (-1,best_idx), colors.HexColor('#e8f5e9')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f7f9fc'), colors.white]),
        ('GRID',   (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ALIGN',  (1,0), (-1,-1), 'CENTER'),
        ('FONTSIZE',(0,0),(-1,-1), 9),
        ('PADDING',(0,0),(-1,-1), 6),
    ]
    t2.setStyle(TableStyle(t2_style))
    story.append(t2)
    story.append(Spacer(1, 0.15*inch))

    # ── Figures ──
    def add_figure(fig_bytes, caption, width=5.5*inch):
        if fig_bytes:
            img_buf = io.BytesIO(fig_bytes)
            img = RLImage(img_buf, width=width, height=width*0.65)
            story.append(img)
            story.append(Paragraph(f"<i>{caption}</i>", label_style))
            story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("3. Confusion Matrices", h1_style))
    add_figure(report_data.get('cm_fig'), "Confusion matrices for all four classifiers.")

    story.append(Paragraph("4. ROC Curves", h1_style))
    add_figure(report_data.get('roc_fig'), "ROC curves showing discriminative power of each model.")

    story.append(Paragraph("5. Metric Comparison", h1_style))
    add_figure(report_data.get('metric_fig'), "Side-by-side comparison of Accuracy, Precision, Recall, and F1-Score.")

    story.append(PageBreak())

    story.append(Paragraph("6. Explainable AI — SHAP Analysis", h1_style))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution "
        "to the model's predictions globally and per customer.",
        body_style))
    story.append(Spacer(1, 0.1*inch))
    add_figure(report_data.get('shap_bar_fig'), "Global SHAP feature importance (mean absolute SHAP values).")
    add_figure(report_data.get('shap_dot_fig'), "SHAP beeswarm plot showing direction and magnitude of each feature's impact.")
    add_figure(report_data.get('shap_waterfall_fig'), "SHAP waterfall plot for a sample churned customer — showing exactly why the model predicted churn.")

    story.append(Paragraph("7. Feature Importance", h1_style))
    add_figure(report_data.get('feat_imp_fig'), f"Feature importance scores from {best_model_name}.")

    # ── Top Factors Table ──
    top_factors = report_data.get('top_factors')
    if top_factors is not None and len(top_factors) > 0:
        story.append(Paragraph("8. Top Churn Drivers", h1_style))
        story.append(Paragraph(
            "The following features have the highest average impact on churn predictions:",
            body_style))
        story.append(Spacer(1, 0.1*inch))
        fac_data = [['Rank','Feature','SHAP Impact Score']]
        for i, row in top_factors.head(10).iterrows():
            fac_data.append([str(i+1), str(row['Feature']), f"{row['Impact']:.5f}"])
        ft = Table(fac_data, colWidths=[0.7*inch, 3.5*inch, 2*inch])
        ft.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [colors.HexColor('#f7f9fc'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('ALIGN', (2,0), (2,-1), 'CENTER'),
            ('FONTSIZE',(0,0),(-1,-1), 9),
            ('PADDING',(0,0),(-1,-1), 6),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.2*inch))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#cccccc'), spaceAfter=8))
    story.append(Paragraph(
        "Generated by Churn Prediction XAI App  •  Powered by SHAP, XGBoost, LightGBM & ReportLab",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.HexColor('#999999'),
                       alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════
#   MAIN APP
# ════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Customer Churn Prediction XAI</h1>
        <p>Upload your dataset · Auto-train 4 ML models · Get an Explainable AI PDF report</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        test_size    = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
        corr_thresh  = st.slider("Correlation Threshold", 0.7, 0.99, 0.90, 0.01)
        drop_thresh  = st.slider("Missing-Value Drop Threshold", 0.3, 0.9, 0.50, 0.05)
        st.markdown("---")
        st.markdown("**Supported file types:** CSV · SQL")
        st.markdown("**Models trained:** Random Forest · Gradient Boosting · XGBoost · LightGBM")

    # ── STEP 1: Upload ──────────────────────────────────────────────
    st.markdown("### <span class='step-badge'>1</span> Upload Your Dataset", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or SQL file", type=['csv','sql'])

    if uploaded_file is None:
        st.info("👆 Upload a `.csv` or `.sql` file to get started.")
        return

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    content  = uploaded_file.read()
    df_raw   = None

    if file_ext == '.csv':
        df_raw = pd.read_csv(io.BytesIO(content))
        st.success(f"✅ CSV loaded — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    elif file_ext == '.sql':
        conn, tables, err = load_sql_bytes(content)
        if err:
            st.error(f"SQL parse error: {err}")
            return
        if len(tables) == 0:
            st.error("No tables found in SQL file.")
            return
        if len(tables) == 1:
            df_raw = pd.read_sql_query(f'SELECT * FROM "{tables[0]}"', conn)
            st.success(f"✅ SQL table `{tables[0]}` loaded — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        else:
            selected_table = st.selectbox("Select table", tables)
            if st.button("Load Table"):
                df_raw = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)
                st.success(f"✅ Loaded `{selected_table}` — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    else:
        st.error("Unsupported file type.")
        return

    if df_raw is None:
        return

    with st.expander("🔍 Preview raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # ── STEP 2: Target Column ──────────────────────────────────────
    st.markdown("### <span class='step-badge'>2</span> Select Churn Column", unsafe_allow_html=True)
    target_col = st.selectbox("Which column is the churn label?", df_raw.columns)
    st.info(f"Distribution: {df_raw[target_col].value_counts().to_dict()}")

    if st.button("🚀 Run Full Pipeline & Generate PDF Report"):
        report_data = {
            'dataset_summary': {
                'total_rows': df_raw.shape[0],
                'total_cols': df_raw.shape[1],
                'target_col': target_col,
                'file_type':  file_ext.upper(),
                'missing':    int(df_raw.isnull().sum().sum()),
            }
        }

        progress = st.progress(0)
        status   = st.empty()

        # ── Preprocessing ──────────────────────────────────────────
        status.text("🔢 Step 3/4/5/6: Preprocessing data…")
        progress.progress(10)

        df1 = smart_numeric_convert(df_raw)
        df2 = convert_dates_to_years(df1)
        df3 = handle_missing(df2, target_col, drop_thresh)
        df4 = smart_encode(df3, target_col)

        vc = df4[target_col].value_counts(normalize=True)
        report_data['dataset_summary']['churn_rate'] = (
            f"{vc.iloc[1]*100:.1f}%" if len(vc) > 1 else "N/A")

        # ── Train/Test Split ───────────────────────────────────────
        status.text("✂️ Step 7: Splitting data…")
        progress.progress(20)

        X = df4.drop(columns=[target_col])
        y = df4[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        # ── Scaling & Feature Engineering ─────────────────────────
        status.text("⚙️ Step 8: Scaling & feature engineering…")
        progress.progress(30)

        # interaction feature
        num_cols = X_train.columns.tolist()
        if len(num_cols) >= 2:
            X_train[f"{num_cols[0]}_x_{num_cols[1]}"] = X_train[num_cols[0]] * X_train[num_cols[1]]
            X_test[f"{num_cols[0]}_x_{num_cols[1]}"]  = X_test[num_cols[0]]  * X_test[num_cols[1]]

        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

        # Correlation removal
        corr = X_train_sc.corr().abs()
        to_remove = set()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i,j] >= corr_thresh:
                    c1, c2 = corr.columns[i], corr.columns[j]
                    to_remove.add(c1 if abs(X_train_sc[c1].corr(y_train)) < abs(X_train_sc[c2].corr(y_train)) else c2)
        if to_remove:
            X_train_sc = X_train_sc.drop(columns=list(to_remove))
            X_test_sc  = X_test_sc.drop(columns=list(to_remove))

        # Feature selection
        rf_sel = RandomForestClassifier(n_estimators=100, random_state=42,
                                        class_weight='balanced', max_depth=10, n_jobs=-1)
        rf_sel.fit(X_train_sc, y_train)
        fi = pd.DataFrame({'Feature': X_train_sc.columns,
                           'Importance': rf_sel.feature_importances_}).sort_values('Importance', ascending=False)
        n_keep = max(8, int(X_train_sc.shape[1]*0.7))
        sel_feats = fi.head(n_keep)['Feature'].tolist()
        X_train_sel = X_train_sc[sel_feats]
        X_test_sel  = X_test_sc[sel_feats]

        # SMOTE
        status.text("⚖️ Step 10: Balancing classes with SMOTE…")
        progress.progress(40)
        try:
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
            X_tr, y_tr = smote.fit_resample(X_train_sel, y_train)
            X_tr = pd.DataFrame(X_tr, columns=sel_feats)
        except Exception:
            X_tr, y_tr = X_train_sel, y_train

        X_te, y_te = X_test_sel, y_test

        # ── Train Models ───────────────────────────────────────────
        status.text("🚀 Step 11: Training 4 models…")
        progress.progress(55)

        model_defs = {
            'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost':           XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'LightGBM':          LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        }
        results, preds_dict = [], {}
        for name, mdl in model_defs.items():
            mdl.fit(X_tr, y_tr)
            yp   = mdl.predict(X_te)
            yppr = mdl.predict_proba(X_te)[:,1]
            preds_dict[name] = {'y_pred': yp, 'y_pred_proba': yppr}
            results.append({
                'Model':     name,
                'Accuracy':  accuracy_score(y_te, yp),
                'Precision': precision_score(y_te, yp, zero_division=0),
                'Recall':    recall_score(y_te, yp, zero_division=0),
                'F1-Score':  f1_score(y_te, yp, zero_division=0),
                'ROC-AUC':   roc_auc_score(y_te, yppr),
            })

        results_df = pd.DataFrame(results).round(4).sort_values('F1-Score', ascending=False)
        best_name  = results_df.iloc[0]['Model']
        best_model = model_defs[best_name]

        # ── Show Results ───────────────────────────────────────────
        st.markdown("### 📊 Model Results")
        cols = st.columns(4)
        metrics_show = ['Accuracy','Precision','Recall','F1-Score']
        for c, m in zip(cols, metrics_show):
            val = results_df[results_df['Model']==best_name][m].values[0]
            c.markdown(f"""
            <div class="metric-card">
                <h3>{val:.3f}</h3>
                <p>{m}<br><small>({best_name})</small></p>
            </div>""", unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # ── Visualizations ─────────────────────────────────────────
        status.text("📈 Step 12: Generating visualizations…")
        progress.progress(70)

        plt.style.use('default')
        sns.set_style('whitegrid')

        # Confusion matrices
        fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, (name, prd) in enumerate(preds_dict.items()):
            cm = confusion_matrix(y_te, prd['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Active','Churned'],
                        yticklabels=['Active','Churned'],
                        ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
            axes[idx].set_ylabel('Actual'); axes[idx].set_xlabel('Predicted')
        plt.tight_layout()
        report_data['cm_fig'] = fig_to_bytes(fig_cm)
        st.pyplot(fig_cm); plt.close(fig_cm)

        # ROC curves
        fig_roc, ax = plt.subplots(figsize=(9, 6))
        for name, prd in preds_dict.items():
            fpr, tpr, _ = roc_curve(y_te, prd['y_pred_proba'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.4f})", lw=2)
        ax.plot([0,1],[0,1],'k--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Models', fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        plt.tight_layout()
        report_data['roc_fig'] = fig_to_bytes(fig_roc)
        st.pyplot(fig_roc); plt.close(fig_roc)

        # Metric bar charts
        fig_met, axes2 = plt.subplots(2, 2, figsize=(13, 9))
        for idx, metric in enumerate(['Accuracy','Precision','Recall','F1-Score']):
            ax2 = axes2[idx//2, idx%2]
            data = results_df.sort_values(metric, ascending=True)
            colors_list = ['#3498db','#e74c3c','#2ecc71','#f39c12']
            ax2.barh(data['Model'], data[metric], color=colors_list[idx], alpha=0.8)
            ax2.set_xlabel(metric, fontweight='bold')
            ax2.set_title(f'{metric} Comparison', fontweight='bold')
            ax2.set_xlim([0,1])
            for i, v in enumerate(data[metric]):
                ax2.text(v+0.01, i, f'{v:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        report_data['metric_fig'] = fig_to_bytes(fig_met)
        st.pyplot(fig_met); plt.close(fig_met)

        # Feature Importance
        feat_imp_df = pd.DataFrame({
            'Feature':    sel_feats,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        sns.barplot(data=feat_imp_df, y='Feature', x='Importance',
                    hue='Feature', palette='viridis', legend=False, ax=ax_fi)
        ax_fi.set_title(f'Feature Importance — {best_name}', fontweight='bold')
        plt.tight_layout()
        report_data['feat_imp_fig'] = fig_to_bytes(fig_fi)
        st.pyplot(fig_fi); plt.close(fig_fi)

        # ── SHAP ──────────────────────────────────────────────────
        status.text("🤖 Step 13: Computing SHAP values…")
        progress.progress(85)

        try:
            explainer   = shap.TreeExplainer(best_model)
            shap_vals   = explainer.shap_values(X_te)
            if isinstance(shap_vals, list): shap_vals = shap_vals[1]

            # SHAP bar
            fig_sb, ax_sb = plt.subplots(figsize=(9, 5))
            shap.summary_plot(shap_vals, X_te, plot_type="bar", show=False)
            ax_sb = plt.gca()
            ax_sb.set_title('SHAP Feature Importance (Global)', fontweight='bold')
            plt.tight_layout()
            report_data['shap_bar_fig'] = fig_to_bytes(plt.gcf())
            st.pyplot(plt.gcf()); plt.close()

            # SHAP dot
            fig_sd = plt.figure(figsize=(9, 5))
            shap.summary_plot(shap_vals, X_te, show=False)
            plt.gca().set_title('SHAP Summary — Impact on Churn', fontweight='bold')
            plt.tight_layout()
            report_data['shap_dot_fig'] = fig_to_bytes(plt.gcf())
            st.pyplot(plt.gcf()); plt.close()

            # SHAP waterfall for one churned customer
            churned_idx_list = y_te[y_te == 1].index
            if len(churned_idx_list) > 0:
                cust_idx = churned_idx_list[0]
                shap_exp = explainer(X_te.loc[[cust_idx]])
                if shap_exp.values.ndim == 3:
                    shap_exp = shap_exp[:,:,1]
                fig_wf = plt.figure(figsize=(9, 5))
                shap.plots.waterfall(shap_exp[0], show=False)
                plt.title(f'SHAP Waterfall — Sample Churned Customer', fontweight='bold')
                plt.tight_layout()
                report_data['shap_waterfall_fig'] = fig_to_bytes(plt.gcf())
                st.pyplot(plt.gcf()); plt.close()

            # Top factors
            shap_exp_all = explainer(X_te)
            sv2 = shap_exp_all.values[:,:,1] if shap_exp_all.values.ndim == 3 else shap_exp_all.values
            top_factors = pd.DataFrame({
                'Feature': sel_feats,
                'Impact':  abs(sv2).mean(axis=0)
            }).sort_values('Impact', ascending=False).reset_index(drop=True)
            report_data['top_factors'] = top_factors

            st.markdown("#### 🏆 Top Churn Drivers (SHAP)")
            st.dataframe(top_factors.head(10), use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"SHAP analysis partial error: {e}")

        # ── Generate PDF ───────────────────────────────────────────
        status.text("📄 Generating PDF report…")
        progress.progress(95)

        pdf_bytes = generate_pdf_report(results_df, best_name, report_data)
        progress.progress(100)
        status.text("✅ Done!")

        st.success("🎉 Analysis complete! Download your PDF report below.")

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

        # Classification report
        with st.expander(f"📋 Full Classification Report — {best_name}"):
            bp = preds_dict[best_name]['y_pred']
            st.text(classification_report(y_te, bp, target_names=['Active','Churned'], digits=4))


if __name__ == "__main__":
    main()
