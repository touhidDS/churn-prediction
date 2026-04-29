"""
Streamlit Churn Prediction App with XAI
Single-page app: upload → configure → train → explain → download PDF
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.file_loader import load_csv, load_sql_file, sql_table_to_dataframe
from src.preprocessor import (
    smart_convert_to_numeric,
    convert_dates_to_years,
    simple_missing_handler,
    smart_categorical_encoder
)
from src.trainer import engineer_features, prune_correlated_features, select_features, run_training
from src.explainer import run_shap_analysis
from src.pdf_report import generate_pdf
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Churn Prediction XAI", page_icon="🔮", layout="wide")
st.title("🔮 Customer Churn Prediction with Explainable AI")

# ─────────────────────────────────────────────
# STEP 1: File Upload
# ─────────────────────────────────────────────
st.markdown("## 📂 Step 1 — Upload File")
uploaded_file = st.file_uploader(
    "Upload a `.csv` or `.sql` file",
    type=['csv', 'sql']
)

df = None
sql_conn = None
sql_tables = []
file_ext = None

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    st.session_state['file_ext'] = file_ext

    if file_ext == '.csv':
        # Save temp file for pandas
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        df = load_csv(tmp_path)
        st.session_state['df'] = df
        st.success(f"✅ CSV loaded — {df.shape[0]} rows × {df.shape[1]} columns")

    elif file_ext == '.sql':
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        sql_conn, sql_tables = load_sql_file(tmp_path)
        if sql_conn is None:
            st.error("❌ Could not load SQL file.")
        else:
            st.success(f"✅ SQL loaded — {len(sql_tables)} table(s) found: {sql_tables}")
            st.session_state['sql_conn'] = sql_conn
            st.session_state['sql_tables'] = sql_tables

# ─────────────────────────────────────────────
# STEP 2: SQL Table Selection (if multi-table)
# ─────────────────────────────────────────────
if uploaded_file is not None and file_ext == '.sql' and sql_conn is not None:
    st.markdown("## 📋 Step 2 — Select Table")
    st.markdown("*" if len(sql_tables) <= 1 else "Your SQL file has multiple tables. Pick one:*")

    if len(sql_tables) == 1:
        df = sql_table_to_dataframe(sql_conn, sql_tables[0])
        st.session_state['df'] = df
        st.success(f"✅ Auto-loaded table: `{sql_tables[0]}` — {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        selected_table = st.selectbox("Available tables:", sql_tables)
        if st.button("✅ Load Selected Table"):
            df = sql_table_to_dataframe(sql_conn, selected_table)
            st.session_state['df'] = df
            st.success(f"✅ Loaded: `{selected_table}` — {df.shape[0]} rows × {df.shape[1]} columns")
            st.rerun()

# ─────────────────────────────────────────────
# STEP 3: Target Column
# ─────────────────────────────────────────────
if df is not None:
    st.markdown("## 🎯 Step 3 — Select Target Column")
    target_col = st.selectbox("Choose the churn/target column:", df.columns.tolist())
    st.session_state['target_col'] = target_col

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Unique values:**")
        st.write(df[target_col].unique())
    with col2:
        st.markdown("**Distribution:**")
        st.write(df[target_col].value_counts())

# ─────────────────────────────────────────────
# STEP 4: Run Pipeline
# ─────────────────────────────────────────────
if df is not None and 'target_col' in st.session_state:
    st.markdown("## 🚀 Step 4 — Run Pipeline")

    if st.button("▶️  Start Training", type="primary"):
        target_column = st.session_state['target_col']
        file_type = st.session_state.get('file_ext', 'CSV')

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ── Preprocessing ──
        status_text.text("🔍 Converting numeric columns...")
        progress_bar.progress(10)
        df_num = smart_convert_to_numeric(df)

        status_text.text("📅 Converting dates to years...")
        progress_bar.progress(20)
        df_dates = convert_dates_to_years(df_num)

        status_text.text("🔧 Handling missing values...")
        progress_bar.progress(30)
        df_miss = simple_missing_handler(df_dates, target_column)

        status_text.text("🔤 Encoding categorical columns...")
        progress_bar.progress(40)
        df_enc = smart_categorical_encoder(df_miss, target_column)

        # ── Train/Test Split ──
        status_text.text("✂️  Splitting train/test...")
        progress_bar.progress(50)
        from sklearn.model_selection import train_test_split
        X = df_enc.drop(columns=[target_column])
        y = df_enc[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── Feature Engineering ──
        status_text.text("⚙️  Engineering features...")
        progress_bar.progress(55)
        X_train_fe, X_test_fe = engineer_features(X_train.copy(), X_test.copy(), y_train)

        # ── Scaling ──
        scaler = StandardScaler()
        scaler.fit(X_train_fe)
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train_fe),
            columns=X_train_fe.columns, index=X_train_fe.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_fe),
            columns=X_test_fe.columns, index=X_test_fe.index
        )

        # ── Correlation Pruning ──
        status_text.text("📊 Pruning correlated features...")
        progress_bar.progress(60)
        X_train_pruned, X_test_pruned = prune_correlated_features(
            X_train_scaled, X_test_scaled, y_train, threshold=0.90
        )

        # ── Feature Selection ──
        status_text.text("🎯 Selecting features...")
        progress_bar.progress(65)
        selected_features, _ = select_features(X_train_pruned, y_train, X_test_pruned)

        # ── Training ──
        status_text.text("🚀 Training models...")
        progress_bar.progress(70)
        results_df, trained_models, selected_features = run_training(
            X_train_pruned, X_test_pruned, y_train, y_test, selected_features
        )

        # ── SHAP ──
        status_text.text("🤖 Running SHAP analysis...")
        progress_bar.progress(85)
        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        feature_imp_df, mean_abs_shap, explainer, shap_values = run_shap_analysis(
            best_model, X_test_pruned, selected_features
        )

        progress_bar.progress(100)
        status_text.text("✅ Done!")

        # Store in session
        st.session_state['results_df'] = results_df
        st.session_state['trained_models'] = trained_models
        st.session_state['best_model_name'] = best_model_name
        st.session_state['best_model'] = best_model
        st.session_state['feature_imp_df'] = feature_imp_df
        st.session_state['mean_abs_shap'] = mean_abs_shap
        st.session_state['file_type'] = file_type
        st.session_state['pipeline_done'] = True

# ─────────────────────────────────────────────
# STEP 5: Results
# ─────────────────────────────────────────────
if st.session_state.get('pipeline_done'):
    st.markdown("---")
    st.markdown("## 📊 Results")

    results_df = st.session_state['results_df']
    best_model_name = st.session_state['best_model_name']
    best_model = st.session_state['best_model']
    feature_imp_df = st.session_state['feature_imp_df']
    mean_abs_shap = st.session_state['mean_abs_shap']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🏆 Best Model: {best_model_name}")
        best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
        st.metric("F1-Score", f"{best_row['F1-Score']:.4f}")
        st.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}")
        st.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
        st.metric("Precision", f"{best_row['Precision']:.4f}")
        st.metric("Recall", f"{best_row['Recall']:.4f}")

    with col2:
        st.markdown("### 📋 All Models Comparison")
        st.dataframe(results_df.set_index('Model'))

    st.markdown("### 📈 SHAP Feature Importance (Global)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_importance_plot = plt.imread('shap_importance.png')
    ax.imshow(shap_importance_plot)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("### 🔥 SHAP Summary Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap_summary_plot_img = plt.imread('shap_summary.png')
    ax2.imshow(shap_summary_plot_img)
    ax2.axis('off')
    st.pyplot(fig2)

    st.markdown("### 📈 Top Churn Drivers (SHAP)")
    st.dataframe(mean_abs_shap.head(15).reset_index(drop=True))

    st.markdown("### 🎯 Model Feature Importance")
    st.dataframe(feature_imp_df.head(15).reset_index(drop=True))

    # ─────────────────────────────────────────────
    # STEP 6: Download PDF
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📄 Download Report")

    file_type = st.session_state.get('file_type', 'CSV')

    if st.button("📄 Generate PDF Report"):
        pdf_path = generate_pdf(
            results_df,
            best_model_name,
            best_model,
            feature_imp_df,
            mean_abs_shap,
            file_type
        )
        st.success("✅ PDF generated!")

    with open('churn_report.pdf', 'rb') as f:
        pdf_bytes = f.read()

    st.download_button(
        label="⬇️  Download Churn Report (PDF)",
        data=pdf_bytes,
        file_name="churn_report.pdf",
        mime="application/pdf"
    )