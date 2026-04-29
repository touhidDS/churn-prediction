"""
Feature engineering, scaling, SMOTE, model training.
Adapted from Churn_Prediction_XAI_with_SQL_Support.ipynb
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def engineer_features(df_train, df_test, y_train):
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df_fe_train = df_train.copy()
    df_fe_test  = df_test.copy()
    created_features = []
    numeric_cols = df_fe_train.columns.tolist()

    revenue_cols = [c for c in numeric_cols if any(w in c.lower() for w in ['revenue', 'income', 'sales', 'earning'])]
    debt_cols    = [c for c in numeric_cols if any(w in c.lower() for w in ['debt', 'debit', 'loan', 'liability', 'payment'])]

    if revenue_cols and debt_cols:
        df_fe_train['Debt_Revenue_Ratio'] = df_fe_train[debt_cols[0]] / (df_fe_train[revenue_cols[0]] + 0.0001)
        df_fe_test['Debt_Revenue_Ratio']  = df_fe_test[debt_cols[0]]  / (df_fe_test[revenue_cols[0]]  + 0.0001)
        created_features.append('Debt_Revenue_Ratio')
        print(f"✅ Created: Debt_Revenue_Ratio")

    pct_cols = [c for c in numeric_cols if '%' in c or 'percent' in c.lower() or 'ownership' in c.lower()]
    for col in pct_cols:
        new_col = f"{col}_High"
        threshold = df_fe_train[col].median()
        df_fe_train[new_col] = (df_fe_train[col] > threshold).astype(int)
        df_fe_test[new_col]  = (df_fe_test[col]  > threshold).astype(int)
        created_features.append(new_col)
        print(f"✅ Created: {new_col}")

    if len(numeric_cols) >= 2:
        new_col = f"{numeric_cols[0]}_x_{numeric_cols[1]}"
        df_fe_train[new_col] = df_fe_train[numeric_cols[0]] * df_fe_train[numeric_cols[1]]
        df_fe_test[new_col]  = df_fe_test[numeric_cols[0]]  * df_fe_test[numeric_cols[1]]
        created_features.append(new_col)
        print(f"✅ Created: {new_col} (interaction)")

    print(f"\n📊 New features: {len(created_features)} | Train: {df_fe_train.shape} | Test: {df_fe_test.shape}")
    return df_fe_train, df_fe_test


def prune_correlated_features(X_train, X_test, y_train, threshold=0.90):
    corr_matrix = X_train.corr()
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr1 = abs(X_train[col1].corr(y_train))
                corr2 = abs(X_train[col2].corr(y_train))
                to_remove.add(col1 if corr1 < corr2 else col2)

    if to_remove:
        X_train = X_train.drop(columns=list(to_remove))
        X_test  = X_test.drop(columns=list(to_remove))
        print(f"🗑️  Removed {len(to_remove)} correlated features: {sorted(to_remove)}")
    else:
        print("✅ No highly correlated features found")
    return X_train, X_test


def select_features(X_train, y_train, X_test):
    print("🔍 Feature Selection...")
    rf_selector = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight='balanced', max_depth=10, n_jobs=-1
    )
    rf_selector.fit(X_train, y_train)

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_selector.feature_importances_
    }).sort_values('Importance', ascending=False)

    n_keep = max(8, int(X_train.shape[1] * 0.7))
    selected_features = feature_importance.head(n_keep)['Feature'].tolist()
    print(f"✅ Selected {n_keep} features from {X_train.shape[1]}")

    return selected_features, rf_selector


def run_training(X_train, X_test, y_train, y_test, selected_features):
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    print("\n⚖️ Applying SMOTE...")
    before_minority = (y_train == 1).sum()
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sel, y_train)
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=selected_features)
    after_minority = (y_train_resampled == 1).sum()
    print(f"✅ Created {after_minority - before_minority} synthetic samples")
    print(f"   Train: {X_train_resampled.shape} | Test: {X_test_sel.shape}")

    print("\n🚀 Training Multiple Models...\n")

    models = {
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        print(f"⏳ Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred       = model.predict(X_test_sel)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        results.append({
            'Model':     name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall':    recall_score(y_test, y_pred),
            'F1-Score':  f1_score(y_test, y_pred),
            'ROC-AUC':   roc_auc_score(y_test, y_pred_proba)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).round(4).sort_values('F1-Score', ascending=False)

    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    print(f"\n🏆 Best Model: {best_row['Model']} | F1: {best_row['F1-Score']} | AUC: {best_row['ROC-AUC']}")

    return results_df, trained_models, selected_features