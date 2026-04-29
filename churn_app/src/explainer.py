"""
SHAP-based explainability.
"""
import shap
import matplotlib.pyplot as plt


def run_shap_analysis(model, X_test, selected_features):
    print("🤖 Explainable AI — SHAP Analysis...\n")

    feature_imp = dict(zip(
        selected_features,
        model.feature_importances_
    ))
    feature_imp_df = pd.DataFrame({
        'Feature':    selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n🔍 SHAP Global Analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[selected_features])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[selected_features], plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Global)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_importance.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ Saved shap_importance.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[selected_features], show=False)
    plt.title('SHAP Summary — Impact on Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ Saved shap_summary.png")

    shap_exp = explainer(X_test[selected_features])
    if shap_exp.values.ndim == 3:
        shap_values_churn = shap_exp.values[:, :, 1]
    else:
        shap_values_churn = shap_exp.values

    mean_abs_shap = pd.DataFrame({
        'Feature': selected_features,
        'Impact':  abs(shap_values_churn).mean(axis=0)
    }).sort_values('Impact', ascending=False)

    print("\n📈 Top Factors Driving Churn:")
    print(mean_abs_shap.to_string(index=False))

    return feature_imp_df, mean_abs_shap, explainer, shap_values