"""
Text-only PDF report generation.
"""
import pandas as pd
from fpdf import FPDF


class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Customer Churn Prediction Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def generate_pdf(results_df, best_model_name, best_model, feature_imp_df, mean_abs_shap, file_type):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Summary
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '1. Summary', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    pdf.multi_cell(0, 6, (
        f"File type processed: {file_type}\n"
        f"Best model: {best_model_name}\n"
        f"F1-Score: {best_row['F1-Score']}\n"
        f"ROC-AUC: {best_row['ROC-AUC']}\n"
        f"Accuracy: {best_row['Accuracy']}\n"
        f"Precision: {best_row['Precision']}\n"
        f"Recall: {best_row['Recall']}\n"
    ))
    pdf.ln(4)

    # Model Comparison
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '2. Model Comparison', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', 'B', 9)
    col_widths = [40, 28, 28, 28, 28, 28]
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    for _, row in results_df.iterrows():
        pdf.cell(col_widths[0], 6, str(row['Model']), border=1)
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            pdf.cell(col_widths[headers.index(col)], 6, str(row[col]), border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    # Feature Importance
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '3. Feature Importance (Model-Based)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(80, 7, 'Feature', border=1)
    pdf.cell(30, 7, 'Importance', border=1, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in feature_imp_df.head(15).iterrows():
        pdf.cell(80, 6, str(row['Feature'])[:40], border=1)
        pdf.cell(30, 6, f"{row['Importance']:.4f}", border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    # Top Churn Drivers
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '4. Top Churn Drivers (SHAP)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(80, 7, 'Feature', border=1)
    pdf.cell(30, 7, 'Avg |SHAP|', border=1, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in mean_abs_shap.head(15).iterrows():
        pdf.cell(80, 6, str(row['Feature'])[:40], border=1)
        pdf.cell(30, 6, f"{row['Impact']:.4f}", border=1, align='C')
        pdf.ln()

    pdf.ln(4)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 6, 'Report generated automatically by Churn Prediction XAI App.', align='C')

    pdf.output('churn_report.pdf')
    print("✅ PDF saved: churn_report.pdf")
    return 'churn_report.pdf'