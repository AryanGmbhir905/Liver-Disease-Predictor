import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

# Load pipeline and encoder
pipeline = joblib.load("liver_pipeline.pkl")
encoder = joblib.load("Label_Encoder.pkl")

# Page config
st.set_page_config(page_title="Liver Disease Prediction", page_icon="🩺", layout="wide")

# CSS (FIXED)
st.markdown("""
<style>
.stApp .block-container {
    background-color: white;
    color: #001f3f;
    padding-top: 20px;
}
label {
    color: #001f3f !important;
    font-weight: bold;
}
.stNumberInput input {
    color: #001f3f !important;
    background-color: #ffffff !important;
}

/* Predict Button */
.stButton>button {
    color: white !important;
    font-weight: bold;
    background-color: #001f3f !important;
    border: none;
    padding: 10px 20px;
}

/* Download Button FIX */
.stDownloadButton>button {
    color: white !important;
    background-color: #001f3f !important;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='background-color:#001f3f; padding: 50px 20px;'>
<h1 style='color:white;'>🩺 Liver Disease Prediction App</h1>
<h3 style='color:white;'>Enter patient details below</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Inputs
st.header("Patient Demographics")
age = st.number_input("Age", 0, 120)

st.header("Biochemical Test Results")
col1, col2, col3 = st.columns(3)

with col1:
    alb = st.number_input("Albumin")
    alp = st.number_input("Alkaline Phosphatase")
    alt = st.number_input("Alanine Aminotransferase")

with col2:
    ast = st.number_input("Aspartate Aminotransferase")
    bil = st.number_input("Bilirubin")
    che = st.number_input("Cholinesterase")

with col3:
    chol = st.number_input("Cholesterol")
    crea = st.number_input("Creatinine")
    ggt = st.number_input("Gamma-Glutamyl Transferase")
    prot = st.number_input("Total Protein")

features = np.array([[age, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])

# Prediction
if st.button("Predict"):
    try:
        prediction = pipeline.predict(features)
        result = encoder.inverse_transform(prediction)[0]

        import shap
        import matplotlib.pyplot as plt
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from datetime import datetime

        model = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]

        explainer = shap.Explainer(model)
        shap_values = explainer(features)

        single_explanation = shap_values[0, :, prediction[0]]

        # FIX: Proper feature names
        single_explanation.feature_names = [
            "Age",
            "Albumin",
            "Alkaline Phosphatase",
            "Alanine Aminotransferase",
            "Aspartate Aminotransferase",
            "Bilirubin",
            "Cholinesterase",
            "Cholesterol",
            "Creatinine",
            "Gamma-Glutamyl Transferase",
            "Total Protein"
        ]

        shap_vals = single_explanation.values

        shap.plots.bar(single_explanation, show=False)
        plt.savefig("shap_plot.png", bbox_inches='tight')
        plt.close()

        feature_display = single_explanation.feature_names

        doc = SimpleDocTemplate("Liver_Report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("AI Clinical Decision Support Report", styles['Title']))
        content.append(Spacer(1, 10))
        content.append(Paragraph("Department: Hepatology", styles['Normal']))
        content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        content.append(Spacer(1, 20))

        table_data = [["Parameter", "Value"]]
        for i in range(len(feature_display)):
            table_data.append([feature_display[i], str(features[0][i])])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        content.append(Paragraph("Patient Biochemical Profile", styles['Heading2']))
        content.append(table)
        content.append(Spacer(1, 20))

        content.append(Paragraph(f"Diagnosis: {result}", styles['Heading2']))

        if result.lower() == "cirrhosis":
            risk = "High Risk"
        elif result.lower() in ["fibrosis", "hepatitis"]:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        content.append(Paragraph(f"Risk Category: {risk}", styles['Heading2']))
        content.append(Spacer(1, 20))

        content.append(Paragraph("Key Clinical Insights", styles['Heading2']))

        pairs = list(zip(feature_display, shap_vals, features[0]))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, val, actual in pairs[:5]:
            val = float(val)
            if val > 0:
                content.append(Paragraph(f"{name} is elevated ({actual}), contributing to disease risk.", styles['Normal']))
            else:
                content.append(Paragraph(f"{name} is within range ({actual}), reducing risk.", styles['Normal']))

        content.append(Spacer(1, 20))

        content.append(Paragraph("Clinical Recommendation", styles['Heading2']))
        content.append(Paragraph(
            "Further liver function tests (LFT), ultrasound imaging, and consultation with a hepatologist is advised.",
            styles['Normal']
        ))

        content.append(Spacer(1, 20))
        content.append(Image("shap_plot.png", width=400, height=200))

        doc.build(content)

        # UI result
        if result.lower() == "blood donor":
            st.markdown(f"<h1 style='color:green'>Patient is a : {result}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color:red'>Patient has : {result}</h1>", unsafe_allow_html=True)

        # Download button FIXED
        with open("Liver_Report.pdf", "rb") as f:
            st.download_button("📄 Download Clinical Report", f, file_name="Liver_Report.pdf")

    except Exception as e:
        st.error(f"Error predicting: {e}")