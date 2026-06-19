import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# ==========================================
# 1. CORE LAYOUT AND COMPONENT CONFIGURATION
# ==========================================
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Prediction App")
st.write("Enter the employee's metrics below to evaluate their retention risk.")

@st.cache_resource
def load_model_objects():
    try:
        # Check both local path and Model directory hierarchy
        if os.path.exists("Model"):
            base_path = "Model"
        elif os.path.exists("../Model"):
            base_path = "../Model"
        else:
            base_path = "."

        # Match the notebook model names ('attrition_model.pkl' vs 'rf_model.pkl')
        model_filename = 'attrition_model.pkl' if os.path.exists(os.path.join(base_path, 'attrition_model.pkl')) else 'rf_model.pkl'

        model = pickle.load(open(os.path.join(base_path, model_filename), 'rb'))
        scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
        expected_features = pickle.load(open(os.path.join(base_path, 'features.pkl'), 'rb'))
        return model, scaler, expected_features
    except FileNotFoundError:
        st.error("❌ Error: Missing required pipeline components. Ensure your saved pkl files are in the directory.")
        return None, None, None

model, scaler, expected_features = load_model_objects()

# Speedometer/Gauge Chart Component Construction
def create_gauge_chart(probability, title="Risk Spectrum Index", height=320):
    prob_percent = probability * 100
    level_color = "#28a745" if prob_percent < 30 else ("#ffc107" if prob_percent < 70 else "#dc3545")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#4a5568'}},
        number={'font': {'color': level_color, 'size': 42}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a5568"},
            'bar': {'color': "#4a5568"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 30], 'color': '#e2fbe2'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#fde8eb'}
            ],
            'threshold': {'line': {'color': level_color, 'width': 5}, 'value': prob_percent}
        }))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ==========================================
# SINGLE EMPLOYEE EVALUATION
# ==========================================
if model is not None:
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        num_companies = st.number_input("Number of Companies Worked", min_value=0, value=2)
        years_at_company = st.number_input("Years At Company", min_value=0, value=4)

    with col2:
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2, format_func=lambda x: f"{x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"], index=0)

    business_travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0)

    st.divider()

    if st.button("Predict Attrition Risk", type="primary", use_container_width=True):
        raw_data = {
            'Age': [float(age)],
            'MonthlyIncome': [float(income)],
            'NumCompaniesWorked': [float(num_companies)],
            'YearsAtCompany': [float(years_at_company)],
            'YearsSinceLastPromotion': [float(years_since_promotion)],
            'EnvironmentSatisfaction': [float(env_satisfaction)],
            'JobInvolvement': [float(job_involvement)],
            'OverTime': [overtime],
            'BusinessTravel': [business_travel]
        }

        input_df = pd.DataFrame(raw_data)

        # CRITICAL FIX: Explicitly match the training drop_first encoding schema
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Map elements into baseline structure tracking all 0s initializations
        final_features = pd.DataFrame(0, index=[0], columns=expected_features)
        for col in input_encoded.columns:
            if col in final_features.columns:
                final_features[col] = input_encoded[col].values

        final_features = final_features.astype(float)
        scaled_input = scaler.transform(final_features)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        retention_probability = 1.0 - probability

        # Display Prediction Outcomes
        if prediction == 1:
            st.error("### ⚠️ Attrition Risk Alert Flagged")
            st.write("The model categorizes this user profile within higher relative turnover boundaries.")
        else:
            st.success("### ✅ Low Risk Retention Profile")
            st.write("Operational thresholds are balanced. High likelihood of continued organizational alignment.")

        st.divider()

        # GAUGE FIRST (BIGGER, FULL WIDTH) — then both probabilities below it
        st.write("#### Graphical Risk Assessment")
        st.plotly_chart(create_gauge_chart(probability, title="Attrition Risk Spectrum", height=320), use_container_width=True)

        st.write("#### Core Operational Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Probability of Attrition (Leaving)", value=f"{probability * 100:.2f}%", delta="Risk Level", delta_color="inverse" if probability > 0.5 else "normal")
        with col_m2:
            st.metric(label="Probability of Retention (Staying)", value=f"{retention_probability * 100:.2f}%")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Deployment Core Architecture | Production Stable v2.2")
