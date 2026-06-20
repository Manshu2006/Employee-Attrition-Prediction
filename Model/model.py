import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🎯 Employee Attrition Prediction App")
st.write("Enter employee metrics below to evaluate retention risk.")

def load_model_objects():
    paths = ["Model", "../Model"]
    
    for base_path in paths:
        if os.path.exists(base_path):
            try:
                model = pickle.load(open(os.path.join(base_path, 'attrition_model.pkl'), 'rb'))
                scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
                expected_features = pickle.load(open(os.path.join(base_path, 'features.pkl'), 'rb'))
                categorical_cols = pickle.load(open(os.path.join(base_path, 'categorical_cols.pkl'), 'rb'))
                return model, scaler, expected_features, categorical_cols
            except:
                continue
    
    st.error("❌ Model files not found! Run Employee_Attrition.ipynb first.")
    return None, None, None, None

model, scaler, expected_features, categorical_cols = load_model_objects()

def create_gauge_chart(probability, title="Risk Spectrum", height=320):
    prob_percent = probability * 100
    level_color = "#28a745" if prob_percent < 30 else ("#ffc107" if prob_percent < 70 else "#dc3545")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'font': {'color': level_color, 'size': 42}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4a5568"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#e2fbe2'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#fde8eb'}
            ],
            'threshold': {'line': {'color': level_color, 'width': 5}, 'value': prob_percent}
        }))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
    return fig

if model is not None:
    st.success("✅ Model loaded successfully!")
    st.divider()
    st.subheader("📋 Employee Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        total_working_years = st.number_input("Total Working Years", min_value=0, value=10)
        num_companies = st.number_input("Number Companies Worked", min_value=0, value=2)
        years_at_company = st.number_input("Years at Company", min_value=0, value=4)
        education = st.number_input("Education Level", min_value=1, max_value=5, value=2)
        
    with col2:
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
        education_field = st.selectbox("Education Field", ["Medical", "Technology", "Law", "Earth Sciences", "Marketing", "Human Resources"], index=0)
    
    st.subheader("🏢 Work Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        department = st.selectbox("Department", ["Sales", "Research and Development", "Human Resources"], index=0)
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Sales Representative", "Research Director", "Manufacturing Director", "Healthcare Representative", "Manager", "Trainee"], index=0)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
    
    with col4:
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0)
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    
    st.divider()
    
    if st.button("🎯 Predict Attrition Risk", type="primary", use_container_width=True):
        raw_data = {
            'Age': [age], 'MonthlyIncome': [monthly_income], 'TotalWorkingYears': [total_working_years],
            'NumCompaniesWorked': [num_companies], 'YearsAtCompany': [years_at_company], 'Education': [education],
            'YearsSinceLastPromotion': [years_since_promotion], 'EnvironmentSatisfaction': [env_satisfaction],
            'JobSatisfaction': [job_satisfaction], 'JobInvolvement': [job_involvement], 'WorkLifeBalance': [work_life_balance],
            'BusinessTravel': [business_travel], 'Department': [department], 'EducationField': [education_field],
            'Gender': [gender], 'JobRole': [job_role], 'MaritalStatus': [marital_status], 'OverTime': [overtime]
        }
        
        input_df = pd.DataFrame(raw_data)
        input_df['TotalWorkingYears_Replaced'] = input_df['TotalWorkingYears'].replace(0, 1)
        input_df['YearsAtCompanyRatio'] = input_df['YearsAtCompany'] / input_df['TotalWorkingYears_Replaced']
        input_df['PromotionDelay'] = input_df['YearsAtCompany'] - input_df['YearsSinceLastPromotion']
        input_df['WorkLifeRiskIndex'] = input_df['JobSatisfaction'] + input_df['EnvironmentSatisfaction'] + input_df['WorkLifeBalance']
        
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        final_features = pd.DataFrame(0, index=[0], columns=expected_features)
        for col in input_encoded.columns:
            if col in final_features.columns:
                final_features[col] = input_encoded[col].values
        
        final_features = final_features.astype(float)
        scaled_input = scaler.transform(final_features)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        retention_probability = 1.0 - probability
        
        st.divider()
        
        if prediction == 1:
            st.error("### ⚠️ HIGH ATTRITION RISK")
            st.warning("🔴 Recommendation: Immediate retention intervention needed")
        else:
            st.success("### ✅ LOW ATTRITION RISK")
            st.info("🟢 Recommendation: Continue current engagement")
        
        st.divider()
        st.write("#### 📊 Risk Assessment")
        st.plotly_chart(create_gauge_chart(probability, title="Attrition Risk %", height=320), use_container_width=True)
        
        st.write("#### 📈 Core Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Probability of Leaving", value=f"{probability * 100:.1f}%")
        with col_m2:
            st.metric(label="Probability of Staying", value=f"{retention_probability * 100:.1f}%")

else:
    st.error("❌ Model not loaded!")
    st.code("Run Employee_Attrition.ipynb first to create Model files")

st.caption("Employee Attrition Predictor v3.1")
