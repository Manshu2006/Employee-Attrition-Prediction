import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🎯 Employee Attrition Prediction App")
st.write("Enter employee metrics below to evaluate retention risk.")

# ==========================================
# 2. LOAD MODEL OBJECTS (with caching)
# ==========================================
@st.cache_resource
def load_model_objects():
    try:
        # Find Model directory
        if os.path.exists("Model"):
            base_path = "Model"
        elif os.path.exists("../Model"):
            base_path = "../Model"
        elif os.path.exists("/mount/src/employee-attrition-prediction/Model"):
            base_path = "/mount/src/employee-attrition-prediction/Model"
        else:
            st.error("❌ Model directory not found!")
            return None
        
        # Load all required files
        model = pickle.load(open(os.path.join(base_path, 'attrition_model.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
        expected_features = pickle.load(open(os.path.join(base_path, 'features.pkl'), 'rb'))
        categorical_cols = pickle.load(open(os.path.join(base_path, 'categorical_cols.pkl'), 'rb'))
        
        return model, scaler, expected_features, categorical_cols
    except FileNotFoundError as e:
        st.error(f"❌ Missing files: {e}")
        return None, None, None, None

model, scaler, expected_features, categorical_cols = load_model_objects()

# ==========================================
# 3. GAUGE CHART COMPONENT
# ==========================================
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
# 4. MAIN PREDICTION LOGIC
# ==========================================
if model is not None:
    st.divider()
    
    # ═══════════════════════════════════════════════
    # INPUT FORM - ALL FIELDS FROM TRAINING DATA
    # ═══════════════════════════════════════════════
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
        education_field = st.selectbox(
            "Education Field",
            ["Medical", "Technology", "Law", "Earth Sciences", "Marketing", "Human Resources"],
            index=0
        )
    
    # CATEGORICAL FIELDS
    st.subheader("🏢 Work Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        department = st.selectbox(
            "Department",
            ["Sales", "Research and Development", "Human Resources"],
            index=0
        )
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician", 
             "Sales Representative", "Research Director", "Manufacturing Director",
             "Healthcare Representative", "Manager", "Trainee"],
            index=0
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"],
            index=0
        )
    
    with col4:
        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
            index=0
        )
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # PREDICTION BUTTON
    # ═══════════════════════════════════════════════
    if st.button("🎯 Predict Attrition Risk", type="primary", use_container_width=True):
        
        # Create raw input DataFrame with ALL required columns
        raw_data = {
            'Age': [age],
            'MonthlyIncome': [monthly_income],
            'TotalWorkingYears': [total_working_years],
            'NumCompaniesWorked': [num_companies],
            'YearsAtCompany': [years_at_company],
            'Education': [education],
            'YearsSinceLastPromotion': [years_since_promotion],
            'EnvironmentSatisfaction': [env_satisfaction],
            'JobSatisfaction': [job_satisfaction],
            'JobInvolvement': [job_involvement],
            'WorkLifeBalance': [work_life_balance],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'EducationField': [education_field],
            'Gender': [gender],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status],
            'OverTime': [overtime]
        }
        
        input_df = pd.DataFrame(raw_data)
        
        # STEP 1: Create engineered features (MUST MATCH TRAINING)
        input_df['TotalWorkingYears_Replaced'] = input_df['TotalWorkingYears'].replace(0, 1)
        input_df['YearsAtCompanyRatio'] = input_df['YearsAtCompany'] / input_df['TotalWorkingYears_Replaced']
        input_df['PromotionDelay'] = input_df['YearsAtCompany'] - input_df['YearsSinceLastPromotion']
        input_df['WorkLifeRiskIndex'] = input_df['JobSatisfaction'] + input_df['EnvironmentSatisfaction'] + input_df['WorkLifeBalance']
        
        # STEP 2: Encode categorical variables (EXACTLY LIKE TRAINING)
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # STEP 3: Create DataFrame with ALL expected columns (filled with 0)
        final_features = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # STEP 4: Fill in the values from encoded input
        for col in input_encoded.columns:
            if col in final_features.columns:
                final_features[col] = input_encoded[col].values
        
        # STEP 5: Ensure correct data type
        final_features = final_features.astype(float)
        
        # STEP 6: Scale features (using saved scaler)
        scaled_input = scaler.transform(final_features)
        
        # STEP 7: Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        retention_probability = 1.0 - probability
        
        # ═══════════════════════════════════════════════
        # DISPLAY RESULTS
        # ═══════════════════════════════════════════════
        st.divider()
        
        if prediction == 1:
            st.error("### ⚠️ HIGH ATTRITION RISK DETECTED")
            st.write("This employee shows strong indicators of potential turnover.")
            st.warning("🔴 **Recommendation**: Immediate retention intervention needed")
        else:
            st.success("### ✅ LOW ATTRITION RISK")
            st.write("This employee shows stable retention indicators.")
            st.info("🟢 **Recommendation**: Continue current engagement strategies")
        
        st.divider()
        
        # Gauge Chart
        st.write("#### 📊 Risk Assessment Visual")
        st.plotly_chart(create_gauge_chart(probability, title="Attrition Risk Percentage", height=320), use_container_width=True)
        
        # Metrics
        st.write("#### 📈 Core Metrics")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric(
                label="Probability of Leaving",
                value=f"{probability * 100:.1f}%",
                delta="Risk Level",
                delta_color="inverse" if probability > 0.5 else "normal"
            )
        
        with col_m2:
            st.metric(
                label="Probability of Staying",
                value=f"{retention_probability * 100:.1f}%"
            )
        
        # Additional insights
        st.divider()
        st.write("#### 💡 Actionable Insights")
        if probability > 0.7:
            st.warning("• High risk - Schedule exit interview prevention meeting")
            st.warning("• Review compensation and career growth opportunities")
            st.warning("• Assess work-life balance and job satisfaction")
        elif probability > 0.4:
            st.info("• Medium risk - Monitor closely")
            st.info("• Engage in career development discussions")
            st.info("• Check satisfaction with management")
        else:
            st.success("• Low risk - Employee appears stable")
            st.success("• Continue current engagement practices")
            st.success("• Consider for leadership development")

else:
    st.error("❌ Model not loaded. Please check your Model directory.")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Employee Attrition Predictor | Production Stable v3.0 | Fixed Feature Engineering")
