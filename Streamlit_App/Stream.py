import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
MAX_UPLOAD_SIZE = 150 * 1024 * 1024  

# ==========================================
# 1. CORE LAYOUT AND COMPONENT CONFIGURATION
# ==========================================
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("📊 Employee Attrition Prediction App")
st.write("Enter the employee's metrics below or upload an HR data spreadsheet to evaluate retention risks.")


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
def create_gauge_chart(probability, title="Risk Spectrum Index", height=280):
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


# UI Setup if model elements are valid
if model is not None:
    tab1, tab2 = st.tabs(["👤 Single Profile Evaluation", "📂 Batch Analytics Module (CSV)"])

    # ==========================================
    # WORKSPACE MODULE 1: SINGLE EMPLOYEE EVALUATION
    # ==========================================
    with tab1:
        st.write("Adjust the parameters below to evaluate a specific profile's retention probability metrics.")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
            num_companies = st.number_input("Number of Companies Worked", min_value=0, value=2)
            years_at_company = st.number_input("Years At Company", min_value=0, value=4)

        with col2:
            years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
            env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
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
                st.error(f"### ⚠️ Attrition Risk Alert Flagged")
                st.write("The model categorizes this user profile within higher relative turnover boundaries.")
            else:
                st.success(f"### ✅ Low Risk Retention Profile")
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

    # ==========================================
    # WORKSPACE MODULE 2: BATCH OPERATIONS INTERFACE
    # ==========================================
    with tab2:
        st.subheader("Dynamic Spreadsheet Screener")
        st.write("Upload an active HR dataset matrix (.csv) to isolate enterprise trends and extract probability scoring across metrics.")

        # ✅ CHANGED: Shows "150 MB" next to upload option
        uploaded_file = st.file_uploader(
            "Upload Target Document (CSV Format) - Max 150 MB",  # Added text showing size
            type=["csv"]
        )

        if uploaded_file is not None:
            # Check file size
            file_size = uploaded_file.size
            if file_size > MAX_UPLOAD_SIZE:
                st.error(f"❌ File size ({file_size / (1024*1024):.2f} MB) exceeds 150 MB limit!")
                st.stop()
            
            try:
                raw_batch = pd.read_csv(uploaded_file)
                st.info(f"✅ Data parsed smoothly. File size: {file_size / (1024*1024):.2f} MB | Extracted {raw_batch.shape[0]} candidate observation profiles.")

                if st.button("Run Bulk Prediction Engine", use_container_width=True):
                    # Ensure matching transformation processes on uploaded file data
                    encoded_batch = pd.get_dummies(raw_batch, drop_first=True)

                    aligned_matrix = pd.DataFrame(0, index=range(len(raw_batch)), columns=expected_features)
                    for col in expected_features:
                        if col in encoded_batch.columns:
                            aligned_matrix[col] = encoded_batch[col].values

                    aligned_matrix = aligned_matrix.astype(float)
                    scaled_matrix = scaler.transform(aligned_matrix)

                    batch_predictions = model.predict(scaled_matrix)
                    batch_probabilities = model.predict_proba(scaled_matrix)[:, 1]

                    output_df = raw_batch.copy()
                    output_df['Attrition Probability (%)'] = (batch_probabilities * 100).round(2)
                    output_df['Retention Probability (%)'] = ((1.0 - batch_probabilities) * 100).round(2)
                    output_df['AI Risk Status'] = ['⚠️ High Risk' if p >= 0.50 else '✅ Low Risk' for p in batch_predictions]
                    output_df = output_df.sort_values(by='Attrition Probability (%)', ascending=False)

                    st.divider()

                    # AVERAGE RISK GAUGE FOR THE WHOLE BATCH
                    avg_probability = float(np.mean(batch_probabilities))
                    avg_retention = 1.0 - avg_probability

                    st.subheader("📈 Overall Batch Risk Gauge")
                    st.plotly_chart(
                        create_gauge_chart(avg_probability, title="Average Attrition Risk (Whole Batch)", height=300),
                        use_container_width=True
                    )

                    col_avg1, col_avg2 = st.columns(2)
                    with col_avg1:
                        st.metric(label="Average Attrition Probability", value=f"{avg_probability * 100:.2f}%")
                    with col_avg2:
                        st.metric(label="Average Retention Probability", value=f"{avg_retention * 100:.2f}%")

                    st.divider()
                    st.subheader("📊 Bars & Analytics Dashboard Insights")

                    col_g1, col_g2 = st.columns(2)

                    with col_g1:
                        st.markdown("##### Attrition Risk Classification Breakdown")
                        fig_pie = px.pie(output_df, names='AI Risk Status',
                                        color='AI Risk Status',
                                        color_discrete_map={'⚠️ High Risk': '#dc3545', '✅ Low Risk': '#28a745'},
                                        hole=0.4)
                        fig_pie.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_g2:
                        if 'OverTime' in output_df.columns:
                            st.markdown("##### OverTime Impact vs Average Attrition Probability")
                            fig_bar = px.bar(output_df.groupby('OverTime')['Attrition Probability (%)'].mean().reset_index(),
                                            x='OverTime', y='Attrition Probability (%)',
                                            labels={'Attrition Probability (%)': 'Avg Attrition Score (%)'},
                                            color='OverTime', color_discrete_sequence=['#2563eb', '#cbd5e1'])
                            fig_bar.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260, showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("Notice: 'OverTime' column missing from sheet. Skipping bar plot.")

                    st.divider()
                    st.subheader("🎯 Risk Evaluation Matrix Output")

                    flagged_total = int((batch_predictions == 1).sum())
                    st.metric(label="Profiles Flagged with Urgent Turnover Risk", value=f"{flagged_total} / {len(raw_batch)}")

                    render_columns = ['EmployeeNumber', 'Age', 'MonthlyIncome', 'OverTime', 'Attrition Probability (%)', 'Retention Probability (%)', 'AI Risk Status']
                    available_render = [c for c in render_columns if c in output_df.columns]

                    st.dataframe(output_df[available_render] if available_render else output_df, use_container_width=True)

                    csv_stream = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download System Retention Evaluation Report (CSV)", data=csv_stream, file_name="Attrition_Risk_Report.csv", mime="text/csv", use_container_width=True)

            except Exception as batch_error:
                st.error(f"Operational pipeline parsing exception error: {batch_error}")


st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Deployment Core Architecture | Production Stable v2.2")
