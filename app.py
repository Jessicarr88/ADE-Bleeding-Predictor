import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bleeding Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        color: #155724;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model():
    with open('bleeding_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)

    return model, features

model, feature_names = load_model()

# Title and description
st.title("🏥 Adverse Drug Event (ADE) Bleeding Risk Predictor")
st.markdown("""
This tool predicts the probability of a bleeding event in a hospitalized patient based on
clinical and demographic characteristics. Use this to identify high-risk patients and guide
prophylactic interventions.
""")

# Create tabs for organization
tab1, tab2, tab3 = st.tabs(["Patient Assessment", "Risk Scale Info", "About Model"])

with tab1:
    st.header("Patient Information & Risk Assessment")

    # Organize inputs in columns for better UX
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Comorbidities")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=70)
        sex = st.selectbox("Sex", ["Male", "Female"], index=1)
        sex_encoded = 1 if sex == "Female" else 0

        st.subheader("Chronic Conditions")
        htn = st.checkbox("Hypertension", value=False)
        hld = st.checkbox("Hyperlipidemia", value=False)
        ihd = st.checkbox("Ischemic Heart Disease", value=False)
        hf = st.checkbox("Heart Failure", value=False)
        dm = st.checkbox("Diabetes Mellitus", value=False)
        ckd = st.checkbox("Chronic Kidney Disease", value=False)
        copd = st.checkbox("COPD", value=False)
        afib = st.checkbox("Atrial Fibrillation", value=False)
        dementia = st.checkbox("Dementia", value=False)
        stroke = st.checkbox("Prior Stroke", value=False)
        baseline_anemia = st.checkbox("Baseline Anemia", value=False)
        liver_disease = st.checkbox("Liver Disease", value=False)
        active_cancer = st.checkbox("Active Cancer", value=False)

    with col2:
        st.subheader("Medications")
        warfarin = st.checkbox("Warfarin", value=False)
        doac = st.checkbox("DOAC (Direct Oral Anticoagulant)", value=False)
        antiplatelet = st.checkbox("Antiplatelet Agent", value=False)
        nsaid = st.checkbox("NSAID", value=False)
        ssri_snri = st.checkbox("SSRI/SNRI", value=False)
        steroid = st.checkbox("Systemic Steroid", value=False)
        heparin_lmwh = st.checkbox("Heparin/LMWH", value=False)

        num_home_meds = st.number_input("Number of Home Medications", min_value=0, max_value=30, value=5)
        num_bleeding_risk_meds = st.number_input("Number of Bleeding Risk Medications", min_value=0, max_value=10, value=2)

        st.subheader("Risk Factors")
        recent_fall = st.checkbox("Recent Fall", value=False)
        prior_bleed_12mo = st.checkbox("Prior Bleeding in Last 12 Months", value=False)
        renal_dose_issue = st.checkbox("Renal Dosing Issue", value=False)
        drug_interaction_flag = st.checkbox("Drug Interaction Flag", value=False)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Vital Signs")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=220, value=140)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)

    with col4:
        st.subheader("Laboratory Values")
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.5)
        platelets = st.number_input("Platelets (K/µL)", min_value=10, max_value=500, value=200)
        inr = st.number_input("INR", min_value=0.5, max_value=15.0, value=1.0)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=10.0, value=1.0)
        gfr = st.number_input("GFR (mL/min)", min_value=5, max_value=150, value=60)
        ast = st.number_input("AST (U/L)", min_value=5, max_value=500, value=30)
        alt = st.number_input("ALT (U/L)", min_value=5, max_value=500, value=30)

    # Prepare prediction input
    input_data = {
        'age': age,
        'sex': sex_encoded,
        'htn': int(htn),
        'hld': int(hld),
        'ihd': int(ihd),
        'hf': int(hf),
        'dm': int(dm),
        'ckd': int(ckd),
        'copd': int(copd),
        'afib': int(afib),
        'dementia': int(dementia),
        'stroke': int(stroke),
        'baseline_anemia': int(baseline_anemia),
        'liver_disease': int(liver_disease),
        'active_cancer': int(active_cancer),
        'warfarin': int(warfarin),
        'doac': int(doac),
        'antiplatelet': int(antiplatelet),
        'nsaid': int(nsaid),
        'ssri_snri': int(ssri_snri),
        'steroid': int(steroid),
        'heparin_lmwh': int(heparin_lmwh),
        'num_home_meds': num_home_meds,
        'num_bleeding_risk_meds': num_bleeding_risk_meds,
        'systolic_bp': systolic_bp,
        'heart_rate': heart_rate,
        'hemoglobin': hemoglobin,
        'platelets': platelets,
        'inr': inr,
        'creatinine': creatinine,
        'gfr': gfr,
        'ast': ast,
        'alt': alt,
        'recent_fall': int(recent_fall),
        'prior_bleed_12mo': int(prior_bleed_12mo),
        'renal_dose_issue': int(renal_dose_issue),
        'drug_interaction_flag': int(drug_interaction_flag)
    }

    # Create DataFrame with correct feature order
    X_input = pd.DataFrame([input_data])[feature_names]

    # Make prediction
    if st.button("🔮 Predict Bleeding Risk", type="primary", use_container_width=True):
        try:
            prediction_proba = model.predict_proba(X_input)[0][1]
            risk_percentage = prediction_proba * 100

            st.divider()
            st.subheader("📊 Risk Assessment Result")

            # Display probability as percentage
            col_risk1, col_risk2 = st.columns([2, 1])
            with col_risk1:
                # Create a visual progress bar
                st.metric(
                    label="Predicted Bleeding Risk",
                    value=f"{risk_percentage:.1f}%",
                    delta=None
                )

            # Risk categorization
            st.divider()

            if risk_percentage < 10:
                st.markdown("""
                    <div class="risk-low">
                    <h3>✅ LOW RISK</h3>
                    <p><strong>Risk: &lt;10%</strong></p>
                    <p><strong>Recommendation:</strong> Standard monitoring. Continue routine care with attention to bleeding precautions.</p>
                    </div>
                    """, unsafe_allow_html=True)

            elif risk_percentage < 30:
                st.markdown("""
                    <div class="risk-moderate">
                    <h3>⚠️ MODERATE RISK</h3>
                    <p><strong>Risk: 10-30%</strong></p>
                    <p><strong>Recommendation:</strong> Enhanced monitoring. Consider reviewing medication regimen. Monitor for signs/symptoms of bleeding. Consider prophylaxis if appropriate.</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div class="risk-high">
                    <h3>🔴 HIGH RISK</h3>
                    <p><strong>Risk: &gt;30%</strong></p>
                    <p><strong>Recommendation:</strong> Close monitoring and intervention. Consider medication adjustments, reversal agents on hand, or prophylactic measures. Multidisciplinary review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

            # Risk factors summary
            st.subheader("Summary of Risk Factors")
            risk_factors = []
            if age > 75:
                risk_factors.append(f"Advanced age ({age} years)")
            if baseline_anemia:
                risk_factors.append("Baseline anemia")
            if warfarin or doac or heparin_lmwh:
                risk_factors.append("Anticoagulation therapy")
            if antiplatelet:
                risk_factors.append("Antiplatelet therapy")
            if nsaid:
                risk_factors.append("NSAID use")
            if prior_bleed_12mo:
                risk_factors.append("Prior bleeding event")
            if num_bleeding_risk_meds >= 2:
                risk_factors.append(f"Multiple bleeding risk medications ({num_bleeding_risk_meds})")
            if hemoglobin < 10:
                risk_factors.append(f"Low hemoglobin ({hemoglobin} g/dL)")
            if platelets < 100:
                risk_factors.append(f"Thrombocytopenia ({platelets} K/µL)")
            if inr > 2.5:
                risk_factors.append(f"Elevated INR ({inr})")
            if ckd or gfr < 30:
                risk_factors.append("Severe renal impairment")

            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• No major bleeding risk factors identified")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

with tab2:
    st.header("Risk Categories Explained")

    st.subheader("🟢 Low Risk (<10%)")
    st.write("""
    Bleeding events are unlikely. The patient has minimal risk factors and stable clinical parameters.
    - **What to do:** Continue standard monitoring. No additional interventions typically needed.
    - **Monitoring:** Routine clinical assessment.
    """)

    st.subheader("🟡 Moderate Risk (10-30%)")
    st.write("""
    Bleeding events are possible. The patient has several risk factors or unstable parameters.
    - **What to do:**
        - Review medication list for potential interactions
        - Consider dose adjustments if applicable
        - Monitor more closely for warning signs
        - Consider prophylactic measures based on clinical judgment
    - **Monitoring:** Enhanced assessment every 4-6 hours or per protocol.
    """)

    st.subheader("🔴 High Risk (>30%)")
    st.write("""
    Bleeding events are significant risk. The patient has multiple risk factors and/or critical parameters.
    - **What to do:**
        - Urgent medication review (especially anticoagulants/antiplatelets)
        - Consider holding or reducing dose of high-risk agents
        - Ensure reversal agents are available (Vitamin K, tranexamic acid, etc.)
        - Consider transfusion support if appropriate
        - Multidisciplinary discussion (cardiology, nephrology, pharmacy)
    - **Monitoring:** Close monitoring. Frequent assessments and labs as clinically indicated.
    """)

with tab3:
    st.header("About This Model")

    st.subheader("Model Information")
    st.write("""
    - **Algorithm:** Random Forest Classifier (100 trees)
    - **Training Data:** 500 synthetic patient encounters
    - **Input Features:** 36 clinical and demographic variables
    - **Target:** Adverse drug event - bleeding event occurrence
    - **Date Generated:** 2026
    """)

    st.subheader("Model Performance")
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    with col_perf1:
        st.metric("Accuracy", "84.0%")
    with col_perf2:
        st.metric("ROC-AUC", "0.89")
    with col_perf3:
        st.metric("Recall", "75.0%")

    st.subheader("Important Disclaimers")
    st.warning("""
    ⚠️ **Clinical Use Notice:**
    - This model is designed for **research and educational purposes** only
    - It is **NOT approved for clinical decision-making** without physician review
    - Model predictions should always be combined with clinical judgment
    - Patient safety is the primary concern - when in doubt, consult senior clinicians
    - This is an **MVP tool** and should be validated in clinical settings before deployment
    - Regular model retraining and validation are recommended as new data becomes available
    """)

    st.info("""
    **Data Privacy:** This tool processes patient data locally. No information is transmitted or stored externally.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
<p>ADE Bleeding Risk Prediction Tool | MVP v1.0 | For Research Use Only</p>
<p>Physician-developed | Always verify with clinical judgment and institutional protocols</p>
</div>
""", unsafe_allow_html=True)
