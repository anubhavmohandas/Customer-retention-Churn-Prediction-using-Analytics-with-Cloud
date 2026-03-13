import streamlit as st
import pandas as pd
import pickle
import os

st.title("Page 3: Quick Predict")
st.write("### Predict Customer Churn Using Pre-Trained Model")

# ========== LOAD PRE-TRAINED MODEL ==========
MODEL_PATH = "customer_churn_model.pkl"
ENCODERS_PATH = "encoders.pkl"

model_loaded = False

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        feature_names = model_data["features_names"]
        
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        
        st.success("✅ Pre-trained model loaded successfully!")
        model_loaded = True
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
else:
    st.warning("⚠️ Pre-trained model not found!")
    st.info("""
    **To use this feature:**
    1. Run `python main.py` first to train and save the model
    2. This will generate `customer_churn_model.pkl` and `encoders.pkl`
    3. Then come back to this page
    
    **Or** use Page 1 & 2 to upload your own data and train a new model.
    """)

# ========== RISK RULES BASED ON DATASET ANALYSIS ==========
# These are based on actual churn rates from the Telco dataset
RISK_RULES = {
    "Contract": {
        "Month-to-month": {"risk": "HIGH", "churn_rate": "42%", "suggestion": "Offer 1-year or 2-year contract discount"},
        "One year": {"risk": "MEDIUM", "churn_rate": "11%", "suggestion": "Encourage upgrade to 2-year contract"},
        "Two year": {"risk": "LOW", "churn_rate": "3%", "suggestion": "Customer is locked in - low risk"}
    },
    "PaymentMethod": {
        "Electronic check": {"risk": "HIGH", "churn_rate": "45%", "suggestion": "Encourage automatic bank transfer or credit card"},
        "Mailed check": {"risk": "MEDIUM", "churn_rate": "19%", "suggestion": "Offer paperless billing incentive"},
        "Bank transfer (automatic)": {"risk": "LOW", "churn_rate": "17%", "suggestion": "Good payment method"},
        "Credit card (automatic)": {"risk": "LOW", "churn_rate": "15%", "suggestion": "Good payment method"}
    },
    "InternetService": {
        "Fiber optic": {"risk": "HIGH", "churn_rate": "42%", "suggestion": "Fiber customers churn more - offer retention discount"},
        "DSL": {"risk": "LOW", "churn_rate": "19%", "suggestion": "DSL customers are more stable"},
        "No": {"risk": "LOW", "churn_rate": "7%", "suggestion": "No internet = less likely to churn"}
    },
    "OnlineSecurity": {
        "No": {"risk": "HIGH", "churn_rate": "42%", "suggestion": "Offer free online security trial"},
        "Yes": {"risk": "LOW", "churn_rate": "15%", "suggestion": "Security add-on reduces churn"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "TechSupport": {
        "No": {"risk": "HIGH", "churn_rate": "42%", "suggestion": "Offer free tech support trial"},
        "Yes": {"risk": "LOW", "churn_rate": "15%", "suggestion": "Tech support reduces churn"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "OnlineBackup": {
        "No": {"risk": "MEDIUM", "churn_rate": "40%", "suggestion": "Offer online backup bundle"},
        "Yes": {"risk": "LOW", "churn_rate": "22%", "suggestion": "Good - has backup service"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "DeviceProtection": {
        "No": {"risk": "MEDIUM", "churn_rate": "39%", "suggestion": "Offer device protection bundle"},
        "Yes": {"risk": "LOW", "churn_rate": "23%", "suggestion": "Good - has device protection"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "StreamingTV": {
        "No": {"risk": "MEDIUM", "churn_rate": "33%", "suggestion": "Offer streaming bundle discount"},
        "Yes": {"risk": "MEDIUM", "churn_rate": "30%", "suggestion": "Has streaming - check satisfaction"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "StreamingMovies": {
        "No": {"risk": "MEDIUM", "churn_rate": "33%", "suggestion": "Offer streaming bundle discount"},
        "Yes": {"risk": "MEDIUM", "churn_rate": "30%", "suggestion": "Has streaming - check satisfaction"},
        "No internet service": {"risk": "LOW", "churn_rate": "7%", "suggestion": "N/A"}
    },
    "PaperlessBilling": {
        "Yes": {"risk": "MEDIUM", "churn_rate": "34%", "suggestion": "Paperless users churn slightly more"},
        "No": {"risk": "LOW", "churn_rate": "16%", "suggestion": "Traditional billing - lower churn"}
    },
    "SeniorCitizen": {
        "Yes": {"risk": "MEDIUM", "churn_rate": "41%", "suggestion": "Offer senior discount or simplified plan"},
        "No": {"risk": "LOW", "churn_rate": "24%", "suggestion": "Non-senior - standard risk"}
    },
    "Partner": {
        "No": {"risk": "MEDIUM", "churn_rate": "33%", "suggestion": "Single customers churn more - offer family plan"},
        "Yes": {"risk": "LOW", "churn_rate": "20%", "suggestion": "Has partner - more stable"}
    },
    "Dependents": {
        "No": {"risk": "MEDIUM", "churn_rate": "31%", "suggestion": "No dependents - less sticky"},
        "Yes": {"risk": "LOW", "churn_rate": "15%", "suggestion": "Has dependents - family plan opportunity"}
    }
}

# Tenure risk thresholds
def get_tenure_risk(tenure):
    if tenure <= 6:
        return {"risk": "VERY HIGH", "churn_rate": "50%+", "suggestion": "New customer - needs immediate engagement"}
    elif tenure <= 12:
        return {"risk": "HIGH", "churn_rate": "35-40%", "suggestion": "Still in early phase - offer loyalty reward"}
    elif tenure <= 24:
        return {"risk": "MEDIUM", "churn_rate": "25-30%", "suggestion": "Building loyalty - check satisfaction"}
    elif tenure <= 48:
        return {"risk": "LOW", "churn_rate": "15-20%", "suggestion": "Established customer - maintain service quality"}
    else:
        return {"risk": "VERY LOW", "churn_rate": "<10%", "suggestion": "Long-term loyal customer"}

# Monthly charges risk
def get_charges_risk(monthly_charges):
    if monthly_charges > 90:
        return {"risk": "HIGH", "churn_rate": "35%+", "suggestion": "High spender - ensure value perception, offer discount"}
    elif monthly_charges > 70:
        return {"risk": "MEDIUM", "churn_rate": "28-32%", "suggestion": "Above average charges - review plan fit"}
    elif monthly_charges > 50:
        return {"risk": "LOW", "churn_rate": "20-25%", "suggestion": "Moderate charges - standard risk"}
    else:
        return {"risk": "VERY LOW", "churn_rate": "15-18%", "suggestion": "Low charges - satisfied or minimal services"}


# ========== INPUT FORM ==========
if model_loaded:
    st.write("---")
    st.write("### 📝 Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    with col2:
        st.write("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    with col3:
        st.write("**Account Info**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)
    
    st.write("---")
    
    # ========== PREDICT BUTTON ==========
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        
        # Convert Senior Citizen to numeric
        senior_citizen_val = 1 if senior_citizen == "Yes" else 0
        
        # Create input dataframe
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen_val,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        df_input = pd.DataFrame([input_data])
        
        # Encode categorical columns using saved encoders
        try:
            for col in df_input.columns:
                if col in encoders:
                    df_input[col] = encoders[col].transform(df_input[col])
            
            # Ensure columns are in correct order
            df_input = df_input[feature_names]
            
            # Make prediction
            prediction = model.predict(df_input)[0]
            prediction_proba = model.predict_proba(df_input)[0]
            
            st.write("---")
            st.write("### 🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("## ⚠️ HIGH RISK")
                    st.write("### Customer is likely to **CHURN**")
                else:
                    st.success("## ✅ LOW RISK")
                    st.write("### Customer is likely to **STAY**")
            
            with col2:
                st.write("**Confidence Scores:**")
                stay_prob = prediction_proba[0] * 100
                churn_prob = prediction_proba[1] * 100
                
                st.metric("Stay Probability", f"{stay_prob:.1f}%")
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
            
            # ========== DETAILED RISK ANALYSIS ==========
            st.write("---")
            st.write("### 📊 Risk Factor Analysis")
            st.write("Each input value is analyzed based on historical churn patterns:")
            
            # Collect all risk factors
            high_risk_factors = []
            medium_risk_factors = []
            low_risk_factors = []
            
            # Check each categorical input against risk rules
            inputs_to_check = {
                "Contract": contract,
                "PaymentMethod": payment_method,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "TechSupport": tech_support,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "PaperlessBilling": paperless_billing,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents
            }
            
            for feature, value in inputs_to_check.items():
                if feature in RISK_RULES and value in RISK_RULES[feature]:
                    rule = RISK_RULES[feature][value]
                    factor_info = {
                        "feature": feature,
                        "value": value,
                        "risk": rule["risk"],
                        "churn_rate": rule["churn_rate"],
                        "suggestion": rule["suggestion"]
                    }
                    
                    if rule["risk"] in ["HIGH", "VERY HIGH"]:
                        high_risk_factors.append(factor_info)
                    elif rule["risk"] == "MEDIUM":
                        medium_risk_factors.append(factor_info)
                    else:
                        low_risk_factors.append(factor_info)
            
            # Check tenure
            tenure_risk = get_tenure_risk(tenure)
            tenure_info = {
                "feature": "Tenure",
                "value": f"{tenure} months",
                "risk": tenure_risk["risk"],
                "churn_rate": tenure_risk["churn_rate"],
                "suggestion": tenure_risk["suggestion"]
            }
            if tenure_risk["risk"] in ["HIGH", "VERY HIGH"]:
                high_risk_factors.append(tenure_info)
            elif tenure_risk["risk"] == "MEDIUM":
                medium_risk_factors.append(tenure_info)
            else:
                low_risk_factors.append(tenure_info)
            
            # Check monthly charges
            charges_risk = get_charges_risk(monthly_charges)
            charges_info = {
                "feature": "MonthlyCharges",
                "value": f"${monthly_charges:.2f}",
                "risk": charges_risk["risk"],
                "churn_rate": charges_risk["churn_rate"],
                "suggestion": charges_risk["suggestion"]
            }
            if charges_risk["risk"] in ["HIGH", "VERY HIGH"]:
                high_risk_factors.append(charges_info)
            elif charges_risk["risk"] == "MEDIUM":
                medium_risk_factors.append(charges_info)
            else:
                low_risk_factors.append(charges_info)
            
            # Display HIGH RISK factors
            if high_risk_factors:
                st.write("#### 🔴 High Risk Factors")
                for factor in high_risk_factors:
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: #fee2e2; color: black;padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #ef4444;">
                            <b>⚠️ {factor['feature']}: {factor['value']}</b><br>
                            <small>Historical churn rate: {factor['churn_rate']} | 💡 {factor['suggestion']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display MEDIUM RISK factors
            if medium_risk_factors:
                with st.expander(f"🟡 Medium Risk Factors ({len(medium_risk_factors)})"):
                    for factor in medium_risk_factors:
                        st.markdown(f"""
                        **{factor['feature']}: {factor['value']}**  
                        Churn rate: {factor['churn_rate']} | 💡 {factor['suggestion']}
                        """)
            
            # Display LOW RISK factors
            if low_risk_factors:
                with st.expander(f"🟢 Low Risk Factors ({len(low_risk_factors)})"):
                    for factor in low_risk_factors:
                        st.markdown(f"""
                        **{factor['feature']}: {factor['value']}**  
                        Churn rate: {factor['churn_rate']} | ✓ {factor['suggestion']}
                        """)
            
            # ========== SUMMARY TABLE ==========
            st.write("---")
            st.write("### 📋 Complete Risk Summary")
            
            all_factors = high_risk_factors + medium_risk_factors + low_risk_factors
            summary_df = pd.DataFrame(all_factors)
            summary_df = summary_df.rename(columns={
                "feature": "Feature",
                "value": "Your Input",
                "risk": "Risk Level",
                "churn_rate": "Churn Rate",
                "suggestion": "Recommendation"
            })
            
            # Color code the risk levels with dark text
            def highlight_risk(row):
                if row['Risk Level'] in ['HIGH', 'VERY HIGH']:
                    return ['background-color: #fee2e2; color: #991b1b'] * len(row)
                elif row['Risk Level'] == 'MEDIUM':
                    return ['background-color: #fef3c7; color: #92400e'] * len(row)
                else:
                    return ['background-color: #d1fae5; color: #166534'] * len(row)
            
            styled_df = summary_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # ========== TOP RECOMMENDATIONS ==========
            st.write("---")
            st.write("### 💡 Top Recommendations to Reduce Churn Risk")
            
            if high_risk_factors:
                st.write("**Priority actions based on high-risk factors:**")
                for i, factor in enumerate(high_risk_factors[:5], 1):
                    st.write(f"{i}. **{factor['feature']}**: {factor['suggestion']}")
            else:
                st.success("✅ No high-risk factors identified! Customer has a good retention profile.")
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Make sure the model was trained on the same feature set.")
