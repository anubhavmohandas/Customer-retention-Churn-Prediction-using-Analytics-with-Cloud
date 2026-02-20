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
            
            # Risk factors
            st.write("---")
            with st.expander("📊 Risk Analysis"):
                risk_factors = []
                
                if contract == "Month-to-month":
                    risk_factors.append("⚠️ Month-to-month contract (high churn rate)")
                if tenure < 12:
                    risk_factors.append("⚠️ Low tenure (< 12 months)")
                if internet_service == "Fiber optic" and online_security == "No":
                    risk_factors.append("⚠️ Fiber optic without online security")
                if payment_method == "Electronic check":
                    risk_factors.append("⚠️ Electronic check payment (higher churn)")
                if monthly_charges > 70:
                    risk_factors.append("⚠️ High monthly charges (> $70)")
                
                if risk_factors:
                    st.write("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("✅ No major risk factors identified")
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Make sure the model was trained on the same feature set.")
