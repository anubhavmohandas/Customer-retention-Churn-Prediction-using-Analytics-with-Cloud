import streamlit as st

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SESSION STATE INITIALIZATION ==========
if "data" not in st.session_state:
    st.session_state["data"] = None
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "label_encoders" not in st.session_state:
    st.session_state["label_encoders"] = None
if "feature_columns" not in st.session_state:
    st.session_state["feature_columns"] = None

# ========== MAIN PAGE ==========
st.title("🔮 Customer Churn Prediction System")
st.write("### Machine Learning-Based Customer Retention Tool")

st.markdown("""
---
#### 📌 How to Use:
1. **Page 1 - Data Upload**: Upload your customer dataset (CSV format)
2. **Page 2 - Model Training**: Select algorithm and train the model
3. **Page 3 - Quick Predict**: Predict churn for individual customers

#### 🔒 Security Features:
- File size limit: 10MB
- Supported formats: CSV only
- Input validation and sanitization
- Secure data handling

---
""")

# Display current status
st.write("### Current Status")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state["data"] is not None:
        st.success(f"✅ Data Loaded ({len(st.session_state['data'])} rows)")
    else:
        st.warning("⚠️ No data uploaded")

with col2:
    if st.session_state["trained_model"] is not None:
        st.success("✅ Model Trained")
    else:
        st.warning("⚠️ No model trained")

with col3:
    if st.session_state["feature_columns"] is not None:
        st.success(f"✅ {len(st.session_state['feature_columns'])} Features")
    else:
        st.info("ℹ️ Train model first")

st.markdown("""
---
#### 👥 Group G05 | IBM Project 2024-25
**Institute of Computer Technology, Ganpat University**
""")

# Sidebar info
with st.sidebar:
    st.write("### 📊 Navigation")
    st.info("Use the sidebar to navigate between pages.")

    st.write("### ℹ️ About")
    st.write("""
    This system predicts customer churn using:
    - Decision Tree
    - Random Forest
    - XGBoost

    Dataset: Kaggle Telco Customer Churn
    """)