import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Optional: SMOTE for class balancing
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

st.title("Page 2: Model Training")

# ========== SECURITY CHECK: Session State ==========
if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ Please upload data first from Page 1.")
    st.stop()

df = st.session_state["data"].copy()

st.write("### Dataset Preview")
st.dataframe(df.head())

# ========== SECURITY CHECK: Minimum Data Requirements ==========
MIN_ROWS = 50
if len(df) < MIN_ROWS:
    st.error(f"❌ Dataset too small! Minimum {MIN_ROWS} rows required for reliable training. Current: {len(df)} rows.")
    st.stop()

# ========== Target Column Selection ==========
st.write("### Select Target Column")

# Auto-detect common target column names
common_targets = ['Churn', 'churn', 'Target', 'target', 'Label', 'label', 'Class', 'class']
default_index = 0
for i, col in enumerate(df.columns):
    if col in common_targets:
        default_index = i
        break

target_column = st.selectbox(
    "Choose Target Column",
    df.columns,
    index=default_index
)

# ========== SECURITY CHECK: Target Column Validation ==========
if target_column not in df.columns:
    st.error("❌ Selected target column not found in dataset!")
    st.stop()

# Check if target is binary
unique_values = df[target_column].nunique()
if unique_values > 10:
    st.error(f"❌ Target column has {unique_values} unique values. This system supports binary classification (max 2 classes).")
    st.stop()

if unique_values < 2:
    st.error("❌ Target column must have at least 2 unique values!")
    st.stop()

# ========== Algorithm Selection ==========
st.write("### Select Algorithm")
algorithm = st.selectbox(
    "Choose Algorithm",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

# ========== Advanced Options ==========
with st.expander("⚙️ Advanced Options"):
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    use_cv = st.checkbox("Use 5-Fold Cross-Validation", value=True)
    
    if SMOTE_AVAILABLE:
        use_smote = st.checkbox("Apply SMOTE (Class Balancing)", value=True)
    else:
        use_smote = False
        st.info("ℹ️ SMOTE not available. Install: pip install imbalanced-learn")
    
    save_model = st.checkbox("Save Trained Model", value=True)

# ========== Training Button ==========
if st.button("🚀 Start Training"):
    
    with st.spinner("Preprocessing data..."):
        
        # ========== DATA PREPROCESSING ==========
        
        # 1. Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column].copy()
        
        # 2. Handle target encoding
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            st.info(f"ℹ️ Target encoded: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")
        
        # 3. Handle missing values
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            st.warning(f"⚠️ Found {missing_before} missing values. Handling...")
            
            # Numeric columns: fill with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].median())
            
            # Categorical columns: fill with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # 4. Handle TotalCharges-like columns (string to numeric)
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to numeric (handles ' ' as NaN)
                    X[col] = pd.to_numeric(X[col].str.strip(), errors='coerce')
                    X[col] = X[col].fillna(0)
                except:
                    pass
        
        # 5. Encode categorical columns
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            st.info(f"ℹ️ Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # 6. Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            st.error("❌ No valid features remaining after preprocessing!")
            st.stop()
        
        # ========== SECURITY CHECK: Feature Validation ==========
        if X.shape[1] < 1:
            st.error("❌ No features available for training!")
            st.stop()
        
        st.success(f"✅ Preprocessing complete! Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # ========== TRAIN-TEST SPLIT ==========
    progress = st.progress(0)
    progress.progress(10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    progress.progress(20)
    
    # ========== SMOTE BALANCING ==========
    if use_smote and SMOTE_AVAILABLE:
        try:
            st.info("ℹ️ Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.success(f"✅ SMOTE applied! Training samples: {len(y_train)}")
        except Exception as e:
            st.warning(f"⚠️ SMOTE failed: {str(e)}. Continuing without balancing.")
    
    progress.progress(30)
    
    # ========== MODEL TRAINING ==========
    st.info(f"🔄 Training {algorithm}...")
    
    try:
        if algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif algorithm == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # XGBoost
            model = XGBClassifier(
                use_label_encoder=False, 
                eval_metric="logloss", 
                random_state=42,
                n_jobs=-1
            )
        
        progress.progress(50)
        
        # Cross-validation
        if use_cv:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            st.info(f"📊 Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        progress.progress(70)
        
        # Fit model
        model.fit(X_train, y_train)
        
        progress.progress(90)
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.stop()
    
    # ========== PREDICTIONS ==========
    y_pred = model.predict(X_test)
    
    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.0
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    progress.progress(100)
    
    # ========== DISPLAY RESULTS ==========
    st.success("✅ Model Training Completed!")
    
    st.write("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("AUC-ROC", f"{auc:.2f}")
    
    # Confusion Matrix
    with st.expander("📊 Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(
            cm,
            columns=['Predicted 0', 'Predicted 1'],
            index=['Actual 0', 'Actual 1']
        ))
    
    # Feature Importance (for tree-based models)
    if algorithm in ["Random Forest", "XGBoost"]:
        with st.expander("📈 Feature Importance"):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importance.set_index('Feature').head(10))
    
    # ========== SAVE MODEL ==========
    if save_model:
        try:
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "churn_model.pkl")
            encoders_path = os.path.join(model_dir, "encoders.pkl")
            
            joblib.dump(model, model_path)
            joblib.dump(label_encoders, encoders_path)
            
            st.success(f"✅ Model saved to `{model_path}`")
            
            # Store in session state for predictions page
            st.session_state["trained_model"] = model
            st.session_state["label_encoders"] = label_encoders
            st.session_state["feature_columns"] = list(X.columns)
            
        except Exception as e:
            st.warning(f"⚠️ Could not save model: {str(e)}")
