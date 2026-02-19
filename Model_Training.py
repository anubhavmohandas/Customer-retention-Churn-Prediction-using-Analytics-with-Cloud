import streamlit as st
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

st.title("Page 2: Model Training")

if "data" not in st.session_state:
    st.warning("⚠ Please upload data first from Page 1.")
else:
    df = st.session_state["data"]


    st.write("### Dataset Preview")
    st.dataframe(df.head())

    default_index = df.columns.get_loc("Churn") if "Churn" in df.columns else 0

    st.write("### Select Target Column")
    target_column = st.selectbox(
        "Choose Target Column",
        df.columns,
        index=default_index
    )

    st.write("### Select Algorithm")
    algorithm = st.selectbox(
        "Choose Algorithm",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )

    if st.button("Start Training"):

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

      
        if algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif algorithm == "Random Forest":
            model = RandomForestClassifier()

        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)


        if hasattr(model, "predict_proba") and len(y.unique()) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        st.success("Model Training Completed!")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("AUC-ROC", f"{auc:.2f}")
