import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)

# 1. Load Dataset
DATA_PATH = "dataset/TelecoCustomerChurn.csv"
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found.")
    exit()

df = pd.read_csv(DATA_PATH)

# 2. Data Cleaning
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)

# 3. Categorical Encoding
object_columns = df.select_dtypes(include=["object", "string"]).columns
encoders = {}

for column in object_columns:
    if column != "Churn":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

# 4. Target formatting
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])
df["Churn"] = df["Churn"].astype(np.int64)

# 5. Split Data
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. SMOTE on training set only
print("⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 7. Define models
models = {
    "decision_tree_model.pkl": DecisionTreeClassifier(random_state=42),
    "random_forest_model.pkl": RandomForestClassifier(random_state=42),
    "xgboost_model.pkl":       XGBRFClassifier(random_state=42),
}

if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 Created 'models' folder.")

# 8. Train, save, and evaluate each model
print("🚀 Training models and computing real metrics...")

# Map pkl filename → key used in metrics.json / views.py
MODEL_KEYS = {
    "decision_tree_model.pkl": "decision_tree",
    "random_forest_model.pkl": "random_forest",
    "xgboost_model.pkl":       "xgboost",
}

metrics_output = {}

for filename, model in models.items():
    model.fit(X_train_smote, y_train_smote)

    # Save model
    path = os.path.join('models', filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {path}")

    # Evaluate on held-out test set (never seen during training)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred) * 100, 1)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 1)
    rec  = round(recall_score(y_test, y_pred, zero_division=0) * 100, 1)
    f1   = round(f1_score(y_test, y_pred, zero_division=0) * 100, 1)
    auc  = round(float(roc_auc_score(y_test, y_prob)), 4)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Downsample ROC curve to 6 representative points for the chart
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    idx = np.linspace(0, len(fpr) - 1, 6, dtype=int)
    roc_points = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in idx]

    key = MODEL_KEYS[filename]
    metrics_output[key] = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc": roc_points,
    }

    print(f"   {key}: acc={acc}% prec={prec}% rec={rec}% f1={f1}% auc={auc}")

# 9. Save metrics.json — views.py reads this instead of hardcoded values
metrics_path = os.path.join('models', 'metrics.json')
with open(metrics_path, "w") as f:
    json.dump(metrics_output, f, indent=2)
print(f"✅ Saved: {metrics_path}")

# 10. Save encoders and feature names
with open("models/encoders.pkl", "wb") as f:
    pickle.dump({
        "encoders": encoders,
        "feature_names": X.columns.tolist()
    }, f)
print("✅ Saved: models/encoders.pkl")

print("\n✨ Success! All models trained, evaluated, and metrics saved.")
