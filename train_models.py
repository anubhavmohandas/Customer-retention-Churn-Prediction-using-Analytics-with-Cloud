"""
Telco Customer Churn — Full Pipeline
EDA → Preprocessing → SMOTE → Model Training → Evaluation → Save Artifacts
"""

import os
import pickle
import json
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRFClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────
DATA_PATH  = "dataset/TelecoCustomerChurn.csv"
MODELS_DIR = "models"
PLOTS_DIR  = "plots"

for d in [MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

PALETTE   = {"No": "#2ecc71", "Yes": "#e74c3c"}
BLUE      = "#3498db"
RED       = "#e74c3c"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False})

# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at '{DATA_PATH}'")

df_raw = pd.read_csv(DATA_PATH)
print("=" * 60)
print(f"✅ Loaded dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────
df = df_raw.copy()
df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
df.reset_index(drop=True, inplace=True)

print(f"\n✅ After cleaning: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 3. EDA Plots
# ─────────────────────────────────────────────
print("\n📊 Generating EDA plots ...")

churn_labels = df_raw["Churn"].value_counts()

# 3-A: Churn distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(churn_labels.index, churn_labels.values,
            color=[PALETTE["No"], PALETTE["Yes"]], edgecolor="white", width=0.5)
axes[0].set_title("Churn Count", fontweight="bold")
axes[0].set_ylabel("Customers")
for i, v in enumerate(churn_labels.values):
    axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=10)

pct = churn_labels / churn_labels.sum() * 100
axes[1].pie(pct.values, labels=pct.index, autopct="%1.1f%%",
            colors=[PALETTE["No"], PALETTE["Yes"]],
            startangle=90, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Churn Share", fontweight="bold")
plt.suptitle("Target Variable — Churn", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_churn_distribution.png", bbox_inches="tight")
plt.close()

# 3-B: Correlation heatmap (For Plotting Only)
df_corr = df.copy()
obj_cols = df_corr.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df_corr[c] = LabelEncoder().fit_transform(df_corr[c].astype(str))
corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  ✅ Saved visual plots to {PLOTS_DIR}/")

# ─────────────────────────────────────────────
# 4. Preprocessing (Upgraded for Django Integration)
# ─────────────────────────────────────────────
print("\n🔧 Preprocessing Data ...")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ONE-HOT ENCODING (Crucial for Logistic Regression)
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)
X_encoded = X_encoded.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING (Crucial for total_charges vs tenure)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 5. SMOTE
# ─────────────────────────────────────────────
print("\n⚖️  Applying SMOTE ...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ─────────────────────────────────────────────
# 6. Define Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": {
        "file": "logistic_regression_model.pkl",
        "model": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    },
    "Random Forest": {
        "file": "random_forest_model.pkl",
        "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    },
    "XGBoost": {
        "file": "xgboost_model.pkl",
        "model": XGBRFClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    },
}

# ─────────────────────────────────────────────
# 7. Train, Evaluate & Save
# ─────────────────────────────────────────────
print("\n🚀 Training models ...\n")

results = {}
roc_data = {}
json_metrics = {}
COLORS_ROC = [BLUE, "#9b59b6", RED]

for name, cfg in models.items():
    print(f"  ▶ {name}")
    clf = cfg["model"]
    clf.fit(X_train_sm, y_train_sm)

    y_pred    = clf.predict(X_test)
    y_proba   = clf.predict_proba(X_test)[:, 1]
    auc       = roc_auc_score(y_test, y_proba)
    report    = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name]  = {"report": report, "auc": auc, "y_pred": y_pred}
    roc_data[name] = (fpr, tpr, auc)

    # Compile JSON data for Django Dashboard
    json_name = name.lower().replace(" ", "_")
    json_metrics[json_name] = {
        "accuracy": round((report["accuracy"]) * 100, 1),
        "precision": round(report["1"]["precision"] * 100, 1),
        "recall": round(report["1"]["recall"] * 100, 1),
        "f1": round(report["1"]["f1-score"] * 100, 1),
    }

    # Save model .pkl file
    path = os.path.join(MODELS_DIR, cfg["file"])
    with open(path, "wb") as f:
        pickle.dump(clf, f)

# Save JSON file for Dashboard
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(json_metrics, f)
print(f"\n    ✅ Saved API Metrics → {MODELS_DIR}/metrics.json")

# ─────────────────────────────────────────────
# 8. Save Django Metadata Artifacts
# ─────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "numeric_cols": num_cols,
        "feature_names": X_train.columns.tolist()
    }, f)
print(f"    ✅ Saved API Metadata → {MODELS_DIR}/metadata.pkl")

print("\n" + "=" * 60)
print("✨ Done! Models and APIs are fully compiled.")
print("=" * 60)