import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier

# 1. Load Dataset
DATA_PATH = "dataset/TelecoCustomerChurn.csv"
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found.")
    exit()

df = pd.read_csv(DATA_PATH)

# 2. Data Cleaning
df = df.drop(columns=["customerID"])

# Fix TotalCharges (Handling the " " strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)

# 3. Categorical Encoding
object_columns = df.select_dtypes(include=["object", "string"]).columns
encoders = {}

for column in object_columns:
    if column != "Churn":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

# --- BUG FIX START: Strict Target Formatting ---
# Convert Churn to 1/0 and force it to be an Integer
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop any rows that failed to map (prevents 'unknown' label error)
df = df.dropna(subset=["Churn"])

# Explicitly cast to int64
df["Churn"] = df["Churn"].astype(np.int64)
# --- BUG FIX END ---

# 4. Split Data
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. SMOTE (Should now work without the ValueError)
print("⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 6. Define and Train the 3 Models
models = {
    "decision_tree_model.pkl": DecisionTreeClassifier(random_state=42),
    "random_forest_model.pkl": RandomForestClassifier(random_state=42),
    "xgboost_model.pkl": XGBRFClassifier(random_state=42)
}

if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 Created 'models' folder.")

print("🚀 Training 3 separate models...")
for filename, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    path = os.path.join('models', filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {path}")

# 7. Save Encoders and Feature Names for Django
with open("models/encoders.pkl", "wb") as f:
    pickle.dump({
        "encoders": encoders,
        "feature_names": X.columns.tolist()
    }, f)
print("✅ Saved: models/encoders.pkl")

print("\n✨ Success! All models trained and saved properly.")