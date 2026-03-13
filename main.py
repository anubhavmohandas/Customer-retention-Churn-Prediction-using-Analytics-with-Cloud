import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import sys


def load_and_clean_data(filepath):
    """Load dataset and perform initial cleaning."""
    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at '{filepath}'")
        print("Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"Place it at: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(df.info())
    print(df.columns)

    df = df.drop(columns=["customerID"])

    # Print unique values for categorical columns
    numerical_feature_list = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in df.columns:
        if col not in numerical_feature_list:
            print(col, df[col].unique())
            print("-" * 50)

    # Fix TotalCharges: replace whitespace with 0.0 and convert to float
    print(f"Rows with blank TotalCharges: {len(df[df['TotalCharges'] == ' '])}")
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    print(df["Churn"].value_counts())
    print(df.describe())

    return df


def plot_histogram(df, column_name, output_dir="."):
    """Plot histogram with mean and median lines."""
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="purple", linestyle="-", label="Median")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column_name}_hist.png"))
    plt.close()


def plot_boxplot(df, column_name, output_dir="."):
    """Plot boxplot for outlier detection."""
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.savefig(os.path.join(output_dir, f"{column_name}_boxplot.png"))
    plt.close()


def generate_visualizations(df, output_dir="."):
    """Generate all EDA visualizations."""
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in numerical_features:
        plot_histogram(df, col, output_dir)
        plot_boxplot(df, col, output_dir)

    # Correlation heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        df[numerical_features].corr(),
        annot=True, cmap="coolwarm", fmt=".2f"
    )
    plt.savefig(os.path.join(output_dir, "heatmap.png"))
    plt.close()

    # Categorical count plots
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    object_cols = ["SeniorCitizen"] + object_cols

    for col in object_cols:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=df[col])
        plt.title(f"Count plot of {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_countplot.png"))
        plt.close()


def preprocess_and_encode(df):
    """Encode target and categorical features, return encoders."""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print(df["Churn"].value_counts())

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    encoders = {}
    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder

    return df, encoders


def train_and_evaluate(df):
    """Train models, evaluate, and return best model."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {y_train.shape}")
    print(y_train.value_counts())

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_smote.shape}")
    print(pd.Series(y_train_smote).value_counts())

    # Train multiple models with cross-validation
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
    }

    cv_scores = {}
    for model_name, model in models.items():
        print(f"Training {model_name} with default parameters...")
        scores = cross_val_score(
            model, X_train_smote, y_train_smote, cv=5, scoring="accuracy"
        )
        cv_scores[model_name] = scores
        print(f"{model_name} CV accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
        print("-" * 70)

    print(cv_scores)

    # Train best model (Random Forest) on full SMOTE training set
    print("\nTraining final Random Forest model...")
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train_smote, y_train_smote)

    # Evaluate on test set
    y_test_pred = rfc.predict(X_test)
    print(f"\nTest set class distribution:\n{y_test.value_counts()}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")

    return rfc, X.columns.tolist()


MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")


def save_artifacts(model, feature_names, encoders):
    """Save model and encoders using joblib to saved_models/."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_data = {
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(model_data, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")


def verify_saved_model():
    """Load and verify saved artifacts."""
    model_data = joblib.load(MODEL_PATH)
    loaded_model = model_data["model"]
    feature_names = model_data["feature_names"]
    print(f"\nVerification - Model type: {type(loaded_model).__name__}")
    print(f"Verification - Features: {feature_names}")


if __name__ == "__main__":
    DATASET_PATH = "dataset/TelecoCustomerChurn.csv"

    df = load_and_clean_data(DATASET_PATH)
    generate_visualizations(df)

    df, encoders = preprocess_and_encode(df)
    model, feature_names = train_and_evaluate(df)

    save_artifacts(model, feature_names, encoders)
    verify_saved_model()