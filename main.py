import pickle
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import cool
from seaborn import histplot
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("dataset/TelecoCustomerChurn.csv")
print(df.info())

print(df.columns)
df = df.drop(columns=["customerID"])
#printing uniques in all columns
numerical_feature_list = ["tenure","MonthlyCharges","TotalCharges"]
for col in df.columns:
    if col not in numerical_feature_list:
        print(col,df[col].unique())
        print("-"*50)

df[df["TotalCharges"]==" "]
len(df[df["TotalCharges"]==" "])
df["TotalCharges"] = df["TotalCharges"].replace({" ":"0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
#df.info()
# checking the class distribution of target class
print(df["Churn"].value_counts())
print(df.describe())
#desribe()only works on numerical data types
"""
numerical feature analysis
-> understand the distribution of tech numerical features
"""
def plot_histogram(df,column_name):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name],kde=True)
    plt.title(f"Distribution of {column_name}")
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    plt.axvline(col_mean,color="red",linestyle="--",label="Mean")
    plt.axvline(col_median, color="purple",linestyle="-",label="Median")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{column_name}_hist.png")
    plt.close()
plot_histogram(df,column_name="tenure")
plot_histogram(df,column_name="MonthlyCharges")
plot_histogram(df,column_name="TotalCharges")
"""
Box plot for numerical features 
"""
def plot_boxplot(df, column_name):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Boxplot of{column_name}")
    plt.ylabel(column_name)
    plt.savefig(f"{column_name}_boxplot.png")
    plt.close()
plot_boxplot(df,column_name="tenure")
plot_boxplot(df,column_name="MonthlyCharges")
plot_boxplot(df,column_name="TotalCharges")

"""
correlation heatmap for numerical columns 
"""
plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.savefig(f"heatmap.png")
plt.close()


"""
categorical features analysis
"""
print(df.dtypes)
object_cols = df.select_dtypes(include=["object","string"]).columns.to_list()
object_cols = ["SeniorCitizen"]+object_cols
for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"Count plot of{col}")
    plt.savefig(f"{col}_countplot.png")
    plt.close()

"""
Data preprocessing
"""
df["Churn"] = df["Churn"].replace(({"Yes":1,"No":0}))
print(df["Churn"].value_counts())

"""
Label encoding of categorical features
"""
object_columns = df.select_dtypes(include=["object","string"]).columns
encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

with open("encoders.pkl","wb") as f:
    pickle.dump(encoders, f)
"""
train and test data split
"""
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(y_train.shape)
print(y_train.value_counts())

"""
Synthetic Minority oversampling technique smote 
"""
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(X_train_smote.shape)
print(y_train_smote.shape)
print(pd.Series(y_train_smote).value_counts())
"""
Model Training 
"""
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "RandomForest":RandomForestClassifier(random_state=42),
    "XGBoost":XGBRFClassifier(random_state=42)
}
"""cross validation results"""
cv_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name} with default parameters ")
    scores = cross_val_score(model,X_train_smote,y_train_smote,cv=5,scoring="accuracy")
    cv_scores[model_name]=scores
    print(f"{model_name} cross validation accuracy: {np.mean(scores):.2f}")
    print("-"*70)
print(cv_scores)

"""
random forest gives the highest accuracy compared to other models with default parameters
"""
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)
print(y_test.value_counts())
"""
Model evaluation
"""
y_test_pred = rfc.predict(X_test)
print("Accuracy score: \n ", accuracy_score(y_test,y_test_pred))
print("Confusion matrix: \n",  confusion_matrix(y_test,y_test_pred))
print("Classification:\n", classification_report(y_test,y_test_pred))


model_data = {"model": rfc,"features_names": X.columns.tolist()}

with open("customer_churn_model.pkl","wb") as f:
    pickle.dump(model_data,f)

"""
Load the saved model and build predictive system
"""
with open("customer_churn_model.pkl","rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]
print(loaded_model)
print(feature_names)