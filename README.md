# Customer Churn Prediction System

A machine learning-based web application that predicts customer churn for telecom companies using classification algorithms.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

-----

## About

Customer churn is when customers stop using a service. Telecom companies lose 15-25% of customers annually, and acquiring new customers costs 5x more than retaining existing ones.

This project builds a predictive system that identifies at-risk customers before they leave, enabling proactive retention strategies.

**Key Results:**

- 84% Cross-Validation Accuracy
- 76% Test Accuracy
- 73% Recall (catches 73% of actual churners)

-----

## Features

- **Data Upload**: Drag-and-drop CSV file upload with validation
- **Multiple Algorithms**: Decision Tree, Random Forest, XGBoost
- **SMOTE Balancing**: Handles imbalanced datasets automatically
- **5-Fold Cross-Validation**: Reliable accuracy estimates
- **Quick Predict**: Instant predictions for individual customers
- **Security**: File size limits, format validation, input sanitization

-----

## Tech Stack

|Category       |Tools                                  |
|---------------|---------------------------------------|
|Language       |Python 3.11                            |
|ML Libraries   |Scikit-learn, XGBoost, Imbalanced-learn|
|Web Framework  |Streamlit                              |
|Data Processing|Pandas, NumPy                          |
|Visualization  |Matplotlib, Seaborn, Plotly            |
|Model Storage  |Joblib, Pickle                         |

-----

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/anubhavmohandas/Customer-retention-Churn-Prediction-using-Analytics.git
cd Customer-retention-Churn-Prediction-using-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it in:

```
dataset/TelecoCustomerChurn.csv
```

### 4. Train the model (one-time setup)

```bash
python main.py
```

This generates:

- `customer_churn_model.pkl` (trained model)
- `encoders.pkl` (label encoders)
- Visualization PNGs (histograms, heatmaps, etc.)

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

-----

## Project Structure

```
├── app.py                    # Streamlit main entry
├── main.py                   # ML training pipeline
├── pages/
│   ├── Data_Upload.py        # Page 1: CSV upload
│   ├── Model_Training.py     # Page 2: Train models
│   └── Predict.py            # Page 3: Quick predictions
├── dataset/
│   └── TelecoCustomerChurn.csv
├── customer_churn_model.pkl  # Trained model (generated)
├── encoders.pkl              # Label encoders (generated)
├── requirements.txt          # Dependencies
└── README.md
```

-----

## Usage

### Option 1: Use Pre-trained Model (Quick Predict)

1. Run `python main.py` once to generate model files
1. Run `streamlit run app.py`
1. Go to **Page 3: Quick Predict**
1. Enter customer details and get instant prediction

### Option 2: Train on Your Own Data

1. Run `streamlit run app.py`
1. Go to **Page 1: Data Upload** and upload your CSV
1. Go to **Page 2: Model Training**
1. Select algorithm and click Train
1. View results and metrics

-----

## Dashboard Pages

### Page 1: Data Upload

- Upload CSV files (max 10MB)
- Automatic validation and preview
- Security checks on file format

### Page 2: Model Training

- Select target column (Churn)
- Choose algorithm (Decision Tree / Random Forest / XGBoost)
- Enable SMOTE balancing
- View accuracy, precision, recall, confusion matrix

### Page 3: Quick Predict

- 19 input fields for customer details
- Loads pre-trained model
- Shows churn probability and risk level
- Identifies risk factors

-----

## Model Performance

|Model            |CV Accuracy|Notes                |
|-----------------|-----------|---------------------|
|Decision Tree    |78%        |Simple, interpretable|
|**Random Forest**|**84%**    |Best performance     |
|XGBoost          |81%        |Gradient boosting    |

### Final Results (Random Forest on Test Set)

|Metric           |Value|
|-----------------|-----|
|Accuracy         |76%  |
|Precision (Churn)|53%  |
|Recall (Churn)   |73%  |
|AUC-ROC          |0.81 |

**Confusion Matrix:**

```
              Predicted
              Stay    Churn
Actual Stay   797     239
Actual Churn   99     274
```

-----

## Dataset

**Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

|Info      |Value          |
|----------|---------------|
|Records   |7,043 customers|
|Features  |21 columns     |
|Churn Rate|26.5%          |

**Feature Categories:**

- Demographics (gender, senior citizen, partner, dependents)
- Services (phone, internet, security, streaming, etc.)
- Account (tenure, contract, billing, payment method)
- Charges (monthly, total)

-----

## Team

**Group G05 - IBM Project 2024-25**

|Name            |Enrollment |Branch|Contribution                      |
|----------------|-----------|------|----------------------------------|
|Anubhav Mohandas|23162192013|CS    |Security, Preprocessing, Dashboard|
|Vishv Munjapara |23162122002|BDA   |ML Pipeline, Visualizations       |
|Khushang Patel  |22162581005|CSE   |Streamlit UI Setup                |

**Guides:**

- Prof. Tejas Kadiya (Internal)
- Prof. Umesh Lakhtariya (Internal)
- Mr. Anoj Dixit (External)

**Institute:** Institute of Computer Technology, Ganpat University

-----

## Future Scope

- [ ] AWS EC2 Deployment
- [ ] Deep Learning (LSTM)
- [ ] Real-time Streaming (Kafka)
- [ ] SHAP Explainability
- [ ] Mobile App (React Native)

-----

## License

MIT License - feel free to use and modify.

-----

## References

1. [Kaggle Telco Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
1. [Scikit-learn Documentation](https://scikit-learn.org/)
1. [XGBoost Documentation](https://xgboost.readthedocs.io/)
1. [Streamlit Documentation](https://docs.streamlit.io/)
1. [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
