<div align="center">

# 🧠 Retain.Ai
### *Predict who's leaving. Before they do.*

[![Django](https://img.shields.io/badge/Django-6.0-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Production-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/ec2)
[![HTTPS](https://img.shields.io/badge/HTTPS-Secured-00C853?style=for-the-badge&logo=letsencrypt&logoColor=white)](https://letsencrypt.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Retain.Ai** is a full-stack, production-deployed AI platform that predicts telecom customer churn using an ensemble of Machine Learning models — complete with 2FA authentication, bulk CSV analysis, real-time risk scoring, and one-click model retraining. Built for scale. Secured like it matters.

**🌐 Live Demo → [churnprediction.duckdns.org](https://churnprediction.duckdns.org)**

</div>

---

## ✨ What makes this different?

Most churn prediction projects are Jupyter notebooks. **This is a production application.**

| Feature | Status |
|---|---|
| 3 ML models running in parallel (LR + RF + XGBoost) | ✅ Live |
| Bulk CSV prediction — thousands of customers at once | ✅ Live |
| Auto-train new models from your own dataset | ✅ Live |
| 2FA login via email OTP | ✅ Live |
| Forgot password with secure email link | ✅ Live |
| Real-time risk scoring with animated UI | ✅ Live |
| PostgreSQL + AWS EC2 + HTTPS deployment | ✅ Live |
| SMOTE for handling class imbalance | ✅ Live |
| Rate limiting, CSRF protection, joblib serialization | ✅ Live |

---

## 🚀 Features

### 🤖 AI Prediction Engine
- **3 models** trained on the IBM Telco Customer Churn Dataset — Logistic Regression, Random Forest, and XGBoost
- **SMOTE** (Synthetic Minority Oversampling) to fix class imbalance — so the model actually learns to detect churners, not just predict everyone stays
- **Auto-Train** — upload your own CSV and retrain all 3 models in one click. No code. No terminal.
- **Ensemble risk scoring** — models vote, you get a confidence-backed churn probability with color-coded risk bands (Low / Medium / High)
- **joblib serialization** — safe, fast model loading. No pickle vulnerabilities.

### 📊 Dashboard & Analytics
- Live metrics: total customers, churn rate, high-risk count, MRR at risk
- Risk distribution breakdown across customer segments
- Activity feed with real-time prediction logs
- AI model registry — compare accuracy, precision, recall, F1 across all 3 models

### 📁 Bulk CSV Processing
- Upload a `.csv` with thousands of subscribers
- Get back per-customer churn probability, risk tier, and contributing factors
- 10 MB file size guard with instant client-side validation
- Export high-risk customer list as CSV for your retention team

### 🔐 Security — Production Grade
- **2FA via Email OTP** — every login triggers a 6-digit code sent to the user's inbox. Expires in 10 minutes.
- **Forgot Password** — UUID token-based reset link. Expires in 1 hour. Immune to email enumeration.
- **SMOTE + joblib** — no arbitrary code execution on model load (unlike pickle)
- **CSRF protection** on all API endpoints
- **Rate limiting** — 3 training requests per hour per user
- **DB indexes** on all foreign keys for query performance
- **HSTS, SSL redirect, X-Frame-Options DENY, Content-Type nosniff** in production
- **Session + CSRF cookies** — Secure, HttpOnly in production

### ☁️ Cloud Deployment
- **AWS EC2** — Ubuntu 22.04, systemd-managed gunicorn service
- **PostgreSQL** — production database with connection pooling
- **Nginx** — reverse proxy, static file serving, SSL termination
- **Let's Encrypt + DuckDNS** — free HTTPS with auto-renewal
- **WhiteNoise** — compressed static file serving
- **Zero-downtime deploys** via `deploy.sh` — git pull → pip install → migrate → collectstatic → restart

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Django 6.0, Django REST Framework |
| **ML** | XGBoost, scikit-learn, imbalanced-learn (SMOTE) |
| **Database** | PostgreSQL (prod), SQLite (dev) |
| **Auth** | Custom User Model, 2FA OTP, Password Reset |
| **Email** | Resend API |
| **Frontend** | Tailwind CSS, Vanilla JS |
| **Server** | Gunicorn (gthread workers), Nginx |
| **Cloud** | AWS EC2, Let's Encrypt SSL |
| **Storage** | WhiteNoise (static), joblib (models) |

---

## ⚡ Quick Start (Local Dev)

```bash
# 1. Clone
git clone https://github.com/anubhavmohandas/Customer-retention-Churn-Prediction-using-Analytics-with-Cloud.git
cd Customer-retention-Churn-Prediction-using-Analytics-with-Cloud

# 2. Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Edit .env with your values

# 5. Migrate & run
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

### Environment Variables

```dotenv
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Optional — leave empty to use SQLite locally
DATABASE_URL=postgres://user:pass@localhost:5432/dbname

# Resend — for 2FA OTP and password reset emails
RESEND_API_KEY=re_your_key_here
RESEND_FROM_EMAIL=Retain.Ai <noreply@yourdomain.com>
```

---

## 🧬 ML Pipeline

```
Raw CSV (IBM Telco Dataset)
        │
        ▼
  Feature Engineering
  - Encode categoricals
  - Scale numericals
  - Handle missing values
        │
        ▼
     SMOTE
  (Balance churners vs non-churners)
        │
        ├──► Logistic Regression
        ├──► Random Forest
        └──► XGBoost
                │
                ▼
        Ensemble Risk Score
        (0–100% churn probability)
                │
          ┌─────┴─────┐
        Low        Medium      High
       <30%       30–70%      >70%
```

**Training metrics (on IBM Telco Dataset):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~80% | ~67% | ~55% | ~60% |
| Random Forest | ~79% | ~65% | ~48% | ~55% |
| XGBoost | ~78% | ~62% | ~52% | ~57% |

---

## 🔒 Authentication Flow

```
User enters email + password
        │
        ▼
  Credentials verified
        │
        ▼
  6-digit OTP generated
  → sent to email via Resend
  → stored in DB with 10-min expiry
        │
        ▼
  User enters OTP on /verify-otp/
        │
        ▼
  ✅ Logged in → Dashboard
```

---

## 📁 Project Structure

```
├── apps/
│   └── customer/
│       ├── models.py          # Customer, PredictionReport, OTPCode, PasswordResetToken
│       ├── views.py           # All views + API endpoints
│       ├── urls.py            # URL routing
│       ├── admin.py           # Admin panel config
│       ├── resend_utils.py    # Email sending via Resend API
│       └── templates/         # HTML templates (Tailwind CSS)
├── churn_prediction/
│   ├── settings.py            # Production-hardened settings
│   └── urls.py
├── static/
│   └── assets/js/
│       ├── prediction.js      # ML prediction UI logic
│       ├── analysis.js        # Risk analysis charts
│       └── main.js
├── models/                    # Trained .joblib model files
├── train_models.py            # Standalone training script
├── deploy/                    # Gunicorn config
└── deploy.sh                  # One-command deployment script
```

---

## 🌐 Deployment (EC2)

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Deploy latest
sudo ./deploy.sh
```

`deploy.sh` does everything:
```
git pull → pip install → migrate → collectstatic → restart gunicorn
```

---

## 👥 Team

Built with 🔥 as a college capstone project.

| Name | Role |
|---|---|
| **Vishv** | Backend architecture, Django setup, ML model training (LR + RF + XGBoost) |
| **Anubhav Mohandas** | Security hardening, cloud deployment (EC2 + HTTPS), full refactor, 2FA + email integration, API fixes, DevOps |
| **Khushang** | Initial Streamlit prototype, documentation, project reporting |

---

## 📄 Dataset

**IBM Telco Customer Churn Dataset** — 7,043 customers, 21 features including contract type, tenure, monthly charges, internet service type, and more.

Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 📜 License

MIT License — feel free to fork, star ⭐, and build on top of this.

---

<div align="center">

**If this project impressed you, drop a ⭐ — it means a lot.**

Made with ❤️ and way too many `git push` attempts.

</div>
