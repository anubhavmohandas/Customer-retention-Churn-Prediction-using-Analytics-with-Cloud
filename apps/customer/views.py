import logging
import os
import joblib
import random
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import Q, Avg, Sum
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.paginator import Paginator

logger = logging.getLogger(__name__)

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from django.utils import timezone
from django.core.cache import cache
from django.contrib.auth import update_session_auth_hash
from .models import Customer, PredictionReport, ReportHistory, ActivityLog, OTPCode, PasswordResetToken
from .resend_utils import send_otp_email, send_password_reset_email
import csv
import secrets

# Risk threshold constants
RISK_HIGH = 0.7
RISK_MEDIUM = 0.3

# --- AUTH VIEWS ---

def signup_page(request):
    return render(request, 'signup.html')


def signup(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("work_email")
            password = data.get("password")

            if not email or not password:
                return JsonResponse({"error": "Email and password are required"}, status=400)

            if Customer.objects.filter(email=email).exists():
                return JsonResponse({"error": "Email already registered"}, status=400)

            try:
                validate_password(password)
            except ValidationError as ve:
                return JsonResponse({"error": list(ve.messages)}, status=400)

            user = Customer.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=data.get("first_name", ""),
                last_name=data.get("last_name", ""),
                company=data.get("company", ""),
                role=data.get("role", "")
            )
            return JsonResponse({"message": "Signup successful", "user_id": user.id}, status=201)
        except Exception:
            logger.exception("Signup failed")
            return JsonResponse({"error": "Something went wrong. Please try again."}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)


def login_page(request):
    return render(request, 'login.html')


def login_view(request):
    if request.method == "POST":
        try:
            # Rate limiting: 5 attempts per IP per 15 minutes
            ip = (request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
                  or request.META.get('REMOTE_ADDR', ''))
            cache_key = f"login_attempts_{ip}"
            attempts = cache.get(cache_key, 0)
            if attempts >= 5:
                return JsonResponse({"error": "Too many login attempts. Please wait 15 minutes."}, status=429)

            data = json.loads(request.body)
            email = data.get("email")
            password = data.get("password")
            user = authenticate(request, username=email, password=password)

            if user is not None:
                cache.delete(cache_key)  # Reset attempts on success

                # Generate 6-digit OTP and send via Resend
                otp_code = f"{secrets.randbelow(1000000):06d}"
                OTPCode.objects.create(user=user, code=otp_code)
                send_otp_email(user.email, otp_code, user.first_name)

                # Store pending user in session (not logged in yet)
                request.session['pending_2fa_user_id'] = user.id

                return JsonResponse({
                    "requires_otp": True,
                    "redirect": "/accounts/verify-otp/"
                }, status=200)

            cache.set(cache_key, attempts + 1, 900)  # 900s = 15 minutes
            return JsonResponse({"error": "Invalid email or password"}, status=400)
        except Exception:
            logger.exception("Login failed")
            return JsonResponse({"error": "Something went wrong. Please try again."}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)


# --- PAGE NAVIGATION ---

@login_required(login_url='/accounts/login/')
def dashboard_page(request):
    user_reports = PredictionReport.objects.filter(user=request.user)

    high_risk_count = user_reports.filter(risk_score__gte=RISK_HIGH).count()
    mrr_at_risk = (user_reports.filter(risk_score__gte=RISK_HIGH).aggregate(Sum('monthly_charges'))[
                       'monthly_charges__sum'] or 0) / 1000

    raw_contract_stats = user_reports.values('contract_type').annotate(
        avg_risk=Avg('risk_score')
    )

    contract_stats = []
    for item in raw_contract_stats:
        risk_val = item['avg_risk'] or 0
        contract_stats.append({
            'contract_type': item['contract_type'],
            'risk_percent': round(risk_val * 100),
            'color': 'red-500' if risk_val >= RISK_HIGH else 'yellow-500' if risk_val >= RISK_MEDIUM else 'emerald-500'
        })

    context = {
        'high_risk_count': high_risk_count,
        'mrr_at_risk': mrr_at_risk,
        'potential_recovery': mrr_at_risk * 0.2,
        'contract_stats': contract_stats,
        'priority_queue': user_reports.filter(risk_score__gte=RISK_HIGH).order_by('-monthly_charges')[:5],
        'total_predictions': user_reports.aggregate(t=Sum('record_count'))['t'] or 0,
    }
    return render(request, 'dashboard.html', context)


@login_required(login_url='/accounts/login/')
def prediction_page(request):
    return render(request, 'prediction.html')


# --- PREDICTION ENGINE (API) ---

class BulkPredictionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            file_obj = request.FILES.get('file')
            requested_model = request.data.get('model', 'random_forest')

            if not file_obj:
                return Response({"error": "No file uploaded"}, status=400)

            MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
            if file_obj.size > MAX_UPLOAD_BYTES:
                return Response(
                    {"error": f"File too large ({file_obj.size // (1024*1024)} MB). Maximum allowed is 10 MB."},
                    status=400
                )

            # 1. Model Selection
            model_map = {
                'random_forest':        'random_forest_model.pkl',
                'xgboost':              'xgboost_model.pkl',
                'logistic_regression':  'logistic_regression_model.pkl',
            }
            is_ensemble = (requested_model == 'ensemble_stack')
            final_model_name = requested_model

            # 2. Load Assets
            # (model loaded later — either single or ensemble)
            meta = joblib.load(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'))

            # 3. Process Data
            df = pd.read_csv(file_obj)
            orig_df = df.copy()

            if 'customerID' in df.columns: df = df.drop(columns=['customerID'])
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors='coerce').fillna(0)

            scaler = meta["scaler"]
            num_cols = meta["numeric_cols"]
            feature_names = meta["feature_names"]

            # Auto-detect categorical columns from the uploaded CSV so any
            # standard telecom churn CSV works — not just the training file.
            categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            # Remove label/target columns if present (won't be in feature_names anyway)
            for drop_col in ["Churn", "churn"]:
                if drop_col in categorical_cols:
                    categorical_cols.remove(drop_col)

            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

            # Align to training feature space: fill missing cols with 0, drop extras
            for col in feature_names:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0.0
            df_encoded = df_encoded[feature_names].astype(float)

            # Scale numeric columns with the fitted StandardScaler
            df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

            # 4. Predict (single model or ensemble average)
            if is_ensemble:
                all_probs = []
                for fname in model_map.values():
                    p = os.path.join(settings.BASE_DIR, 'models', fname)
                    if os.path.exists(p):
                        all_probs.append(joblib.load(p).predict_proba(df_encoded)[:, 1])
                predictions = np.mean(all_probs, axis=0) if all_probs else np.zeros(len(df_encoded))
                final_model_name = 'ensemble_stack'
            else:
                model_filename = model_map.get(requested_model, 'random_forest_model.pkl')
                model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
                if not os.path.exists(model_path):
                    model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')
                    final_model_name = 'random_forest'
                model = joblib.load(model_path)
                predictions = model.predict_proba(df_encoded)[:, 1]
            avg_risk = float(predictions.mean())
            high_risk_count = int((predictions > RISK_HIGH).sum())

            # 5. Save snapshot, then trim bulk records to last 10
            # Handle NaN protection for tenure and monthly_charges
            tenure_val = orig_df['tenure'].mean() if 'tenure' in orig_df.columns else 0
            tenure_val = 0 if math.isnan(tenure_val) else int(tenure_val)
            charges_val = orig_df['MonthlyCharges'].mean() if 'MonthlyCharges' in orig_df.columns else 0.0
            charges_val = 0.0 if math.isnan(charges_val) else float(charges_val)

            with transaction.atomic():
                PredictionReport.objects.create(
                    user=request.user,
                    subscriber_id=f"BATCH-{random.randint(100, 999)}",
                    source_file=file_obj.name,
                    record_count=len(df),
                    tenure=tenure_val,
                    monthly_charges=charges_val,
                    contract_type="Bulk Upload",
                    risk_score=avg_risk,
                    model_version=final_model_name
                )

                # Trim bulk PredictionReport records to last 10 (keep Single Analysis untouched)
                bulk_qs = PredictionReport.objects.filter(
                    user=request.user
                ).exclude(source_file="Single Analysis").order_by('-created_at')
                excess_ids = list(bulk_qs.values_list('id', flat=True)[10:])
                if excess_ids:
                    PredictionReport.objects.filter(id__in=excess_ids).delete()

                # 6. Save to permanent ReportHistory
                ReportHistory.objects.create(
                    user=request.user,
                    batch_name=f"Batch_{timezone.now().strftime('%b_%d_%H%M')}",
                    source_file=file_obj.name,
                    total_records=len(df),
                    avg_risk_score=avg_risk,
                    critical_count=high_risk_count,
                    model_version=final_model_name
                )

                # 7. Audit log
                try:
                    from apps.customer.models import ActivityLog
                    ActivityLog.objects.create(
                        user=request.user,
                        action='BULK_PREDICT',
                        ip_address=request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
                                    or request.META.get('REMOTE_ADDR'),
                        detail=f"{len(predictions)} records processed from '{file_obj.name}' using {final_model_name}",
                    )
                except Exception:
                    logger.exception("Failed to log bulk prediction activity")

            return Response({
                "avg_risk": round(avg_risk * 100, 2),
                "total_processed": len(predictions),
                "high_risk_count": high_risk_count,
                "med_risk_count": int(((predictions <= RISK_HIGH) & (predictions > RISK_MEDIUM)).sum()),
                "low_risk_count": int((predictions <= RISK_MEDIUM).sum()),
            })
        except Exception:
            logger.exception("Bulk prediction failed")
            return Response({"error": "Prediction failed. Please check your file and try again."}, status=500)


class SinglePredictionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            tenure = float(request.data.get('tenure', 0))
            monthly_charges = float(request.data.get('monthly_charges', 0))
            contract = request.data.get('contract', 'Month-to-month')
            requested_model = request.data.get('model', 'random_forest')
            save_val = request.data.get('save_to_history', False)
            save_to_history = str(save_val).lower() == 'true' or save_val is True

            # Load metadata (scaler + feature names)
            meta = joblib.load(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'))

            scaler = meta["scaler"]
            numeric_cols = meta["numeric_cols"]
            feature_names = meta["feature_names"]

            # Build feature vector — all zeros, then set known inputs
            input_dict = {feat: 0.0 for feat in feature_names}
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = tenure * monthly_charges  # reasonable estimate

            # Set One-Hot contract flag
            contract_col = f"Contract_{contract}"
            if contract_col in input_dict:
                input_dict[contract_col] = 1.0

            df_input = pd.DataFrame([input_dict])[feature_names]
            df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

            # Ensemble: average all 3 models, or run single model
            if requested_model == "ensemble_stack":
                model_files = [
                    'random_forest_model.pkl',
                    'xgboost_model.pkl',
                    'logistic_regression_model.pkl',
                ]
                probs = []
                for m_file in model_files:
                    path = os.path.join(settings.BASE_DIR, 'models', m_file)
                    if os.path.exists(path):
                        m_obj = joblib.load(path)
                        probs.append(float(m_obj.predict_proba(df_input)[0, 1]))
                prob = sum(probs) / len(probs) if probs else 0.0
                final_model_name = "Ensemble Stack"
            else:
                model_map = {
                    'random_forest':        'random_forest_model.pkl',
                    'xgboost':              'xgboost_model.pkl',
                    'logistic_regression':  'logistic_regression_model.pkl',
                }
                model_filename = model_map.get(requested_model, 'random_forest_model.pkl')
                model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
                if not os.path.exists(model_path):
                    model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')
                model_obj = joblib.load(model_path)
                prob = float(model_obj.predict_proba(df_input)[0, 1])
                final_model_name = requested_model.replace('_', ' ').title()

            # Dynamic confidence from metrics.json
            confidence_score = round(max(prob, 1 - prob) * 100, 1)
            try:
                metrics_path = os.path.join(settings.BASE_DIR, 'models', 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    if requested_model == "ensemble_stack":
                        accs = [m.get('accuracy', 85.0) for m in metrics.values()]
                        confidence_score = round(sum(accs) / len(accs), 1) if accs else confidence_score
                    else:
                        model_acc = metrics.get(requested_model, {}).get('accuracy')
                        if model_acc:
                            confidence_score = model_acc
            except Exception:
                logger.exception("Could not load metrics for confidence score")

            # Save live prediction (replace previous single analysis)
            with transaction.atomic():
                PredictionReport.objects.filter(
                    user=request.user, source_file="Single Analysis"
                ).delete()
                PredictionReport.objects.create(
                    user=request.user,
                    subscriber_id=f"SUB-{random.randint(1000, 9999)}",
                    source_file="Single Analysis",
                    record_count=1,
                    tenure=int(tenure),
                    monthly_charges=monthly_charges,
                    contract_type=contract,
                    risk_score=prob,
                    model_version=final_model_name
                )

                # Optionally save to permanent ReportHistory
                if save_to_history:
                    ReportHistory.objects.create(
                        user=request.user,
                        batch_name=f"Manual_Run_{timezone.now().strftime('%H%M%S')}",
                        source_file="Manual Entry",
                        total_records=1,
                        avg_risk_score=prob,
                        critical_count=1 if prob >= RISK_HIGH else 0,
                        model_version=final_model_name
                    )

            return Response({
                "probability": prob,
                "saved": save_to_history,
                "model_used": final_model_name,
                "confidence": confidence_score,
            })
        except Exception:
            logger.exception("Single prediction failed")
            return Response({"error": "Prediction failed. Please try again."}, status=500)


# --- HISTORY VIEW ---

@login_required(login_url='/accounts/login/')
def report_history_page(request):
    query = request.GET.get('q', '').strip()

    # Auto-delete records older than 30 days — throttled to once per day per user
    cleanup_key = f"cleanup_done_{request.user.id}"
    if not cache.get(cleanup_key):
        ReportHistory.cleanup_old_reports()
        cache.set(cleanup_key, True, 86400)  # 24 hours

    reports = ReportHistory.objects.filter(user=request.user).order_by('-created_at')

    if query:
        reports = reports.filter(
            Q(batch_name__icontains=query) |
            Q(model_version__icontains=query) |
            Q(source_file__icontains=query)
        )

    # Aggregate stats over full queryset before paginating
    stats = reports.aggregate(
        total_rec=Sum('total_records'),
        avg_risk=Avg('avg_risk_score'),
        crit_count=Sum('critical_count')
    )

    # Annotate display helpers
    annotated = []
    for report in reports:
        report.display_risk = (report.avg_risk_score or 0) * 100
        elapsed = (timezone.now() - report.created_at).days if report.created_at else 0
        report.expiry_days = max(0, 30 - elapsed)
        annotated.append(report)

    # Paginate — 30 records per page
    paginator = Paginator(annotated, 30)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'reports': page_obj,
        'page_obj': page_obj,
        'query': query,
        'total_sims': reports.count(),
        'total_records': stats['total_rec'] or 0,
        'avg_risk': round((stats['avg_risk'] or 0) * 100, 1),
        'critical_count': int(stats['crit_count'] or 0),
    }
    return render(request, 'reports.html', context)


@login_required(login_url='/accounts/login/')
def risk_analysis_page(request):
    reports = PredictionReport.objects.filter(user=request.user)

    total_mrr = reports.aggregate(Sum('monthly_charges'))['monthly_charges__sum'] or 0
    revenue_at_risk = reports.filter(risk_score__gte=RISK_HIGH).aggregate(Sum('monthly_charges'))['monthly_charges__sum'] or 0

    contract_stats = reports.values('contract_type').annotate(avg_risk=Avg('risk_score'))
    contract_labels = [item['contract_type'] for item in contract_stats]
    contract_values = [round(item['avg_risk'] * 100, 1) for item in contract_stats]

    correlation_data = []
    for r in reports.order_by('-created_at')[:100]:
        correlation_data.append({
            'x': r.tenure,
            'y': r.monthly_charges,
            'r': r.risk_score * 20
        })

    context = {
        'total_mrr': round(total_mrr / 1000, 1),
        'revenue_at_risk': round(revenue_at_risk / 1000, 1),
        'contract_labels_json': json.dumps(contract_labels),
        'contract_values_json': json.dumps(contract_values),
        'correlation_data_json': json.dumps(correlation_data),
    }
    return render(request, 'risk_analysis.html', context)


@login_required(login_url='/accounts/login/')
def export_risk_list(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="critical_risk_list.csv"'

    writer = csv.writer(response)
    writer.writerow(['Subscriber ID', 'Tenure', 'Monthly Charges', 'Contract', 'Risk Score', 'Model'])

    critical_reports = PredictionReport.objects.filter(
        user=request.user,
        risk_score__gte=RISK_HIGH
    )

    for report in critical_reports:
        writer.writerow([
            report.subscriber_id,
            report.tenure,
            report.monthly_charges,
            report.contract_type,
            f"{report.risk_percent}%",
            report.model_version
        ])

    # Audit log
    try:
        from apps.customer.models import ActivityLog
        ActivityLog.objects.create(
            user=request.user,
            action='CSV_EXPORT',
            ip_address=request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
                        or request.META.get('REMOTE_ADDR'),
            detail=f"{critical_reports.count()} critical-risk records exported as critical_risk_list.csv",
        )
    except Exception:
        logger.exception("Failed to log CSV export activity")

    return response


@login_required(login_url='/accounts/login/')
def ai_models_page(request):
    # Model metrics are training-time results — always show them regardless of user activity
    empty_metrics = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        "roc": [[0, 0], [1, 1]]
    }

    metrics_path = os.path.join(settings.BASE_DIR, 'models', 'metrics.json')
    comparison_data = {}

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                real_metrics = json.load(f)
            for model_key in ["logistic_regression", "random_forest", "xgboost"]:
                md = real_metrics.get(model_key, {})
                comparison_data[model_key] = {
                    "accuracy":  md.get("accuracy",  0.0),
                    "precision": md.get("precision", 0.0),
                    "recall":    md.get("recall",    0.0),
                    "f1":        md.get("f1",        0.0),
                    "tn":        md.get("tn",        0),
                    "fp":        md.get("fp",        0),
                    "fn":        md.get("fn",        0),
                    "tp":        md.get("tp",        0),
                    "roc":       md.get("roc",       [[0, 0], [1, 1]]),
                }
            status = "Optimized"
        except Exception:
            logger.exception("Error parsing metrics.json")
            comparison_data = {m: empty_metrics for m in ["logistic_regression", "random_forest", "xgboost"]}
            status = "Error"
    else:
        logger.warning("models/metrics.json not found — run train_models.py to generate it.")
        comparison_data = {m: empty_metrics for m in ["logistic_regression", "random_forest", "xgboost"]}
        status = "Not Trained"

    has_data = PredictionReport.objects.filter(user=request.user).exists()

    return render(request, 'models.html', {
        'comparison_json': json.dumps(comparison_data),
        'has_data': has_data,
        'model_status': status
    })


@login_required(login_url='/accounts/login/')
def settings_page(request):
    user = request.user

    if request.method == 'POST':
        action = request.POST.get('action', 'profile')

        if action == 'password':
            current_password = request.POST.get('current_password', '')
            new_password = request.POST.get('new_password', '')
            confirm_password = request.POST.get('confirm_password', '')

            if not user.check_password(current_password):
                messages.error(request, "Current password is incorrect.")
            elif new_password != confirm_password:
                messages.error(request, "New passwords do not match.")
            else:
                try:
                    validate_password(new_password, user)
                    user.set_password(new_password)
                    user.save()
                    update_session_auth_hash(request, user)  # Keep session alive after pw change
                    messages.success(request, "Password updated successfully.")
                except ValidationError as ve:
                    for err in ve.messages:
                        messages.error(request, err)
            return redirect('settings_page')

        # Default: profile update
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        risk_threshold = request.POST.get('risk_threshold')

        try:
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.username = email

            if risk_threshold:
                threshold_val = int(risk_threshold)
                if not 0 <= threshold_val <= 100:
                    messages.error(request, "Risk threshold must be between 0 and 100.")
                    return redirect('settings_page')
                user.risk_threshold = threshold_val

            user.save()
            messages.success(request, "Your profile and AI preferences have been updated.")
            return redirect('settings_page')

        except IntegrityError:
            messages.error(request, "This email address is already in use by another account.")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")

    return render(request, 'settings.html')


# ==========================================
# AUTO-TRAIN VIEW
# ==========================================

class TrainCustomModelView(APIView):
    """
    POST /api/train-models/
    Accepts a CSV with a 'Churn' column. Auto-detects all features,
    trains LR + RF + XGB with SMOTE balancing, saves model artifacts
    and updates metrics.json. Training takes 10-30 seconds.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Rate limiting: max 3 training requests per user per hour
        cache_key = f"train_attempts_{request.user.id}"
        attempts = cache.get(cache_key, 0)
        if attempts >= 3:
            return Response({"error": "Training limit reached. Please wait before retraining."}, status=429)

        try:
            file_obj = request.FILES.get('file')
            if not file_obj:
                return Response({"error": "No file uploaded"}, status=400)

            df = pd.read_csv(file_obj)

            if 'Churn' not in df.columns:
                return Response({"error": "Dataset must contain a 'Churn' column."}, status=400)

            df['Churn'] = (
                df['Churn'].astype(str).str.lower()
                .map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
                .fillna(0).astype(int)
            )

            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])

            X = df.drop(columns=["Churn"])
            y = df["Churn"]

            cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
            num_cols = [c for c in X.columns if c not in cat_cols]

            for col in num_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(float)
            feature_names = X_encoded.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            if num_cols:
                scaled_num = [c for c in num_cols if c in X_train.columns]
                if scaled_num:
                    X_train[scaled_num] = scaler.fit_transform(X_train[scaled_num])
                    X_test[scaled_num] = scaler.transform(X_test[scaled_num])

            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

            models_to_train = {
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
            }

            MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')
            os.makedirs(MODELS_DIR, exist_ok=True)
            json_metrics = {}

            for name, clf in models_to_train.items():
                clf.fit(X_train_sm, y_train_sm)

                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1]

                acc   = accuracy_score(y_test, y_pred)
                prec  = precision_score(y_test, y_pred, zero_division=0)
                rec   = recall_score(y_test, y_pred, zero_division=0)
                f1    = f1_score(y_test, y_pred, zero_division=0)

                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                idx = np.linspace(0, len(fpr) - 1, 6, dtype=int)
                roc_points = [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in idx]

                joblib.dump(clf, os.path.join(MODELS_DIR, f"{name}_model.pkl"))

                json_metrics[name] = {
                    "accuracy":  round(acc * 100, 1),
                    "precision": round(prec * 100, 1),
                    "recall":    round(rec * 100, 1),
                    "f1":        round(f1 * 100, 1),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                    "roc": roc_points,
                }

            with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
                json.dump(json_metrics, f, indent=2)

            joblib.dump({
                "scaler": scaler,
                "numeric_cols": num_cols,
                "feature_names": feature_names,
            }, os.path.join(MODELS_DIR, "metadata.pkl"))

            try:
                ActivityLog.objects.create(
                    user=request.user,
                    action='MODEL_RETRAIN',
                    detail=f"Retrained LR + RF + XGB on {len(df):,} rows from {file_obj.name}",
                )
            except Exception:
                pass

            # Increment rate limit counter on successful training
            cache.set(cache_key, attempts + 1, 3600)  # 3600 seconds = 1 hour

            return Response({"message": "AutoML Training Complete!", "metrics": json_metrics})

        except Exception as e:
            logger.exception("Auto-train failed")
            return Response({"error": str(e)}, status=500)


# ==========================================
# 2FA OTP VERIFY
# ==========================================

def otp_verify_page(request):
    if request.method == 'GET':
        if 'pending_2fa_user_id' not in request.session:
            return redirect('login')
        return render(request, 'otp_verify.html')

    if request.method == 'POST':
        user_id = request.session.get('pending_2fa_user_id')
        if not user_id:
            return JsonResponse({"error": "Session expired. Please login again."}, status=400)

        entered_code = request.POST.get('otp', '').strip()

        try:
            user = Customer.objects.get(id=user_id)
        except Customer.DoesNotExist:
            return JsonResponse({"error": "Invalid session."}, status=400)

        # Find latest valid OTP for this user
        otp_obj = OTPCode.objects.filter(user=user, is_used=False).order_by('-created_at').first()

        if not otp_obj or not otp_obj.is_valid():
            return JsonResponse({"error": "OTP expired. Please login again to get a new code."}, status=400)

        if otp_obj.code != entered_code:
            return JsonResponse({"error": "Incorrect code. Please try again."}, status=400)

        # Valid — mark used, log the user in
        otp_obj.is_used = True
        otp_obj.save()
        del request.session['pending_2fa_user_id']

        login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        return JsonResponse({"message": "Login successful", "redirect": "/dashboard/"}, status=200)


def resend_otp(request):
    """Resend a fresh OTP to the pending user."""
    user_id = request.session.get('pending_2fa_user_id')
    if not user_id:
        return JsonResponse({"error": "Session expired."}, status=400)

    try:
        user = Customer.objects.get(id=user_id)
    except Customer.DoesNotExist:
        return JsonResponse({"error": "Invalid session."}, status=400)

    otp_code = f"{secrets.randbelow(1000000):06d}"
    OTPCode.objects.create(user=user, code=otp_code)
    send_otp_email(user.email, otp_code, user.first_name)
    return JsonResponse({"message": "New code sent."}, status=200)


# ==========================================
# PASSWORD RESET
# ==========================================

def password_reset_request(request):
    if request.method == 'GET':
        return render(request, 'password_reset_request.html')

    email = request.POST.get('email', '').strip().lower()
    try:
        user = Customer.objects.get(email=email)
        token_obj = PasswordResetToken.objects.create(user=user)
        reset_url = request.build_absolute_uri(f"/accounts/password-reset/confirm/{token_obj.token}/")
        send_password_reset_email(user.email, reset_url, user.first_name)
    except Customer.DoesNotExist:
        pass  # Don't reveal whether email exists

    # Always show the same page to prevent email enumeration
    return render(request, 'password_reset_sent.html', {'email': email})


def password_reset_confirm(request, token):
    try:
        token_obj = PasswordResetToken.objects.get(token=token)
    except PasswordResetToken.DoesNotExist:
        return render(request, 'password_reset_confirm.html', {'error': 'Invalid or expired link.'})

    if not token_obj.is_valid():
        return render(request, 'password_reset_confirm.html', {'error': 'This link has expired. Please request a new one.'})

    if request.method == 'GET':
        return render(request, 'password_reset_confirm.html', {'token': token})

    new_password = request.POST.get('new_password', '')
    confirm_password = request.POST.get('confirm_password', '')

    if new_password != confirm_password:
        return render(request, 'password_reset_confirm.html', {'token': token, 'error': 'Passwords do not match.'})

    try:
        from django.contrib.auth.password_validation import validate_password
        validate_password(new_password, token_obj.user)
    except ValidationError as e:
        return render(request, 'password_reset_confirm.html', {'token': token, 'error': ' '.join(e.messages)})

    token_obj.user.set_password(new_password)
    token_obj.user.save()
    token_obj.is_used = True
    token_obj.save()

    return render(request, 'password_reset_confirm.html', {'success': True})
