import logging
import os
import pickle
import random
import json
import numpy as np
import pandas as pd
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models import Q, Avg, Sum
from django.http import JsonResponse, HttpResponse
from django.conf import settings

logger = logging.getLogger(__name__)

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from django.utils import timezone
from .models import Customer, PredictionReport, ReportHistory
import csv
from django.db.models import Avg, Sum, Count

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
            data = json.loads(request.body)
            email = data.get("email")
            password = data.get("password")
            user = authenticate(request, username=email, password=password)

            if user is not None:
                login(request, user)
                return JsonResponse({
                    "message": "Login successful",
                    "user_id": user.id,
                    "full_name": f"{user.first_name} {user.last_name}"
                }, status=200)
            return JsonResponse({"error": "Invalid email or password"}, status=400)
        except Exception:
            logger.exception("Login failed")
            return JsonResponse({"error": "Something went wrong. Please try again."}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)


# --- PAGE NAVIGATION ---

# def home_page(request):
#     return render(request, 'home.html')


@login_required(login_url='/accounts/login/')
def dashboard_page(request):
    user_reports = PredictionReport.objects.filter(user=request.user)

    # Existing logic...
    high_risk_count = user_reports.filter(risk_score__gte=0.7).count()
    mrr_at_risk = (user_reports.filter(risk_score__gte=0.7).aggregate(Sum('monthly_charges'))[
                       'monthly_charges__sum'] or 0) / 1000

    # Contract Stats with Pre-calculated Percentages
    raw_contract_stats = user_reports.values('contract_type').annotate(
        avg_risk=Avg('risk_score')
    )

    contract_stats = []
    for item in raw_contract_stats:
        risk_val = item['avg_risk'] or 0
        contract_stats.append({
            'contract_type': item['contract_type'],
            'risk_percent': round(risk_val * 100),
            # Determine color here to keep the template clean
            'color': 'red-500' if risk_val >= 0.7 else 'yellow-500' if risk_val >= 0.3 else 'emerald-500'
        })

    context = {
        'high_risk_count': high_risk_count,
        'mrr_at_risk': mrr_at_risk,
        'potential_recovery': mrr_at_risk * 0.2,
        'contract_stats': contract_stats,
        'priority_queue': user_reports.filter(risk_score__gte=0.7).order_by('-monthly_charges')[:5],
        'total_predictions': user_reports.count(),
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

            # 1. Model Selection
            model_filename = f'{requested_model}_model.pkl'
            model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
            if not os.path.exists(model_path):
                model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')
                final_model_name = 'random_forest'
            else:
                final_model_name = requested_model

            # 2. Load Assets
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

            # 3. Process Data
            df = pd.read_csv(file_obj)
            orig_df = df.copy()

            if 'customerID' in df.columns: df = df.drop(columns=['customerID'])
            if 'Churn' in df.columns: df = df.drop(columns=['Churn'])
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors='coerce').fillna(0)

            # ONE-HOT ENCODE the same categorical columns used during training
            cat_cols = meta.get("cat_cols", [])
            df_encoded = pd.get_dummies(
                df, columns=[c for c in cat_cols if c in df.columns], drop_first=False
            ).astype(float)

            # Align to training feature space: add missing cols as 0, drop extra cols
            df_encoded = df_encoded.reindex(columns=meta["feature_names"], fill_value=0)

            # Apply the same StandardScaler fitted during training
            num_cols = meta["numeric_cols"]
            df_encoded[num_cols] = meta["scaler"].transform(df_encoded[num_cols])

            # 4. Predict
            predictions = model.predict_proba(df_encoded)[:, 1]
            avg_risk = float(predictions.mean())
            high_risk_count = int((predictions > 0.7).sum())

            # 5. Save to LIVE PredictionReport (For Dashboard/Analysis)
            PredictionReport.objects.create(
                user=request.user,
                subscriber_id=f"BATCH-{random.randint(100, 999)}",
                source_file=file_obj.name,
                record_count=len(df),
                tenure=int(orig_df['tenure'].mean()) if 'tenure' in orig_df.columns else 0,
                monthly_charges=float(orig_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in orig_df.columns else 0.0,
                contract_type="Bulk Upload",
                risk_score=avg_risk,
                model_version=final_model_name
            )

            # 6. Save to PERMANENT ReportHistory (For History Page)
            ReportHistory.objects.create(
                user=request.user,
                batch_name=f"Batch_{timezone.now().strftime('%b_%d_%H%M')}",
                source_file=file_obj.name,
                total_records=len(df),
                avg_risk_score=avg_risk,
                critical_count=high_risk_count,
                model_version=final_model_name
            )

            # Log the bulk prediction run for audit trail
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
                "med_risk_count": int(((predictions <= 0.7) & (predictions > 0.3)).sum()),
                "low_risk_count": int((predictions <= 0.3).sum()),
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

            # Model Selection logic
            model_filename = f'{requested_model}_model.pkl'
            model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
            if not os.path.exists(model_path):
                model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')
                final_model_name = 'random_forest'
            else:
                final_model_name = requested_model

            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

            # Build feature vector: all zeros (absent / default category), then set known inputs
            input_dict = {col: 0.0 for col in meta["feature_names"]}
            input_dict['tenure'] = float(tenure)
            input_dict['MonthlyCharges'] = float(monthly_charges)
            input_dict['TotalCharges'] = 0.0  # not provided in single prediction

            # Set One-Hot contract flag (e.g. "Contract_Month-to-month")
            contract_col = f"Contract_{contract}"
            if contract_col in input_dict:
                input_dict[contract_col] = 1.0

            df_input = pd.DataFrame([input_dict])

            # Apply StandardScaler to numeric columns
            num_cols = meta["numeric_cols"]
            df_input[num_cols] = meta["scaler"].transform(df_input[num_cols])

            prob = float(model.predict_proba(df_input)[0, 1])

            # 1. Save to LIVE PredictionReport
            PredictionReport.objects.create(
                user=request.user,
                subscriber_id=f"SUB-{random.randint(1000, 9999)}",
                source_file="Manual Slider",
                record_count=1,
                tenure=int(tenure),
                monthly_charges=monthly_charges,
                contract_type=contract,
                risk_score=prob,
                model_version=final_model_name
            )

            # 2. Save to PERMANENT ReportHistory
            ReportHistory.objects.create(
                user=request.user,
                batch_name=f"Single_Calc_{timezone.now().strftime('%H%M')}",
                source_file="Manual Entry",
                total_records=1,
                avg_risk_score=prob,
                critical_count=1 if prob >= 0.7 else 0,
                model_version=final_model_name
            )

            return Response({"probability": prob, "status": "success"})
        except Exception:
            logger.exception("Single prediction failed")
            return Response({"error": "Prediction failed. Please try again."}, status=500)

# --- HISTORY VIEW ---

@login_required(login_url='/accounts/login/')
def report_history_page(request):
    query = request.GET.get('q', '').strip()

    # 1. Maintenance: Auto-delete records older than 30 days every time page is loaded
    ReportHistory.cleanup_old_reports()

    # 2. User isolation - Pull from the ARCHIVE, not the live predictions
    reports = ReportHistory.objects.filter(user=request.user)

    if query:
        reports = reports.filter(
            Q(batch_name__icontains=query) |
            Q(model_version__icontains=query) |
            Q(source_file__icontains=query)
        )

    # 3. Stats Aggregation (Now based on the frozen history)
    total_sims = reports.count()
    # Note: We pull totals directly from the history rows we saved earlier
    total_records = reports.aggregate(Sum('total_records'))['total_records__sum'] or 0
    avg_risk_raw = reports.aggregate(Avg('avg_risk_score'))['avg_risk_score__avg'] or 0
    critical_count = reports.aggregate(Sum('critical_count'))['critical_count__sum'] or 0

    context = {
        'reports': reports, # Ordered by -created_at in Meta
        'query': query,
        'total_sims': total_sims,
        'total_records': total_records,
        'avg_risk': round(avg_risk_raw * 100, 1),
        'critical_count': int(critical_count), # Convert from float sum to int
    }
    return render(request, 'reports.html', context)

@login_required(login_url='/accounts/login/')
def risk_analysis_page(request):
    # Base queryset for the logged-in user
    reports = PredictionReport.objects.filter(user=request.user)

    # 1. Financial Impact
    total_mrr = reports.aggregate(Sum('monthly_charges'))['monthly_charges__sum'] or 0
    revenue_at_risk = reports.filter(risk_score__gte=0.7).aggregate(Sum('monthly_charges'))['monthly_charges__sum'] or 0

    # 2. Contract Risk (Data for Bar Chart)
    # Get average risk per contract type
    contract_stats = reports.values('contract_type').annotate(avg_risk=Avg('risk_score'))
    contract_labels = [item['contract_type'] for item in contract_stats]
    contract_values = [round(item['avg_risk'] * 100, 1) for item in contract_stats]

    # 3. Correlation Data (Data for Bubble Chart)
    # We take a sample of the last 100 reports to plot
    correlation_data = []
    for r in reports.order_by('-created_at')[:100]:
        correlation_data.append({
            'x': r.tenure,
            'y': r.monthly_charges,
            'r': r.risk_score * 20  # Scaling radius by risk
        })

    context = {
        'total_mrr': round(total_mrr / 1000, 1),  # displayed as 'k'
        'revenue_at_risk': round(revenue_at_risk / 1000, 1),
        # Pass JSON strings to JavaScript
        'contract_labels_json': json.dumps(contract_labels),
        'contract_values_json': json.dumps(contract_values),
        'correlation_data_json': json.dumps(correlation_data),
    }
    return render(request, 'risk_analysis.html', context)


@login_required(login_url='/accounts/login/')
def export_risk_list(request):
    # 1. Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="critical_risk_list.csv"'

    writer = csv.writer(response)

    # 2. Write the Header Row
    writer.writerow(['Subscriber ID', 'Tenure', 'Monthly Charges', 'Contract', 'Risk Score', 'Model'])

    # 3. Fetch only Critical reports for the logged-in user (Risk > 70%)
    critical_reports = PredictionReport.objects.filter(
        user=request.user,
        risk_score__gte=0.7
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

    # Log the CSV export for audit trail
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
    # 1. Check if this user has actually processed any data
    has_data = PredictionReport.objects.filter(user=request.user).exists()

    if not has_data:
        # 2. Provide an "Empty State" dictionary
        empty_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "roc": [[0,0], [1,1]]
        }
        comparison_data = {
            "logistic_regression": empty_metrics,
            "random_forest":       empty_metrics,
            "xgboost":             empty_metrics,
        }
        status = "No Data"
    else:
        # Load real metrics from models/metrics.json (generated by train_models.py)
        # This file is updated automatically every time models are retrained.
        status = "Optimized"
        metrics_path = os.path.join(settings.BASE_DIR, 'models', 'metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                comparison_data = json.load(f)
        except FileNotFoundError:
            logger.warning("models/metrics.json not found — run train_models.py to generate it.")
            _empty = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                      "tn": 0, "fp": 0, "fn": 0, "tp": 0, "roc": [[0,0],[1,1]]}
            comparison_data = {
                "logistic_regression": _empty,
                "random_forest":       _empty,
                "xgboost":             _empty,
            }

    return render(request, 'models.html', {
        'comparison_json': json.dumps(comparison_data),
        'has_data': has_data,
        'model_status': status
    })


@login_required(login_url='/accounts/login/')
def settings_page(request):
    user = request.user

    if request.method == 'POST':
        # 1. Extract data from the POST request
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        risk_threshold = request.POST.get('risk_threshold')

        try:
            # 2. Update Personal Info
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.username = email  # Keeping username synced with email if needed

            # 3. Update AI Sensitivity
            # Check if risk_threshold exists before assigning
            if risk_threshold:
                user.risk_threshold = int(risk_threshold)

            user.save()
            messages.success(request, "Your profile and AI preferences have been updated.")
            return redirect('settings_page')

        except IntegrityError:
            # This triggers if the email already exists in the database
            messages.error(request, "This email address is already in use by another account.")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")

    # Pass the current user to the template (though request.user is available by default)
    return render(request, 'settings.html')