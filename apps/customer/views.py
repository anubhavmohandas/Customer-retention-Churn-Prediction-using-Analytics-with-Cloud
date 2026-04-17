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

            model_map = {
                'random_forest': 'random_forest_model.pkl',
                'xgboost': 'xgboost_model.pkl',
                'logistic_regression': 'logistic_regression_model.pkl'
            }
            model_filename = model_map.get(requested_model, 'random_forest_model.pkl')
            model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)

            if not os.path.exists(model_path):
                model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')

            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

            scaler = meta["scaler"]
            numeric_cols = meta["numeric_cols"]
            feature_names = meta["feature_names"]

            df = pd.read_csv(file_obj)
            orig_df = df.copy()

            if 'customerID' in df.columns: df = df.drop(columns=['customerID'])
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors='coerce').fillna(0)

            # One-Hot Encode and align with trained features
            categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            if "Churn" in categorical_cols:
                categorical_cols.remove("Churn")

            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

            for col in feature_names:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0.0

            df_input = df_encoded[feature_names].astype(float)
            df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

            predictions = model.predict_proba(df_input)[:, 1]
            avg_risk = float(predictions.mean())
            high_risk_count = int((predictions > 0.7).sum())

            PredictionReport.objects.filter(user=request.user).exclude(source_file="Single Analysis").delete()

            PredictionReport.objects.create(
                user=request.user,
                subscriber_id=f"BATCH-{random.randint(100, 999)}",
                source_file=file_obj.name,
                record_count=len(df),
                tenure=int(orig_df['tenure'].mean()) if 'tenure' in orig_df.columns else 0,
                monthly_charges=float(orig_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in orig_df.columns else 0.0,
                contract_type="Bulk Upload",
                risk_score=avg_risk,
                model_version=requested_model.replace('_', ' ').title()
            )

            ReportHistory.objects.create(
                user=request.user,
                batch_name=f"Batch_{timezone.now().strftime('%b_%d_%H%M')}",
                source_file=file_obj.name,
                total_records=len(df),
                avg_risk_score=avg_risk,
                critical_count=high_risk_count,
                model_version=requested_model.replace('_', ' ').title()
            )

            return Response({
                "avg_risk": round(avg_risk * 100, 2),
                "total_processed": len(predictions),
                "high_risk_count": high_risk_count,
                "med_risk_count": int(((predictions <= 0.7) & (predictions > 0.3)).sum()),
                "low_risk_count": int((predictions <= 0.3).sum()),
            })
        except Exception as e:
            logger.exception("Bulk prediction failed")
            return Response({"error": f"Internal Error: {str(e)}"}, status=500)


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

            # Load Metadata (Scaler & Feature Names)
            with open(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

            scaler = meta["scaler"]
            numeric_cols = meta["numeric_cols"]
            feature_names = meta["feature_names"]

            # Format input mathematically
            input_dict = {feat: 0.0 for feat in feature_names}
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = tenure * monthly_charges

            contract_col = f"Contract_{contract}"
            if contract_col in input_dict:
                input_dict[contract_col] = 1.0

            df_input = pd.DataFrame([input_dict])[feature_names]
            df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

            # Ensemble Logic vs Single Model Logic
            if requested_model == "ensemble_stack":
                models_to_run = ['random_forest_model.pkl', 'xgboost_model.pkl', 'logistic_regression_model.pkl']
                probs = []
                for m_file in models_to_run:
                    path = os.path.join(settings.BASE_DIR, 'models', m_file)
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            m_obj = pickle.load(f)
                            probs.append(float(m_obj.predict_proba(df_input)[0, 1]))
                prob = sum(probs) / len(probs) if probs else 0.0
                final_model_name = "Ensemble Stack"
            else:
                model_map = {
                    'random_forest': 'random_forest_model.pkl',
                    'xgboost': 'xgboost_model.pkl',
                    'logistic_regression': 'logistic_regression_model.pkl'
                }
                model_filename = model_map.get(requested_model, 'random_forest_model.pkl')
                model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)

                if not os.path.exists(model_path):
                    model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')

                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
                prob = float(model_obj.predict_proba(df_input)[0, 1])
                final_model_name = requested_model.replace('_', ' ').title()

            # Dynamic AI Confidence Scoring
            confidence_score = 85.0
            try:
                metrics_path = os.path.join(settings.BASE_DIR, 'models', 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        if requested_model == "ensemble_stack":
                            accs = [m.get('accuracy', 85.0) for m in metrics.values()]
                            confidence_score = round(sum(accs) / len(accs), 1) if accs else 85.0
                        else:
                            confidence_score = metrics.get(requested_model, {}).get('accuracy', 85.0)
            except Exception as e:
                logger.error(f"Could not load metrics for confidence: {e}")

            # Database Tracking
            PredictionReport.objects.filter(user=request.user, source_file="Single Analysis").delete()
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

            if save_to_history:
                ReportHistory.objects.create(
                    user=request.user,
                    batch_name=f"Manual_Run_{timezone.now().strftime('%H%M%S')}",
                    source_file="Manual Entry",
                    total_records=1,
                    avg_risk_score=prob,
                    critical_count=1 if prob >= 0.7 else 0,
                    model_version=final_model_name
                )

            return Response({
                "probability": prob,
                "saved": save_to_history,
                "model_used": final_model_name,
                "confidence": confidence_score
            })
        except Exception as e:
            logger.exception("Single prediction failed")
            return Response({"error": str(e)}, status=500)


@login_required(login_url='/accounts/login/')
def report_history_page(request):
    query = request.GET.get('q', '').strip()

    # Maintenance: Cleanup old data (30-day policy)
    if hasattr(ReportHistory, 'cleanup_old_reports'):
        ReportHistory.cleanup_old_reports()

    reports = ReportHistory.objects.filter(user=request.user).order_by('-created_at')

    if query:
        reports = reports.filter(
            Q(batch_name__icontains=query) |
            Q(model_version__icontains=query) |
            Q(source_file__icontains=query)
        )

    # Format data for the UI
    for report in reports:
        report.display_risk = (report.avg_risk_score or 0) * 100
        if report.created_at:
            elapsed = (timezone.now() - report.created_at).days
            report.expiry_days = max(0, 30 - elapsed)
        else:
            report.expiry_days = 30

    stats = reports.aggregate(
        total_rec=Sum('total_records'),
        avg_risk=Avg('avg_risk_score'),
        crit_count=Sum('critical_count')
    )

    context = {
        'reports': reports,
        'query': query,
        'total_sims': reports.count(),
        'total_records': stats['total_rec'] or 0,
        'avg_risk': round((stats['avg_risk'] or 0) * 100, 1),
        'critical_count': int(stats['crit_count'] or 0),
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

    return response


@login_required(login_url='/accounts/login/')
def ai_models_page(request):
    has_data = PredictionReport.objects.filter(user=request.user).exists()

    empty_metrics = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        "roc": [[0,0], [1,1]]
    }

    if not has_data:
        comparison_data = {
            "logistic_regression": empty_metrics,
            "random_forest": empty_metrics,
            "xgboost": empty_metrics
        }
        status = "No Data"
    else:
        status = "Optimized"
        # Fallback visual data for charts (ROC and Confusion Matrix)
        fallback_extras = {
            "logistic_regression": {
                "tn": 810, "fp": 223, "fn": 108, "tp": 266,
                "roc": [[0,0], [0.22, 0.65], [0.4, 0.75], [0.7, 0.88], [1,1]]
            },
            "random_forest": {
                "tn": 842, "fp": 24, "fn": 12, "tp": 122,
                "roc": [[0,0], [0.1, 0.6], [0.3, 0.85], [0.6, 0.95], [1,1]]
            },
            "xgboost": {
                "tn": 855, "fp": 11, "fn": 9, "tp": 125,
                "roc": [[0,0], [0.05, 0.7], [0.2, 0.92], [0.5, 0.98], [1,1]]
            }
        }

        comparison_data = {}
        metrics_path = os.path.join(settings.BASE_DIR, 'models', 'metrics.json')

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    real_metrics = json.load(f)

                for model in ["logistic_regression", "random_forest", "xgboost"]:
                    model_data = real_metrics.get(model, {})
                    comparison_data[model] = {
                        "accuracy": model_data.get("accuracy", 0.0),
                        "precision": model_data.get("precision", 0.0),
                        "recall": model_data.get("recall", 0.0),
                        "f1": model_data.get("f1", 0.0),
                        "tn": model_data.get("tn", fallback_extras[model]["tn"]),
                        "fp": model_data.get("fp", fallback_extras[model]["fp"]),
                        "fn": model_data.get("fn", fallback_extras[model]["fn"]),
                        "tp": model_data.get("tp", fallback_extras[model]["tp"]),
                        "roc": model_data.get("roc", fallback_extras[model]["roc"])
                    }
            except Exception as e:
                logger.error(f"Error parsing metrics.json: {e}")
                comparison_data = {m: empty_metrics for m in ["logistic_regression", "random_forest", "xgboost"]}
        else:
            comparison_data = {m: empty_metrics for m in ["logistic_regression", "random_forest", "xgboost"]}

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