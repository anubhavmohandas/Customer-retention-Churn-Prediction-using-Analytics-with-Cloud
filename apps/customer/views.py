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
from django.core.paginator import Paginator

logger = logging.getLogger(__name__)

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from django.utils import timezone
from django.core.cache import cache
from django.contrib.auth import update_session_auth_hash
from .models import Customer, PredictionReport, ReportHistory, ActivityLog
import csv

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
                login(request, user)
                return JsonResponse({
                    "message": "Login successful",
                    "user_id": user.id,
                    "full_name": f"{user.first_name} {user.last_name}"
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

    high_risk_count = user_reports.filter(risk_score__gte=0.7).count()
    mrr_at_risk = (user_reports.filter(risk_score__gte=0.7).aggregate(Sum('monthly_charges'))[
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
            'color': 'red-500' if risk_val >= 0.7 else 'yellow-500' if risk_val >= 0.3 else 'emerald-500'
        })

    context = {
        'high_risk_count': high_risk_count,
        'mrr_at_risk': mrr_at_risk,
        'potential_recovery': mrr_at_risk * 0.2,
        'contract_stats': contract_stats,
        'priority_queue': user_reports.filter(risk_score__gte=0.7).order_by('-monthly_charges')[:5],
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

            # 4. Predict (single model or ensemble average)
            if is_ensemble:
                all_probs = []
                for fname in model_map.values():
                    p = os.path.join(settings.BASE_DIR, 'models', fname)
                    if os.path.exists(p):
                        with open(p, 'rb') as f:
                            all_probs.append(pickle.load(f).predict_proba(df_encoded)[:, 1])
                predictions = np.mean(all_probs, axis=0) if all_probs else np.zeros(len(df_encoded))
                final_model_name = 'ensemble_stack'
            else:
                model_filename = model_map.get(requested_model, 'random_forest_model.pkl')
                model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
                if not os.path.exists(model_path):
                    model_path = os.path.join(settings.BASE_DIR, 'models', 'random_forest_model.pkl')
                    final_model_name = 'random_forest'
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                predictions = model.predict_proba(df_encoded)[:, 1]
            avg_risk = float(predictions.mean())
            high_risk_count = int((predictions > 0.7).sum())

            # 5. Save snapshot, then trim bulk records to last 10
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
            save_val = request.data.get('save_to_history', False)
            save_to_history = str(save_val).lower() == 'true' or save_val is True

            # Load metadata (scaler + feature names)
            with open(os.path.join(settings.BASE_DIR, 'models', 'metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)

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
                        with open(path, 'rb') as f:
                            m_obj = pickle.load(f)
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
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
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
                    critical_count=1 if prob >= 0.7 else 0,
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
    revenue_at_risk = reports.filter(risk_score__gte=0.7).aggregate(Sum('monthly_charges'))['monthly_charges__sum'] or 0

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
                user.risk_threshold = int(risk_threshold)

            user.save()
            messages.success(request, "Your profile and AI preferences have been updated.")
            return redirect('settings_page')

        except IntegrityError:
            messages.error(request, "This email address is already in use by another account.")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")

    return render(request, 'settings.html')
