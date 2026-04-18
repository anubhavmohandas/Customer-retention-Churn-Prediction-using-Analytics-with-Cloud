from django.urls import path
from django.contrib.auth.views import LogoutView
from apps.customer.views import (
    signup,
    signup_page,
    login_page,
    dashboard_page,
    login_view,
    SinglePredictionView,
    BulkPredictionView,
    prediction_page,
    report_history_page,
    risk_analysis_page, export_risk_list, ai_models_page, settings_page,
    TrainCustomModelView
)

urlpatterns = [
    # --- Authentication ---
    path('signup/', signup_page, name='signup_page'),
    path('api/signup/', signup, name='signup_api'),
    path('accounts/login/', login_page, name='login'),  # Standard Django login path
    path('accounts/logout/', LogoutView.as_view(next_page='/accounts/login/'), name='logout'),
    path('api/login/', login_view, name='login_api'),

    # --- Dashboard & Core Pages ---
    path('', dashboard_page, name='dashboard_page'),  # Setting Dashboard as the default landing
    path('dashboard/', dashboard_page, name='dashboard_page'),  # Alternative path
    path('prediction/', prediction_page, name='prediction_page'),
    path('reports/', report_history_page, name='report_history'),
    path('risk-analysis/',risk_analysis_page, name='risk_analysis'),
    path('export-risk-list/', export_risk_list, name='export_risk_list'),
    path('ai-models/', ai_models_page, name='ai_models_page'),
    path('settings/', settings_page, name='settings_page'),


    # --- AI Prediction APIs ---
    path('api/predict-single/', SinglePredictionView.as_view(), name='predict_single'),
    path('api/predict-bulk/', BulkPredictionView.as_view(), name='predict_bulk'),
    path('api/train-models/', TrainCustomModelView.as_view(), name='train_models'),


]