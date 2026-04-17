from django.contrib import admin
from apps.customer.models import Customer, PredictionReport, ReportHistory

# Register your models here.

admin.site.register(Customer)
admin.site.register(PredictionReport)
admin.site.register(ReportHistory)