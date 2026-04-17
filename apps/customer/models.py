from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from datetime import timedelta

# 1. The SaaS User
class Customer(AbstractUser):
    company = models.CharField(max_length=255, blank=True, null=True)
    ROLE_CHOICES = [
        ('founder_ceo', 'Founder / CEO'),
        ('product_manager', 'Product Manager'),
        ('growth_marketing', 'Growth / Marketing'),
        ('data_analyst', 'Data Analyst'),
        ('customer_success', 'Customer Success'),
        ('engineer', 'Engineer'),
        ('other', 'Other'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, blank=True, null=True)
    email = models.EmailField(unique=True)
    risk_threshold = models.IntegerField(default=70)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.company}"




# 2. Prediction Report (The source for your History Table)
class PredictionReport(models.Model):
    # Attribution: Linking to the Customer model
    # related_name='reports' allows you to do: request.user.reports.all()
    user = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='reports', null=True)

    # Identification
    subscriber_id = models.CharField(max_length=50, default="SIM-AUTO", db_index=True)

    # Execution Details
    source_file = models.CharField(max_length=255, default="Manual Entry")
    record_count = models.IntegerField(default=1)

    # Scenario Details
    tenure = models.IntegerField()
    monthly_charges = models.FloatField()
    contract_type = models.CharField(max_length=50)  # e.g., "Month-to-month"

    # AI Engine Details
    risk_score = models.FloatField()  # Stores as 0.8523
    model_version = models.CharField(max_length=50, default="RF_CORE_V1")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.subscriber_id} | {self.risk_level}"

    # --- UI HELPERS (Used directly in your Table Template) ---

    @property
    def risk_level(self):
        """Used for the CSS conditional classes in the HTML table"""
        if self.risk_score >= 0.7:
            return "Critical"
        elif self.risk_score >= 0.3:
            return "Warning"
        return "Stable"

    @property
    def risk_percent(self):
        """Converts decimal score (0.85) to percentage (85.0) for the UI"""
        try:
            return round(self.risk_score * 100, 1)
        except (TypeError, ValueError):
            return 0.0


class LoginHistory(models.Model):
    user = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='login_history')
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)
    session_key = models.CharField(max_length=40, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Login History'
        verbose_name_plural = 'Login History'

    def __str__(self):
        return f"{self.user.email} — {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"


class ReportHistory(models.Model):
    user = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='history_archives')

    # Store the core stats as a "Frozen" snapshot
    batch_name = models.CharField(max_length=100)  # e.g., "Batch_April_16"
    source_file = models.CharField(max_length=255)
    total_records = models.IntegerField()
    avg_risk_score = models.FloatField()
    critical_count = models.IntegerField()

    # Metadata
    model_version = models.CharField(max_length=50, default="RF_CORE_V1")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def days_remaining(self):
        """Calculates how many days left before this report is deleted"""
        expiry_date = self.created_at + timedelta(days=30)
        remaining = (expiry_date - timezone.now()).days
        return max(0, remaining)

    @staticmethod
    def cleanup_old_reports():
        """Call this in your view to delete anything older than 30 days"""
        thirty_days_ago = timezone.now() - timedelta(days=30)
        ReportHistory.objects.filter(created_at__lt=thirty_days_ago).delete()

