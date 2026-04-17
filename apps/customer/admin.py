from django.contrib import admin
from apps.customer.models import Customer, PredictionReport, ReportHistory, LoginHistory


# ── LoginHistory ─────────────────────────────────────────────────────────────

@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    list_display = ('user_email', 'timestamp', 'ip_address', 'session_key', 'user_agent_short')
    list_filter = ('timestamp', 'user')
    search_fields = ('user__email', 'ip_address', 'session_key')
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)
    readonly_fields = ('user', 'timestamp', 'ip_address', 'user_agent', 'session_key')

    # Disable all write operations — this is a forensic log
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description='User', ordering='user__email')
    def user_email(self, obj):
        return obj.user.email

    @admin.display(description='User Agent')
    def user_agent_short(self, obj):
        return obj.user_agent[:80] + '…' if len(obj.user_agent) > 80 else obj.user_agent


# ── Existing models ───────────────────────────────────────────────────────────

admin.site.register(Customer)
admin.site.register(PredictionReport)
admin.site.register(ReportHistory)
