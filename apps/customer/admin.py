from django.contrib import admin
from apps.customer.models import Customer, PredictionReport, ReportHistory, LoginHistory, ActivityLog


# ── LoginHistory ─────────────────────────────────────────────────────────────

@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    list_display  = ('user_email', 'timestamp', 'ip_address', 'session_key', 'user_agent_short')
    list_filter   = ('timestamp', 'user')
    search_fields = ('user__email', 'ip_address', 'session_key')
    date_hierarchy = 'timestamp'
    ordering      = ('-timestamp',)
    readonly_fields = ('user', 'timestamp', 'ip_address', 'user_agent', 'session_key')

    def has_add_permission(self, request):    return False
    def has_change_permission(self, request, obj=None): return False
    def has_delete_permission(self, request, obj=None): return False

    @admin.display(description='User', ordering='user__email')
    def user_email(self, obj):
        return obj.user.email

    @admin.display(description='User Agent')
    def user_agent_short(self, obj):
        return obj.user_agent[:80] + '…' if len(obj.user_agent) > 80 else obj.user_agent


# ── ActivityLog ───────────────────────────────────────────────────────────────

@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display  = ('who', 'action', 'timestamp', 'ip_address', 'detail')
    list_filter   = ('action', 'timestamp')
    search_fields = ('user__email', 'attempted_email', 'ip_address', 'detail')
    date_hierarchy = 'timestamp'
    ordering      = ('-timestamp',)
    readonly_fields = ('user', 'timestamp', 'action', 'ip_address', 'detail', 'attempted_email')

    def has_add_permission(self, request):    return False
    def has_change_permission(self, request, obj=None): return False
    def has_delete_permission(self, request, obj=None): return False

    @admin.display(description='User / Attempted Email')
    def who(self, obj):
        if obj.user:
            return obj.user.email
        return obj.attempted_email or '—'


# ── Existing models ───────────────────────────────────────────────────────────

admin.site.register(Customer)
admin.site.register(PredictionReport)
admin.site.register(ReportHistory)
