import logging
from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver

logger = logging.getLogger(__name__)


def _get_client_ip(request):
    """Extract real client IP, respecting X-Forwarded-For from nginx."""
    if request is None:
        return None
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


# ── Successful login ──────────────────────────────────────────────────────────

@receiver(user_logged_in)
def log_user_login(sender, request, user, **kwargs):
    from apps.customer.models import LoginHistory
    try:
        LoginHistory.objects.create(
            user=user,
            ip_address=_get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            session_key=request.session.session_key or '',
        )
    except Exception:
        logger.exception("Failed to record login history for user %s", getattr(user, 'email', user.pk))


# ── Failed login attempt ──────────────────────────────────────────────────────

@receiver(user_login_failed)
def log_failed_login(sender, credentials, request, **kwargs):
    from apps.customer.models import ActivityLog
    try:
        attempted_email = credentials.get('email') or credentials.get('username') or ''
        ActivityLog.objects.create(
            user=None,
            action='LOGIN_FAILED',
            ip_address=_get_client_ip(request),
            attempted_email=str(attempted_email)[:254],
            detail=f"Failed login attempt for: {attempted_email}",
        )
    except Exception:
        logger.exception("Failed to record failed login attempt")


# ── Logout ────────────────────────────────────────────────────────────────────

@receiver(user_logged_out)
def log_user_logout(sender, request, user, **kwargs):
    from apps.customer.models import ActivityLog
    if user is None:
        return
    # Skip logout log if this is part of an account deletion (already logged separately)
    if request and request.session.get('_account_being_deleted'):
        return
    try:
        ActivityLog.objects.create(
            user=user,
            action='LOGOUT',
            ip_address=_get_client_ip(request),
            detail="User session ended",
        )
    except Exception:
        logger.exception("Failed to record logout for user %s", getattr(user, 'email', user.pk))
