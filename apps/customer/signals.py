import logging
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver

logger = logging.getLogger(__name__)


def _get_client_ip(request):
    """Extract real client IP, respecting X-Forwarded-For from nginx."""
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


@receiver(user_logged_in)
def log_user_login(sender, request, user, **kwargs):
    """Record every successful login to LoginHistory."""
    from apps.customer.models import LoginHistory  # local import avoids circular

    try:
        LoginHistory.objects.create(
            user=user,
            ip_address=_get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
            session_key=request.session.session_key or '',
        )
    except Exception:
        logger.exception("Failed to record login history for user %s", getattr(user, 'email', user.pk))
