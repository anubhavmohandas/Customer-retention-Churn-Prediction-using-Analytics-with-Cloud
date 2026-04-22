"""
Resend email utility for Retain.Ai.
Replace RESEND_API_KEY in your .env with your real key from resend.com.
"""
import logging
import resend
from django.conf import settings

logger = logging.getLogger(__name__)


def send_email(to: str, subject: str, html: str) -> bool:
    """Send an email via Resend. Returns True on success, False on failure."""
    resend.api_key = settings.RESEND_API_KEY
    try:
        resend.Emails.send({
            "from": settings.RESEND_FROM_EMAIL,
            "to": to,
            "subject": subject,
            "html": html,
        })
        return True
    except Exception as e:
        logger.error(f"Resend email failed to {to}: {e}")
        return False


def send_otp_email(to: str, otp: str, name: str = "") -> bool:
    html = f"""
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px;">
        <div style="text-align: center; margin-bottom: 32px;">
            <div style="display: inline-block; width: 48px; height: 48px; background: #000; border-radius: 16px; transform: rotate(3deg);"></div>
        </div>
        <h1 style="font-size: 24px; font-weight: 800; color: #000; margin-bottom: 8px;">Your login code</h1>
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 32px;">
            Hi{' ' + name if name else ''}, use the code below to complete your sign-in to Retain.Ai.
        </p>
        <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 16px; padding: 32px; text-align: center; margin-bottom: 24px;">
            <p style="font-size: 48px; font-weight: 900; letter-spacing: 12px; color: #000; margin: 0;">{otp}</p>
            <p style="color: #9ca3af; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-top: 12px;">Expires in 10 minutes</p>
        </div>
        <p style="color: #9ca3af; font-size: 12px;">If you didn't request this, ignore this email. Your account is safe.</p>
        <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 24px 0;">
        <p style="color: #d1d5db; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">Retain.Ai Intelligence</p>
    </div>
    """
    return send_email(to, "Your Retain.Ai login code", html)


def send_password_reset_email(to: str, reset_url: str, name: str = "") -> bool:
    html = f"""
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px;">
        <div style="text-align: center; margin-bottom: 32px;">
            <div style="display: inline-block; width: 48px; height: 48px; background: #000; border-radius: 16px; transform: rotate(3deg);"></div>
        </div>
        <h1 style="font-size: 24px; font-weight: 800; color: #000; margin-bottom: 8px;">Reset your password</h1>
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 32px;">
            Hi{' ' + name if name else ''}, click the button below to reset your Retain.Ai password.
            This link expires in 1 hour.
        </p>
        <div style="text-align: center; margin-bottom: 32px;">
            <a href="{reset_url}"
               style="display: inline-block; background: #000; color: #fff; padding: 16px 40px;
                      border-radius: 12px; font-weight: 800; font-size: 14px; text-decoration: none;
                      letter-spacing: 1px; text-transform: uppercase;">
                Reset Password
            </a>
        </div>
        <p style="color: #9ca3af; font-size: 12px;">
            If you didn't request a password reset, you can safely ignore this email.
            Your password will not be changed.
        </p>
        <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 24px 0;">
        <p style="color: #d1d5db; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">Retain.Ai Intelligence</p>
    </div>
    """
    return send_email(to, "Reset your Retain.Ai password", html)
