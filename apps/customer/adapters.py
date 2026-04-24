# apps/customer/adapters.py
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def get_login_url(self, request, **kwargs):
        return '/accounts/login/'