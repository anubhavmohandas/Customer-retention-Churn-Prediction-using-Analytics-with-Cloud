from django.apps import AppConfig


class CustomerConfig(AppConfig):
    name = 'apps.customer'

    def ready(self):
        import apps.customer.signals  # noqa: F401 — registers signal handlers
