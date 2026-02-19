from apps.customers.views import customer
from django.urls import path

urlpatterns=[
    path('', customer, name='customer')
]