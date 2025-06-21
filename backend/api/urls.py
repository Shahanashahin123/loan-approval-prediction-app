from django.urls import path
from .views import predict_loan

urlpatterns = [
    path('predict/', predict_loan),
]
