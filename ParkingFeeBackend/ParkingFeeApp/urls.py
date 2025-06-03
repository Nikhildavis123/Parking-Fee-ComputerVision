from django.urls import path
from . import views

from ParkingFeeApp.viewsets.live_detection_viewset import LiveDetectionView



urlpatterns = [
    path('live_detection/', LiveDetectionView.as_view(), name='live_detection'),
]  
