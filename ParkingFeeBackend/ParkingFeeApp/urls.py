from django.urls import path
from . import views

from ParkingFeeApp.viewsets.live_detection_viewset import LiveDetectionView
from ParkingFeeApp.viewsets.ocr_viewset import OCRView



urlpatterns = [
    path('live_detection/', LiveDetectionView.as_view(), name='live_detection'),
    path('ocr_extraction/', OCRView.as_view(), name='ocr_extraction'),
]  
