import os
import json
from django.views import View
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated


# @method_decorator(csrf_exempt, name='dispatch')
class LiveDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Get form data from POST request
            model_name      = request.POST.get('model_name')
            base_path       = r"C:\Users\clint\OneDrive\coding"
            model_path      = os.path.join(base_path, model_name)



            return JsonResponse({
                "status": "Success",
                "message": "Live Detection complete.",
            }, status=200)
        
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"})
