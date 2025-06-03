import os
import json
import cv2
import time
import easyocr
from django.views import View
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from path import model_path,ip_camera,saved_plate_path


class LiveDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Get form data from POST request
            model_name      = request.POST.get('model_name')

            # if using webcam
            # cap= cv2.VideoCapture(0)
            # IF using DroidCam
            cap = cv2.VideoCapture(ip_camera)
            cap.set(3, 640)  # Width
            cap.set(4, 480)  # Height

            min_area          = 500
            count             = 0
            start_time        = time.time()  # Capture start time
            detected_plates   = []  # Store detected plates
            extracted_numbers = []  # Store extracted text from plate
            plate_cascade     = cv2.CascadeClassifier(model_path)

            while time.time() - start_time < 10:  # Run for 30 seconds
                success, img = cap.read()
                if not success or img is None:
                    print("Error: Unable to capture frame from camera")
                    break
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
                for (x, y, w, h) in plates:
                    area = w * h
                    if area > min_area:
                        plate_coordinates = (x, y, w, h)
                        # **Check for duplicate detections**
                        if plate_coordinates not in detected_plates:
                            detected_plates.append(plate_coordinates)  # Store unique plate detection
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                            # Extract plate region
                            img_roi = img[y:y+h, x:x+w]
                            
                            save_path = os.path.join(saved_plate_path, f"plate_{count}.jpg") 
                            cv2.imwrite(save_path, img_roi)  # Auto-save only if unique
                            count += 1
                cv2.imshow("Result", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' press
                    break
            cap.release()
            cv2.destroyAllWindows()

            return JsonResponse({
                "status": "Success",
                "message": "Live Detection complete.",
                "plates_detected": count +1,
            }, status=200)
        
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"})
