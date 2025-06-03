import os
import easyocr
import cv2
from django.http import JsonResponse
from rest_framework.views import APIView
from path import model_path,ip_camera,saved_plate_path

class OCRView(APIView):
    def post(self, request, *args, **kwargs):
        try:

            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])  # English OCR

            # Array to store extracted text from all images
            extracted_texts = []

            # Loop through all images in the folder
            for filename in os.listdir(saved_plate_path):
                if filename.endswith((".jpg", ".png", ".jpeg")):  # Process only image files
                    image_path = os.path.join(saved_plate_path, filename)
                    
                    # Read the image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error loading {filename}")
                        continue
                    
                    # Convert to grayscale (optional, improves detection)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Run OCR on the image
                    ocr_result = reader.readtext(img_gray, detail=0)  # Extract only text
                    
                    if ocr_result:
                        extracted_texts.append(ocr_result[0])  # Store plate number

            # Print final array with all extracted text
            print("Extracted License Plates:", extracted_texts)
            return JsonResponse({
                "status": "Success",
                "message": "EasyOCR extraction complete.",
                "extracted number plate":extracted_texts,
            }, status=200)
        
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"})