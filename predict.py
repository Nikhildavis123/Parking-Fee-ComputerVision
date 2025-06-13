import cv2
import time
import pandas as pd
import atexit
import os
from datetime import datetime
from ultralytics import YOLO
import easyocr

# Define model paths
CAR_MODEL_PATH = r"C:\Users\clint\OneDrive\coding\ComputerVision\car_detection_model.pt"
PLATE_MODEL_PATH = r"C:\Users\clint\OneDrive\coding\ComputerVision\license_plate_detection.pt"

# Initialize the YOLO models
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define folder to save car images
SAVE_FOLDER = r"C:\Users\clint\OneDrive\coding\ComputerVision\car_detected"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Initialize DataFrame for saving results
excel_path = "license_plate_log.xlsx"
df_columns = ["car_id", "serial_number", "confidence_score", "license_number_plate", "plate_score", "date", "time"]
serial_number = 1  # Increment with every detection

# Try to load existing data to maintain car ID continuity
try:
    existing_df = pd.read_excel(excel_path)
    if not existing_df.empty:
        car_id = existing_df["car_id"].max() + 1  # Increment car_id for the new batch
    else:
        car_id = 1
except FileNotFoundError:
    car_id = 1  # Initialize if no previous data exists

# Confidence threshold
CONF_THRESHOLD = 0.7  # Only process detections above 70% confidence

# Initialize FPS tracking
previous_time = time.time()
frame_count = 0
fps = 0

def save_log(df):
    """Ensure the log is saved before exit, appending instead of overwriting."""
    if not df.empty:
        try:
            existing_df = pd.read_excel(excel_path)
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=df_columns)

        df_final = pd.concat([existing_df, df], ignore_index=True)
        df_final.to_excel(excel_path, index=False)
        print("License plate log updated successfully.")

def filter_highest_plate_score():
    """Filter rows, keeping only the highest plate score row from the highest car_id."""
    try:
        df = pd.read_excel(excel_path)

        if df.empty:
            print("No data found in the Excel file.")
            return

        # Find the highest car_id value
        highest_car_id = df["car_id"].max()

        # Separate untouched data + rows to filter
        df_untouched = df[df["car_id"] != highest_car_id]  # Keep all other car_id values
        df_filtered = df[df["car_id"] == highest_car_id]  # Filter only highest car_id rows

        # Find the row with the highest plate score among the filtered rows
        highest_score_row = df_filtered.loc[df_filtered["plate_score"].idxmax()]

        # Combine untouched data with the highest-scoring row for highest car_id
        df_final = pd.concat([df_untouched, pd.DataFrame([highest_score_row])], ignore_index=True)

        # Save back to Excel
        df_final.to_excel(excel_path, index=False)
        print(f"Filtered successfully! Kept all other car IDs. Only the highest plate score row for car_id {highest_car_id} remains.")
    except FileNotFoundError:
        print("Excel file not found. Skipping filtering.")

def ocr_image(img, coordinates):
    """Extracts text from detected license plates using EasyOCR."""
    x1, y1, x2, y2 = map(int, coordinates)
    cropped_img = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    result = reader.readtext(gray)

    text, plate_score = "Plate Not Detected", 0.0
    for res in result:
        if len(res[1]) > 6 and res[2] > 0.2:  # Filter based on confidence
            text, plate_score = res[1], res[2]

    return text, plate_score

# Open webcam
cap = cv2.VideoCapture(0)

df = pd.DataFrame(columns=df_columns)  # Store new entries

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            current_time = time.time()
            fps = 10 / (current_time - previous_time)
            previous_time = current_time

        # Run car detection
        car_results = car_model.predict(frame)
        detected_car = False

        for result in car_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf_score = confs[i]

                if conf_score >= CONF_THRESHOLD:
                    detected_car = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Car ({conf_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Save detected car image
                    img_filename = os.path.join(SAVE_FOLDER, f"car_detected_{serial_number:03d}.png")
                    cv2.imwrite(img_filename, frame[y1:y2, x1:x2])

                    # Run license plate detection
                    plate_results = plate_model.predict(frame)
                    detected_plate = False

                    for plate_result in plate_results:
                        plate_boxes = plate_result.boxes.xyxy.cpu().numpy()
                        plate_confs = plate_result.boxes.conf.cpu().numpy()

                        for j, plate_box in enumerate(plate_boxes):
                            px1, py1, px2, py2 = map(int, plate_box)
                            plate_conf_score = plate_confs[j]

                            if plate_conf_score >= CONF_THRESHOLD:
                                detected_plate = True
                                license_text, plate_score = ocr_image(frame, plate_box)

                                # Draw bounding box
                                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{license_text} ({plate_score:.2f})", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Save results to DataFrame
                                df.loc[len(df)] = [car_id, serial_number, conf_score, license_text, plate_score, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")]

                    serial_number += 1

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display live detection feed
        cv2.imshow("Live ANPR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted. Saving detected data...")
    save_log(df)

finally:
    save_log(df)
    
    # Uncomment to enable filtering highest plate score for the highest car ID
    filter_highest_plate_score()

    cap.release()
    cv2.destroyAllWindows()
