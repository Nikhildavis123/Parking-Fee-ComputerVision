import cv2
import time
import pandas as pd
import os
import re
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ==================== Model paths ====================
CAR_MODEL_PATH = r"C:\Users\clint\OneDrive\coding\ComputerVision\car_detection_model.pt"
PLATE_MODEL_PATH = r"C:\Users\clint\OneDrive\coding\ComputerVision\license_plate_detection.pt"

car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

# ==================== File and format setup ====================
excel_path = "license_plate_log.xlsx"
df_columns = ["plate_number", "in_time", "out_time", "plate_score", "duration", "parking_fee"]
if not os.path.exists(excel_path):
    pd.DataFrame(columns=df_columns).to_excel(excel_path, index=False)
log_df = pd.read_excel(excel_path)

# ==================== Preprocessing folder setup ====================
BASE_PREPROCESS_DIR = "preprocessed"
STAGES = ["grayscale", "clahe", "bilateral", "gaussian", "sharpen"]
for stage in STAGES:
    os.makedirs(os.path.join(BASE_PREPROCESS_DIR, stage), exist_ok=True)

# ==================== Config ====================
CONF_THRESHOLD = 0.5
PLATE_THRESHOLD = 0.3
COOLDOWN_SECONDS = 5
PARKING_RATE_PER_SECOND = 0.10  # €0.10 per second
PLATE_FORMAT_REGEX = r"^[A-Z]{3}[0-9]{2}[A-Z]{1}[0-9]{4}$"

# ==================== Helpers ====================
def normalize_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def validate_plate_format(plate):
    return re.fullmatch(PLATE_FORMAT_REGEX, plate) is not None

def preprocess_for_easyocr(plate_img, plate_id):
    # Resize small plates
    h, w = plate_img.shape[:2]
    TARGET_WIDTH = 300
    if w < TARGET_WIDTH:
        scale_factor = TARGET_WIDTH / w
        plate_img = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 1. Grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(BASE_PREPROCESS_DIR, "grayscale", f"{plate_id}.png"), gray)

    # 2. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    cv2.imwrite(os.path.join(BASE_PREPROCESS_DIR, "clahe", f"{plate_id}.png"), contrast)

    # 3. Bilateral filter (denoise)
    bilateral = cv2.bilateralFilter(contrast, 11, 17, 17)
    cv2.imwrite(os.path.join(BASE_PREPROCESS_DIR, "bilateral", f"{plate_id}.png"), bilateral)

    # 4. Gaussian blur
    gaussian = cv2.GaussianBlur(bilateral, (5, 5), 0)
    cv2.imwrite(os.path.join(BASE_PREPROCESS_DIR, "gaussian", f"{plate_id}.png"), gaussian)

    # 5. Sharpening
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gaussian, -1, sharpen_kernel)
    cv2.imwrite(os.path.join(BASE_PREPROCESS_DIR, "sharpen", f"{plate_id}.png"), sharpened)

    return sharpened

# ==================== Track current session ====================
current_session = {
    "start_time": None,
    "last_seen": None,
    "plates": []
}

# ==================== Main Loop ====================
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_plate_this_frame = False
        car_results = car_model.predict(frame)

        for result in car_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                if confs[i] >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    plate_results = plate_model.predict(frame)
                    for presult in plate_results:
                        plate_boxes = presult.boxes.xyxy.cpu().numpy()
                        plate_confs = presult.boxes.conf.cpu().numpy()

                        for j, pbox in enumerate(plate_boxes):
                            if plate_confs[j] >= CONF_THRESHOLD:
                                px1, py1, px2, py2 = map(int, pbox)
                                cropped = frame[py1:py2, px1:px2]

                                plate_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                preprocessed_img = preprocess_for_easyocr(cropped, plate_id)

                                ocr_results = reader.readtext(preprocessed_img)

                                for res in ocr_results:
                                    if isinstance(res[1], str):
                                        raw = res[1].strip()
                                        norm_plate = normalize_plate(raw)
                                        score = res[2]

                                        if (
                                            len(norm_plate) >= 6 and
                                            score > PLATE_THRESHOLD and
                                            validate_plate_format(norm_plate)
                                        ):
                                            now = datetime.now()
                                            detected_plate_this_frame = True

                                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                                            cv2.putText(frame, f"{norm_plate} ({score:.2f})", (px1, py1 - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                            if current_session["start_time"] is None:
                                                current_session["start_time"] = now
                                                current_session["plates"] = [(norm_plate, score)]
                                            else:
                                                current_session["plates"].append((norm_plate, score))
                                            current_session["last_seen"] = time.time()

        # Cooldown check
        if current_session["last_seen"] and not detected_plate_this_frame:
            if time.time() - current_session["last_seen"] > COOLDOWN_SECONDS:
                best_plate, best_score = max(current_session["plates"], key=lambda x: x[1])
                in_time = current_session["start_time"]
                out_time = datetime.now()
                duration = out_time - in_time
                duration_str = str(duration).split('.')[0]
                parking_seconds = int(duration.total_seconds())
                parking_fee = round(parking_seconds * PARKING_RATE_PER_SECOND, 2)
                parking_fee_str = f"€{parking_fee:.2f}"

                new_row = pd.DataFrame([[best_plate,
                                         in_time.strftime("%Y-%m-%d %H:%M:%S"),
                                         out_time.strftime("%Y-%m-%d %H:%M:%S"),
                                         round(best_score, 3),
                                         duration_str,
                                         parking_fee_str]],
                                       columns=df_columns)
                log_df = pd.concat([log_df, new_row], ignore_index=True)
                print(f"[CAR LOGGED] {best_plate} | Duration: {duration_str} | Fee: {parking_fee_str}")
                current_session = {"start_time": None, "last_seen": None, "plates": []}

        cv2.imshow("Live ANPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted — saving final log...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_df.to_excel(excel_path, index=False)
    print(f"✅ Log saved at: {os.path.abspath(excel_path)}")


