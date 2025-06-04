import cv2
import time
import re
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR with GPU acceleration, English language, and angle classification.
ocr = PaddleOCR(use_gpu=True, lang="en", use_angle_cls=True)

# Open the camera (device 0).
cap = cv2.VideoCapture(0)

# Set the camera resolution to 1280 x 720.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Choose a system font path.
# For Windows, you might use: 'C:\\Windows\\Fonts\\Arial.ttf'
# For Linux, a common choice is: '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
# font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Update accordingly
font_path = 'C:\\Windows\\Fonts\\Arial.ttf'

print("Starting camera stream. Press 'q' to quit.")

# Initialize time for FPS calculations.
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Convert the frame from BGR to RGB as PaddleOCR expects RGB images.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run PaddleOCR on the frame.
    result = ocr.ocr(frame_rgb, cls=True)

    # Filter results to include only those that match exactly 10 digits.
    ten_digit_numbers = []
    for line in result:
        # Each line is structured as [bounding_box, (text, confidence)]
        bbox = line[0]
        text, confidence = line[1]
        if re.fullmatch(r'\d{10}', text):
            ten_digit_numbers.append((bbox, text, confidence))

    # Optionally, draw filtered bounding boxes and texts on the original frame.
    for bbox, text, confidence in ten_digit_numbers:
        pts = np.array(bbox).astype(np.int32)
        cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"Detected 10-digit number: {text} (Confidence: {confidence:.2f})")

    # Use draw_ocr to annotate the frame with all OCR results (using the system font).
    annotated_frame = draw_ocr(frame, result, font_path=font_path)

    # Calculate FPS.
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay the FPS on the annotated frame.
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting annotated frame.
    cv2.imshow("OCR Detection", annotated_frame)

    # Quit the stream when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
