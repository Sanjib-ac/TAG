import cv2
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os
import re
import time
import queue
import threading

# ============================================================================
# Configuration: Toggle image preprocessing (for improved OCR accuracy)
# ============================================================================
APPLY_PREPROCESSING = False  # Set to True to apply preprocessing; False to skip.

# ============================================================================
# Setup OCR model and device
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
model.to(device)

# ============================================================================
# Global variables and thread synchronization
# ============================================================================
ocr_result_lock = threading.Lock()
global_ocr_result = ""  # This string holds the latest OCR result (digits only)

# Define a queue to pass frames from the main loop to the OCR worker thread.
# The maxsize limits the number of pending frames so that old frames are dropped if OCR falls behind.
frame_queue = queue.Queue(maxsize=4)


# ============================================================================
# OCR Worker Function: Every frame in the queue is processed for OCR.
# ============================================================================
def perform_ocr(frame):
    """
    Processes a single frame:
      - Optionally applies preprocessing (grayscale, blur, adaptive threshold)
      - Converts from BGR to RGB,
      - Converts to a PIL image,
      - Saves to a temporary PNG file (workaround for DocumentFile),
      - Creates a Document from the file,
      - Runs OCR inference, and finally
      - Returns a string containing numerical digits only,
        with individual numbers separated by commas.
    """
    # Optionally preprocess the image for enhanced OCR accuracy.
    if APPLY_PREPROCESSING:
        # Convert the frame from BGR to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use adaptive thresholding to highlight the digits.
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        # Convert thresholded image back to BGR (3 channels) so that subsequent conversion works normally.
        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Convert frame from BGR to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the image to a PIL Image.
    pil_img = Image.fromarray(frame_rgb)
    # Save the PIL image to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_filename = tmp.name
        pil_img.save(tmp, format="PNG")
    # Create a DocumentFile using the temporary file path.
    doc = DocumentFile.from_images([tmp_filename])
    os.remove(tmp_filename)
    result = model(doc)

    recognized_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    recognized_text += word.value + " "

    # Extract each digit sequence and join them with commas.
    numbers = re.findall(r'\d+', recognized_text.strip())
    digits_only = ",".join(numbers)
    return digits_only


def ocr_worker():
    global global_ocr_result
    while True:
        frame = frame_queue.get()
        if frame is None:  # Sentinel value to terminate the worker.
            frame_queue.task_done()
            break
        try:
            result = perform_ocr(frame)
            with ocr_result_lock:
                global_ocr_result = result
        except Exception as e:
            print("OCR processing error:", e)
        frame_queue.task_done()


# Start one OCR worker thread.
worker_thread = threading.Thread(target=ocr_worker, daemon=True)
worker_thread.start()

# ============================================================================
# Open camera feed and process every frame
# ============================================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Set a desired resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Camera resolution is set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print("Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
        break

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Always push every frame to the OCR queue.
    # If the queue is full, drop this frame.
    try:
        frame_queue.put(frame.copy(), block=False)
    except queue.Full:
        pass

    # Get the latest OCR result.
    with ocr_result_lock:
        ocr_text = global_ocr_result

    overlay_text = f"Digits: {ocr_text}  FPS: {fps:.2f}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Add a slight sleep to yield control and reduce CPU usage.
    time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()

# Signal the worker thread to exit and wait for it to finish.
frame_queue.put(None)
worker_thread.join()



# import cv2
# import torch
# from doctr.models import ocr_predictor
# from doctr.io import DocumentFile
# from PIL import Image
# import tempfile
# import os
# import re
# import time
# import queue
# import threading
# from collections import deque, Counter
#
# # ============================================================================
# # Configuration
# # ============================================================================
# APPLY_PREPROCESSING = True     # Enable OCR-friendly preprocessing
# ENABLE_SMOOTHING = True        # Stabilize OCR results
# OCR_HISTORY_LENGTH = 5         # Number of results to use for smoothing
#
# # ============================================================================
# # OCR Model Setup
# # ============================================================================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)
# model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
# model.to(device)
#
# # ============================================================================
# # OCR Result and Frame Queue Handling
# # ============================================================================
# ocr_result_lock = threading.Lock()
# global_ocr_result = ""
# frame_queue = queue.Queue(maxsize=50)
# recent_results = deque(maxlen=OCR_HISTORY_LENGTH)
#
# # ============================================================================
# # Preprocessing Function (for enhanced OCR on low-contrast, small text)
# # ============================================================================
# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     scale = 2.0
#     resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(resized)
#     thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
#     return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#
# # ============================================================================
# # OCR Frame Processing
# # ============================================================================
# def perform_ocr(frame):
#     if APPLY_PREPROCESSING:
#         frame = preprocess_for_ocr(frame)
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(frame_rgb)
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#         tmp_filename = tmp.name
#         pil_img.save(tmp, format="PNG")
#
#     doc = DocumentFile.from_images([tmp_filename])
#     os.remove(tmp_filename)
#     result = model(doc)
#
#     recognized_text = ""
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     recognized_text += word.value + " "
#
#     numbers = re.findall(r'\d+', recognized_text.strip())
#     digits_only = ",".join(numbers)
#     return digits_only
#
# def smoothed_result(new_result):
#     recent_results.append(new_result)
#     if not ENABLE_SMOOTHING:
#         return new_result
#     most_common, _ = Counter(recent_results).most_common(1)[0]
#     return most_common
#
# # ============================================================================
# # OCR Worker Thread
# # ============================================================================
# def ocr_worker():
#     global global_ocr_result
#     while True:
#         frame = frame_queue.get()
#         if frame is None:
#             frame_queue.task_done()
#             break
#         try:
#             result = perform_ocr(frame)
#             smoothed = smoothed_result(result)
#             with ocr_result_lock:
#                 global_ocr_result = smoothed
#         except Exception as e:
#             print("OCR error:", e)
#         frame_queue.task_done()
#
# # Start OCR thread
# worker_thread = threading.Thread(target=ocr_worker, daemon=True)
# worker_thread.start()
#
# # ============================================================================
# # Webcam Feed
# # ============================================================================
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Unable to open camera")
#     exit()
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
# print("Press 'q' to quit.")
#
# prev_time = time.time()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Frame capture error")
#         break
#
#     current_time = time.time()
#     fps = 1.0 / (current_time - prev_time)
#     prev_time = current_time
#
#     try:
#         frame_queue.put(frame.copy(), block=False)
#     except queue.Full:
#         pass
#
#     with ocr_result_lock:
#         ocr_text = global_ocr_result
#
#     overlay_text = f"Digits: {ocr_text}  FPS: {fps:.2f}"
#     cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     cv2.imshow("OCR Camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     time.sleep(0.005)
#
# cap.release()
# cv2.destroyAllWindows()
# frame_queue.put(None)
# worker_thread.join()
#
