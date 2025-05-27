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
model = ocr_predictor(pretrained=True)
model.to(device)

# ============================================================================
# Global variables and thread synchronization
# ============================================================================
ocr_result_lock = threading.Lock()
global_ocr_result = ""  # This string holds the latest OCR result (digits only)

# Define a queue to pass frames from the main loop to the OCR worker thread.
# The maxsize limits the number of pending frames so that old frames are dropped if OCR falls behind.
frame_queue = queue.Queue(maxsize=50)


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
#
# # ------------------------------------------------------------------
# # Setup model and device
# # ------------------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)
#
# # Create the OCR model and send it to the device.
# model = ocr_predictor(pretrained=True)
# model.to(device)
#
# # ------------------------------------------------------------------
# # Global variables and synchronization primitives
# # ------------------------------------------------------------------
# ocr_result_lock = threading.Lock()
# global_ocr_result = ""  # Holds the OCR result (digits-only string)
#
# # Queue for frames; adjust maxsize as needed.
# frame_queue = queue.Queue(maxsize=100)
#
#
# # ------------------------------------------------------------------
# # OCR processing function.
# # ------------------------------------------------------------------
# def perform_ocr(frame):
#     """
#     Given an OpenCV frame (BGR), this function converts it to RGB,
#     then to a PIL image, saves it to a temporary file (workaround for docTR),
#     creates a DocumentFile, runs the OCR model, and returns the recognized
#     numerical digits (all non-digits are removed).
#     """
#     # Convert BGR to RGB.
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # Convert array to a PIL Image.
#     pil_img = Image.fromarray(frame_rgb)
#
#     # Save the image to a temporary file.
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#         tmp_filename = tmp.name
#         pil_img.save(tmp, format="PNG")
#
#     # Use the temporary file path to create the DocumentFile.
#     doc = DocumentFile.from_images([tmp_filename])
#     os.remove(tmp_filename)
#
#     # Run OCR inference.
#     result = model(doc)
#
#     recognized_text = ""
#     # Traverse the structure: pages -> blocks -> lines -> words.
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     recognized_text += word.value + " "
#
#     # Keep only numerical digits.
#     digits_only = re.sub(r"\D", "", recognized_text.strip())
#     return digits_only
#
#
# # ------------------------------------------------------------------
# # OCR Worker Thread Function
# # ------------------------------------------------------------------
# def ocr_worker():
#     global global_ocr_result
#     while True:
#         frame = frame_queue.get()
#         if frame is None:
#             # None is our sentinel to shut down the thread.
#             frame_queue.task_done()
#             break
#         try:
#             # Process OCR on the frame.
#             result = perform_ocr(frame)
#             # Update the global OCR result (safely).
#             with ocr_result_lock:
#                 global_ocr_result = result
#         except Exception as e:
#             print("Error during OCR processing:", e)
#         frame_queue.task_done()
#
#
# # Start the worker thread.
# worker_thread = threading.Thread(target=ocr_worker, daemon=True)
# worker_thread.start()
#
# # ------------------------------------------------------------------
# # Open camera feed and main loop.
# # ------------------------------------------------------------------
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     print("Unable to open camera")
#     exit()
#
# # Set camera resolution (e.g., 1280 x 720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Camera resolution is set to: {width}x{height}")
# print("Press 'q' to quit.")
#
# prev_frame_time = time.time()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error capturing frame")
#         break
#
#     # Calculate FPS.
#     new_frame_time = time.time()
#     fps = 1.0 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#
#     # Attempt to push the current frame into the processing queue.
#     # If the queue is full, discard this frame.
#     try:
#         frame_queue.put(frame.copy(), block=False)
#     except queue.Full:
#         pass  # Frame skipped because the queue is full.
#
#     # Retrieve the latest OCR result (safely).
#     with ocr_result_lock:
#         ocr_text = global_ocr_result
#
#     # Overlay the OCR result (digits only) and FPS on the frame.
#     overlay_text = f"Digits: {ocr_text}  FPS: {fps:.2f}"
#     cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     print("Detected digits:", ocr_text)
#     cv2.imshow("Camera Feed", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Signal the worker thread to exit by putting None into the queue.
# frame_queue.put(None)
# worker_thread.join()
