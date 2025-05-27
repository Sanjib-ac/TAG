import cv2
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os
import re
import time

# Select device: GPU if available, else CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create the OCR model and send it to the device.
model = ocr_predictor(pretrained=True)
model.to(device)

# Open camera feed.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Set the desired resolution (for example: 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# (Optional) confirm the resolution
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution is set to: {width}x{height}")

print("Press 'q' to quit.")

# Initialize a variable for FPS calculation.
prev_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
        break

    # Convert frame from BGR (OpenCV format) to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the RGB frame to a PIL Image.
    pil_img = Image.fromarray(frame_rgb)

    # Write the PIL image to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_filename = tmp.name
        pil_img.save(tmp, format="PNG")

    # Create a DocumentFile using the file path.
    doc = DocumentFile.from_images([tmp_filename])
    os.remove(tmp_filename)

    # Run OCR inference.
    result = model(doc)

    # Concatenate recognized words from the hierarchical output.
    recognized_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    recognized_text += word.value + " "

    # Use a regular expression to keep only numerical digits.
    digits_only = re.sub(r"\D", "", recognized_text.strip())

    # Calculate FPS.
    new_frame_time = time.time()
    fps = 1.0 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Overlay the detected digits and FPS onto the frame.
    overlay_text = f"Digits: {digits_only}  FPS: {fps:.2f}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

    # Also print the detected digits to the console.
    print("Detected digits:", digits_only)

    # Display the annotated frame.
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
