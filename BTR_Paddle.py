import cv2
import paddle
from paddleocr import PaddleOCR


# Set device to GPU if available
# paddle.set_device('gpu')

print("Paddle version:", paddle.__version__)
print("Using device:", paddle.device.get_device())

# Initialize OCR system
ocr = PaddleOCR(use_textline_orientation=False, lang='en')  # updated parameter

# Open webcam
cap = cv2.VideoCapture(0)

# Path to a suitable font for drawing OCR results
font_path = 'path/to/your/font.ttf'  # <-- update this path accordingly

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for PaddleOCR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # OCR inference, disable classifier (cls) if not needed
    result = ocr.ocr(rgb, cls=False)

    # Parse OCR results
    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # Draw OCR results on the frame
    frame = ocr.visualize(frame, boxes, texts, scores, font_path=font_path)

    # Show frame with OCR results
    cv2.imshow("OCR", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
