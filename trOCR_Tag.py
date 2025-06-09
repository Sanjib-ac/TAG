from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import cv2
import time
import re


def run_multiple_camera_ocr(camera_indices, inference_interval=1.0, width=640, height=480):
    """
    Runs real-time OCR using TrOCR from multiple cameras using a desired capture resolution.

    Parameters:
      - camera_indices: List of integers for camera indices (e.g., [0, 1]).
      - inference_interval: Time (in seconds) between OCR inferences for each camera.
      - width: Desired frame width.
      - height: Desired frame height.
    """
    # Load the pretrained processor and model with the fast processor.
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.eval()

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open multiple video capture devices with desired resolution.
    captures = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Unable to open camera index {idx}")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            captures.append(cap)

    if not captures:
        print("No valid cameras found. Exiting.")
        return

    num_cams = len(captures)
    print("Press 'q' to quit.")

    # Initialize OCR text storage and timers.
    current_texts = ["" for _ in range(num_cams)]
    last_inference_time = time.time()
    prev_frame_time = time.time()  # For FPS calculation

    while True:
        current_frame_time = time.time()
        # Calculate frame rate (FPS) for each loop iteration.
        frame_fps = 1.0 / max(current_frame_time - prev_frame_time, 1e-6)
        prev_frame_time = current_frame_time

        current_time = time.time()
        frames = []
        # Capture one frame per camera.
        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {camera_indices[i]}: Error capturing frame.")
                frames.append(None)
            else:
                frames.append(frame)

        # Run OCR inference only when the inference interval has elapsed.
        if (current_time - last_inference_time) >= inference_interval:
            for i, frame in enumerate(frames):
                if frame is not None:
                    # Convert frame from BGR (OpenCV) to RGB.
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert the NumPy array to a PIL image.
                    pil_img = Image.fromarray(frame_rgb)
                    # Preprocess the image for TrOCR.
                    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    # Debug output to see what the model returns.
                    print(f"Raw OCR output from camera {camera_indices[i]}:", generated_text)

                    # Clean up the output by removing spaces.
                    clean_text = "".join(filter(str.isdigit, generated_text))
                    if len(clean_text) == 10:
                        current_texts[i] = clean_text
                    else:
                        current_texts[i] = f"Detected: {generated_text}"
            last_inference_time = current_time

        # Overlay recognized text and FPS on each frame.
        for i, frame in enumerate(frames):
            if frame is not None:
                overlay = f"Cam {camera_indices[i]} | OCR: {current_texts[i]} | FPS: {frame_fps:.2f}"
                cv2.putText(frame, overlay, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"Camera {camera_indices[i]}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release cameras and close windows.
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


def main():
    # Example usage: process camera indices 0 and 1 with a resolution of 640x480
    # and an inference interval of 1 second.
    camera_indices = [0]
    run_multiple_camera_ocr(camera_indices, inference_interval=1.0, width=640, height=480)


if __name__ == "__main__":
    main()
