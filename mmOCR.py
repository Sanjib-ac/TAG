import cv2
import torch
from mmocr.apis import MMOCRInferencer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the MMOCRInferencer with detection and recognition models
    ocr = MMOCRInferencer(det='dbnet', rec='crnn', device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = ocr(frame)

        # Access the OCR results
        instances = results['predictions'][0]  # List of dicts with 'text', 'box', etc.
        for inst in instances:
            text = inst['text']
            if text.isdigit() and len(text) == 10:
                box = inst['box']
                pts = [tuple(map(int, pt)) for pt in box]
                for i in range(4):
                    cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                print(f"Detected 10-digit: {text}")

        cv2.imshow("Digits Only", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
