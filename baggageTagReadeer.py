import cv2
import numpy as np
import pytesseract
import time
from cv2 import dnn_superres

# Set Tesseract’s executable path if it's not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """
    Enhance image sharpness using an unsharp mask.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def detect_text_east(image, net, min_confidence=0.5):
    """
    Uses the EAST text detector to detect text regions in an image.
    Returns a list of bounding boxes (startX, startY, endX, endY).
    """
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize the image and create a blob
    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over rows and columns to build rectangles
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            score = scoresData[x]
            if score < min_confidence:
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))

    # Apply non-maxima suppression to suppress overlapping boxes
    indices = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            (startX, startY, endX, endY) = rects[i]
            # Scale the bounding box coordinates based on the resizing ratios.
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            results.append((startX, startY, endX, endY))
    return results


def main():
    # Open a connection to the default camera (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # --- Load the Super Resolution Model ---
    sr = dnn_superres.DnnSuperResImpl_create()
    sr_model_path = "./EDSR_x4.pb"  # Update the path to your SR model file
    sr.readModel(sr_model_path)
    sr.setModel("edsr", 4)  # Using EDSR with a scaling factor of 4

    # --- Load the EAST Text Detection Model ---
    east_model_path = "./frozen_east_text_detection.pb"  # Update with your EAST model file
    east_net = cv2.dnn.readNet(east_model_path)

    # --- Tesseract Config: detect digits only ---
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'

    while True:
        # Start timer for FPS calculation
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # --- Enhance the Frame ---
        upscaled = sr.upsample(frame)
        denoised = cv2.fastNlMeansDenoisingColored(
            upscaled, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
        sharpened = unsharp_mask(denoised)

        # --- Detect Text Regions ---
        boxes = detect_text_east(sharpened, east_net, min_confidence=0.5)

        recognized_numbers = []
        for (startX, startY, endX, endY) in boxes:
            roi = sharpened[startY:endY, startX:endX]
            text = pytesseract.image_to_string(roi, config=custom_config)
            detected_text = text.strip()
            recognized_numbers.append(detected_text)
            # Draw the bounding box
            cv2.rectangle(sharpened, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Print detected text above the bounding box
            cv2.putText(
                sharpened,
                detected_text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)

        # Display the FPS on the image
        cv2.putText(
            sharpened,
            f"FPS: {fps:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # (Optional) Also display all detected numbers at the top left
        cv2.putText(
            sharpened,
            "Numbers: " + ", ".join(recognized_numbers),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # --- Show the Processed Frame ---
        cv2.imshow("Camera Feed - Detected Text", sharpened)
        print(f"****Done*****")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#
# import cv2
# import numpy as np
# import pytesseract
# from cv2 import dnn_superres
#
#
# # If Tesseract is not in your system path, uncomment and set the full path:
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5):
#     """
#     # Enhance image sharpness using an unsharp mask.
#     """
#     blurred = cv2.GaussianBlur(image, kernel_size, sigma)
#     sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
#     return sharpened
#
#
# def detect_text_east(image, net, min_confidence=0.5):
#     """
#     Uses the EAST text detector to detect text regions in an image.
#     Returns a list of bounding boxes (startX, startY, endX, endY).
#     """
#     (H, W) = image.shape[:2]
#     newW, newH = (320, 320)
#     rW = W / float(newW)
#     rH = H / float(newH)
#
#     # Resize the image and create a blob
#     resized = cv2.resize(image, (newW, newH))
#     blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
#                                  (123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#     (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
#
#     (numRows, numCols) = scores.shape[2:4]
#     rects = []
#     confidences = []
#
#     # Loop over rows and columns to build rectangles
#     for y in range(numRows):
#         scoresData = scores[0, 0, y]
#         xData0 = geometry[0, 0, y]
#         xData1 = geometry[0, 1, y]
#         xData2 = geometry[0, 2, y]
#         xData3 = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]
#
#         for x in range(numCols):
#             score = scoresData[x]
#             if score < min_confidence:
#                 continue
#             offsetX = x * 4.0
#             offsetY = y * 4.0
#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)
#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]
#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)
#
#             rects.append((startX, startY, endX, endY))
#             confidences.append(float(score))
#
#     # Apply non-maxima suppression to suppress overlapping boxes
#     indices = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
#     results = []
#     if len(indices) > 0:
#         for i in indices.flatten():
#             (startX, startY, endX, endY) = rects[i]
#             # Scale the bounding box coordinates based on the resizing ratios.
#             startX = int(startX * rW)
#             startY = int(startY * rH)
#             endX = int(endX * rW)
#             endY = int(endY * rH)
#             results.append((startX, startY, endX, endY))
#     return results
#
#
# def main():
#     # === Step 1: Load and Enhance the Image ===
#     image_path = "baggage_tag.jpg"  # Update with your image path
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Image not found. Please check the image path.")
#         return
#
#     # --- Deep Super Resolution ---
#     sr = dnn_superres.DnnSuperResImpl_create()
#     sr_model_path = "EDSR_x4.pb"  # Update with the correct path to your SR model
#     sr.readModel(sr_model_path)
#     sr.setModel("edsr", 4)  # Using EDSR with a scaling factor of 4
#     upscaled = sr.upsample(image)
#
#     # --- Denoising ---
#     denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, h=10, hColor=10, templateWindowSize=7,
#                                                searchWindowSize=21)
#
#     # --- Sharpening ---
#     sharpened = unsharp_mask(denoised)
#
#     # === Step 2: Text Detection using EAST ===
#     east_model_path = "frozen_east_text_detection.pb"  # Update with the correct path to your EAST model
#     east_net = cv2.dnn.readNet(east_model_path)
#     boxes = detect_text_east(sharpened, east_net, min_confidence=0.5)
#
#     # === Step 3: OCR with Tesseract ===
#     custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
#     recognized_numbers = []
#     for (startX, startY, endX, endY) in boxes:
#         roi = sharpened[startY:endY, startX:endX]
#         text = pytesseract.image_to_string(roi, config=custom_config)
#         recognized_numbers.append(text.strip())
#
#     print("Detected Numbers:", recognized_numbers)
#
#     # --- Optional: Visualize the results ---
#     for (startX, startY, endX, endY) in boxes:
#         cv2.rectangle(sharpened, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
#     cv2.imshow("Enhanced Image with Detected Text", sharpened)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
