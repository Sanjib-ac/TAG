from paddleocr import PaddleOCR
import numpy as np
import cv2
import time
import easyocr


def straighten_image(img):
    """
    Detects the angle of the text block in an image and rotates it to be horizontal.
    """
    # Load the image
    # img = cv2.imread(img)
    print(f"Img shape: {img.shape}")
    # Convert the image to grayscale
    # Preprocess the image for contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2_imshow(binary)

    # 2. Find contours of the text block
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours into a single one to find the overall bounding box
    if not contours:
        print("No contours found.")
        return img  # Return original image if no text is found

    all_points = np.concatenate(contours, axis=0)

    # 3. Calculate the angle using the minimum area rectangle
    # This function returns ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(all_points)
    angle = rect[-1]

    # The angle returned by minAreaRect can be in [-90, 0).
    # We need to adjust it to get the correct rotation.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # print(f"Detected Angle: {angle:.2f} degrees")

    # 4. Rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the actual rotation
    rotated_img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    # cv2_imshow(binary)
    # cv2_imshow(rotated_img)
    # cv2.imshow("Ori", img)
    # cv2.imshow("Bina", binary)
    # cv2.imshow("rotated", rotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return rotated_img


def ocrPaddle(croImg):
    ocr = PaddleOCR(lang='en', use_textline_orientation=False)
    straightened_image = straighten_image(croImg)
    start_time_ocr = time.time()
    result = ocr.predict(straightened_image)
    if result and result[0]:
        # The recognized texts are in a list under the 'rec_texts' key
        recognized_texts = result[0]['rec_texts']
        # The confidence scores are in a list under the 'rec_scores' key
        confidence_scores = result[0]['rec_scores']
        # The coordinates of the text box are under 'dt_polys'
        text_location = result[0]['dt_polys']
         # Since there's only one line of text, we access the first item
        if recognized_texts:
            text = recognized_texts[0]
            score = confidence_scores[0]
            print(f"Recognized Text: {text}")
            print(f"Confidence Score: {score:.2f}")
            # print(f"Location: {text_location[0]}")
        end_time_ocr = time.time()
        duration_ocr = end_time_ocr - start_time_ocr
        # print(f"reader.readtext() took:  {duration_ocr:.4f} seconds")

def easOcr(img):
    alphanumeric = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    reader = easyocr.Reader(['en'], gpu=True)
    start_time_ocr = time.time()
    straightened_images = straighten_image(croImg)  # [straighten_image(o) for o in cropped_images[1:]]
    result = reader.readtext(straightened_images, decoder='greedy',
                             detail=0,paragraph=False,
                             canvas_size=720,
                             workers=0,
                             allowlist=alphanumeric)

    if result:
        print(f"Recognized Text: {result[0]}")
    end_time_ocr = time.time()

    duration_ocr = end_time_ocr - start_time_ocr
    print(f"easOcr took:  {duration_ocr:.4f} seconds")


if __name__ == '__main__':
    croImg = cv2.imread(".\cropped\ocrCrop.png")
    # ocrPaddle(croImg)
    easOcr(croImg)

