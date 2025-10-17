import glob
import easyocr
import cv2
import time
import numpy as np


ALPHANUMERIC_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
NUM_ITERS = 10

def rotate_img(img, angle):
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # rotate our image by 45 degrees around the center of the images
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with a black background
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))


def rotate_frame(frame, img_path):
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)

    # angle = get_text_orientation(frame)
    # print(f"Angle: {angle}")

    angle = -45
    if 'right' in img_path:
        angle += -90

    # rotate our image by 45 degrees around the center of the images
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with a black background
    rotated_frame = cv2.warpAffine(frame, M, (new_w, new_h), borderValue=(0, 0, 0))

    return rotated_frame


def get_text_orientation(frame):
    # kernel = np.ones((15, 15), np.uint8)
    # frame = cv2.dilate(frame, kernel, iterations=2)
    # kernel = np.ones((19, 19), np.uint8)
    # frame = cv2.erode(frame, kernel, iterations=2)
    # cv2.imshow('Thicker Outline', frame)
    # cv2.waitKey(100)
    # cv2.imshow('Thicker Outline', frame)
    # cv2.waitKey(100)
    # 2. Find contours of the text block
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # green contours with thickness=2
    #
    # # Show the result
    # cv2.imshow("Contours", frame)
    # cv2.waitKey(100)

    # Combine all contours into a single one to find the overall bounding box
    if not contours:
        print("No contours found.")
        return 0  # Return original image if no text is found

    all_points = np.concatenate(contours, axis=0)

    # 3. Calculate the angle using the minimum area rectangle
    # This function returns ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(all_points)
    angle = rect[-1]

    # The angle returned by minAreaRect can be in [-90, 0).
    # We need to adjust it to get the correct rotation.
    if angle < -45:
        return -(90 + angle)

    return -angle


def remove_only_long_lines(gray):
    """Remove long straight lines without affecting text."""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Detect line segments (no morphology this time!)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=int(gray.shape[1] * 0.6),
                            maxLineGap=15)

    mask = np.ones_like(gray, np.uint8) * 255
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # Compute slope (angle) to ensure it’s a real straight line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle) > 80:  # mostly horizontal or vertical
                cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=4)

    cleaned = cv2.bitwise_and(gray, mask)
    return cleaned

def get_text_orientation2(img):
    # 1. Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise + edge detection
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imshow("Edges", edges)
    cv2.waitKey(100)

    # 3. Compute gradients
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(sobely, sobelx) * 180 / np.pi

    # Mask zero edges
    valid = edges > 0
    if np.count_nonzero(valid) == 0:
        print("No edges")
        return 0

    # 4. Compute dominant orientation histogram
    hist, bins = np.histogram(angles[valid], bins=180, range=(-90, 90))
    dominant_angle = bins[np.argmax(hist)]

    return dominant_angle

def preprocess_frame(frame, img_path="", sharpen_edges=True, convert2binary=True):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if sharpen_edges:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    if convert2binary:
        # 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
        # _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # rotate image to straighten out text
    # frame = rotate_frame(frame, img_path)

    # rotated_imgs = [rotate_img(frame, angle) for angle in range(15, 360, 15)]
    rotated_imgs = [rotate_img(frame, angle) for angle in range(45, 360, 45)]
    return rotated_imgs

    # cv2.imshow("Frame", frame)
    # cv2.waitKey(100)

    # return frame

def validate_text(result):
    if (len(result) == 7 and result.isdecimal()) or (len(result) == 8 and result[:7].isdecimal() and result[7:].isalpha()):
        return result

    return f"INVALID {result}"

def main():
    easyocr_reader = easyocr.Reader(['en'], gpu=True)

    img_paths = glob.glob("cropped/*.bmp")

    num_batches = int(len(img_paths) / 2)
    total_time = 0

    for j in range(NUM_ITERS + 1):
        for i in range(num_batches):
            img0_path = img_paths[2 * i]
            img1_path = img_paths[2 * i + 1]

            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)

            # print(f"Orientation: {get_text_orientation2(img0)}, {get_text_orientation2(img1)}")
            # gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            # sans = remove_only_long_lines(gray)
            # cv2.imshow("frame", sans)
            # cv2.waitKey(100)

            # gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            #
            # # 2. Find contours of the text block
            # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # cv2.drawContours(img0, contours, -1, (0, 255, 0), 2)  # green contours with thickness=2
            #
            # # Show the result
            # cv2.imshow("Contours", img0)
            # cv2.waitKey(100)

            start_time = time.perf_counter_ns()

            sharpen_edges = True
            convert2binary = True

            # img0 = preprocess_frame(img0, img0_path, sharpen_edges, convert2binary)
            # img1 = preprocess_frame(img1, img1_path, sharpen_edges, convert2binary)

            imgs0 = preprocess_frame(img0, img0_path, sharpen_edges, convert2binary)
            imgs1 = preprocess_frame(img1, img1_path, sharpen_edges, convert2binary)

            # imgs = imgs0.extend(imgs1)

            # print(f"Orientation: {get_text_orientation2(img0)}, {get_text_orientation2(img1)}")

            result = easyocr_reader.readtext_batched(
                # [img0, img1],
                imgs0,
                n_width=250,
                n_height=250,
                decoder='greedy',
                detail=0,
                paragraph=False,
                canvas_size=250,
                workers=0,
                allowlist=ALPHANUMERIC_CHARS
            )

            results = ["".join(r) for r in result]
            print(results)

            # text0 = validate_text(results[0])
            # text1 = validate_text(results[1])

            stop_time = time.perf_counter_ns()
            elapsed_time_ms = (stop_time - start_time) / 1000000

            # print(f"Time: {elapsed_time_ms:.2f} ms | Text: {text0}, {text1}")

            print(f"Time: {elapsed_time_ms:.2f} ms")

            if j != 0:
                total_time += elapsed_time_ms

    avg_time = total_time / (num_batches * NUM_ITERS)
    print(f"Avg Time: {avg_time:.2f} ms")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
