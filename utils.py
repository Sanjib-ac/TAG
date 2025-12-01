import easyocr
import cv2
import numpy as np


ALPHANUMERIC_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def sharpen_edges(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    return cv2.filter2D(frame, -1, kernel)

def convert2binary(frame):
    _, binary_frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
    return binary_frame

def get_frame_middle_area(frame):
    # Get image dimensions
    h, w = frame.shape[:2]

    # Define 50% crop boundaries (centered)
    start_x = w // 4
    end_x   = w - w // 4
    start_y = h // 4
    end_y   = h - h // 4

    # Crop the middle 50%
    cropped = frame[start_y : end_y, start_x : end_x]

    return cropped

def find_frame_angle(frame):
    middle_frame = get_frame_middle_area(frame)

    contours, _ = cv2.findContours(middle_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(middle_frame, contours, -1, (0, 255, 0), 2)  # green contours with thickness=2
    #
    # # Show the result
    # cv2.imshow("Contours", middle_frame)
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

    # # Convert to box points (4 corners)
    # box = cv2.boxPoints(rect)
    # box = np.intp(box)  # convert to integer coordinates
    #
    # color = cv2.cvtColor(middle_frame, cv2.COLOR_GRAY2RGB)
    #
    # # Draw the rectangle
    # cv2.drawContours(color, [box], 0, (0, 255, 0), 2)
    # cv2.imshow("Frame", color)
    # cv2.waitKey(100)

    # print("Raw angle =", angle)

    # The angle returned by minAreaRect can be in [-90, 0).
    # We need to adjust it to get the correct rotation.
    if angle < -45:
        return -(90 + angle)

    return -angle

def rotate_frame(frame, angle):
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)

    # rotate our image by * degrees around the center of the images
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
    return cv2.warpAffine(frame, M, (new_w, new_h), borderValue=(0, 0, 0))

def preprocess_frame(frame, img_path):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = sharpen_edges(frame)
    frame = convert2binary(frame)

    # angle = find_frame_angle(frame)
    # print(angle)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(100)

    angle = -45
    if 'right' in img_path:
        angle += -90

    frame = rotate_frame(frame, angle)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(100)
    # print(angle)

    return frame

def validate_text(result):
    if (len(result) == 7 and result.isdecimal()) or (len(result) == 8 and result[:7].isdecimal() and result[7:].isalpha()):
        return result

    return f"INVALID {result}"

def init_easyocr():
    return easyocr.Reader(['en'], gpu=True, cudnn_benchmark=True)

def read_text_batched(easyocr_reader, imgs):
    result = easyocr_reader.readtext_batched(
        imgs,
        n_width=250,
        n_height=250,
        decoder='greedy',
        detail=0,
        paragraph=False,
        canvas_size=250,
        workers=0,
        allowlist=ALPHANUMERIC_CHARS
    )

    return ["".join(r) for r in result]