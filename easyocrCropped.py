import glob
import easyocr
import cv2
import time
import numpy as np


ALPHANUMERIC_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
NUM_ITERS = 10


def rotate_frame(frame, img_path):
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)

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

def main():
    easyocr_reader = easyocr.Reader(['en'], gpu=True)

    img_paths = glob.glob("cropped/*.bmp")
    imgs = []

    for img_path in img_paths:
        print(img_path)
        img = cv2.imread(img_path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = rotate_frame(frame, img_path)
        imgs.append(frame)

    num_batches = int(len(imgs) / 2)
    total_time = 0

    for _ in range(NUM_ITERS):
        for i in range(num_batches):
            img0 = imgs[2 * i]
            img1 = imgs[2 * i + 1]

            start_time = time.perf_counter_ns()

            result = easyocr_reader.readtext_batched(
                [img0, img1],
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
            result0 = results[0]
            result1 = results[1]

            stop_time = time.perf_counter_ns()
            elapsed_time_ms = (stop_time - start_time) / 1000000

            if (len(result0) == 7 or len(result0) == 8) or (len(result1) == 7 or len(result1) == 8):
                print(f"Time: {elapsed_time_ms:.2f} | Text: {result0}, {result1}")

            total_time += elapsed_time_ms

    avg_time = total_time / (num_batches * NUM_ITERS)
    print(f"Avg Time: {avg_time:.2f}")

if __name__ == "__main__":
    main()
