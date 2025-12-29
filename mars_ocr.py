import glob
import easyocr
import cv2
import time
import numpy as np
import utils
import copy

BATCH_SIZE = 2

def main():
    easyocr_reader = utils.init_easyocr()

    img_paths = glob.glob("cropped/*.bmp")
    img_paths += copy.deepcopy(img_paths)

    imgs = [cv2.imread(img_path) for img_path in img_paths]

    total_time = 0
    num_batches = int(len(imgs) / BATCH_SIZE)

    for i in range(num_batches):
        img_batch = imgs[i : i + BATCH_SIZE]
        img_batch_paths = img_paths[i : i + BATCH_SIZE]

        start_time = time.perf_counter_ns()

        img_batch = [utils.preprocess_frame(img, img_path) for img, img_path in zip(img_batch, img_batch_paths)]

        results = utils.read_text_batched(easyocr_reader, img_batch)

        stop_time = time.perf_counter_ns()

        elapsed_time_ms = (stop_time - start_time) / 1000000

        print(f"Time: {elapsed_time_ms:.2f} ms | Results: {results}")

        if i != 0:
            total_time += elapsed_time_ms

    avg_batch_time = total_time / (num_batches - 1)
    avg_img_time = avg_batch_time / BATCH_SIZE
    print(f"Avg Img Time: {avg_img_time:.2f} | Avg Batch Time: {avg_batch_time:.2f} ms")


if __name__ == "__main__":
    main()