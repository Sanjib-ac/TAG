import cv2
import utils
import time
import os
import logging
import numpy as np

def get_output_dir(timestamp):
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir)
    return output_dir


def main():
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = get_output_dir(timestamp)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{timestamp}.log"), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug("Obtaining camera")

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_y_start = int(frame_height * 0.45)
    frame_y_end = int(frame_height* 0.65)
    frame_x_start = int(frame_width * 0.125)
    frame_x_end = int(frame_width * 0.55)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    video_path = os.path.join(output_dir, f"{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_x_end - frame_x_start, frame_y_end - frame_y_start))

    logger.debug("Initializing ocr reader")

    easyocr_reader = utils.init_easyocr()

    count = 0
    total_time_ms = 0
    total_ocr_time_ms = 0

    loop_continue = True

    while loop_continue:
        start_time = time.perf_counter_ns()

        ret, frame = cap.read()

        frame = frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

        camera_time = time.perf_counter_ns()

        result = easyocr_reader.readtext(
            frame,
            # decoder='greedy',
            # detail=0,
            # paragraph=True,
            # canvas_size=frame_width,
            # workers=0,
            allowlist=utils.ALPHANUMERIC_CHARS
        )

        ocr_time = time.perf_counter_ns()

        elapsed_time_ms = (ocr_time - start_time) / 1e6
        camera_time_ms = (camera_time - start_time) / 1e6
        ocr_time_ms = (ocr_time - camera_time) / 1e6

        if count != 0:
            total_time_ms += elapsed_time_ms
            total_ocr_time_ms += ocr_time_ms

        count += 1

        logger.info(f"Frame: {count} | Elapsed Time: {elapsed_time_ms:.2f} ms | Camera Time: {camera_time_ms:.2f} ms | OCR Time: {ocr_time_ms:.2f} ms")

        for (bbox, text, conf) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            logger.info(f'Text: {text!r} | Confidence: {(conf * 100):.2f}%')
            cv2.rectangle(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (255, 0, 0), 2)

        out.write(frame)

        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            loop_continue = False

    avg_time_ms = total_time_ms / (count - 1)
    avg_ocr_time_ms = total_ocr_time_ms / (count - 1)
    logger.info(f"Avg Elapsed Time: {avg_time_ms:.2f} ms | Avg OCR Time: {avg_ocr_time_ms:.2f} ms")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()