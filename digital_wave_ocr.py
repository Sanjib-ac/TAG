import cv2
import utils
import time
import os
import logging
import numpy as np

BATCH_SIZE = 15
MAX_FRAMES = 1000

def get_output_dir(timestamp_date, timestamp):
    output_dir = os.path.join("output", timestamp_date, f"{timestamp}_{BATCH_SIZE}")
    os.makedirs(output_dir)
    return output_dir


def main():
    timestamp_date = time.strftime("%Y-%m-%d")
    timestamp_time = time.strftime("%H-%M-%S")
    timestamp = f"{timestamp_date}-{timestamp_time}"
    output_dir = get_output_dir(timestamp_date, timestamp)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{timestamp}_{BATCH_SIZE}.log"), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug(f"Batch Size: {BATCH_SIZE}")

    logger.debug("Obtaining camera...")

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_y_start = int(frame_height * 0.4)
    frame_y_end = int(frame_height * 0.7)
    frame_x_start = int(frame_width * 0.3)
    frame_x_end = int(frame_width * 0.7)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    video_path = os.path.join(output_dir, f"{timestamp}_{BATCH_SIZE}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_x_end - frame_x_start, frame_y_end - frame_y_start))

    logger.debug("Initializing ocr reader...")

    easyocr_reader = utils.init_easyocr()

    count = 0
    total_time_ms = 0
    total_ocr_time_ms = 0

    loop_continue = True
    is_first = True

    while loop_continue:
        frames = []

        start_time = time.perf_counter_ns()

        for i in range(BATCH_SIZE):
            ret, frame = cap.read()

            if ret:
                frame = frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
                # _, frame = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
                frames.append(frame)
            else:
                logger.error("Camera could not capture a frame!")
                loop_continue = False

        camera_time = time.perf_counter_ns()

        results = easyocr_reader.readtext_batched(
            frames,
            # decoder='greedy',
            # detail=0,
            # paragraph=True,
            # canvas_size=frame_width,
            # workers=0,
            allowlist=utils.ALPHANUMERIC_CHARS
        )

        ocr_time = time.perf_counter_ns()

        if not is_first:
            elapsed_time_ms = (ocr_time - start_time) / 1e6
            camera_time_ms = (camera_time - start_time) / 1e6
            ocr_time_ms = (ocr_time - camera_time) / 1e6

            total_time_ms += elapsed_time_ms
            total_ocr_time_ms += ocr_time_ms

            num_frames = len(frames)

            # logger.info(f"Frames: {count}-{count + num_frames} | Elapsed Time: {elapsed_time_ms:.2f} ms | Camera Time: {camera_time_ms:.2f} ms | OCR Time: {ocr_time_ms:.2f} ms")

            logger.info(f"Frames: {count + 1}-{count + num_frames} | OCR Time: {ocr_time_ms:.2f} ms")

            count += num_frames

            for i in range(num_frames):
                frame = frames[i]
                result = results[i]

                for (bbox, text, conf) in result:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    logger.debug(f'Text: {text!r} | Confidence: {(conf * 100):.2f}%')
                    cv2.rectangle(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (255, 0, 0), 2)

                out.write(frame)

                cv2.imshow('Camera', frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) == ord('q') or count >= MAX_FRAMES:
                    loop_continue = False

            if num_frames == 0:
                loop_continue = False

        else:
            is_first = False

    avg_time_ms = total_time_ms / count
    avg_ocr_time_ms = total_ocr_time_ms / count
    logger.info(f"Avg Elapsed Time: {avg_time_ms:.2f} ms | Avg OCR Time per Img: {avg_ocr_time_ms:.2f} ms")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()