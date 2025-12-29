import glob
import easyocr
import cv2
import time
import numpy as np
import sys
import torch
import os
import argparse
import json

ALPHANUMERIC_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def sharpen_edges(frame, kernel):
    # default:
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])

    return cv2.filter2D(frame, -1, kernel)


def convert2binary(frame, thresh):
    # default thresh: 75
    _, binary_frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)
    return binary_frame


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


def preprocess_frame(frame, img_path, sharpen_kernel, thresh):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if sharpen_kernel is not None:
        frame = sharpen_edges(frame, sharpen_kernel)

    if thresh is not False:
        frame = convert2binary(frame, thresh)

    angle = -45
    if 'right' in img_path:
        angle += -90

    frame = rotate_frame(frame, angle)

    return frame


def read_text_batched(easyocr_reader, imgs, args):
    result = easyocr_reader.readtext_batched(
        imgs,
        n_width=args.n_width,
        n_height=args.n_height,
        decoder=args.decoder,
        detail=args.detail,
        paragraph=args.paragraph,
        canvas_size=args.canvas_size,
        workers=args.workers,
        allowlist=args.chars
    )

    return ["".join(r) for r in result]


def ocr(args, img_paths, sharpen_kernel, thresh):
    use_gpu = not args.cpu
    easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, cudnn_benchmark=use_gpu)

    imgs = [cv2.imread(img_path) for img_path in img_paths]

    total_time = 0
    num_batches = 0

    for i in range(0, len(imgs), args.batch):
        img_batch = imgs[i: i + args.batch]
        img_batch_paths = img_paths[i: i + args.batch]

        start_time = time.perf_counter_ns()

        img_batch = [preprocess_frame(img, img_path, sharpen_kernel, thresh) for img, img_path in
                     zip(img_batch, img_batch_paths)]

        results = read_text_batched(easyocr_reader, img_batch, args)

        stop_time = time.perf_counter_ns()

        elapsed_time_ms = (stop_time - start_time) / 1000000

        output = {"results": results, "time": elapsed_time_ms}

        print(json.dumps(output), flush=True)

        total_time += elapsed_time_ms
        num_batches += 1

    avg_batch_time = total_time / num_batches
    avg_img_time = avg_batch_time / args.batch

    if args.show_time:
        print(f"Avg Img Time: {avg_img_time:.2f} | Avg Batch Time: {avg_batch_time:.2f} ms", flush=True)


class PrefixStdout:
    def __init__(self, orig):
        self.orig = orig
        self.at_line_start = True

    def write(self, data):
        # This line MUST be removed or commented out
        # data = data.replace('\r', '\n')

        parts = data.splitlines(keepends=True)
        for chunk in parts:
            # This logic correctly passes through the output
            self.orig.write(chunk)
            self.at_line_start = chunk.endswith("\n")
        self.orig.flush()

    def flush(self):
        self.orig.flush()


def is_multiprocessing_worker():
    """Check if this is a multiprocessing worker process"""
    return '--multiprocessing-fork' in sys.argv or 'parent_pid=' in ' '.join(sys.argv)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, required=True, help="Directory of cropped image files on disk")

    parser.add_argument("--ext", type=str, default="bmp",
                        help="File extension of the image files on disk (default: 'bmp')")
    parser.add_argument("--batch", type=int, default=1, help="Number of images per batch (default: 1)")

    parser.add_argument("--show-time", action='store_true', help="Show timing information")

    parser.add_argument("--sharpen-kernel", type=str, default="0,-1,0;-1,5,-1;0,-1,0",
                        help="Kernel for sharpening edges for preprocessing. Disable sharpening by setting value to 'false' (default: '0,-1,0;-1,5,-1;0,-1,0')")
    parser.add_argument("--thresh", type=int, default=75,
                        help="Threshold value for preprocessing. Disable thresholding by setting value to -1 (default: 75)")

    parser.add_argument("--cpu", action='store_true', help="Use CPU for OCR processing")

    parser.add_argument("--n-width", type=int, default=250, help="Width for OCR processing (default: 250)")
    parser.add_argument("--n-height", type=int, default=250, help="Height for OCR processing (default: 250)")
    parser.add_argument("--decoder", type=str, default="greedy", choices=["greedy", "beamsearch", "wordbeamsearch"],
                        help="Decoder for OCR processing (default: 'greedy')")
    parser.add_argument("--detail", type=int, default=0, help="Detail level for OCR processing (default: 0)")
    parser.add_argument("--paragraph", action='store_true', help="Enable paragraph mode for OCR processing")
    parser.add_argument("--canvas-size", type=int, default=250, help="Canvas size for OCR processing (default: 250)")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for OCR processing (default: 0)")
    parser.add_argument("--chars", type=str, default=ALPHANUMERIC_CHARS,
                        help="Allowed characters for OCR (default: alphanumeric characters)")

    return parser.parse_args()


def main():
    sys.stdout = PrefixStdout(sys.stdout)
    sys.stderr = PrefixStdout(sys.stderr)
    sys.stdout.isatty = lambda: False
    sys.stderr.isatty = lambda: False

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA version:", torch.version.cuda, flush=True)
    # print("CWD:", os.getcwd(), flush=True)
    # print("Python exe:", sys.executable, flush=True)

    try:
        args = parse_args()
        if args is None:
            sys.exit(0)

        print(f"Checking image directory path: {args.dir}", flush=True)
        if not os.path.exists(args.dir):
            print(f"ERROR: Image directory path not found: {args.dir}", flush=True)
            print(f"Full path: {os.path.abspath(args.dir)}", flush=True)
            sys.exit(1)
        else:
            print(f"Image directory path exists: {args.dir}", flush=True)

        print(f"Checking images in directory path with extension: {args.ext}", flush=True)
        img_dir_path_glob = os.path.join(args.dir, f"*.{args.ext}")
        img_paths = glob.glob(img_dir_path_glob)
        if len(img_paths) <= 0:
            print(f"ERROR: Images with extension not found: {args.ext}", flush=True)
            print(f"Full glob path: {os.path.abspath(img_dir_path_glob)}", flush=True)
            sys.exit(1)
        else:
            print(f"Images with extension exist: {args.ext}", flush=True)
            print(f"Number of images with extension: {len(img_paths)}", flush=True)
            print(f"Full glob path: {os.path.abspath(img_dir_path_glob)}", flush=True)

        print(f"Checking batch size: {args.batch}", flush=True)
        if len(img_paths) % args.batch != 0:
            print(
                f"WARNING: Batch size does not evenly divide the number of images: {args.batch} (batch size), {len(img_paths)} (number of images)",
                flush=True)
        else:
            print(
                f"Batch size evenly divides the number of images: {args.batch} (batch size), {len(img_paths)} (number of images)",
                flush=True)

        sharpen_kernel = None
        print(f"Checking sharpen kernel: {args.sharpen_kernel}", flush=True)
        if args.sharpen_kernel.lower() == "false":
            print(f"Sharpening disabled", flush=True)
        else:
            sharpen_kernel = np.array(
                [[int(x) for x in row.split(",")]
                 for row in args.sharpen_kernel.split(";")],
                dtype=np.int32
            )
            if sharpen_kernel.shape[0] != sharpen_kernel.shape[1] or sharpen_kernel.shape[0] % 2 == 0:
                print(f"ERROR: Sharpen kernel must be square and have odd dimensions: {sharpen_kernel.shape}",
                      flush=True)
                sys.exit(1)
            else:
                print(f"Sharpen kernel is valid: {sharpen_kernel.shape}", flush=True)

        thresh = False
        print(f"Checking threshold value: {args.thresh}", flush=True)
        if args.thresh == -1:
            print(f"Thresholding disabled", flush=True)
        elif args.thresh < 0 or args.thresh > 255:
            print(f"ERROR: Threshold value must be between 0 and 255: {args.thresh}", flush=True)
            sys.exit(1)
        else:
            thresh = args.thresh
            print(f"Threshold value is valid: {args.thresh}", flush=True)

        try:
            print(f"Starting OCR...", flush=True)

            ocr(args, img_paths, sharpen_kernel, thresh)

            print("OCR completed successfully!", flush=True)
            sys.exit(0)
            # print(f"Results: {results}", flush=True)

        except Exception as e:
            print(f"ERROR during OCR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"ERROR in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        # input("Press Enter to exit...")  # Keep console open to see error


if __name__ == "__main__":
    main()
