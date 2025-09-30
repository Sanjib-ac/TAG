import cv2
import torch
import torch.profiler as profiler
import torch.nn as nn
from doctr.models import ocr_predictor  # pip install "python-doctr[torch]"
from doctr.io import DocumentFile
from PIL import Image
from io import BytesIO
import os
import re
import time

# Ensure a local cache for doctr is available.
os.environ["DOCTR_CACHE_DIR"] = "./"


# -------------------------------------------------------------------
# Custom SAR Decoder Components
# -------------------------------------------------------------------
class SimpleAttention(nn.Module):
    """
    A simple attention mechanism over encoder outputs.
    """

    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.attn_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_state, encoder_outputs):
        hidden_expanded = hidden_state.unsqueeze(1).expand_as(encoder_outputs)
        concat = torch.cat([encoder_outputs, hidden_expanded], dim=2)  # [B, T, 2*hidden_size]
        attn_weights = self.attn_fc(concat)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)  # [B, hidden_size]
        return context


class SARDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of output classes (for digits: 10)
            embedding_dim: int = 256,
            hidden_size: int = 512,
            max_seq_len: int = 10,  # Maximum output length (e.g. 10 digits)
            min_seq_len: int = None,  # Minimum sequence length (if decoding stops early)
            teacher_forcing: bool = True,
            eos_token: int = None  # Optional: token which indicates "end-of-sequence"
    ):
        """
        Initialize the SAR decoder with a fixed output sequence length.
        """
        super(SARDecoder, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.min_seq_len = max_seq_len if min_seq_len is None else min_seq_len
        self.teacher_forcing = teacher_forcing
        self.eos_token = eos_token

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.rnn_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.attention = SimpleAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.start_token = 0

    def forward(self, encoder_outputs, targets=None):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
        outputs = []

        for t in range(self.max_seq_len):
            context = self.attention(hidden_state, encoder_outputs)  # [B, hidden_size]
            token_embedding = self.embedding(prev_token)  # [B, embedding_dim]
            rnn_input = torch.cat([token_embedding, context], dim=1)  # [B, embedding_dim+hidden_size]
            hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
            logits = self.fc(hidden_state)  # [B, num_classes]
            outputs.append(logits.unsqueeze(1))
            if self.teacher_forcing and targets is not None:
                prev_token = targets[:, t]
            else:
                predicted = logits.argmax(dim=1)
                if (self.eos_token is not None) and (t >= self.min_seq_len):
                    if (predicted == self.eos_token).all():
                        break
                prev_token = predicted

        outputs = torch.cat(outputs, dim=1)
        return outputs


# -------------------------------------------------------------------
# MultiCameraOCR Class (Batch Processing)
# -------------------------------------------------------------------
class MultiCameraOCR:
    """
    A class to capture video from multiple cameras, process frames from all cameras in a batch
    with OCR (using docTR and mixed-precision), and display an overlay with recognized digits and FPS.
    """
    def __init__(self, camera_indices, width=1280, height=720, apply_preprocessing=False, ocr_interval=1.0):
        """
        Parameters:
            camera_indices (list of int): List of camera indices to capture.
            width (int): Frame width.
            height (int): Frame height.
            apply_preprocessing (bool): Whether to preprocess images.
            ocr_interval (float): Seconds between running OCR.
        """
        self.camera_indices = camera_indices
        self.width = width
        self.height = height
        self.apply_preprocessing = apply_preprocessing
        self.ocr_interval = ocr_interval

        # Initialize VideoCapture objects.
        self.captures = []
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise Exception(f"Unable to open camera with index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Camera {idx} resolution is set to: {width} x {height}")
            self.captures.append(cap)

        # Determine torch device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Instantiate the OCR model (docTR).
        self.model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='sar_resnet31',
            pretrained=True,
            assume_straight_pages=False
        ).to(self.device)

        # Patch the recognition module with our custom SAR decoder.
        num_classes = 10  # Recognize digits 0-9.
        self.model.reco_predictor.decoder = SARDecoder(
            num_classes=num_classes,
            max_seq_len=10,
            min_seq_len=10,
            teacher_forcing=True
        ).to(self.device)

    def perform_ocr_batch(self, frames):
        """
        Processes a batch of frames (one frame per active camera) with optimizations:
          - Optionally applies preprocessing,
          - Converts each frame from BGR to RGB,
          - Converts each frame to a PIL Image,
          - Converts each image to PNG bytes using a reusable BytesIO buffer,
          - Creates a DocumentFile from the list of bytes,
          - Runs OCR inference with mixed-precision inference,
          - Returns a list of recognized digit strings (one per camera).
        """
        img_bytes_list = []
        # Reuse a single BytesIO buffer to avoid repeatedly creating new ones.
        buffer = BytesIO()

        for frame in frames:
            if self.apply_preprocessing:
                # Preprocess: grayscale, blur, adaptive threshold.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                # Convert threshold output to a PIL image in RGB mode.
                pil_img = Image.fromarray(thresh).convert("RGB")
            else:
                # Directly convert from BGR to RGB then to a PIL image.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

            # Reset the buffer and save the PIL image as PNG.
            buffer.seek(0)
            buffer.truncate(0)
            pil_img.save(buffer, format="PNG")
            img_bytes_list.append(buffer.getvalue())

        # Create a DocumentFile from the list of PNG bytes.
        doc = DocumentFile.from_images(img_bytes_list)

        # Run inference with no_grad and mixed precision.
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.model(doc)

        # Extract OCR text from each processed page.
        ocr_batch_results = []
        for page in result.pages:
            digit_words = [
                word.value
                for block in page.blocks
                for line in block.lines
                for word in line.words
                if word.value.isdigit()
            ]
            ocr_text = ",".join(digit_words)
            ocr_batch_results.append(ocr_text)
        return ocr_batch_results

    def run(self):
        """
        Main loop: reads frames from all cameras, performs batch OCR (at a throttled rate),
        overlays recognized digits (only if exactly 10 digits are detected) and FPS,
        and displays each camera feed in its own window.
        Press 'q' to quit.
        """
        print("Press 'q' to quit.")
        prev_time = time.time()
        last_ocr_time = time.time()
        # Initialize OCR result storage for each camera.
        ocr_results = ["" for _ in self.captures]

        while True:
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            frames = []
            # Read one frame per camera.
            for idx, cap in enumerate(self.captures):
                ret, frame = cap.read()
                if not ret:
                    print(f"Camera {self.camera_indices[idx]}: Error capturing frame.")
                    frames.append(None)
                else:
                    frames.append(frame)

            # If sufficient time has elapsed, run OCR on the batch for cameras that provided a frame.
            if (current_time - last_ocr_time) >= self.ocr_interval:
                valid_frames = []
                valid_indices = []  # Keep track of which camera corresponds to which frame.
                for i, frame in enumerate(frames):
                    if frame is not None:
                        valid_frames.append(frame)
                        valid_indices.append(i)
                if valid_frames:
                    try:
                        batch_results = self.perform_ocr_batch(valid_frames)
                        # Update OCR result for each valid camera.
                        for j, i in enumerate(valid_indices):
                            ocr_results[i] = batch_results[j]
                    except Exception as e:
                        print("OCR processing error:", e)
                last_ocr_time = current_time

            # All 10 digits numbers
            for i, frame in enumerate(frames):
                if frame is not None:
                    ocr_text = ocr_results[i]

                    # Extract all 10-digit numbers from the OCR result
                    all_10_digit_numbers = re.findall(r'\b\d{10}\b', ocr_text)

                    # Join them with a separator ('|') for display
                    digits_to_display = " | ".join(all_10_digit_numbers)

                    overlay_text = f"Digits: {digits_to_display}  FPS: {fps:.2f}"
                    cv2.putText(frame, overlay_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    window_name = f"Camera Feed {self.camera_indices[i]}"
                    cv2.imshow(window_name, frame)

            # Overlay text and display each camera's frame.
            # for i, frame in enumerate(frames):
            #     if frame is not None:
            #         ocr_text = ocr_results[i]
            #         # Only display if exactly 10 digits are detected.
            #         if re.fullmatch(r'\d{10}', ocr_text):
            #             digits_to_display = ocr_text
            #         else:
            #             digits_to_display = ""
            #         overlay_text = f"Digits: {digits_to_display}  FPS: {fps:.2f}"
            #         cv2.putText(frame, overlay_text, (10, 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #         window_name = f"Camera Feed {self.camera_indices[i]}"
            #         cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release all camera resources.
        for cap in self.captures:
            cap.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Example usage: Multiple cameras processed in batch.
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # For example, process cameras with index 0 and 1.
    cam_indices = [0]
    multi_cam = MultiCameraOCR(camera_indices=cam_indices, width=1280, height=720, apply_preprocessing=False)
    multi_cam.run()
