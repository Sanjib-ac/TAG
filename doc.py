import cv2
import torch
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
        # Expand the hidden state and concatenate with encoder outputs.
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
        If teacher_forcing is False and eos_token is defined, decoding will stop early
        only if the EOS token is predicted _and_ the decoded length is at least min_seq_len.
        """
        super(SARDecoder, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.min_seq_len = max_seq_len if min_seq_len is None else min_seq_len
        self.teacher_forcing = teacher_forcing
        self.eos_token = eos_token  # For instance, if your vocabulary included an EOS token

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.rnn_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.attention = SimpleAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        # Define a start token (adjust if necessary)
        self.start_token = 0

    def forward(self, encoder_outputs, targets=None):
        """
        Args:
            encoder_outputs: Tensor of shape [B, seq_len_enc, hidden_size]
            targets: Tensor of shape [B, max_seq_len] (if available during training)
        Returns:
            outputs: Tensor of shape [B, L, num_classes], where L is up to max_seq_len.
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)

        # Initialize with the start token.
        prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
        outputs = []
        # Iterate for a maximum of max_seq_len steps.
        for t in range(self.max_seq_len):
            context = self.attention(hidden_state, encoder_outputs)  # [B, hidden_size]
            token_embedding = self.embedding(prev_token)  # [B, embedding_dim]
            rnn_input = torch.cat([token_embedding, context], dim=1)  # [B, embedding_dim+hidden_size]
            hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
            logits = self.fc(hidden_state)  # [B, num_classes]
            outputs.append(logits.unsqueeze(1))  # Append along time dimension.

            # Determine next token.
            if self.teacher_forcing and targets is not None:
                prev_token = targets[:, t]
            else:
                predicted = logits.argmax(dim=1)
                # If an EOS token is defined, allow early stopping only if we've reached the minimum length.
                if (self.eos_token is not None) and (t >= self.min_seq_len):
                    if (predicted == self.eos_token).all():
                        break
                prev_token = predicted

        outputs = torch.cat(outputs, dim=1)  # [B, L, num_classes]
        return outputs


# -------------------------------------------------------------------
# CameraOCR Class
# -------------------------------------------------------------------
class CameraOCR:
    """
    A class to capture video from a specified camera, process each frame
    with OCR (using docTR), and display an overlay with recognized digits and FPS.
    """

    def __init__(self, camera_index=0, width=1280, height=720, apply_preprocessing=False):
        """
        Initialize the camera and the OCR model.

        Parameters:
            camera_index (int): Index of the camera (0, 1, etc.).
            width (int): Desired frame width.
            height (int): Desired frame height.
            apply_preprocessing (bool): Whether to apply image preprocessing before OCR.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.apply_preprocessing = apply_preprocessing

        # Set up camera capture.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception(f"Unable to open camera with index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Camera {camera_index} resolution is set to: {width} x {height}")

        # Set the device for torch.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Instantiate the OCR model.
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31',
                                   pretrained=True, assume_straight_pages=False).to(self.device)
        # Patch the recognition module with our custom SAR decoder.
        num_classes = 10  # For digits 0-9
        self.model.reco_predictor.decoder = SARDecoder(num_classes=num_classes,
                                                       max_seq_len=10,
                                                       min_seq_len=10,
                                                       teacher_forcing=True).to(self.device)

    def perform_ocr(self, frame):
        """
        Process a single frame:
          - Optionally applies preprocessing,
          - Converts the image from BGR to RGB and to a PIL Image,
          - Saves the image in an in-memory buffer,
          - Runs OCR inference via docTR,
          - Returns a string of recognized digits.
        """
        if self.apply_preprocessing:
            # Convert to grayscale, blur, threshold, then back to BGR.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Convert from BGR to RGB and create a PIL Image.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Save the image to an in-memory buffer as PNG.
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Create a DocumentFile from the image bytes.
        doc = DocumentFile.from_images([img_bytes])

        # Run OCR inference.
        with torch.no_grad():
            result = self.model(doc)

        # Collect recognized words that contain only digits.
        digit_words = [
            word.value
            for page in result.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
            if word.value.isdigit()
        ]
        return ",".join(digit_words)

    def run(self):
        """
        Main loop: captures frames, performs OCR on each frame synchronously,
        overlays recognized digits (if exactly 10 are present) and FPS, and displays the result.
        Press 'q' to exit the loop.
        """
        print("Press 'q' to quit.")
        prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.camera_index}: Error capturing frame.")
                break

            # Compute the FPS.
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Run OCR synchronously for the current frame.
            try:
                ocr_text = self.perform_ocr(frame)
            except Exception as e:
                print(f"Camera {self.camera_index}: OCR processing error:", e)
                ocr_text = ""

            # Only display if exactly 10 digits are detected.
            if re.fullmatch(r'\d{10}', ocr_text):
                digits_to_display = ocr_text
            else:
                digits_to_display = ""

            overlay_text = f"Digits: {digits_to_display}  FPS: {fps:.2f}"
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame in a window named by the camera index.
            cv2.imshow(f"Camera Feed {self.camera_index}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create an instance for camera 0.
    camera0 = CameraOCR(camera_index=0, width=680, height=420, apply_preprocessing=False)
    camera0.run()

    # To run another camera concurrently, you could create another instance:
    # camera1 = CameraOCR(camera_index=1, width=1280, height=720, apply_preprocessing=False)
    # camera1.run()

# import cv2
# import torch
# import torch.nn as nn
# from doctr.models import ocr_predictor  # pip install "python-doctr[torch]" ; pip install python-doctr
# from doctr.io import DocumentFile
# from PIL import Image
# from io import BytesIO
# import tempfile
# import os
# import re
# import time
# import queue
# import threading
#
# os.environ["DOCTR_CACHE_DIR"] = "./"
#
#
# # -------------------------------------------------------------------
# # Custom SAR Decoder Components
# # -------------------------------------------------------------------
# class SimpleAttention(nn.Module):
#     """
#     A simple attention mechanism over encoder outputs.
#     """
#
#     def __init__(self, hidden_size):
#         super(SimpleAttention, self).__init__()
#         self.attn_fc = nn.Linear(hidden_size * 2, 1)
#
#     def forward(self, hidden_state, encoder_outputs):
#         # hidden_state: [B, hidden_size]
#         # encoder_outputs: [B, seq_len_enc, hidden_size]
#         hidden_expanded = hidden_state.unsqueeze(1).expand_as(encoder_outputs)
#         concat = torch.cat([encoder_outputs, hidden_expanded], dim=2)  # [B, T, 2*hidden_size]
#         attn_weights = self.attn_fc(concat)  # [B, T, 1]
#         attn_weights = torch.softmax(attn_weights, dim=1)
#         context = torch.sum(attn_weights * encoder_outputs, dim=1)  # [B, hidden_size]
#         return context
#
#
# class SARDecoder(nn.Module):
#     def __init__(
#             self,
#             num_classes: int,  # Number of output classes (for digits: 10)
#             embedding_dim: int = 256,
#             hidden_size: int = 512,
#             max_seq_len: int = 10,  # Maximum output length (e.g. 10 digits)
#             min_seq_len: int = None,  # Minimum sequence length (if decoding stops early)
#             teacher_forcing: bool = True,
#             eos_token: int = None  # Optional: token which indicates "end-of-sequence"
#     ):
#         """
#         Initialize the SAR decoder with a fixed output sequence length.
#         If teacher_forcing is False and eos_token is defined, decoding will stop early
#         only if the EOS token is predicted _and_ the decoded length is at least min_seq_len.
#         """
#         super(SARDecoder, self).__init__()
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.hidden_size = hidden_size
#         self.max_seq_len = max_seq_len
#         # If a minimum length is not provided, default to max_seq_len.
#         self.min_seq_len = max_seq_len if min_seq_len is None else min_seq_len
#         self.teacher_forcing = teacher_forcing
#         self.eos_token = eos_token  # For instance, if your vocabulary included an EOS token
#
#         self.embedding = nn.Embedding(num_classes, embedding_dim)
#         self.rnn_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
#         self.attention = SimpleAttention(hidden_size)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#         # Define a start token (adjust if necessary)
#         self.start_token = 0
#
#     def forward(self, encoder_outputs, targets=None):
#         """
#         Args:
#             encoder_outputs: Tensor of shape [B, seq_len_enc, hidden_size]
#             targets: Tensor of shape [B, max_seq_len] (if available during training)
#         Returns:
#             outputs: Tensor of shape [B, L, num_classes], where L is up to max_seq_len.
#         """
#         batch_size = encoder_outputs.size(0)
#         device = encoder_outputs.device
#
#         hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
#         cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
#
#         # Initialize with the start token.
#         prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
#         outputs = []
#         # We'll use a for-loop for a maximum of max_seq_len iterations.
#         for t in range(self.max_seq_len):
#             context = self.attention(hidden_state, encoder_outputs)  # [B, hidden_size]
#             token_embedding = self.embedding(prev_token)  # [B, embedding_dim]
#             rnn_input = torch.cat([token_embedding, context], dim=1)  # [B, embedding_dim+hidden_size]
#             hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
#             logits = self.fc(hidden_state)  # [B, num_classes]
#             outputs.append(logits.unsqueeze(1))  # Append along time dimension.
#
#             # Determine next token.
#             if self.teacher_forcing and targets is not None:
#                 prev_token = targets[:, t]
#             else:
#                 predicted = logits.argmax(dim=1)
#                 # If an EOS token is defined, allow early stopping only if we've reached the minimum length.
#                 if (self.eos_token is not None) and (t >= self.min_seq_len):
#                     # Check element-wise if EOS has been predicted.
#                     # For simplicity, if all items in the batch predict EOS, then exit early.
#                     if (predicted == self.eos_token).all():
#                         break
#                 prev_token = predicted
#
#         outputs = torch.cat(outputs, dim=1)  # [B, L, num_classes], where L <= max_seq_len
#         return outputs
#
#
# APPLY_PREPROCESSING = False  # Set to True to apply preprocessing; False to skip.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)
# # Instantiate the OCR model
# model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31',
#                       pretrained=True, assume_straight_pages=False).to(device)
# # model.to(device)
#
# # Diagnostic: Print available attributes to confirm submodules.
# # print("OCRPredictor attributes:", dir(model))
#
# # Patch the recognition module—here available as "reco_predictor"—with custom SAR decoder.
# num_classes = 10  # For digits 0-9
# model.reco_predictor.decoder = SARDecoder(num_classes=num_classes, max_seq_len=10, min_seq_len=10,
#                                           teacher_forcing=True).to(device)
#
# # Global variables and thread synchronization
# ocr_result_lock = threading.Lock()
# global_ocr_result = ""  # This will hold the latest OCR result (digits only)
#
# frame_queue = queue.Queue(maxsize=1)  # Queue to pass frames from the main loop to the worker.
#
#
# def perform_ocr(frame):
#     """
#     Processes a single frame:
#       - Optionally applies preprocessing,
#       - Converts from BGR to RGB and to a PIL Image,
#       - Uses an in-memory buffer (instead of saving to disk),
#       - Runs OCR inference via docTR, and returns a string of recognized digits.
#     """
#     # Optional preprocessing.
#     if APPLY_PREPROCESSING:
#         # Convert to grayscale, blur, threshold, then convert back to BGR.
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         thresh = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
#         frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#
#     # Convert from BGR to RGB then to a PIL image.
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(frame_rgb)
#
#     # Save the image in-memory as PNG to a BytesIO buffer.
#     buffer = BytesIO()
#     pil_img.save(buffer, format="PNG")
#     # Retrieve the image as bytes.
#     img_bytes = buffer.getvalue()
#
#     # Create a DocumentFile from the image bytes.
#     doc = DocumentFile.from_images([img_bytes])
#
#     # Run OCR inference.
#     with torch.no_grad():
#         result = model(doc)
#
#     # Collect recognized words that are digits.
#     digit_words = [
#         word.value
#         for page in result.pages
#         for block in page.blocks
#         for line in block.lines
#         for word in line.words
#         if word.value.isdigit()
#     ]
#     return ",".join(digit_words)
#
#
# # # OCR Worker Function: Processes each frame for OCR.
# # def perform_ocr(frame):
# #     """
# #     Processes a single frame:
# #       - Optionally applies preprocessing,
# #       - Converts from BGR to RGB and to a PIL Image,
# #       - Saves to a temporary PNG file (for DocumentFile),
# #       - Runs OCR inference, and returns a string of recognized digits.
# #     """
# #     if APPLY_PREPROCESSING:
# #         # Preprocessing: convert to grayscale, blur, threshold, then convert back to BGR.
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #         thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                        cv2.THRESH_BINARY_INV, 11, 2)
# #         frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
# #
# #     # Convert frame from BGR to RGB and then to a PIL image.
# #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     pil_img = Image.fromarray(frame_rgb)
# #
# #     # Save the image temporarily as PNG, then load it via DocumentFile.
# #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
# #         tmp_filename = tmp.name
# #         pil_img.save(tmp, format="PNG")
# #     doc = DocumentFile.from_images([tmp_filename])
# #     os.remove(tmp_filename)
# #
# #     # Run OCR inference.
# #     with torch.no_grad():
# #         result = model(doc)
# #
# #     # Collect recognized words that are digits.
# #     digit_words = [
# #         word.value
# #         for page in result.pages
# #         for block in page.blocks
# #         for line in block.lines
# #         for word in line.words
# #         if word.value.isdigit()
# #     ]
# #     return ",".join(digit_words)
#
#
# # def perform_ocr(frame):
# #     """
# #     Processes a single frame:
# #       - Optionally applies preprocessing,
# #       - Converts from BGR to RGB and to a PIL Image,
# #       - Saves to a temporary PNG file (for DocumentFile),
# #       - Runs OCR inference, and returns a string of recognized digits.
# #     """
# #     if APPLY_PREPROCESSING:
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #         thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                        cv2.THRESH_BINARY_INV, 11, 2)
# #         frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
# #
# #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     pil_img = Image.fromarray(frame_rgb)
# #
# #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
# #         tmp_filename = tmp.name
# #         pil_img.save(tmp, format="PNG")
# #
# #     doc = DocumentFile.from_images([tmp_filename])
# #     os.remove(tmp_filename)
# #     result = model(doc)
# #
# #     recognized_text = ""
# #     for page in result.pages:
# #         for block in page.blocks:
# #             for line in block.lines:
# #                 for word in line.words:
# #                     recognized_text += word.value + " "
# #
# #     numbers = re.findall(r'\d+', recognized_text.strip())
# #     digits_only = ",".join(numbers)
# #     return digits_only
#
# def ocr_worker():
#     global global_ocr_result
#     while True:
#         frame = frame_queue.get()
#         if frame is None:  # Sentinel value to exit.
#             frame_queue.task_done()
#             break
#         try:
#             result = perform_ocr(frame)
#             with ocr_result_lock:
#                 global_ocr_result = result
#         except Exception as e:
#             print("OCR processing error:", e)
#         frame_queue.task_done()
#
#
# # Start the OCR worker thread.
# worker_thread = threading.Thread(target=ocr_worker, daemon=True)
# worker_thread.start()
#
# # Set up camera capture.
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Adjust the camera index (e.g., 0 or 1) as needed.
# if not cap.isOpened():
#     print("Unable to open camera")
#     exit()
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# print(f"Camera resolution is set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
# print("Press 'q' to quit.")
#
# prev_time = time.time()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error capturing frame")
#         break
#
#     # Calculate frames per second.
#     current_time = time.time()
#     fps = 1.0 / (current_time - prev_time)
#     prev_time = current_time
#
#     try:
#         frame_queue.put(frame.copy(), block=False)
#     except queue.Full:
#         pass
#
#     # Retrieve the latest OCR result.
#     with ocr_result_lock:
#         ocr_text = global_ocr_result
#
#     # Only display if exactly 10 digits are detected.
#     if re.fullmatch(r'\d{10}', ocr_text):
#         digits_to_display = ocr_text
#     else:
#         digits_to_display = ""
#
#     overlay_text = f"Digits: {digits_to_display}  FPS: {fps:.2f}"
#     cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     cv2.imshow("Camera Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Signal the worker thread to exit and wait for it.
# frame_queue.put(None)
# worker_thread.join()
