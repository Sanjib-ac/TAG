import cv2
import torch
import torch.nn as nn
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os
import re
import time
import queue
import threading

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
        # hidden_state: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len_enc, hidden_size]
        hidden_expanded = hidden_state.unsqueeze(1).expand_as(encoder_outputs)
        # Concatenate along the feature dimension
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
            max_seq_len: int = 10,  # Fixed output length (10 digits)
            teacher_forcing: bool = True
    ):
        """
        Initialize the SAR decoder with a fixed output sequence length.
        """
        super(SARDecoder, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.teacher_forcing = teacher_forcing

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
            outputs: Tensor of shape [B, max_seq_len, num_classes]
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)

        # Initialize with the start token
        prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
        outputs = []
        for t in range(self.max_seq_len):
            context = self.attention(hidden_state, encoder_outputs)  # [B, hidden_size]
            token_embedding = self.embedding(prev_token)  # [B, embedding_dim]
            rnn_input = torch.cat([token_embedding, context], dim=1)  # [B, embedding_dim + hidden_size]
            hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
            logits = self.fc(hidden_state)  # [B, num_classes]
            outputs.append(logits.unsqueeze(1))  # Append along time dimension.
            if self.teacher_forcing and targets is not None:
                prev_token = targets[:, t]
            else:
                prev_token = logits.argmax(dim=1)
        outputs = torch.cat(outputs, dim=1)  # [B, max_seq_len, num_classes]
        return outputs


# ============================================================================
# Configuration: Toggle image preprocessing (for improved OCR accuracy)
# ============================================================================
APPLY_PREPROCESSING = False  # Set to True to apply preprocessing; False to skip.

# ============================================================================
# Setup OCR model and device
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Instantiate the OCR model without passing 'max_seq_len'
model = ocr_predictor(
    det_arch='db_resnet50',
    reco_arch='sar_resnet31',
    pretrained=True
)
model.to(device)

# Diagnostic: Print available attributes to confirm submodules.
# print("OCRPredictor attributes:", dir(model))

# Patch the recognition module—here available as "reco_predictor"—with our custom SAR decoder.
num_classes = 10  # For digits 0-9
model.reco_predictor.decoder = SARDecoder(num_classes=num_classes, max_seq_len=10).to(device)

# ============================================================================
# Global variables and thread synchronization
# ============================================================================
ocr_result_lock = threading.Lock()
global_ocr_result = ""  # This will hold the latest OCR result (digits only)

frame_queue = queue.Queue(maxsize=3)  # Queue to pass frames from the main loop to the worker.


# ============================================================================
# OCR Worker Function: Processes each frame for OCR.
# ============================================================================
def perform_ocr(frame):
    """
    Processes a single frame:
      - Optionally applies preprocessing,
      - Converts from BGR to RGB and to a PIL Image,
      - Saves to a temporary PNG file (for DocumentFile),
      - Runs OCR inference, and returns a string of recognized digits.
    """
    if APPLY_PREPROCESSING:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_filename = tmp.name
        pil_img.save(tmp, format="PNG")

    doc = DocumentFile.from_images([tmp_filename])
    os.remove(tmp_filename)
    result = model(doc)

    recognized_text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    recognized_text += word.value + " "

    numbers = re.findall(r'\d+', recognized_text.strip())
    digits_only = ",".join(numbers)
    return digits_only


def ocr_worker():
    global global_ocr_result
    while True:
        frame = frame_queue.get()
        if frame is None:  # Sentinel value to terminate the worker.
            frame_queue.task_done()
            break
        try:
            result = perform_ocr(frame)
            with ocr_result_lock:
                global_ocr_result = result
        except Exception as e:
            print("OCR processing error:", e)
        frame_queue.task_done()


worker_thread = threading.Thread(target=ocr_worker, daemon=True)
worker_thread.start()

# ============================================================================
# Camera Capture and Processing Loop
# ============================================================================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Camera resolution is set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print("Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
        break

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    try:
        frame_queue.put(frame.copy(), block=False)
    except queue.Full:
        pass

    with ocr_result_lock:
        ocr_text = global_ocr_result

    overlay_text = f"Digits: {ocr_text}  FPS: {fps:.2f}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()

# Signal the worker thread to exit, then wait for it to finish.
frame_queue.put(None)
worker_thread.join()
