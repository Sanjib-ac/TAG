# TAG

For custom Training:

MyOCRProject/

│
├── dataset/
│   ├── images/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── … 
│   ├── train.csv
│   └── val.csv         # (Optional: for validation)
│
├── train.py
└── requirements.txt

train.csv

image,label

img1.png,1234567890

img2.png,0987654321

img3.png,5678901234

...

# How to use:
from doctr.models import ocr_predictor

import torch

# Instantiate the model (architecture remains the same)
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)

# Patch the recognizer with the custom SAR decoder.
num_classes = 10
model.reco_predictor.decoder = SARDecoder(num_classes=num_classes, max_seq_len=10).to("cuda")

# Load fine-tuned weights
model.load_state_dict(torch.load("model_finetuned.pth"))
model.eval()

