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


