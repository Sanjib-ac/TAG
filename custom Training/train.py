import os
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Import the DOctr OCR predictor and DocumentFile API
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


# -------------------------------------------------------------------
# Custom SAR Decoder Components with Attention
# -------------------------------------------------------------------
class SimpleAttention(nn.Module):
    """
    A simple attention mechanism that computes a weighted sum of encoder outputs.
    """

    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.attn_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_state, encoder_outputs):
        # hidden_state: [B, hidden_size]
        # encoder_outputs: [B, seq_len_enc, hidden_size]
        hidden_expanded = hidden_state.unsqueeze(1).expand_as(encoder_outputs)  # [B, T, hidden_size]
        concat = torch.cat([encoder_outputs, hidden_expanded], dim=2)  # [B, T, 2*hidden_size]
        attn_weights = self.attn_fc(concat)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)  # [B, hidden_size]
        return context


class SARDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of output classes (for digits: typically 10)
            embedding_dim: int = 256,
            hidden_size: int = 512,
            max_seq_len: int = 10,  # Fixed output length (e.g., 10 for a 10-digit number)
            teacher_forcing: bool = True
    ):
        """
        A SAR decoder that runs for a fixed number of steps.
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

        # Define a start token (usually index 0; adjust if necessary)
        self.start_token = 0

    def forward(self, encoder_outputs, targets=None):
        """
        Run decoding for fixed steps using teacher forcing (if targets provided).
        Args:
            encoder_outputs: Tensor of shape [B, seq_len_enc, hidden_size]
            targets: Tensor of shape [B, max_seq_len] during training
        Returns:
            outputs: Tensor of shape [B, max_seq_len, num_classes] (logits for each time step)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)

        prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
        outputs = []
        for t in range(self.max_seq_len):
            context = self.attention(hidden_state, encoder_outputs)  # [B, hidden_size]
            token_embedding = self.embedding(prev_token)  # [B, embedding_dim]
            rnn_input = torch.cat([token_embedding, context], dim=1)  # [B, embedding_dim + hidden_size]
            hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
            logits = self.fc(hidden_state)  # [B, num_classes]
            outputs.append(logits.unsqueeze(1))  # [B, 1, num_classes]
            if self.teacher_forcing and targets is not None:
                prev_token = targets[:, t]
            else:
                prev_token = logits.argmax(dim=1)
        outputs = torch.cat(outputs, dim=1)  # [B, max_seq_len, num_classes]
        return outputs


# -------------------------------------------------------------------
# Custom Dataset Definition
# -------------------------------------------------------------------
class CustomOCRDataset(Dataset):
    """
    Expects a CSV file with columns:
      - 'image': image filename (located in img_dir)
      - 'label': string representing the number (e.g., "1234567890")
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_str = row['label'].strip()
        label = [int(ch) for ch in label_str]  # Convert string label into list of integers
        label = torch.tensor(label, dtype=torch.long)  # Shape: [seq_len]
        return image, label


# -------------------------------------------------------------------
# Validation Function
# -------------------------------------------------------------------
def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # Convert tensor images into PIL images
            to_pil = transforms.ToPILImage()
            batch_images = [to_pil(img.cpu()) for img in images]
            doc = DocumentFile.from_images(batch_images)

            features = model.reco_predictor.encoder(doc)  # Get encoded features.
            logits = model.reco_predictor.decoder(features)  # Get predictions from our custom decoder.

            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


# -------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------
def train(args):
    # Comprehensive transform pipeline (data augmentation + normalization)
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.RandomRotation(degrees=10, fill=(255, 255, 255)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, fillcolor=255),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create training and validation datasets/loaders.
    train_dataset = CustomOCRDataset(args.train_csv, args.img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = CustomOCRDataset(args.val_csv, args.img_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Instantiate the DOctr OCR model (pretrained)
    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='sar_resnet31',
        pretrained=True
    )
    model.to(args.device)

    # Freeze detection module if desired (to fine-tune only recognition)
    for param in model.det_predictor.parameters():
        param.requires_grad = False

    # Patch the recognizer with the custom SARDecoder.
    num_classes = 10  # For digits 0-9.
    model.reco_predictor.decoder = SARDecoder(num_classes=num_classes, max_seq_len=args.seq_len).to(args.device)

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            # Convert tensor images back to PIL images for DocumentFile.
            to_pil = transforms.ToPILImage()
            batch_images = [to_pil(img.cpu()) for img in images]
            doc = DocumentFile.from_images(batch_images)

            features = model.reco_predictor.encoder(doc)
            logits = model.reco_predictor.decoder(features)

            # Reshape logits and labels for loss calculation.
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % args.print_freq == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, args.device, criterion)
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.2f}s")

        # Save the model if the validation loss improved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

    print("Training completed. Best Val Loss:", best_val_loss)


# -------------------------------------------------------------------
# Main Function and Argument Parsing
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='dataset/train.csv',
                        help='Path to CSV file with training data')
    parser.add_argument('--val_csv', type=str, default='dataset/val.csv', help='Path to CSV file with validation data')
    parser.add_argument('--img_dir', type=str, default='dataset/images', help='Directory with input images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--height', type=int, default=512, help='Image height for resizing')
    parser.add_argument('--width', type=int, default=512, help='Image width for resizing')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Fixed sequence length (e.g., 10 for a 10-digit number)')
    parser.add_argument('--model_save_path', type=str, default='model_finetuned.pt',
                        help='Path to save the best trained model')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency (in batches) during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    args = parser.parse_args()
    train(args)


# import os
# import time
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
#
# # Import the DOctr OCR predictor and DocumentFile API
# from doctr.models import ocr_predictor
# from doctr.io import DocumentFile
#
# # -------------------------------------------------------------------
# # Custom SAR Decoder Components with Attention
# # -------------------------------------------------------------------
# class SimpleAttention(nn.Module):
#     """
#     A simple attention mechanism that computes a weighted sum of encoder outputs.
#     """
#     def __init__(self, hidden_size):
#         super(SimpleAttention, self).__init__()
#         self.attn_fc = nn.Linear(hidden_size * 2, 1)
#
#     def forward(self, hidden_state, encoder_outputs):
#         # hidden_state: [B, hidden_size]
#         # encoder_outputs: [B, seq_len_enc, hidden_size]
#         hidden_expanded = hidden_state.unsqueeze(1).expand_as(encoder_outputs)  # [B, T, hidden_size]
#         concat = torch.cat([encoder_outputs, hidden_expanded], dim=2)            # [B, T, 2*hidden_size]
#         attn_weights = self.attn_fc(concat)                                      # [B, T, 1]
#         attn_weights = torch.softmax(attn_weights, dim=1)
#         context = torch.sum(attn_weights * encoder_outputs, dim=1)               # [B, hidden_size]
#         return context
#
# class SARDecoder(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,       # Number of output classes (for digits: typically 10)
#         embedding_dim: int = 256,
#         hidden_size: int = 512,
#         max_seq_len: int = 10,    # Fixed output length (e.g., 10 for a 10-digit number)
#         teacher_forcing: bool = True
#     ):
#         """
#         A SAR decoder that runs for a fixed number of steps.
#         """
#         super(SARDecoder, self).__init__()
#         self.num_classes = num_classes
#         self.embedding_dim = embedding_dim
#         self.hidden_size = hidden_size
#         self.max_seq_len = max_seq_len
#         self.teacher_forcing = teacher_forcing
#
#         self.embedding = nn.Embedding(num_classes, embedding_dim)
#         self.rnn_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
#         self.attention = SimpleAttention(hidden_size)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#         # Define a start token (usually index 0; adjust if necessary)
#         self.start_token = 0
#
#     def forward(self, encoder_outputs, targets=None):
#         """
#         Run decoding for fixed steps using teacher forcing (if targets provided).
#         Args:
#             encoder_outputs: Tensor of shape [B, seq_len_enc, hidden_size]
#             targets: Tensor of shape [B, max_seq_len] during training
#         Returns:
#             outputs: Tensor of shape [B, max_seq_len, num_classes] (logits for each time step)
#         """
#         batch_size = encoder_outputs.size(0)
#         device = encoder_outputs.device
#
#         hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
#         cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
#
#         prev_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=device)
#         outputs = []
#         for t in range(self.max_seq_len):
#             context = self.attention(hidden_state, encoder_outputs)       # [B, hidden_size]
#             token_embedding = self.embedding(prev_token)                   # [B, embedding_dim]
#             rnn_input = torch.cat([token_embedding, context], dim=1)         # [B, embedding_dim + hidden_size]
#             hidden_state, cell_state = self.rnn_cell(rnn_input, (hidden_state, cell_state))
#             logits = self.fc(hidden_state)                                 # [B, num_classes]
#             outputs.append(logits.unsqueeze(1))                            # [B, 1, num_classes]
#             if self.teacher_forcing and targets is not None:
#                 prev_token = targets[:, t]
#             else:
#                 prev_token = logits.argmax(dim=1)
#         outputs = torch.cat(outputs, dim=1)  # [B, max_seq_len, num_classes]
#         return outputs
#
# # -------------------------------------------------------------------
# # Custom Dataset Definition
# # -------------------------------------------------------------------
# class CustomOCRDataset(Dataset):
#     """
#     Expects a CSV file with columns:
#       - 'image': image filename (located in img_dir)
#       - 'label': string representing the number (e.g., "1234567890")
#     """
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img_path = os.path.join(self.img_dir, row['image'])
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         label_str = row['label'].strip()
#         label = [int(ch) for ch in label_str]  # Convert string label into list of integers
#         label = torch.tensor(label, dtype=torch.long)  # Shape: [seq_len]
#         return image, label
#
# # -------------------------------------------------------------------
# # Validation Function
# # -------------------------------------------------------------------
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             # Convert tensor images back to PIL images for DocumentFile.
#             to_pil = transforms.ToPILImage()
#             batch_images = [to_pil(img.cpu()) for img in images]
#             doc = DocumentFile.from_images(batch_images)
#
#             features = model.reco_predictor.encoder(doc)  # Get encoded features.
#             logits = model.reco_predictor.decoder(features) # Get predictions from our custom decoder.
#
#             B, T, C = logits.size()
#             logits = logits.view(B * T, C)
#             labels = labels.view(B * T)
#             loss = criterion(logits, labels)
#             total_loss += loss.item()
#     model.train()
#     avg_loss = total_loss / len(val_loader)
#     return avg_loss
#
# # -------------------------------------------------------------------
# # Training Function
# # -------------------------------------------------------------------
# def train(config):
#     # Comprehensive transform pipeline: data augmentation + normalization.
#     transform = transforms.Compose([
#         transforms.Resize((config['height'], config['width'])),
#         transforms.RandomRotation(degrees=10, fill=(255, 255, 255)),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, fillcolor=255),
#         transforms.GaussianBlur(kernel_size=3),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#
#     # Create the training and validation datasets/loaders.
#     train_dataset = CustomOCRDataset(config['train_csv'], config['img_dir'], transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
#
#     val_dataset = CustomOCRDataset(config['val_csv'], config['img_dir'], transform=transform)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
#
#     # Instantiate the DOctr OCR model (pretrained).
#     model = ocr_predictor(
#         det_arch='db_resnet50',
#         reco_arch='sar_resnet31',
#         pretrained=True
#     )
#     model.to(config['device'])
#
#     # (Optional) Freeze the detection module if you want to fine-tune only the recognizer.
#     for param in model.det_predictor.parameters():
#         param.requires_grad = False
#
#     # Patch the recognizer with the custom SARDecoder.
#     num_classes = 10  # For digits 0-9.
#     model.reco_predictor.decoder = SARDecoder(num_classes=num_classes, max_seq_len=config['seq_len']).to(config['device'])
#
#     # Define the loss function and optimizer.
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=config['lr'])
#
#     model.train()
#     best_val_loss = float('inf')
#     for epoch in range(config['epochs']):
#         epoch_loss = 0.0
#         start_time = time.time()
#         for i, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(config['device']), labels.to(config['device'])
#             # Convert tensor images back to PIL images for DocumentFile.
#             to_pil = transforms.ToPILImage()
#             batch_images = [to_pil(img.cpu()) for img in images]
#             doc = DocumentFile.from_images(batch_images)
#
#             features = model.reco_predictor.encoder(doc)
#             logits = model.reco_predictor.decoder(features)
#
#             # Reshape logits and labels for loss calculation.
#             B, T, C = logits.size()
#             logits = logits.view(B * T, C)
#             labels = labels.view(B * T)
#
#             loss = criterion(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             if (i + 1) % config['print_freq'] == 0:
#                 print(f"Epoch [{epoch+1}/{config['epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
#
#         avg_train_loss = epoch_loss / len(train_loader)
#         val_loss = validate(model, val_loader, config['device'], criterion)
#         elapsed = time.time() - start_time
#         print(f"Epoch [{epoch+1}/{config['epochs']}] Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.2f}s")
#
#         # Save the model if the validation loss improves.
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), config['model_save_path'])
#             print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
#
#     print("Training completed. Best Val Loss:", best_val_loss)
#
# # -------------------------------------------------------------------
# # Main Execution (No CLI, just using hard-coded configuration)
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     # Define configuration variables.
#     config = {
#         'train_csv': 'dataset/train.csv',         # Path to CSV file with training data.
#         'val_csv': 'dataset/val.csv',             # Path to CSV file with validation data.
#         'img_dir': 'dataset/images',              # Directory where images are stored.
#         'batch_size': 8,                          # Batch size.
#         'epochs': 10,                             # Number of training epochs.
#         'lr': 1e-4,                               # Learning rate.
#         'height': 512,                            # Image height.
#         'width': 512,                             # Image width.
#         'seq_len': 10,                            # Fixed sequence length (10 digits).
#         'model_save_path': 'model_finetuned.pt',    # Save path with .pt extension.
#         'print_freq': 10,                         # Print frequency (in batches).
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use.
#     }
#
#     train(config)
