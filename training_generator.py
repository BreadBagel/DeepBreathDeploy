import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import numpy as np
import cv2
import noisereduce as nr

# === CONFIGURATION ===
POS_DIR         = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'
NEG_DIR         = r'C:\Users\User\Downloads\organized_cough_dataset\bronchitis'
DEBUG_DIR       = 'debug_images'
MODEL_SAVE_PATH = 'resnet34_finetuned_denoise.pth'
BATCH_SIZE      = 16
EPOCHS          = 10
LR              = 1e-4
IMG_SIZE        = (224, 224)
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sliding-window parameters
WINDOW_SEC    = 1.0   # 1 second window
HOP_SEC       = 0.5   # 50% overlap

os.makedirs(DEBUG_DIR, exist_ok=True)

class SlidingWindowCoughDataset(Dataset):
    def __init__(self, file_pairs, debug=False):
        self.files = file_pairs
        self.debug = debug

    def __len__(self):
        total = 0
        for path, _ in self.files:
            y, sr = librosa.load(path, sr=None)
            win = int(WINDOW_SEC * sr)
            hop = int(HOP_SEC * sr)
            n = 1 + max(0, (len(y) - win) // hop)
            total += n
        return total

    def __getitem__(self, idx):
        cum = 0
        for path, label in self.files:
            y, sr = librosa.load(path, sr=None)
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            # Denoise entire recording using first 0.5s of noise
            noise_clip = y[:int(0.5 * sr)]
            y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)

            win = int(WINDOW_SEC * sr)
            hop = int(HOP_SEC * sr)
            n_windows = 1 + max(0, (len(y) - win) // hop)

            if idx < cum + n_windows:
                local = idx - cum
                start = local * hop
                end = start + win
                segment = y[start:end] if end <= len(y) else np.pad(y[start:], (0, end - len(y)))

                # Augmentation
                if random.random() < 0.5:
                    rate = random.uniform(0.9, 1.1)
                    segment = librosa.effects.time_stretch(segment, rate=rate)
                if random.random() < 0.5:
                    steps = random.randint(-2, 2)
                    segment = librosa.effects.pitch_shift(segment, sr, n_steps=steps)
                if random.random() < 0.3:
                    segment = segment + 0.005 * np.random.randn(len(segment))

                # Dynamic FFT settings
                n_fft = min(2048, len(segment))
                hop_length = n_fft // 4

                # MFCC (top panel)
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13,
                                            n_fft=n_fft, hop_length=hop_length)
                mfcc = np.clip(mfcc, -100, 100)
                mfcc = ((mfcc - mfcc.min()) / (mfcc.max() - mfcc.min()) * 255).astype(np.uint8)
                mfcc = cv2.resize(mfcc, (IMG_SIZE[1], IMG_SIZE[0] // 2))
                mfcc_color = cv2.applyColorMap(mfcc, cv2.COLORMAP_INFERNO)

                # Cochleagram (bottom panel)
                mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=64,
                                                     fmax=sr/2, n_fft=n_fft, hop_length=hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = np.clip(mel_db, -80, 0)
                mel = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
                mel = cv2.resize(mel, (IMG_SIZE[1], IMG_SIZE[0] // 2))
                mel_color = cv2.applyColorMap(mel, cv2.COLORMAP_TURBO)

                # Stack panels
                combined = np.vstack((mfcc_color, mel_color))

                # Debug: save first few windows
                if self.debug and local < 5:
                    base = os.path.basename(path).rsplit('.', 1)[0]
                    cv2.imwrite(f"{DEBUG_DIR}/{base}_win{local}.png", combined)
                    print(f"[DEBUG] Saved denoised window {local} for {base}")

                # To tensor & normalize
                img = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = (img - mean) / std

                return img, torch.tensor(float(label))
            cum += n_windows
        raise IndexError("Index out of range")

# Build file list
all_pairs = []
for f in os.listdir(POS_DIR):
    if f.lower().endswith(('.wav', '.mp3', '.3gp')):
        all_pairs.append((os.path.join(POS_DIR, f), 1))
for f in os.listdir(NEG_DIR):
    if f.lower().endswith(('.wav', '.mp3', '.3gp')):
        all_pairs.append((os.path.join(NEG_DIR, f), 0))

# 80/10/10 split
random.shuffle(all_pairs)
total = len(all_pairs)
train_end = int(0.8 * total)
val_end   = train_end + int(0.1 * total)
train_pairs = all_pairs[:train_end]
val_pairs   = all_pairs[train_end:val_end]
test_pairs  = all_pairs[val_end:]

print(f"Files → Total={total}, Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

# Create datasets & loaders
train_ds = SlidingWindowCoughDataset(train_pairs, debug=True)
val_ds   = SlidingWindowCoughDataset(val_pairs)
test_ds  = SlidingWindowCoughDataset(test_pairs)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# Model setup (transfer learning)
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# Training & validation
for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_ds)

    # Validation
    model.eval()
    val_loss, correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total_count += labels.size(0)
    val_loss /= len(val_ds)
    val_acc   = correct / total_count

    print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {train_loss:.4f}  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1%}")

# Final test evaluation
model.eval()
test_loss, correct, total_count = 0.0, 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
        outputs = model(imgs)
        test_loss += criterion(outputs, labels).item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total_count += labels.size(0)
test_loss /= len(test_ds)
test_acc   = correct / total_count
print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.1%}")

# Save final model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved fine-tuned model to {MODEL_SAVE_PATH}")
