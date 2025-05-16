import os
import random
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Fixed symptom probabilities for simulation
# Example probabilities for symptoms [fever, fatigue] for pneumonia and bronchitis
PNEUM_PROBS = [0.8, 0.6]  # High probability of fever, moderate fatigue for pneumonia
BRONCH_PROBS = [0.2, 0.7]  # Low fever, higher fatigue probability for bronchitis

# Audio parameters
SR = 16000  # target sampling rate
SEGMENT_LENGTH = 1.0  # segment length in seconds
SEGMENT_SAMPLES = int(SR * SEGMENT_LENGTH)

# Directories for data (update these paths as needed)
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Classes (adjust names to your folders/labels)
class_names = ['bronchitis', 'pneumonia']  # label 0 = bronchitis, 1 = pneumonia


# Utility to load file paths and labels from directory structure
def get_audio_filepaths_labels(data_dir):
    filepaths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.wav'):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return filepaths, labels


# Dataset class that loads 1-second segments from audio, applies augmentations, and creates spectrograms
class CoughAudioDataset(Dataset):
    def __init__(self, filepaths, labels, train=True, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.train = train
        self.transform = transform  # waveform-level transforms (noise, shift, etc.)

        # Pre-compute segments (file index, start sample)
        self.segments = []
        for i, file in enumerate(self.filepaths):
            info = torchaudio.info(file)
            total_samples = info.num_frames
            # Number of full segments in this file
            n_seg = total_samples // SEGMENT_SAMPLES
            for j in range(n_seg):
                start = j * SEGMENT_SAMPLES
                self.segments.append((i, start))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        file_idx, start = self.segments[idx]
        file_path = self.filepaths[file_idx]
        label = self.labels[file_idx]

        # Load 1-second segment of audio
        waveform, sr = torchaudio.load(file_path, frame_offset=start, num_frames=SEGMENT_SAMPLES)
        # Resample if needed
        if sr != SR:
            resampler = torchaudio.transforms.Resample(sr, SR)
            waveform = resampler(waveform)
        # Denoise step (e.g., high-pass filter to remove low-frequency noise)
        waveform = torchaudio.functional.highpass_biquad(waveform, SR, cutoff_freq=80.0)

        # Waveform-level augmentation if training
        if self.train:
            # Random time shift
            if random.random() < 0.5:
                shift = random.randint(0, SEGMENT_SAMPLES - 1)
                waveform = torch.roll(waveform, shifts=shift)
            # Add random Gaussian noise
            if random.random() < 0.5:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise

        # Create mel-spectrogram
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=512, n_mels=128
        )(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # SpecAugment: time and frequency masking (apply only if training)
        if self.train:
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=13)
            spectrogram = time_mask(spectrogram)
            spectrogram = freq_mask(spectrogram)

        # Normalize spectrogram to [0,1], then scale to [-1,1]
        spec_min, spec_max = spectrogram.min(), spectrogram.max()
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min + 1e-6)
        spectrogram = spectrogram * 2.0 - 1.0

        # The spectrogram has shape [1, n_mels, time], duplicate channels to [3, ...] for ResNet
        spectrogram = spectrogram.repeat(3, 1, 1)

        # Resize to 224x224 for ResNet input
        spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.squeeze(0)

        return spectrogram, float(label)


# Custom collate function to stack tensors
def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return inputs, targets


# Mixup augmentation function
def mixup_data(x, y, alpha=0.4):
    """Return mixed inputs, pairs of targets, and lambda for mixup."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Prepare data
train_files, train_labels = get_audio_filepaths_labels(train_dir)
val_files, val_labels = get_audio_filepaths_labels(val_dir)
test_files, test_labels = get_audio_filepaths_labels(test_dir)

train_dataset = CoughAudioDataset(train_files, train_labels, train=True)
val_dataset = CoughAudioDataset(val_files, val_labels, train=False)
test_dataset = CoughAudioDataset(test_files, test_labels, train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize model: ResNet18 pretrained, replace final layer for binary classification
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Add dropout and replace final layer (1 output for binary classification)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # dropout to reduce overfitting
    nn.Linear(num_ftrs, 1)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # weight decay for regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0
n_epochs = 30

for epoch in range(n_epochs):
    model.train()
    train_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Apply mixup augmentation in training
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.4)
        outputs = model(inputs).squeeze(1)
        # Mixup loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_resnet18_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    scheduler.step(avg_val_loss)

# Load the best model for inference
model.load_state_dict(torch.load('best_resnet18_model.pth'))


# Function to compute file-level predictions (average of segment probabilities)
def get_file_predictions(model, file_list, device):
    model.eval()
    file_probs = []
    for file_path in file_list:
        waveform, sr = torchaudio.load(file_path)
        if sr != SR:
            waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
        waveform = torchaudio.functional.highpass_biquad(waveform, SR, cutoff_freq=80.0)
        num_segments = waveform.shape[1] // SEGMENT_SAMPLES
        probs = []
        for j in range(num_segments):
            start = j * SEGMENT_SAMPLES
            segment = waveform[:, start:start + SEGMENT_SAMPLES]
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=1024, hop_length=512, n_mels=128
            )(segment)
            spec = torchaudio.transforms.AmplitudeToDB()(spec)
            spec_min, spec_max = spec.min(), spec.max()
            spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)
            spec = spec * 2.0 - 1.0
            spec = spec.repeat(3, 1, 1)
            spec = F.interpolate(spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            spec = spec.to(device)
            with torch.no_grad():
                output = model(spec.unsqueeze(0))
                prob = torch.sigmoid(output).item()
                probs.append(prob)
        file_prob = np.mean(probs) if len(probs) > 0 else 0.0
        file_probs.append(file_prob)
    return np.array(file_probs)


# Compute CNN predicted probabilities for each file
train_probs = get_file_predictions(model, train_files, device)
val_probs = get_file_predictions(model, val_files, device)
test_probs = get_file_predictions(model, test_files, device)


# Simulate symptom features for each file and combine with CNN probabilities
def simulate_symptoms(labels):
    symptom_features = []
    for lbl in labels:
        if lbl == 1:  # pneumonia
            sym = np.random.binomial(1, PNEUM_PROBS)
        else:  # bronchitis
            sym = np.random.binomial(1, BRONCH_PROBS)
        symptom_features.append(sym)
    return np.array(symptom_features)


train_sym = simulate_symptoms(train_labels)
val_sym = simulate_symptoms(val_labels)
test_sym = simulate_symptoms(test_labels)

# Combine CNN probability and symptom features for logistic regression
X_train = np.column_stack((train_probs, train_sym))
y_train = np.array(train_labels)
X_val = np.column_stack((val_probs, val_sym))
y_val = np.array(val_labels)
X_test = np.column_stack((test_probs, test_sym))
y_test = np.array(test_labels)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression for fusion
logistic = LogisticRegression(solver='lbfgs')
logistic.fit(X_train_scaled, y_train)
joblib.dump(logistic, 'logistic_fusion_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Evaluate accuracy and AUC on train/val/test
for name, X_set, y_set in [('Train', X_train_scaled, y_train),
                           ('Validation', X_val_scaled, y_val),
                           ('Test', X_test_scaled, y_test)]:
    preds = logistic.predict(X_set)
    probs = logistic.predict_proba(X_set)[:, 1]
    acc = accuracy_score(y_set, preds)
    auc = roc_auc_score(y_set, probs)
    print(f"{name} Accuracy: {acc:.4f}, AUC: {auc:.4f}")
