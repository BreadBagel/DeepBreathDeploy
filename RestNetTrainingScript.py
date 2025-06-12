import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import librosa
import noisereduce as nr
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve,
                             accuracy_score, cohen_kappa_score)
import matplotlib.pyplot as plt

# === CONFIG ===
PNEUM_DIR = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'
NONPNEUM_DIR = r'C:\Users\User\Downloads\organized_cough_dataset\Negative'
DEBUG_DIR = 'debug_images6'
MODEL_PATH = 'resnet34_full_pipelineLastforCV.pth'
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WINDOW_SEC = 1.0
HOP_SEC = 0.5
EARLY_STOP_PATIENCE = 3

os.makedirs(DEBUG_DIR, exist_ok=True)



class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



class CoughDataset(Dataset):
    def __init__(self, file_label_pairs, augment=False):
        self.files = file_label_pairs
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        y, sr = librosa.load(path, sr=None)
        if y.max() > 0:
            y = y / np.max(np.abs(y))
        y = nr.reduce_noise(y=y, sr=sr, y_noise=y[:int(0.5 * sr)])
        win = int(WINDOW_SEC * sr)
        seg = y[:win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        if self.augment:
            if random.random() < 0.5:
                seg = librosa.effects.time_stretch(seg, rate=random.uniform(0.9, 1.1))
            if random.random() < 0.5:
                seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=random.randint(-2, 2))
            if random.random() < 0.3:
                seg += 0.005 * np.random.randn(len(seg))
        img = self.make_image(seg, sr)
        return img, torch.tensor(label, dtype=torch.float32)

    def make_image(self, y, sr):
        n_fft = min(2048, len(y));
        hop = n_fft // 4
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        mfcc = np.clip(mfcc, -100, 100)
        mfcc = ((mfcc - mfcc.min()) / (mfcc.max() - mfcc.min()) * 255).astype(np.uint8)
        mfcc = cv2.resize(mfcc, (IMG_SIZE[1], IMG_SIZE[0] // 2))
        top = cv2.applyColorMap(mfcc, cv2.COLORMAP_INFERNO)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2, n_fft=n_fft, hop_length=hop)
        db = librosa.power_to_db(mel, ref=np.max)
        db = np.clip(db, -80, 0)
        mel_m = ((db - db.min()) / (db.max() - db.min()) * 255).astype(np.uint8)
        mel_m = cv2.resize(mel_m, (IMG_SIZE[1], IMG_SIZE[0] // 2))
        bot = cv2.applyColorMap(mel_m, cv2.COLORMAP_TURBO)
        img_np = np.vstack((top, bot))
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std


def validate_windows(model, file_label_pairs):
    model.eval()
    y_true, y_prob = [], []
    for path, label in file_label_pairs:
        y, sr = librosa.load(path, sr=None)
        if y.max() > 0: y /= np.max(np.abs(y))
        y = nr.reduce_noise(y=y, sr=sr, y_noise=y[:int(0.5 * sr)])
        win, hop = int(WINDOW_SEC * sr), int(HOP_SEC * sr)
        scores = []
        for i in range(0, len(y) - win + 1, hop):
            seg = y[i:i + win]
            if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)))
            img = CoughDataset([], augment=False).make_image(seg, sr).unsqueeze(0).to(DEVICE)
            scores.append(float(torch.sigmoid(model(img)).item()))
        if not scores: continue
        y_prob.append(sum(scores) / len(scores))
        y_true.append(label)

    # binary predictions
    y_pred = [p > 0.5 for p in y_prob]
    # metrics
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'kappa': kappa,
        'conf_matrix': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn},
        'roc_curve': (fpr, tpr)
    }



def train_resnet(train_loader, val_files):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith('layer4') or name.startswith('fc'):
            p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    crit = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping()
    best_model, best_score = None, 0.0
    for epoch in range(EPOCHS):
        model.train();
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} Train"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = crit(logits, y)
            loss.backward();
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        metrics = validate_windows(model, val_files)
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}")
        scheduler.step(metrics['auc'])
        stopper(metrics['auc'])
        if metrics['auc'] > best_score:
            best_score, best_model = metrics['auc'], model.state_dict()
        if stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    model.load_state_dict(best_model)
    return model


if __name__ == '__main__':
    # prepare file lists
    all_paths = [(os.path.join(PNEUM_DIR, f), 1) for f in os.listdir(PNEUM_DIR)
                 if f.lower().endswith(('.wav', '.mp3', '.3gp'))]
    all_paths += [(os.path.join(NONPNEUM_DIR, f), 0) for f in os.listdir(NONPNEUM_DIR)
                  if f.lower().endswith(('.wav', '.mp3', '.3gp'))]
    paths, labels = zip(*all_paths)
    # hold-out split
    tr_p, te_p, tr_l, te_l = train_test_split(paths, labels, test_size=0.2,
                                              stratify=labels, random_state=42)
    train_files = list(zip(tr_p, tr_l));
    test_files = list(zip(te_p, te_l))

    # Cross-Validation
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_metrics = []
    for fold, (ti, vi) in enumerate(skf.split(tr_p, tr_l), 1):
        print(f"\n=== Fold {fold} ===")
        tr_files = [train_files[i] for i in ti]
        vl_files = [train_files[i] for i in vi]
        tr_loader = DataLoader(CoughDataset(tr_files, augment=True), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4)
        model = train_resnet(tr_loader, vl_files)
        m = validate_windows(model, vl_files)
        print(
            f"Fold {fold}: AUC={m['auc']:.3f}, F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}, Kappa={m['kappa']:.3f}")
        cv_metrics.append({'Fold': fold, **{k: v for k, v in m.items() if k not in ['conf_matrix', 'roc_curve']}})
    cv_df = pd.DataFrame(cv_metrics)
    print("\n=== CV Results ===")
    print(cv_df.to_string(index=False))

    # Final train & test
    full_loader = DataLoader(CoughDataset(train_files, augment=True), batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)
    final_model = train_resnet(full_loader, test_files)
    m_test = validate_windows(final_model, test_files)
    print("\n=== Test Results ===")
    # Confusion Matrix Table
    cm = m_test['conf_matrix']
    cm_table = pd.DataFrame({
        'Predicted Positive': [cm['TP'], cm['FP']],
        'Predicted Negative': [cm['FN'], cm['TN']]
    }, index=['Actual Positive', 'Actual Negative'])
    print("\n3.4.1 Confusion Matrix Table")
    print(cm_table.to_string())
    # Metrics
    print(f"\nAccuracy: {m_test['accuracy']:.3f}")
    print(f"Precision: {m_test['precision']:.3f}")
    print(f"Recall (Sensitivity): {m_test['recall']:.3f}")
    print(f"F1-Score: {m_test['f1']:.3f}")
    print(f"Cohen's Kappa: {m_test['kappa']:.3f}")
    # ROC Curve & AUC
    fpr, tpr = m_test['roc_curve']
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {m_test['auc']:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    print(f"AUC Score: {m_test['auc']:.3f}")

    # Save model
    torch.save(final_model.state_dict(), MODEL_PATH)
