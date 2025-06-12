import torch
import torch.nn as nn
import numpy as np
import librosa
import cv2
import joblib
from torchvision import models

# === CONFIG ===
RESNET_PTH  = 'FINALStablerestNet34.pth'
LOGREG_PKL  = 'FINALSTABLELogisticsReg1.pkl'
IMG_SIZE    = (224, 224)
WINDOW_SEC  = 1.0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === UTILITY: make spectrogram/MFCC image ===
def make_image_np(y, sr):
    n_fft = min(2048, len(y))
    hop = n_fft // 4
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    mfcc = np.clip(mfcc, -100,100)
    mfcc = ((mfcc-mfcc.min())/(mfcc.max()-mfcc.min())*255).astype(np.uint8)
    mfcc = cv2.resize(mfcc, (IMG_SIZE[1], IMG_SIZE[0]//2))
    top = cv2.applyColorMap(mfcc, cv2.COLORMAP_INFERNO)
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr/2,
                                         n_fft=n_fft, hop_length=hop)
    db = librosa.power_to_db(mel, ref=np.max)
    db = np.clip(db, -80,0)
    mel_m = ((db-db.min())/(db.max()-db.min())*255).astype(np.uint8)
    mel_m = cv2.resize(mel_m, (IMG_SIZE[1], IMG_SIZE[0]//2))
    bot = cv2.applyColorMap(mel_m, cv2.COLORMAP_TURBO)
    return np.vstack((top, bot))

# === LOAD MODELS ===
# 1) CNN
cnn = models.resnet34(weights=None)
cnn.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(cnn.fc.in_features,1))
cnn.load_state_dict(torch.load(RESNET_PTH, map_location=DEVICE))
cnn.to(DEVICE).eval()
# 2) Fusion logistic regression
fusion = joblib.load(LOGREG_PKL)

# === INFERENCE FUNCTION ===
def diagnose_from_file(audio_path, symptom_vector):
    # 1) Load and trim/pad to 1s
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) < sr:
        y = np.pad(y, (0, sr-len(y)))
    y = y[:sr]
    # 2) Make image and normalize
    img_np = make_image_np(y, sr)
    img = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    inp = (img - mean) / std
    inp = inp.to(DEVICE)
    # 3) CNN forward to get audio probability
    with torch.no_grad():
        p_audio = torch.sigmoid(cnn(inp))[0,0].item()
    # 4) Fusion predict
    features = [p_audio] + symptom_vector
    label = fusion.predict([features])[0]
    proba = fusion.predict_proba([features])[0][1]
    # 5) Return human-friendly output
    diagnosis = 'Pneumonia' if label==1 else 'Non-Pneumonia'
    return diagnosis, proba, p_audio

if __name__ == "__main__":
    # <— EDIT THESE for your test —
    audio_path = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia\P5.mp3'
    # Example symptom vector: replace with your 16 binary answers in order of
    # ['fever','tachypnea',…,'productive_cough','chest_tightness']
    symptom_vector = [1,0,0,1,0,0,0,0,0,0,1]

    diag, conf, p_audio = diagnose_from_file(audio_path, symptom_vector)
    print(f"Audio-only Probability of Pneumonia: {p_audio*100:.1f}%")
    print(f"Final Diagnosis: {diag}")
    print(f"Model Confidence: {conf*100:.1f}%")
