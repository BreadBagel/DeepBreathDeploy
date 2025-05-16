import os
import random
import numpy as np
import torch
import librosa
import cv2
from restnet import ResNet34
from logreg_model import LogisticFusionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --- Configuration ---
POS_AUDIO_DIR     = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'
NEG_AUDIO_DIR     = r'C:\Users\User\Downloads\organized_cough_dataset\healthy'
RESNET_WEIGHTS    = 'resnet34_final.pth'
LOGREG_SAVE_PATH  = 'logistic_fusion.pkl'
IMG_SIZE          = (224, 224)
PANEL_HEIGHT      = IMG_SIZE[0] // 2
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Symptom simulation probabilities
SYMPTOM_KEYS      = ['fever','tachypnea','chest_retractions','nasal_flaring','poor_feeding','lethargy']
SYMPTOM_PROBS_POS = {'fever':0.8,'tachypnea':0.6,'chest_retractions':0.5,
                     'nasal_flaring':0.4,'poor_feeding':0.3,'lethargy':0.2}
SYMPTOM_PROBS_NEG = {k: min(1-p, 0.2) for k,p in SYMPTOM_PROBS_POS.items()}

# --- Load ResNet model ---
resnet = ResNet34(num_classes=1).to(DEVICE)
resnet.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=DEVICE))
resnet.eval()

def extract_resnet_prob(path):
    y, sr = librosa.load(path, sr=None)
    if np.max(np.abs(y)) > 0:
        y = y/np.max(np.abs(y))
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = ((mfcc-mfcc.min())/(mfcc.max()-mfcc.min())*255).astype(np.uint8)
    mfcc = cv2.resize(mfcc, (IMG_SIZE[1], PANEL_HEIGHT))
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr/2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = ((mel_db-mel_db.min())/(mel_db.max()-mel_db.min())*255).astype(np.uint8)
    mel_db = cv2.resize(mel_db, (IMG_SIZE[1], PANEL_HEIGHT))
    # Color maps & stack
    top    = cv2.applyColorMap(mfcc, cv2.COLORMAP_MAGMA)
    bottom = cv2.applyColorMap(mel_db, cv2.COLORMAP_OCEAN)
    combined = np.vstack((top, bottom))  # (224,224,3)
    # To tensor & normalize
    img = torch.from_numpy(combined).permute(2,0,1).float()/255.0
    mean,std = torch.tensor([0.485,0.456,0.406]).view(3,1,1), torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = (img-mean)/std
    img = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = resnet(img)
        prob  = torch.sigmoid(logit)[0,0].item()
    return prob

def simulate_symptoms(is_pos):
    probs = SYMPTOM_PROBS_POS if is_pos else SYMPTOM_PROBS_NEG
    return [1 if random.random()<probs[k] else 0 for k in SYMPTOM_KEYS]

# --- Build feature matrix ---
X, y = [], []
for label, directory in [(1, POS_AUDIO_DIR), (0, NEG_AUDIO_DIR)]:
    for fname in os.listdir(directory):
        if not fname.lower().endswith(('.wav','.mp3','.3gp')):
            continue
        path = os.path.join(directory, fname)
        p_resnet = extract_resnet_prob(path)
        sym_vec   = simulate_symptoms(label==1)
        X.append([p_resnet] + sym_vec)
        y.append(label)

X, y = np.array(X), np.array(y)

# --- Split, Train & Evaluate ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
fusion_model = LogisticFusionModel()
fusion_model.train(X_train, y_train, save_path=LOGREG_SAVE_PATH)
y_pred = fusion_model.predict(X_val)
y_prob = fusion_model.predict_proba(X_val)

print("Fusion LogReg Accuracy:", accuracy_score(y_val, y_pred))
print("Fusion ROC AUC:",      roc_auc_score(y_val, y_prob))
print(classification_report(y_val, y_pred))
print(f"Saved Logistic Regression Fusion model to {LOGREG_SAVE_PATH}")
