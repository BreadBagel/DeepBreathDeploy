import os
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
import noisereduce as nr
import cv2
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === CONFIG ===
PNEUM_DIR   = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'
BRONCH_DIR  = r'C:\Users\User\Downloads\organized_cough_dataset\bronchitis'
DEBUG_DIR   = 'debug_images6'
RESNET_PTH  = 'resnet34_best.pth'
LOGREG_PKL  = 'fusion_logreg.pkl'
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-4
IMG_SIZE    = (224, 224)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SYMPTOM_KEYS = [
    'fever','tachypnea','chest_retractions','nasal_flaring',
    'poor_feeding','lethargy','grunting','cyanosis','refusal_feeds',
    'stridor','fast_breathing',
    'dry_cough','wheezing','nocturnal_cough','productive_cough','chest_tightness'
]

PNEUM_PROBS = {'fever':0.95, 'tachypnea':0.82, 'chest_retractions':0.50,
    'nasal_flaring':0.30, 'poor_feeding':0.60, 'lethargy':0.15,
    'grunting':0.25, 'cyanosis':0.10, 'refusal_feeds':0.20,
    'stridor':0.05, 'fast_breathing':0.80,
    'dry_cough':0.70, 'wheezing':0.30, 'nocturnal_cough':0.75,
    'productive_cough':0.50, 'chest_tightness':0.00}
BRONCH_PROBS = {'fever':0.40, 'tachypnea':0.05, 'chest_retractions':0.05,
    'nasal_flaring':0.02, 'poor_feeding':0.05, 'lethargy':0.05,
    'grunting':0.00, 'cyanosis':0.00, 'refusal_feeds':0.00,
    'stridor':0.00, 'fast_breathing':0.10,
    'dry_cough':0.95, 'wheezing':0.80, 'nocturnal_cough':0.80,
    'productive_cough':0.50, 'chest_tightness':0.00}

WINDOW_SEC = 1.0
HOP_SEC    = 0.5

os.makedirs(DEBUG_DIR, exist_ok=True)

class CoughWindowDataset(Dataset):
    def __init__(self, file_label_pairs):
        self.files = []
        for path, label in file_label_pairs:
            y, sr = librosa.load(path, sr=None)
            if y.max()>0:
                y = y/np.max(np.abs(y))
            y = nr.reduce_noise(y=y, sr=sr, y_noise=y[:int(0.5*sr)])
            win, hop = int(WINDOW_SEC*sr), int(HOP_SEC*sr)
            count = 1 + max(0, (len(y)-win)//hop)
            self.files.append({'path': path, 'label': float(label), 'y': y, 'sr': sr,
                                'win': win, 'hop': hop, 'count': count})
        counts = [f['count'] for f in self.files]
        self.prefix = np.cumsum(counts) if counts else np.array([], dtype=int)
        self.total_windows = int(self.prefix[-1]) if len(self.prefix)>0 else 0

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.prefix, idx, side='right')
        entry = self.files[file_idx]
        prev_sum = int(self.prefix[file_idx-1]) if file_idx>0 else 0
        win_i = idx - prev_sum

        seg = entry['y'][win_i*entry['hop']:win_i*entry['hop']+entry['win']]
        if len(seg)<entry['win']:
            seg = np.pad(seg, (0, entry['win']-len(seg)))
        if random.random()<0.5:
            seg = librosa.effects.time_stretch(seg, rate=random.uniform(0.9,1.1))
        if random.random()<0.5:
            seg = librosa.effects.pitch_shift(seg, sr=entry['sr'], n_steps=random.randint(-2,2))
        if random.random()<0.3:
            seg += 0.005*np.random.randn(len(seg))

        img_np = CoughWindowDataset.make_image_np(seg, entry['sr'])
        fname = os.path.splitext(os.path.basename(entry['path']))[0]
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{fname}_w{win_i}.png"), img_np)

        img = torch.from_numpy(img_np).permute(2,0,1).float()/255.0
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img = (img-mean)/std
        return img, torch.tensor(entry['label']).unsqueeze(0)

    @staticmethod
    def make_image_np(y, sr):
        n_fft = min(2048, len(y))
        hop = n_fft//4
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
        mfcc = np.clip(mfcc, -100,100)
        mfcc = ((mfcc-mfcc.min())/(mfcc.max()-mfcc.min())*255).astype(np.uint8)
        mfcc = cv2.resize(mfcc, (IMG_SIZE[1], IMG_SIZE[0]//2))
        top = cv2.applyColorMap(mfcc, cv2.COLORMAP_INFERNO)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr/2,
                                             n_fft=n_fft, hop_length=hop)
        db = librosa.power_to_db(mel, ref=np.max)
        db = np.clip(db, -80,0)
        mel_m = ((db-db.min())/(db.max()-db.min())*255).astype(np.uint8)
        mel_m = cv2.resize(mel_m, (IMG_SIZE[1], IMG_SIZE[0]//2))
        bot = cv2.applyColorMap(mel_m, cv2.COLORMAP_TURBO)

        return np.vstack((top, bot))

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, path=RESNET_PTH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_metric, model):
        if val_metric > self.best_loss + self.min_delta:  # CHANGED TO TRACK ACCURACY RISE
            self.best_loss = val_metric
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def simulate_symptoms_for_file(is_pneumo):
    probs = PNEUM_PROBS if is_pneumo else BRONCH_PROBS
    return [1 if random.random()<probs[k] else 0 for k in SYMPTOM_KEYS]


def train_resnet(train_pairs, val_pairs):
    tr_ds = CoughWindowDataset(train_pairs)
    vl_ds = CoughWindowDataset(val_pairs)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl_ld = DataLoader(vl_ds, BATCH_SIZE, num_workers=2, pin_memory=True)

    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad=False
    # UNFREEZE LAST BLOCK AND HEAD
    for name, p in model.named_parameters():  # ADDED
        if name.startswith('layer4') or name.startswith('fc'): p.requires_grad=True  # ADDED
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    model.to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]  # CHANGED to include layer4
    optimizer = optim.Adam(trainable, lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)  # CHANGED to 'max'
    crit = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=3)

    for ep in range(EPOCHS):
        model.train()
        total_train = 0
        for x,y in tqdm(tr_ld, desc=f"Train Epoch {ep+1}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()*x.size(0)
        train_loss = total_train/len(tr_ds)

        # FILE-LEVEL VALIDATION
        model.eval()
        file_scores, file_labels = {}, {}
        with torch.no_grad():
            for idx in range(len(vl_ds)):
                img, _ = vl_ds[idx]
                score = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE)))[0,0].item()
                fi = np.searchsorted(vl_ds.prefix, idx, side='right')
                meta = vl_ds.files[fi]
                path = meta['path']
                file_scores.setdefault(path, []).append(score)
                file_labels[path] = meta['label']

        y_true, y_pred = [], []
        for path,scores in file_scores.items():
            m = sum(scores)/len(scores)
            y_true.append(file_labels[path])
            y_pred.append(1 if m>0.5 else 0)
        val_acc = sum(int(p==t) for p,t in zip(y_pred,y_true))/len(y_true)

        print(f"Epoch {ep+1}/{EPOCHS} — Train Loss: {train_loss:.4f}  File Val Acc: {val_acc:.2%}")

        scheduler.step(val_acc)
        stopper(val_acc, model)
        if stopper.early_stop:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(RESNET_PTH))
    return model


def extract_probs(model, pairs):
    model.eval()
    probs = []
    for path,_ in pairs:
        y,sr = librosa.load(path, sr=None)
        if y.max()>0: y/=np.max(np.abs(y))
        clip = y[:int(sr*WINDOW_SEC)]
        img_np = CoughWindowDataset.make_image_np(clip, sr)
        img = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()/255.0
        mean,std = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1), torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        inp = (img-mean)/std
        with torch.no_grad():
            inp = inp.to(DEVICE)
            probs.append(float(torch.sigmoid(model(inp))[0,0]))
    return np.array(probs)

if __name__ == "__main__":
    all_files = [(os.path.join(PNEUM_DIR,f),1) for f in os.listdir(PNEUM_DIR) if f.lower().endswith(('.wav','.mp3','.3gp'))] + \
                [(os.path.join(BRONCH_DIR,f),0) for f in os.listdir(BRONCH_DIR) if f.lower().endswith(('.wav','.mp3','.3gp'))]
    paths, labels = zip(*all_files)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (ti, tsi) in enumerate(skf.split(paths, labels), 1):
        print(f"\n=== Fold {fold} ===")
        train_pairs = [all_files[i] for i in ti]
        test_pairs  = [all_files[i] for i in tsi]
        v = int(0.1*len(train_pairs))
        val_pairs   = train_pairs[:v]
        tr_pairs    = train_pairs[v:]

        resnet_model = train_resnet(tr_pairs, val_pairs)
        all_pairs = tr_pairs + val_pairs + test_pairs
        X_audio   = extract_probs(resnet_model, all_pairs)
        X_sym     = np.array([simulate_symptoms_for_file(lbl==1) for _,lbl in all_pairs])
        y_all     = np.array([lbl for _,lbl in all_pairs])
        X_full    = np.column_stack((X_audio, X_sym))
        n_tr, n_vl = len(tr_pairs), len(val_pairs)
        X_tr, X_vl, X_te = X_full[:n_tr], X_full[n_tr:n_tr+n_vl], X_full[n_tr+n_vl:]
        y_tr, y_vl, y_te = y_all[:n_tr], y_all[n_tr:n_tr+n_vl], y_all[n_tr+n_vl:]

        fusion = LogisticRegression(solver='liblinear')
        fusion.fit(X_tr, y_tr)
        # SAVE PER-FOLD FUSION MODEL  # ADDED
        joblib.dump(fusion, f"fusion_logreg_fold{fold}.pkl")  # ADDED
        print(f"Saved fusion_logreg_fold{fold}.pkl")  # ADDED

        preds = fusion.predict(X_te)
        acc = accuracy_score(y_te, preds)
        print(f"Fold {fold} Test Acc: {acc:.2%}")
        fold_accs.append(acc)

    print(f"\nAverage Test Acc over 5 folds: {np.mean(fold_accs):.2%}")

    # TRAIN FINAL FUSION ON ALL DATA  # ADDED
    # reload final CNN
    final_cnn = models.resnet34(weights=None)
    final_cnn.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(final_cnn.fc.in_features, 1))
    final_cnn.load_state_dict(torch.load(RESNET_PTH))
    final_cnn.to(DEVICE)
    all_pairs = all_files
    X_audio_all = extract_probs(final_cnn, all_pairs)
    X_sym_all   = np.array([simulate_symptoms_for_file(lbl==1) for _,lbl in all_pairs])
    X_full_all  = np.column_stack((X_audio_all, X_sym_all))
    y_all       = np.array([lbl for _,lbl in all_files])
    fusion_final = LogisticRegression(solver='liblinear')
    fusion_final.fit(X_full_all, y_all)
    joblib.dump(fusion_final, LOGREG_PKL)
    print(f"Saved final fusion model to {LOGREG_PKL}")  # ADDED
