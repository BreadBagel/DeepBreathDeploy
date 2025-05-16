import matplotlib
matplotlib.use('Agg')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr

def normalize_audio(y):
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

def reduce_noise(y, sr):
    return nr.reduce_noise(y=y, sr=sr)

def save_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=min(64, int(sr / 200)), fmax=sr / 2)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_cochleagram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    y = normalize_audio(y)
    y = reduce_noise(y, sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)
    cochleagram = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cochleagram, sr=sr, fmax=sr / 2, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Cochleagram (High-Res Mel Spectrogram)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def extract_mfcc(file_path, num_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    y = normalize_audio(y)
    y = reduce_noise(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    return {
        "mfcc": mfcc.tolist(),
        "sr": sr,
        "num_mfcc": num_mfcc
    }

def extract_cochleagram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = normalize_audio(y)
    y = reduce_noise(y, sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)
    cochleagram = librosa.power_to_db(S, ref=np.max)
    return {
        "cochleagram": cochleagram.tolist(),
        "sr": sr
    }
