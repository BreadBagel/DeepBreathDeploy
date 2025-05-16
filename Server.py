
import sys
print(sys.path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import os
import librosa
import librosa.display
import numpy as np
from werkzeug.utils import secure_filename
from pydub import AudioSegment

import noisereduce as nr  # <- make sure na-import ito sa taas kasama ng ibang import

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SPECTROGRAM_FOLDER = 'spectrograms'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

ALLOWED_EXTENSIONS = {'wav', 'mp3', '3gp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr if sr else 22050,
        n_mels=min(64, int(sr / 200)),
        fmax=sr / 2 if sr else 8000
    )

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


# Function to generate and save cochleagram image
def save_cochleagram(file_path, output_path):
    # Load audio file and normalize
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    y = normalize_audio(y)  # Normalize audio levels to avoid distortion
    y = reduce_noise(y, sr)  # Reduce background noise to improve clarity

    # Generate Mel Spectrogram (Cochleagram)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)  # Generate Mel Spectrogram
    cochleagram = librosa.power_to_db(S, ref=np.max)  # Convert to dB scale for better visualization

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cochleagram, sr=sr, fmax=sr / 2, cmap='coolwarm')  # Different look
    plt.colorbar(format='%+2.0f dB')
    plt.title('Cochleagram (High-Res Mel Spectrogram)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def normalize_audio(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y

def reduce_noise(y, sr):
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    return reduced_noise



def convert_to_wav(file_path):
    #Convert 3GP file to WAV
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    audio = AudioSegment.from_file(file_path, format="3gp")
    audio.export(wav_path, format="wav")
    return wav_path

def extract_mfcc(file_path, num_mfcc=13):
    #Extract MFCC from audio
    y, sr = librosa.load(file_path, sr=None)
    y = normalize_audio(y)  # normalize
    y = reduce_noise(y, sr)  # reduce noise
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    return {
        "mfcc": mfcc.tolist(),
        "sr": sr,
        "num_mfcc": num_mfcc
    }


def extract_cochleagram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = normalize_audio(y)  # normalize
    y = reduce_noise(y, sr) # reduce noise
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr/2)
    cochleagram = librosa.power_to_db(S, ref=np.max)
    return {
        "cochleagram": cochleagram.tolist(),
        "sr": sr
    }


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert 3GP to WAV
        if filename.endswith('.3gp'):
            file_path = convert_to_wav(file_path)
            filename = filename.rsplit('.', 1)[0] + '.wav'

        # spectrogram
        spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], f"{filename}.png")
        save_spectrogram(file_path, spectrogram_path)

        # Extract MFCCs
        mfcc_data = extract_mfcc(file_path)

        # Extract Cochleagram
        cochleagram_data = extract_cochleagram(file_path)

        # Save Cochleagram Image
        cochleagram_image_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], f"{filename}_cochleagram.png")
        save_cochleagram(file_path, cochleagram_image_path)

        return jsonify({
            'message': 'File uploaded successfully',
            'spectrogram_path': spectrogram_path,
            'mfcc_data': mfcc_data,
            'cochleagram_data': cochleagram_data,
            'cochleagram_image_path': cochleagram_image_path
        })

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
