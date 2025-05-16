from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os

from utils.file_utils import allowed_file, convert_to_wav
from utils.audio_processing import (
    extract_mfcc,
    extract_cochleagram,
    save_spectrogram,
    save_cochleagram
)

audio_blueprint = Blueprint('audio', __name__)

@audio_blueprint.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        if filename.endswith('.3gp'):
            upload_path = convert_to_wav(upload_path)
            filename = filename.rsplit('.', 1)[0] + '.wav'

        # Spectrogram paths
        spec_path = os.path.join(current_app.config['SPECTROGRAM_FOLDER'], f"{filename}.png")
        cochleagram_img_path = os.path.join(current_app.config['SPECTROGRAM_FOLDER'], f"{filename}_cochleagram.png")

        save_spectrogram(upload_path, spec_path)
        mfcc_data = extract_mfcc(upload_path)
        cochleagram_data = extract_cochleagram(upload_path)
        save_cochleagram(upload_path, cochleagram_img_path)

        return jsonify({
            'message': 'File uploaded successfully',
            'spectrogram_path': spec_path,
            'mfcc_data': mfcc_data,
            'cochleagram_data': cochleagram_data,
            'cochleagram_image_path': cochleagram_img_path
        })

    return jsonify({'error': 'Invalid file format'}), 400
