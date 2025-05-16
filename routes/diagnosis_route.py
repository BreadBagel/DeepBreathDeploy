# routes/diagnosis_routes.py

import os, uuid, json
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# import your core inference
from fusion_tester import diagnose_from_file

# utils for format‐conversion (reuse your existing helpers)
from utils.file_utils import allowed_file, convert_to_wav

diagnosis_bp = Blueprint('diagnosis', __name__, url_prefix='/api')

@diagnosis_bp.route('/diagnose', methods=['POST'])
def diagnose():
    # 1) verify "file" part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type: {file.filename}'}), 400

    # 2) verify "symptoms" part
    symptoms_raw = request.form.get('symptoms')
    if not symptoms_raw:
        return jsonify({'error': 'Missing symptoms field'}), 400
    try:
        symptom_vector = json.loads(symptoms_raw)
        if not (isinstance(symptom_vector, list) and len(symptom_vector) == 16):
            raise ValueError()
        # normalize to ints 0/1
        symptom_vector = [int(bool(x)) for x in symptom_vector]
    except Exception:
        return jsonify({
            'error': 'Bad symptoms format: expected JSON array of 16 booleans or 0/1 ints'
        }), 400

    # 3) save uploaded file
    filename     = secure_filename(file.filename)
    unique_name  = f"{uuid.uuid4().hex}_{filename}"
    upload_dir   = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    save_path    = os.path.join(upload_dir, unique_name)
    file.save(save_path)

    # 4) convert to WAV if needed
    if save_path.lower().endswith('.3gp'):
        save_path = convert_to_wav(save_path)

    try:
        # 5) run your ResNet + logreg
        diagnosis, confidence, audio_prob = diagnose_from_file(
            save_path, symptom_vector
        )

        # 6) respond
        return jsonify({
            'status': 'success',
            'diagnosis': diagnosis,
            'model_confidence': confidence,
            'audio_only_probability': audio_prob
        }), 200

    except Exception as e:
        # crash protection
        return jsonify({'error': str(e)}), 500

    finally:
        # 7) cleanup
        try:
            os.remove(save_path)
        except OSError:
            pass
