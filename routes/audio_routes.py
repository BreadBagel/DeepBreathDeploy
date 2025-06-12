#audio_routes.py
from flask import Blueprint, request, jsonify, current_app
import mysql.connector
from config import Config
from routes.utils.auth import token_required
from werkzeug.utils import secure_filename
import os
import tempfile
tempfile

from utils.file_utils import allowed_file, convert_to_wav

audio_bp = Blueprint('audio', __name__, url_prefix='/api')

@audio_bp.route('/upload-audio', methods=['POST'])
@token_required
def upload_audio():
    # 1) Validate inputs
    if 'file' not in request.files or 'session_id' not in request.form:
        return jsonify({'error':'Missing file or session_id'}), 400

    session_id = request.form['session_id']
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error':'Invalid or empty file'}), 400

    # 2) Secure and optionally convert
    filename  = secure_filename(file.filename)
    raw_bytes = file.read()
    # If .3gp conversion required, use a safe temp directory
    if filename.lower().endswith('.3gp'):
        temp_dir  = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(raw_bytes)
        wav_path = convert_to_wav(temp_path)
        with open(wav_path, 'rb') as f:
            raw_bytes = f.read()
        filename = os.path.basename(wav_path)
        # clean up temp files
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # 3) Insert into audio_recordings table
    conn = mysql.connector.connect(**Config.get_db_config())
    cur  = conn.cursor()
    sql  = (
      "INSERT INTO audio_recordings"
      " (session_id, filename, audio_blob) VALUES (%s, %s, %s)"
    )
    cur.execute(sql, (session_id, filename, raw_bytes))
    conn.commit()
    rec_id = cur.lastrowid
    cur.close()
    conn.close()

    return jsonify({
      'status': 'success',
      'recording_id': rec_id,
      'filename': filename
    }), 201
