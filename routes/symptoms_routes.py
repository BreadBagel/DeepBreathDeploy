#symtpms_routes.py

from flask import Blueprint, request, jsonify
import mysql.connector
from config import Config
from routes.utils.auth import verify_token

symptoms_bp = Blueprint('symptoms', __name__, url_prefix='/api')

SYMPTOM_KEYS = [

    'fever',
    # 'tachypnea',
    'chest_retractions',
    'nasal_flaring',
    # 'poor_feeding',
    'lethargy',
    'grunting',
    'cyanosis',
    # 'refusal_feeds',
    #'stridor'  # remove,
    #'fast_breathing'  # remove fast breathing,
    'dry_cough',
    'wheezing',
    'nocturnal_cough',
    'productive_cough',
    'rr'
    # 'chest_tightness'
    # apnea add
    # 'fever','tachypnea','chest_retractions','nasal_flaring',
    # 'poor_feeding','lethargy','grunting','cyanosis','refusal_feeds',
    # 'stridor','fast_breathing',
    # 'dry_cough','wheezing','nocturnal_cough','productive_cough','chest_tightness'
]


# Reuse token_required from patient_routes
from routes.patient_routes import token_required

@symptoms_bp.route('/symptoms', methods=['POST'])
@token_required
def submit_symptoms():
    data = request.get_json(force=True)
    # Check for missing keys
    missing = [key for key in ['patient_id'] + SYMPTOM_KEYS if key not in data]
    if missing:
        return jsonify({'status': 'error', 'message': f"Missing fields: {', '.join(missing)}"}), 400

    # Required fields
    required = [
        'patient_id',
        'cough', 'breathing_difficulty', 'fast_breathing', 'chest_pull',
        'nasal_flaring', 'grunting', 'fever', 'feeding_refusal',
        'unresponsive', 'stridor', 'cyanosis'
    ]
    # for fld in required:
    #     if fld not in data:
    #         return jsonify({'status':'error', 'message': f"'{fld}' is required."}), 400

    # Connect to DB
    conn = mysql.connector.connect(**Config.get_db_config())
    cur = conn.cursor()

    # 1) Create session
    cur.execute('INSERT INTO sessions (patient_id) VALUES (%s)', (data['patient_id'],))
    session_id = cur.lastrowid

    # 2) Insert symptoms
    columns = ', '.join(['session_id'] + SYMPTOM_KEYS)
    placeholders = ', '.join(['%s'] * (1 + len(SYMPTOM_KEYS)))
    sql = f"INSERT INTO symptoms ({columns}) VALUES ({placeholders})"


    # Build parameter tuple: session_id followed by int flags
    params = [session_id] + [int(bool(data[key])) for key in SYMPTOM_KEYS]
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'status': 'success', 'session_id': session_id}), 201

@symptoms_bp.route('/patients/<int:pid>/symptoms', methods=['GET'])
@token_required
def list_symptoms(pid):
    # Connect to DB
    conn = mysql.connector.connect(**Config.get_db_config())
    cur = conn.cursor(dictionary=True)

    cols = ', '.join([f's.{key}' for key in SYMPTOM_KEYS])
    query = f'''
        SELECT
            s.id AS symptom_id,
            sess.date_recorded,
            {cols}
        FROM symptoms s
        JOIN sessions sess ON s.session_id = sess.id
        WHERE sess.patient_id = %s
        ORDER BY sess.date_recorded DESC
        '''
    cur.execute(query, (pid,))
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return jsonify(rows), 200
