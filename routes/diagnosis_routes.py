# routes/diagnosis_routes.py
from flask import Blueprint, request, jsonify, current_app
import mysql.connector
from config import Config
from routes.utils.auth import token_required

save_bp = Blueprint('save_diagnosis', __name__, url_prefix='/api')

def get_db():
    return mysql.connector.connect(**Config.get_db_config())

@save_bp.route('/save-diagnosis', methods=['POST'])
@token_required
def save_diagnosis():
    """
    Expects JSON:
      {
        "session_id": 123,
        "diagnosis": "Bronchitis",
        "model_confidence": 0.87,
        "audio_only_probability": 0.65
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'status':'error','message':'Invalid or missing JSON'}), 400

    sid = data.get('session_id')
    diag = data.get('diagnosis')
    mc   = data.get('model_confidence')
    aop  = data.get('audio_only_probability')

    # Validate inputs
    if sid is None or diag is None or mc is None or aop is None:
        return jsonify({
            'status':'error',
            'message':'Must include session_id, diagnosis, model_confidence, audio_only_probability'
        }), 400

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute('SELECT id FROM sessions WHERE id=%s', (sid,))
        if cur.fetchone() is None:
            cur.close(); conn.close()
            return jsonify({'status':'error','message':'Session not found'}), 404

        # Insert into diagnoses table
        sql = (
            "INSERT INTO diagnoses "
            "(session_id, diagnosis, model_confidence, audio_only_probability) "
            "VALUES (%s, %s, %s, %s)"
        )
        cur.execute(sql, (sid, diag, mc, aop))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status':'ok','message':'Diagnosis saved'}), 201

    except Exception as e:
        current_app.logger.error(f"DB insert error: {e}")
        return jsonify({'status':'error','message':'Database error'}), 500



@save_bp.route('/sessions/<int:session_id>/diagnosis', methods=['GET'])
@token_required
def fetch_diagnosis(session_id):
    conn = get_db()
    cur  = conn.cursor(dictionary=True)

    cur.execute("""
      SELECT d.diagnosis,
             d.model_confidence,
             d.audio_only_probability,
             d.recorded_at,
             p.first_name,
             p.last_name,
             p.suffix,
             p.sex,
             p.age_months,
             p.weight,
             p.history,
             s.date_recorded
      FROM diagnoses d
      JOIN sessions s   ON d.session_id = s.id
      JOIN patients p   ON s.patient_id = p.id
      WHERE d.session_id = %s
      ORDER BY d.id DESC
      LIMIT 1
    """, (session_id,))

    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return jsonify({'status':'error','message':'No diagnosis found'}), 404

    return jsonify({
      'status': 'ok',
      'patient_name': f"{row['first_name']} {row['last_name']}" + (f", {row['suffix']}" if row['suffix'] else ""),
      'session_date':    row['date_recorded'].strftime("%Y-%m-%d %H:%M"),
      'sex':             row['sex'],
      'age':             row['age_months'],
      'weight':          row['weight'],
      'history':         row['history'],
      'diagnosis':       row['diagnosis'],
      'confidence':      row['model_confidence'],
      'audio_prob':      row['audio_only_probability']
    }), 200
