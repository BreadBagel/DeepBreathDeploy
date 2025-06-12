#session_routes.py

from flask import Blueprint, jsonify, g, request, current_app
import mysql.connector
from config import Config
from routes.utils.auth import token_required

session_bp = Blueprint('session_routes', __name__, url_prefix='/api')

def get_db():
    return mysql.connector.connect(**Config.get_db_config())

@session_bp.route('/patients/<int:pid>/sessions', methods=['GET'])
@token_required
def list_sessions(pid):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute('SELECT user_id FROM patients WHERE id=%s', (pid,))
    owner = cur.fetchone()
    query = """
    SELECT
      sess.id                         AS session_id,
      IFNULL(MAX(s.recorded_at), sess.date_recorded) AS recorded_at,
      p.first_name,
      p.last_name,
      p.suffix
    FROM sessions sess
    JOIN patients p ON p.id = sess.patient_id
    LEFT JOIN symptoms s ON s.session_id = sess.id
    WHERE sess.patient_id = %s
    GROUP BY sess.id, p.first_name, p.last_name, p.suffix
    ORDER BY recorded_at DESC
    """
    cur.execute(query, (pid,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows), 200

@session_bp.route('/patients/<int:pid>/sessions', methods=['POST'])
@token_required
def create_session(pid):
    # Log who owns the patient and who the JWT user is
    current_app.logger.debug(f"create_session called: g.user_id={g.user_id}")
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute('SELECT user_id FROM patients WHERE id=%s', (pid,))
    owner = cur.fetchone()
    current_app.logger.debug(f"owner from DB: {owner}")

    if not owner or owner['user_id'] != g.user_id:
        cur.close()
        conn.close()
        return jsonify({'status':'error','message':'Forbidden'}), 403

    cur.execute('INSERT INTO sessions (patient_id) VALUES (%s)', (pid,))
    conn.commit()
    session_id = cur.lastrowid
    cur.close()
    conn.close()
    return jsonify({'status':'success','session_id': session_id}), 201

@session_bp.route('/sessions/<int:session_id>/diagnosis', methods=['POST'])
@token_required
def save_diagnosis(session_id):
    conn = get_db()
    cur  = conn.cursor(dictionary=True)
    cur.execute('''
      SELECT s.patient_id, p.user_id 
      FROM sessions s 
      JOIN patients p ON p.id = s.patient_id 
      WHERE s.id = %s
    ''', (session_id,))
    row = cur.fetchone()
    if not row or row['user_id'] != g.user_id:
        cur.close()
        conn.close()
        return jsonify({'status': 'error', 'message': 'Session not found or forbidden'}), 404

    data = request.get_json(silent=True)
    if not data:
        cur.close(); conn.close()
        return jsonify({'status':'error','message':'Invalid or missing JSON'}), 400

    diag = data.get('diagnosis')
    mc   = data.get('model_confidence')
    aop  = data.get('audio_only_probability')
    if diag is None or mc is None or aop is None:
        cur.close(); conn.close()
        return jsonify({
            'status':'error',
            'message':'Must include diagnosis, model_confidence, audio_only_probability'
        }), 400

    try:
        cur.execute('''
          UPDATE sessions
          SET diagnosis = %s,
              model_confidence = %s,
              audio_only_probability = %s
          WHERE id = %s
        ''', (diag, mc, aop, session_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close(); conn.close()
        current_app.logger.error(f"DB error: {e}")
        return jsonify({'status':'error','message':'Database error'}), 500

    cur.close(); conn.close()
    return jsonify({'status':'ok', 'message':'Diagnosis saved'}), 200
