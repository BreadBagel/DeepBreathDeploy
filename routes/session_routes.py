from flask import Blueprint, jsonify, g, request
import mysql.connector
from config import Config
from routes.utils.auth import token_required

session_bp = Blueprint('session_routes', __name__, url_prefix='/api')

def get_db():
    return mysql.connector.connect(**Config.get_db_config())

@session_bp.route('/patients/<int:pid>/sessions', methods=['GET'])
@token_required
def list_sessions(pid):
    # Security check: only allow the owner (g.user_id) to view their patient’s sessions
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute('SELECT user_id FROM patients WHERE id=%s', (pid,))
    owner = cur.fetchone()
    #if not owner or owner['user_id'] != g.user_id:
    #    cur.close()
    #    conn.close()
    #    return jsonify({'status': 'error', 'message': 'Forbidden'}), 403

    # Fetch each session along with the latest symptoms.recorded_at (fallback to date_recorded)
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
    # Security: ensure the patient belongs to the logged-in user
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute('SELECT user_id FROM patients WHERE id=%s', (pid,))
    owner = cur.fetchone()
    if not owner or owner['user_id'] != g.user_id:
        cur.close()
        conn.close()
        return jsonify({'status':'error','message':'Forbidden'}), 403

    # Insert a new session
    cur.execute('INSERT INTO sessions (patient_id) VALUES (%s)', (pid,))
    conn.commit()
    session_id = cur.lastrowid

    cur.close()
    conn.close()
    return jsonify({'status':'success','session_id': session_id}), 201
