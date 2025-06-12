# routes/remarks_routes.py

from flask import Blueprint, request, jsonify, g, current_app as app
import mysql.connector
from config import Config
from routes.utils.auth import token_required
import traceback

remarks_bp = Blueprint('remarks', __name__)

@remarks_bp.route('/sessions/<int:session_id>/remarks', methods=['POST'])
@token_required
def add_doctor_remarks(session_id):
    try:
        # 1. Only allow doctors
        if not hasattr(g, 'user_role') or g.user_role != 'doctor':
            return jsonify({'status':'error','message':'Forbidden: doctors only.'}), 403

        data = request.get_json(force=True)
        remarks = data.get('remarks', '').strip()
        if not remarks:
            return jsonify({'status':'error','message':'`remarks` is required.'}), 400

        app.logger.debug(f"Doctor {g.user_id} adding remarks to session {session_id}: {remarks}")

        conn = mysql.connector.connect(**Config.get_db_config())
        cursor = conn.cursor()

        # 2. Ensure the session exists (and get its session_id). We no longer check patient.user_id = g.user_id,
        #    because the “owner” is the patient’s parent, not the doctor. Instead we only verify that the session is real.
        cursor.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            return jsonify({'status':'error','message':'Session not found.'}), 404

        # 3. Update the diagnosis row tied to this session
        #    We still assume exactly one diagnoses row per session—if none exists, we INSERT instead.
        update_sql = """
        UPDATE diagnoses
           SET doctor_remarks = %s,
               doctor_id      = %s
         WHERE session_id = %s
        """
        cursor.execute(update_sql, (remarks, g.user_id, session_id))

        if cursor.rowcount == 0:
            # No existing diagnosis row: INSERT a new row now
            insert_sql = """
            INSERT INTO diagnoses
                       (session_id, diagnosis, model_confidence, audio_only_probability, doctor_remarks, doctor_id)
                VALUES (%s,         NULL,       NULL,             NULL,                   %s,            %s)
            """
            cursor.execute(insert_sql, (session_id, remarks, g.user_id))

        conn.commit()
        return jsonify({'status':'ok'}), 200

    except mysql.connector.Error:
        app.logger.error("MySQL error:\n" + traceback.format_exc())
        return jsonify({'status':'error','message':'Database error.'}), 500

    except Exception:
        app.logger.error("Unexpected error:\n" + traceback.format_exc())
        return jsonify({'status':'error','message':'Internal server error.'}), 500

    finally:
        try: cursor.close()
        except: pass
        try: conn.close()
        except: pass


@remarks_bp.route('/sessions/<int:session_id>/remarks', methods=['GET'])
@token_required
def get_doctor_remarks(session_id):
    try:
        # Optionally check roles here (e.g. allow doctors & parents)
        conn = mysql.connector.connect(**Config.get_db_config())
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
          SELECT doctor_remarks AS remarks
            FROM diagnoses
           WHERE session_id = %s
        """, (session_id,))
        row = cursor.fetchone()
        if not row or row['remarks'] is None:
            return jsonify({'status':'error','message':'No remarks found.'}), 404

        return jsonify({'status':'ok', 'remarks': row['remarks']}), 200

    except mysql.connector.Error:
        app.logger.error("MySQL error:\n" + traceback.format_exc())
        return jsonify({'status':'error','message':'Database error.'}), 500

    except Exception:
        app.logger.error("Unexpected error:\n" + traceback.format_exc())
        return jsonify({'status':'error','message':'Internal server error.'}), 500

    finally:
        try: cursor.close()
        except: pass
        try: conn.close()
        except: pass
