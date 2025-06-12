import jwt
from functools import wraps
from flask import current_app, request, jsonify, g
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

# changes here
import mysql.connector
from config import Config
import traceback

# Verify the raw JWT token (you can keep or remove this helper if you prefer using flask-jwt-extended)
def verify_token(token: str):
    try:
        data = jwt.decode(
            token,
            current_app.config['SECRET_KEY'],
            algorithms=['HS256']
        )
        return data.get('sub')   # adjust key name if you store user ID elsewhere
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Decorator to require a Bearer token and set g.user_id as an integer
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Ensure flask_jwt_extended parses & verifies the JWT
        verify_jwt_in_request()

        # Cast the identity claim (sub) to int
        raw_id = get_jwt_identity()
        try:
            g.user_id = int(raw_id)
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Invalid token identity'}), 401

        # ──────────── NEW LINES BELOW ────────────
        # 3) Fetch the user’s role from the database and set g.user_role.
        try:
            conn = mysql.connector.connect(**Config.get_db_config())
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT role FROM users WHERE id = %s", (g.user_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
        except mysql.connector.Error:
            current_app.logger.error("DB error in token_required:\n" + traceback.format_exc())
            return jsonify({'status': 'error', 'message': 'Database error'}), 500

        if not row:
            return jsonify({'status': 'error', 'message': 'User not found'}), 401

        g.user_role = row['role']  # e.g. 'doctor' or 'parent'
        current_app.logger.debug(f"[token_required] user_id={g.user_id}, user_role={g.user_role}")
        # ──────────── END NEW LINES ────────────

        return f(*args, **kwargs)
    return decorated
