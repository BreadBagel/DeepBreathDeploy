#login_user.py

from flask import Blueprint, request, jsonify

from werkzeug.security import check_password_hash
from marshmallow import Schema, fields, ValidationError
from flask_jwt_extended import create_access_token
import mysql.connector
from mysql.connector import Error
from config import Config

login_bp = Blueprint('login', __name__)

class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

@login_bp.route('/login', methods=['POST'])
def login_user():
    # 1) Validate JSON payload
    try:
        data = LoginSchema().load(request.get_json(force=True))
    except ValidationError as ve:
        return jsonify({'status':'error', 'errors': ve.messages}), 400

    conn = cursor = None
    try:
        # 2) Connect to MySQL
        conn   = mysql.connector.connect(**Config.get_db_config())
        cursor = conn.cursor(dictionary=True)

        # 3) Fetch user record
        cursor.execute(
            "SELECT id, password_hash, role FROM users WHERE username = %s",
            (data['username'],)
        )
        user = cursor.fetchone()

        # 4) Verify credentials
        if not user or not check_password_hash(user['password_hash'], data['password']):
            return jsonify({'status':'error', 'message':'Invalid credentials'}), 401

        # 5) Create JWT (adds a "sub" claim with user['id'])
        access_token = create_access_token(
            identity=str(user['id']),
            additional_claims={'role': user['role']}
        )

        return jsonify({
            'status': 'success',
            'access_token': access_token,
            'role': user['role']
        }), 200

    except Error as e:
        print("Login DB error:", e)
        return jsonify({'status':'error', 'message':'Database error.'}), 500

    finally:
        # 6) Clean up DB connections
        if cursor:
            cursor.close()
        if conn:
            conn.close()
