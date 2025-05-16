#register_user.py

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from marshmallow import Schema, fields, ValidationError
import mysql.connector
from mysql.connector import IntegrityError, Error
from config import Config

register_bp = Blueprint('register', __name__)

class UserSchema(Schema):
    username = fields.Str(required=True, validate=lambda s: s.isalnum() and 3 <= len(s) <= 30)
    email    = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda s: (
        len(s) >= 8 and
        any(c.isupper() for c in s) and
        any(c.islower() for c in s) and
        any(c.isdigit() for c in s) and
        any(c in '!@#$%^&*+=?_-.' for c in s)
    ))
    role = fields.Str(
        required=True,
        validate=lambda s: s in ('doctor', 'parent')
    )

@register_bp.route('/register', methods=['POST'])
def register_user():
    try:
        data = UserSchema().load(request.get_json(force=True))
    except ValidationError as err:
        return jsonify({'status':'error','errors':err.messages}), 400

    username, email, role  = data['username'], data['email'], data['role']
    pw_hash = generate_password_hash(data['password'], method='pbkdf2:sha256', salt_length=16)

    conn = None
    try:
        conn = mysql.connector.connect(**Config.get_db_config())
        cursor = conn.cursor(dictionary=True)
        sql = "INSERT INTO users (username,email,password_hash,role) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (username, email, pw_hash, role))
        conn.commit()
    except IntegrityError as ie:
        if ie.errno == 1062:
            return jsonify({'status':'error','message':'Username or email already exists.'}),409
        return jsonify({'status':'error','message':'Integrity error.'}),500
    except Error:
        return jsonify({'status':'error','message':'Database error.'}),500
    finally:
        if cursor: cursor.close()
        if conn:    conn.close()

    return jsonify({'status':'success','message':'Registration successful.'}),201