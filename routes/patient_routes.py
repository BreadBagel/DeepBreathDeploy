# patient_routes.py

from flask import Blueprint, request, jsonify, g
import mysql.connector
from config import Config
from functools import wraps
from routes.utils.auth import verify_token
from mysql.connector import errorcode


patient_bp = Blueprint('patients', __name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        token = auth_header.replace('Bearer ', '') if auth_header else ''
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'status':'error','message':'Invalid token.'}), 401
        g.current_user_id = user_id
        return f(*args, **kwargs)
    return decorated

# Create a new patient
@patient_bp.route('/patients', methods=['POST'])
@token_required
def create_patient():
    data = request.get_json(force=True)
    # Validate required fields
    for fld in ('first_name','last_name','age_months','weight','sex','history'):
        if not data.get(fld):
            return jsonify({'status':'error','message':f'{fld} is required.'}), 400

    conn = mysql.connector.connect(**Config.get_db_config())
    cursor = conn.cursor()
    sql = (
        "INSERT INTO patients "
        "(user_id, first_name, last_name, suffix, age_months,weight,sex,history) "
        "VALUES (%s, %s, %s, %s, %s, %s,%s, %s)"
    )
    params = (
        g.current_user_id,
        data['first_name'], data['last_name'], data.get('suffix'),
        data['age_months'], data['weight'], data['sex'], data['history']
    )
    try:
        cursor.execute(sql, params)
        conn.commit()
    except mysql.connector.Error as err:
        # Duplicate entry error
        if err.errno == errorcode.ER_DUP_ENTRY:
            cursor.close()
            conn.close()
            return jsonify({
                'status': 'error',
                'message': 'A patient with that name already exists.'
            }), 409
        else:
            # re‑raise or handle other errors
            cursor.close()
            conn.close()
            raise
    pid = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({'status':'success','patient_id':pid}), 201

# List all patients for the current user
@patient_bp.route('/patients', methods=['GET'])
@token_required
def list_patients():
    conn = mysql.connector.connect(**Config.get_db_config())
    cursor = conn.cursor(dictionary=True)

    # Step A: Look up current user’s role
    cursor.execute(
        "SELECT role FROM users WHERE id = %s",
        (g.current_user_id,)
    )
    user_row = cursor.fetchone()
    if not user_row:
        cursor.close()
        conn.close()
        return jsonify({'status': 'error', 'message': 'User not found.'}), 404

    user_role = user_row['role']

    # Step B: If user is a doctor → return ALL rows from `patients`
    if user_role == 'doctor':
        cursor.execute(
            "SELECT * "
            "FROM deepbreath.patients"
        )
    else:
        # Otherwise (parent) → return only those with user_id = current user
        cursor.execute(
            "SELECT * "
            "FROM deepbreath.patients "
            "WHERE user_id = %s",
            (g.current_user_id,)
        )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows), 200


# --------------------------------------------
# 3) GET A SINGLE PATIENT BY ID (route: GET /patients/<pid>)
#    This remains separate from list_patients().
# --------------------------------------------
@patient_bp.route('/patients/<int:pid>', methods=['GET'])
@token_required
def get_patient(pid):
    conn = mysql.connector.connect(**Config.get_db_config())
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT
            first_name,
            last_name,
            sex AS gender,
            age_months,
            suffix,
            weight,
            history,
            recorded_at,
            user_id
        FROM patients
        WHERE id = %s
        """,
        (pid,)
    )
    patient = cursor.fetchone()
    cursor.close()
    conn.close()

    if not patient:
        return jsonify({'status': 'error', 'message': 'Not found'}), 404
    return jsonify(patient), 200
# def list_patients():
#     conn = mysql.connector.connect(**Config.get_db_config())
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute(
#         "SELECT id, first_name, last_name, suffix, age_months, sex "
#         "FROM patients WHERE user_id = %s",
#         (g.current_user_id,)
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     return jsonify(rows), 200
#
# # Get a single patient's details
# @patient_bp.route('/patients/<int:pid>', methods=['GET'])
# @token_required
# def get_patient(pid):
#     # 1) Open a DB connection
#     conn = mysql.connector.connect(**Config.get_db_config())
#     cursor = conn.cursor(dictionary=True)
#
#     # 2) First, look up this user’s role in the `users` table
#     cursor.execute(
#         "SELECT role FROM users WHERE id = %s",
#         (g.current_user_id,)
#     )
#     user_row = cursor.fetchone()
#     if not user_row:
#         # (Shouldn’t happen if verify_token was correct, but just in case…)
#         cursor.close()
#         conn.close()
#         return jsonify({'status': 'error', 'message': 'User not found.'}), 404
#
#     user_role = user_row['role']
#
#     # 3) If this user is a doctor, select *all* patients.
#     if user_role == 'doctor':
#         cursor.execute(
#             "SELECT * "
#             "FROM deepbreath.patients"
#         )
#     else:
#         # 4) Otherwise (e.g. parents), only select patients whose user_id = them
#         cursor.execute(
#             "SELECT id, first_name, last_name, suffix, age_months, sex, user_id "
#             "FROM patients WHERE user_id = %s",
#             (g.current_user_id,)
#         )
#
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     return jsonify(rows), 200
#     # conn = mysql.connector.connect(**Config.get_db_config())
#     # cursor = conn.cursor(dictionary=True)
#     # cursor.execute(
#     #     """
#     #     SELECT
#     #         first_name,
#     #         last_name,
#     #         sex AS gender,
#     #         age_months
#     #     FROM patients
#     #     WHERE id = %s
#     #     """,
#     #     (pid,)
#     # )
#     # patient = cursor.fetchone()
#     # cursor.close()
#     # conn.close()
#     # if not patient:
#     #     return jsonify({'status':'error','message':'Not found'}), 404
#     # return jsonify(patient), 200
