import jwt
from functools import wraps
from flask import current_app, request, jsonify, g

# Verify the raw JWT token
def verify_token(token: str):
    try:
        data = jwt.decode(token,
                          current_app.config['SECRET_KEY'],
                          algorithms=['HS256'])
        return data.get('sub')   # adjust key name if you store user ID elsewhere
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Decorator to require a Bearer token and set g.user_id
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'status':'error','message':'Missing Authorization header'}), 401

        token = auth_header.split(' ', 1)[1]
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'status':'error','message':'Invalid or expired token'}), 401

        # store the authenticated user’s ID for downstream checks
        g.user_id = user_id
        return f(*args, **kwargs)
    return decorated
