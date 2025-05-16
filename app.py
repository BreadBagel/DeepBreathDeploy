#app.py
from flask import Flask
from flask_jwt_extended import JWTManager

from config import Config
from models import db
from routes.audio_routes import audio_blueprint
from routes.register_user import register_bp#blueprint
from routes.login_user import login_bp
from routes.patient_routes import patient_bp
from routes.session_routes import session_bp
from routes.symptoms_routes import symptoms_bp
from routes.diagnosis_route import diagnosis_bp



from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

db.init_app(app)

jwt = JWTManager(app)

app.register_blueprint(register_bp, url_prefix='/api')
app.register_blueprint(login_bp, url_prefix='/api')
app.register_blueprint(patient_bp,   url_prefix='/api')
app.register_blueprint(session_bp,   url_prefix='/api')
app.register_blueprint(symptoms_bp,  url_prefix='/api')
app.register_blueprint(diagnosis_bp)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SPECTROGRAM_FOLDER'], exist_ok=True)

app.register_blueprint(audio_blueprint)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
