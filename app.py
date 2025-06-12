#app.py
from flask import Flask
from flask_jwt_extended import JWTManager

from config import Config
from models import db
from routes.audio_routes import audio_bp
from routes.register_user import register_bp#blueprint
from routes.login_user import login_bp
from routes.patient_routes import patient_bp
from routes.session_routes import session_bp
from routes.symptoms_routes import symptoms_bp
from routes.diagnosis_routes import save_bp  # at top

from routes.results_data_getter import results_data_getter_bp
from routes.remarks_routes import remarks_bp



from flask_cors import CORS
import os
app = Flask(__name__)
FLASK_DEBUG=0
CORS(app)
app.config.from_object(Config)

db.init_app(app)

jwt = JWTManager(app)
import pprint
print("\n=== REGISTERED ROUTES ===")
pprint.pprint(sorted([r.rule for r in app.url_map.iter_rules()]))
print("=========================\n")


app.register_blueprint(register_bp, url_prefix='/api')
app.register_blueprint(login_bp, url_prefix='/api')
app.register_blueprint(patient_bp,   url_prefix='/api')
app.register_blueprint(session_bp,   url_prefix='/api')
app.register_blueprint(symptoms_bp,  url_prefix='/api')
app.register_blueprint(results_data_getter_bp)
app.register_blueprint(save_bp)
app.register_blueprint(remarks_bp,      url_prefix='/api')    # ← add url_prefix here



os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SPECTROGRAM_FOLDER'], exist_ok=True)

app.register_blueprint(audio_bp)

import pprint
pprint.pprint(app.url_map.iter_rules())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
