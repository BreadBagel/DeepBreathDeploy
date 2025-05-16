#config.py

import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    SPECTROGRAM_FOLDER = 'spectrograms'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', '3gp'}

    # Load from environment or defaults
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_NAME = os.getenv('DB_NAME', 'deepbreath')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASS = os.getenv('DB_PASS', 'Alyas-icebear03')
    DB_CHARSET = 'utf8mb4'
    SECRET_KEY = os.getenv('SECRET_KEY', '')

    # Build one SQLALCHEMY_DATABASE_URI string
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
        f"?charset={DB_CHARSET}"
    )
    # Turn off FSADeprecationWarning
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # === JWT Configuration ===
    JWT_SECRET_KEY = SECRET_KEY
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'

    # (Optional) If you want cookies instead for refresh tokens:
    # JWT_COOKIE_SECURE  = False
    # JWT_COOKIE_CSRF_PROTECT = True

    @classmethod
    def get_db_config(cls):
        return {
            'host': cls.DB_HOST,
            'database': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASS,
            'charset': cls.DB_CHARSET
        }