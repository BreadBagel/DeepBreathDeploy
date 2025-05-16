# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(30),   unique=True, nullable=False)
    email         = db.Column(db.String(255),  unique=True, nullable=False)
    password_hash = db.Column(db.Text,          nullable=False)
    created_at    = db.Column(db.DateTime,      default=datetime.utcnow)

    patients = db.relationship('Patient', back_populates='user', cascade='all, delete-orphan')


class Patient(db.Model):
    __tablename__ = 'patients'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'))
    first_name  = db.Column(db.String(100), nullable=False)
    last_name   = db.Column(db.String(100), nullable=False)
    suffix      = db.Column(db.String(50))
    age_months  = db.Column(db.Integer, nullable=False)
    sex         = db.Column(db.String(10), nullable=False)

    user     = db.relationship('User',    back_populates='patients')
    sessions = db.relationship('Session', back_populates='patient', cascade='all, delete-orphan')


class Session(db.Model):
    __tablename__ = 'sessions'
    id              = db.Column(db.Integer,   primary_key=True)
    patient_id      = db.Column(db.Integer,   db.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    date_recorded   = db.Column(db.DateTime,  default=datetime.utcnow, nullable=False)


    patient = db.relationship('Patient', back_populates='sessions')

    def to_dict(self):
        return {
            'id':               self.id,
            'patient_id':       self.patient_id,
            'date_recorded':    self.date_recorded.isoformat(),
        }


class Symptom:
    __tablename__ = 'symptoms'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id', ondelete='CASCADE'))
    cough = db.Column(db.Boolean, default=False)
    breathing_difficulty = db.Column(db.Boolean, default=False)
    fast_breathing = db.Column(db.Boolean, default=False)
    chest_pull = db.Column(db.Boolean, default=False)
    nasal_flaring = db.Column(db.Boolean, default=False)
    grunting = db.Column(db.Boolean, default=False)
    fever = db.Column(db.Boolean, default=False)
    feeding_refusal = db.Column(db.Boolean, default=False)
    unresponsive = db.Column(db.Boolean, default=False)
    stridor = db.Column(db.Boolean, default=False)
    cyanosis = db.Column(db.Boolean, default=False)


session = db.relationship('Session', backref='symptoms')
