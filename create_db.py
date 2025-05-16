# 4) create_db.py
import mysql.connector
from config import Config

# 1) Connect without a database
cfg = Config.get_db_config()
conn = mysql.connector.connect(
    host=cfg['host'],
    user=cfg['user'],
    password=cfg['password'],
    charset=cfg['charset']
)
conn.autocommit = True
cur = conn.cursor()

# 2) Drop & recreate the deepbreath database
cur.execute(f"DROP DATABASE IF EXISTS `{cfg['database']}`")
cur.execute(
    f"CREATE DATABASE `{cfg['database']}` CHARACTER SET {cfg['charset']}"
)
conn.database = cfg['database']

# 3) users table
cur.execute('''
CREATE TABLE IF NOT EXISTS users (
  id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(30) NOT NULL,
  email VARCHAR(255) NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  role ENUM('doctor','parent') NOT NULL DEFAULT 'parent',
  PRIMARY KEY (id),
  UNIQUE KEY (username),
  UNIQUE KEY (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
''')

# 4) patients table (new schema)
cur.execute('''
CREATE TABLE IF NOT EXISTS patients (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  first_name VARCHAR(100) NOT NULL,
  last_name  VARCHAR(100) NOT NULL,
  suffix     VARCHAR(50)  DEFAULT NULL,
  age_months INT NOT NULL,
  sex        VARCHAR(10)  NOT NULL,
  PRIMARY KEY (id),
  KEY fk_patients_user (user_id),
  CONSTRAINT fk_patients_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
''')

# 5) sessions table
cur.execute('''
CREATE TABLE IF NOT EXISTS sessions (
  id INT NOT NULL AUTO_INCREMENT,
  patient_id INT NOT NULL,
  date_recorded TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY fk_sessions_patient (patient_id),
  CONSTRAINT fk_sessions_patient
    FOREIGN KEY (patient_id) REFERENCES patients(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
''')

# 6) symptoms table
cur.execute('''
CREATE TABLE IF NOT EXISTS symptoms (
  id INT NOT NULL AUTO_INCREMENT,
  session_id INT NOT NULL,
  cough TINYINT(1) NOT NULL DEFAULT 0,
  breathing_difficulty TINYINT(1) NOT NULL DEFAULT 0,
  fast_breathing TINYINT(1) NOT NULL DEFAULT 0,
  chest_pull TINYINT(1) NOT NULL DEFAULT 0,
  nasal_flaring TINYINT(1) NOT NULL DEFAULT 0,
  grunting TINYINT(1) NOT NULL DEFAULT 0,
  fever TINYINT(1) NOT NULL DEFAULT 0,
  feeding_refusal TINYINT(1) NOT NULL DEFAULT 0,
  unresponsive TINYINT(1) NOT NULL DEFAULT 0,
  stridor TINYINT(1) NOT NULL DEFAULT 0,
  cyanosis TINYINT(1) NOT NULL DEFAULT 0,
  recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY fk_symptoms_session (session_id),
  CONSTRAINT fk_symptoms_session
    FOREIGN KEY (session_id) REFERENCES sessions(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
''')

# Close
cur.close()
conn.close()
