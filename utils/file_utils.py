from config import Config
from pydub import AudioSegment

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def convert_to_wav(file_path):
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    audio = AudioSegment.from_file(file_path, format="3gp")
    audio.export(wav_path, format="wav")
    return wav_path
