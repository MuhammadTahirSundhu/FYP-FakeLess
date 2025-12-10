"""
Flask backend for advanced audio protection using defense_trainingv2.py pipeline.
Exposes an API endpoint to upload an audio file and receive the protected version.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import base64
import base64
import traceback
from cryptography.fernet import Fernet
from Crypto.Util.Padding import pad, unpad

# Import the AdvancedAudioProtector from defense_trainingv2.py
# Force a non-interactive matplotlib backend to avoid tkinter/TkAgg GUI
# issues when Flask worker threads create figures. Use 'Agg' for file-based
# rendering (no GUI / no tkinter). Must set before any pyplot import.
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass

from defense_trainingv2 import AdvancedAudioProtector, AdvancedAudioProcessor

# Configuration (adjust as needed)
DELTA_PATH = 'checkpoints_advanced/universal_delta_best.npy'
SAMPLE_RATE = 16000
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Cloud_files_test')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav'}

key = b'0123456789abcdefghijklmnopqrstuv' # 32 chars for AES-256 or fernet
# iv = b'\x00' * 16  # 16 bytes IV for AES

b64_key = base64.urlsafe_b64encode(key)
fernet = Fernet(b64_key)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Minimal config for protector (matches defense_trainingv2.py)
config = {
    'sr': 16000,
    'n_fft': 1024,
    'hop_length': 256,
    'n_mels': 80,
    'delta_T': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Load protector once at startup
protector = AdvancedAudioProtector(DELTA_PATH, config)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encrypt_file(data):
    # cipher = AES.new(key, AES.MODE_CBC, iv)
    # padded_data = pad(data, AES.block_size)
    # encrypted = cipher.encrypt(padded_data)
    # return encrypted
    return fernet.encrypt(data)

def decrypt_file(data):
    # cipher = AES.new(key, AES.MODE_CBC, iv)
    # decrypted_padded = cipher.decrypt(data)
    # decrypted = unpad(decrypted_padded, AES.block_size)
    # return decrypted
    return fernet.decrypt(data)

@app.route('/protect', methods=['POST'])
def protect_audio():
    """
    Expects a multipart/form-data POST with a file field named 'audio'.
    Returns the protected audio as a downloadable .wav file.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400
    file = request.files['audio']
    scale = int(request.form.get('scale', 1))
    print(f"Received scale parameter: {scale}")
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # if not allowed_file(file.filename):
    #     return jsonify({'error': f'Invalid file type. Only .wav allowed.{file.filename}'}), 400

    input_path = None
    output_path = None
    try:
        filename = secure_filename(file.filename)

        # Read encrypted bytes directly (binary). Do NOT decode as UTF-8.
        enc_bytes = file.read()  # bytes (ASCII base64 token written by client)
        # Debug info: length and a short hex preview
        print(f"Received encrypted bytes length: {len(enc_bytes)}")
        print(f"First 8 bytes (hex): {enc_bytes[:8].hex()}")

        # Decrypt bytes using Fernet (expects bytes token)
        raw_bytes = decrypt_file(enc_bytes)

        # Ensure the decrypted input file has a .wav extension so soundfile can infer format
        base_name = os.path.splitext(filename)[0]
        input_filename = f"{base_name}.wav"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        with open(input_path, 'wb') as f:
            f.write(raw_bytes)

        # if len(enc_bytes) % AES.block_size != 0:
        #     print("Warning: Encrypted data length is not a multiple of AES block size!")
        input_path = os.path.abspath(input_path)


        output_filename = f"protected_{input_filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        import time; time.sleep(0.1)
        protector.protect_audio(input_path, output_path, scale)
        import time; time.sleep(1)

        # Read the protected file into memory
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        enc_audio_bytes = encrypt_file(audio_bytes)


        # Only keep the protected audio and comparison image; delete the uploaded input file
        # if input_path and os.path.exists(input_path):
        #     os.remove(input_path)

        # Send the file as a response
        return (
            enc_audio_bytes,
            200,
            {
                'Content-Type': 'audio/wav',
                'Content-Disposition': f'attachment; filename={output_filename}'
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        pass
        # Extra cleanup in case of error
        # try:
        #     if input_path and os.path.exists(input_path):
        #         os.remove(input_path)
        # except Exception:
        #     pass
        # try:
        #     if output_path and os.path.exists(output_path):
        #         os.remove(output_path)
        # except Exception:
        #     pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
