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
import traceback

# Import the AdvancedAudioProtector from defense_trainingv2.py
from defense_trainingv2 import AdvancedAudioProtector, AdvancedAudioProcessor

# Configuration (adjust as needed)
DELTA_PATH = 'checkpoints_advanced/universal_delta_best.npy'
SAMPLE_RATE = 16000
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Cloud_files_test')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'wav'}

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

@app.route('/protect', methods=['POST'])
def protect_audio():
    """
    Expects a multipart/form-data POST with a file field named 'audio'.
    Returns the protected audio as a downloadable .wav file.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .wav allowed.'}), 400

    input_path = None
    output_path = None
    try:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        input_path = 'Cloud_files_test/audio.wav' #os.path.abspath(input_path)

        output_filename = f"protected_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        import time; time.sleep(0.1)
        protector.protect_audio(input_path, output_path, 1)
        import time; time.sleep(1)

        # Read the protected file into memory
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()


        # Only keep the protected audio and comparison image; delete the uploaded input file
        # if input_path and os.path.exists(input_path):
        #     os.remove(input_path)

        # Send the file as a response
        return (
            audio_bytes,
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
