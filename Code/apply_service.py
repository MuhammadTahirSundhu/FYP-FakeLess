from flask import Flask, request, send_file
import numpy as np
import torch
from utils_audio import load_wav, save_wav, SR
from defense_training import PredictorNet
from apply_defense import apply_universal_delta_to_wav
import os, tempfile

app = Flask(__name__)

@app.route("/apply_delta", methods=["POST"])
def apply_delta():
    """Apply universal delta to uploaded WAV."""
    try:
        # Read input file
        file = request.files["file"]
        wav_np = load_wav(file)

        # Load your universal delta
        delta = np.load("checkpoints_phase1/universal_delta_epoch25.npy")

        # Apply perturbation
        protected = apply_universal_delta_to_wav(wav_np, delta)

        # Save temporary protected file
        out_path = os.path.join(tempfile.gettempdir(), "protected.wav")
        save_wav(out_path, protected, SR)

        return send_file(out_path, mimetype="audio/wav")

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
