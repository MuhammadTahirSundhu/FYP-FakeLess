import os
# import torch
# import torchaudio

# # --- FIX 1: BYPASS TORCHCODEC ERROR ---
# # This forces torchaudio to use the stable 'soundfile' backend
# try:
#     if "soundfile" in torchaudio.list_audio_backends():
#         torchaudio.set_audio_backend("soundfile")
# except:
#     # Fallback for newer versions where the above might be deprecated
#     os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# # --- FIX 2: ALLOWLIST FOR PYTORCH 2.6 ---
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
# torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])
# os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="""Hi, my name is sameed.
                """,
                file_path="XTTS_v2_voiceCloner/output.wav",
                speaker_wav="XTTS_v2_voiceCloner/test.wav",
                language="en")
