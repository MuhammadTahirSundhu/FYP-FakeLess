"""
utils_audio.py
Audio utilities for the project: loading/saving, mel/STFT conversion,
simple Griffin-Lim reconstruction (fallback), and small helpers.

Phase-1 role:
 - centralize audio processing parameters (sr, n_fft, hop_length, n_mels)
 - provide reproducible transforms used by training and inference scripts

Notes / Where to improve later:
 - Replace `mel_to_wave_griffinlim` with HiFi-GAN vocoder for demo-quality audio.
 - Keep parameters consistent across all scripts to avoid mismatches.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
# --- canonical config used across scripts (change here to affect all) ---
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
GRIFFIN_LIM_ITERS = 32
# ----------------------------------------------------------------

# --- CRITICAL CORRECTION for v0.5.15 ---
# The correct import path for the base Pretrained class in this older version:
try:
    from speechbrain.pretrained import Pretrained 
    print("Using 'speechbrain.pretrained.Pretrained' import path.")
except ImportError:
    # If the above fails, your version might be highly customized or missing modules.
    # The best solution is to upgrade (see warning below).
    print("WARNING: Could not import Pretrained from speechbrain.pretrained. Proceeding with placeholder.")
    Pretrained = None # Set to None to handle errors gracefully

# --- 1. Load the HiFi-GAN Vocoder (Updated for v0.5.15) ---
print("Loading 16kHz HiFi-GAN Vocoder (SpeechBrain v0.5.15)...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = "speechbrain/tts-hifigan-libritts-16kHz"
hifigan_vocoder = None # Initialize vocoder

if Pretrained is not None:
    try:
        # Load the pretrained system using the corrected path
        hifigan_system = Pretrained.from_hparams(
            source=MODEL_ID,
            savedir=f"pretrained_models/{MODEL_ID.split('/')[-1]}",
            run_opts={"device": DEVICE}
        ).to(DEVICE)
        
        # CRITICAL: Extract the actual generator module from the system's modules
        # The key 'generator' is standard for SpeechBrain vocoders.
        hifigan_vocoder = hifigan_system.mods.generator
        hifigan_vocoder.eval()
        
        print("HiFi-GAN Vocoder loaded successfully.")
        
    except Exception as e:
        print(f"Error during HiFi-GAN loading or module access: {e}")
        print("This is likely due to the age of SpeechBrain v0.5.15.")


# -------------------------
# I/O
# -------------------------
def load_wav(path, sr=SR):
    """Load WAV as mono float32 numpy array at target sample rate."""
    wav, r = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

def save_wav(path, wav, sr=SR):
    """Write float32 waveform to disk."""
    sf.write(path, wav.astype(np.float32), samplerate=sr)

# -------------------------
# Spectrograms / mels
# -------------------------
def wav_to_stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Return complex STFT (shape [freq_bins, frames])."""
    return librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, window='hann')

def stft_mag_phase(stft_complex):
    """Split complex STFT to magnitude and phase."""
    mag = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    return mag, phase

def wav_to_mel(wav, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Compute log-mel spectrogram.
    Returned shape: [n_mels, T]
    """
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, power=1.0)  # magnitude-power=1
    logS = np.log(np.clip(S, 1e-9, None))
    return logS

# -------------------------
# Reconstruction (fallback)
# -------------------------
def mel_to_wave_griffinlim(mel, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=GRIFFIN_LIM_ITERS):
    """
    Convert log-mel back to waveform using librosa's approximate inverse:
      mel -> linear stft approximation -> Griffin-Lim.

    WARNING: Griffin-Lim quality is low. Use HiFi-GAN for demo (see README).
    This is fine for training iterations / debugging.
    """
    S = np.exp(mel)  # undo log
    # approximate linear spectrogram from mel
    linear_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft)
    wav = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=n_fft, n_iter=n_iter)
    return wav

'''
def mel_to_wave_hifigan(mel_tensor, vocoder, sr=SR):
    """
    Convert log-mel back to waveform using the loaded HiFi-GAN generator module.
    
    CRITICAL CHANGE: Accepts a PyTorch Tensor [B, N_MELS, T] and returns a PyTorch Tensor.
    This preserves the computational graph for backpropagation.
    
    :param mel_tensor: PyTorch Tensor of shape [B, N_MELS, T] (or [N_MELS, T] if unsqueeze needed)
    :param vocoder: The loaded HiFi-GAN generator module (nn.Module)
    :param sr: Sample rate (16000)
    :return: PyTorch Tensor of shape [B, 1, T_wav]
    """
    if vocoder is None:
        raise ValueError("HiFi-GAN vocoder is not loaded. Cannot synthesize audio.")
    
    # 1. Ensure input is a 3D tensor: [B, N_MELS, T]
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0) # Add batch dimension if missing

    # 2. **CRITICAL: Clip tensor without losing the gradient history**
    #    The 'mel_clipped_safe' step in the old code used NumPy/clip, which broke the graph.
    #    We now use torch.clamp on the tensor.
    mel_tensor_clipped = torch.clamp(mel_tensor, min=-15.0, max=5.0) 
    
    # 3. Generate waveform using the generator's forward method
    # NOTE: We remove the 'with torch.no_grad():' block here.
    # The vocoder's weights are frozen because they are not in the optimizer's list of parameters.
    # We allow the gradient for the *input* (mel_tensor_clipped) to flow through.
    
    # The generator module is called directly. Output shape is [Batch_size, 1, T_wav]
    wav_tensor = vocoder(mel_tensor_clipped) 

    # 4. Return the waveform tensor. Do NOT convert to NumPy or detach/CPU here.
    return wav_tensor
# In utils_audio.py, def mel_to_wave_hifigan:
'''
def mel_to_wave_hifigan(mel_tensor, vocoder, sr=SR):
    """
    Convert log-mel back to waveform using the loaded HiFi-GAN generator module.
    
    :param mel_tensor: PyTorch Tensor of shape [B, N_MELS, T] or [N_MELS, T]
    :return: PyTorch Tensor of shape [B, 1, T_wav]
    """
    if vocoder is None:
        raise ValueError("HiFi-GAN vocoder is not loaded. Cannot synthesize audio.")
    
    # Ensure input is 3D: [B, N_MELS, T]
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0) # Add batch dimension if missing
    
    # Ensure input is on the correct device (already done in train_universal_delta, but good safeguard)
    mel_tensor = mel_tensor.to(DEVICE)

    # Clamp: This is a differentiable operation.
    mel_tensor_clipped = torch.clamp(mel_tensor, min=-15.0, max=5.0) 
    vocoder.eval()  # no training, just inference mode
    # Generate waveform. Gradient flows through here. Output shape is [Batch_size, 1, T_wav]
    wav_tensor = vocoder(mel_tensor_clipped) 

    # Return the waveform tensor.
    return wav_tensor
# ----------------------------------------------------
# --- Example Usage (How to test) ---
# ----------------------------------------------------

# -------------------------
# Small helpers
# -------------------------
def pad_or_trim(arr, length):
    """Pad with zeros or trim 1D array to `length` samples/frames."""
    if arr.shape[0] >= length:
        return arr[:length]
    pad = length - arr.shape[0]
    return np.pad(arr, (0, pad), mode='constant')

def tile_delta_to_length(delta_np, T):
    """
    Tile a mel-domain delta (shape [n_mels, delta_T]) across frames to reach T frames.
    If delta is shape [n_mels, 1], it is broadcast.
    """
    n_mels, delta_T = delta_np.shape
    reps = int(np.ceil(T / delta_T))
    tiled = np.tile(delta_np, (1, reps))[:, :T]
    return tiled
