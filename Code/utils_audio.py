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

# --- canonical config used across scripts (change here to affect all) ---
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
GRIFFIN_LIM_ITERS = 32

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
