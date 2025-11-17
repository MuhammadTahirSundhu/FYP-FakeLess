"""
utils_audio.py - PHASE 2 ENHANCED (FIXED)
Audio utilities with advanced features - Windows compatible
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import os
import sys

# --- canonical config ---
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
                                       n_mels=n_mels, power=1.0)
    logS = np.log(np.clip(S, 1e-9, None))
    return logS

# -------------------------
# PHASE 2: Advanced Vocoder Integration (FIXED)
# -------------------------
class NeuralVocoder:
    """
    Wrapper for hifigan vocoder (or fallback to Griffin-Lim).
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.hifigan_available = False
        
        # Try to load hifigan from pretrained_models/hifigan
        try:
            # Add hifigan to path
            hifigan_path = os.path.join('pretrained_models', 'hifigan')
            if os.path.exists(hifigan_path):
                sys.path.insert(0, hifigan_path)
            
            from pretrained_models.hifigan.models import Generator
            from pretrained_models.hifigan.env import AttrDict
            import json
            
            # Look for config in hifigan directory
            config_path = os.path.join(hifigan_path, 'config_v1.json')
            if not os.path.exists(config_path):
                config_path = 'config_v1.json'  # Try current directory
            
            # Look for checkpoint
            checkpoint_paths = [
                os.path.join(hifigan_path, 'generator_universal.pth.tar'),
                os.path.join(hifigan_path, 'generator_v1'),
                os.path.join(hifigan_path, 'g_02500000'),
                'pretrained_models/hifigan_universal.pt',
            ]
            
            checkpoint_path = None
            for cp in checkpoint_paths:
                if os.path.exists(cp):
                    checkpoint_path = cp
                    break
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"hifigan config not found at {config_path}")
            if checkpoint_path is None:
                raise FileNotFoundError("hifigan checkpoint not found")
            
            with open(config_path) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            
            self.generator = Generator(h).to(device)
            
            # Load checkpoint
            state_dict_g = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'generator' in state_dict_g:
                self.generator.load_state_dict(state_dict_g['generator'])
            else:
                self.generator.load_state_dict(state_dict_g)
            
            self.generator.eval()
            self.generator.remove_weight_norm()
            
            self.hifigan_available = True
            print(f"[INFO] hifigan vocoder loaded from {checkpoint_path}")
            
        except Exception as e:
            print(f"[WARN] hifigan not available: {e}")
            print("[INFO] Using Griffin-Lim fallback (lower quality)")
            self.hifigan_available = False
    
    def mel_to_audio(self, mel_np, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
        """
        Convert mel spectrogram to audio waveform.
        """
        if self.hifigan_available:
            try:
                # hifigan expects [1, n_mels, T]
                mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    audio = self.generator(mel_tensor)
                return audio.squeeze().cpu().numpy()
            except Exception as e:
                print(f"[WARN] hifigan inference failed: {e}, falling back to Griffin-Lim")
                return mel_to_wave_griffinlim(mel_np, sr, n_fft, hop_length)
        else:
            return mel_to_wave_griffinlim(mel_np, sr, n_fft, hop_length)

# Global vocoder instance (lazy initialization)
_global_vocoder = None

def get_vocoder(device='cpu'):
    """Get or create global vocoder instance."""
    global _global_vocoder
    if _global_vocoder is None:
        _global_vocoder = NeuralVocoder(device=device)
    return _global_vocoder

def mel_to_wave_griffinlim(mel, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=GRIFFIN_LIM_ITERS):
    """
    Griffin-Lim reconstruction (fallback method).
    """
    S = np.exp(mel)
    linear_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft)
    wav = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=n_fft, n_iter=n_iter)
    return wav

# -------------------------
# PHASE 2: Psychoacoustic Masking (SIMPLIFIED for performance)
# -------------------------
class PsychoacousticMasking:
    """
    Simplified psychoacoustic masking for faster computation.
    """
    def __init__(self, sr=SR, n_fft=N_FFT):
        self.sr = sr
        self.n_fft = n_fft
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.loudness_curve = self._compute_loudness_curve()
    
    def _compute_loudness_curve(self):
        """
        Simplified Fletcher-Munson curve (pre-computed).
        """
        freqs = self.freqs
        threshold = np.zeros_like(freqs)
        
        for i, f in enumerate(freqs):
            if f < 20:
                threshold[i] = 80
            elif f < 100:
                threshold[i] = 40 - (f - 20) / 80 * 30
            elif f < 1000:
                threshold[i] = 10 - (f - 100) / 900 * 5
            elif f < 4000:
                threshold[i] = 5
            elif f < 8000:
                threshold[i] = 5 + (f - 4000) / 4000 * 15
            else:
                threshold[i] = 20 + (f - 8000) / 8000 * 30
        
        return 10 ** (threshold / 20.0)
    
    def compute_masking_threshold(self, audio_spectrum):
        """Compute simplified masking threshold."""
        # Fast approximation: use mean spectrum
        avg_spectrum = np.mean(audio_spectrum, axis=1, keepdims=True)
        base_threshold = self.loudness_curve.reshape(-1, 1)
        dynamic_threshold = avg_spectrum * 0.1
        threshold = np.maximum(base_threshold, dynamic_threshold)
        return np.broadcast_to(threshold, audio_spectrum.shape)

# Global masking instance
_global_masking = None

def get_psychoacoustic_masking(sr=SR, n_fft=N_FFT):
    """Get or create global psychoacoustic masking instance."""
    global _global_masking
    if _global_masking is None:
        _global_masking = PsychoacousticMasking(sr=sr, n_fft=n_fft)
    return _global_masking

# -------------------------
# PHASE 2: Multi-Domain Features
# -------------------------
def extract_audio_features(wav, sr=SR):
    """Extract comprehensive audio features."""
    features = {}
    
    try:
        # Spectral features
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=wav, sr=sr)))
        features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=wav, sr=sr)))
        
        # Temporal features
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(wav)))
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=wav)))
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=wav, sr=sr)
        pitch_values = pitches[pitches > 0]
        features['pitch_mean'] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).astype(float)
        
    except Exception as e:
        print(f"[WARN] Feature extraction failed: {e}")
        # Return defaults
        features = {
            'spectral_centroid': 2000.0,
            'spectral_rolloff': 4000.0,
            'spectral_bandwidth': 2000.0,
            'zero_crossing_rate': 0.1,
            'rms_energy': 0.02,
            'pitch_mean': 200.0,
            'mfcc_mean': np.zeros(13, dtype=float)
        }
    
    return features

# -------------------------
# PHASE 2: Quality Metrics (Optional dependencies)
# -------------------------
def compute_snr(original, protected):
    """Compute Signal-to-Noise Ratio."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - protected) ** 2)
    
    if noise_power < 1e-10:
        return 100.0
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)

def compute_lsd(original_mel, protected_mel):
    """Compute Log-Spectral Distance."""
    diff = original_mel - protected_mel
    lsd = np.mean(np.sqrt(np.mean(diff ** 2, axis=0)))
    return float(lsd)

# Optional: PESQ and STOI (need C++ compiler on Windows - skip if not available)
try:
    from pesq import pesq as pesq_metric
    PESQ_AVAILABLE = True
except:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_metric
    STOI_AVAILABLE = True
except:
    STOI_AVAILABLE = False

def compute_quality_metrics(original_wav, protected_wav, sr=SR):
    """Compute comprehensive quality metrics."""
    metrics = {}
    
    # Basic metrics (always available)
    metrics['snr'] = compute_snr(original_wav, protected_wav)
    
    # Mel-domain metrics
    orig_mel = wav_to_mel(original_wav, sr=sr)
    prot_mel = wav_to_mel(protected_wav, sr=sr)
    
    # Ensure same shape
    min_T = min(orig_mel.shape[1], prot_mel.shape[1])
    orig_mel = orig_mel[:, :min_T]
    prot_mel = prot_mel[:, :min_T]
    
    metrics['lsd'] = compute_lsd(orig_mel, prot_mel)
    
    # Advanced metrics (if available)
    if PESQ_AVAILABLE:
        try:
            metrics['pesq'] = float(pesq_metric(sr, original_wav, protected_wav, 'wb'))
        except:
            metrics['pesq'] = None
    else:
        metrics['pesq'] = None
    
    if STOI_AVAILABLE:
        try:
            metrics['stoi'] = float(stoi_metric(original_wav, protected_wav, sr, extended=False))
        except:
            metrics['stoi'] = None
    else:
        metrics['stoi'] = None
    
    return metrics

# -------------------------
# Helpers
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
    """
    n_mels, delta_T = delta_np.shape
    reps = int(np.ceil(T / delta_T))
    tiled = np.tile(delta_np, (1, reps))[:, :T]
    return tiled