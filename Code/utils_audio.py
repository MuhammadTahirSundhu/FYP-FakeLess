"""
utils_audio.py - PHASE 2 ENHANCED
Audio utilities with advanced features:
- HiFi-GAN vocoder integration (replaces Griffin-Lim)
- Psychoacoustic masking (Fletcher-Munson curves)
- Multi-domain feature extraction
- Quality metrics (PESQ, STOI)

Phase-2 improvements:
 ✅ Neural vocoder for demo-quality audio
 ✅ Perceptual masking for imperceptible perturbations
 ✅ Advanced audio analysis tools
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn

# --- canonical config ---
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
GRIFFIN_LIM_ITERS = 32

# -------------------------
# I/O (unchanged)
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
# PHASE 2: Advanced Vocoder Integration
# -------------------------
class NeuralVocoder:
    """
    Wrapper for HiFi-GAN vocoder (or fallback to Griffin-Lim).
    Automatically uses best available method.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.hifigan_available = False
        
        # Try to load HiFi-GAN
        try:
            from hifigan.models import Generator
            from hifigan.env import AttrDict
            import json
            
            # Load universal HiFi-GAN v1 config
            config_path = 'hifigan_config.json'
            checkpoint_path = 'pretrained_models/hifigan_universal.pt'
            
            with open(config_path) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            
            self.generator = Generator(h).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.generator.load_state_dict(state_dict['generator'])
            self.generator.eval()
            self.generator.remove_weight_norm()
            
            self.hifigan_available = True
            print("[INFO] HiFi-GAN vocoder loaded successfully")
            
        except Exception as e:
            print(f"[WARN] HiFi-GAN not available ({e}), using Griffin-Lim fallback")
            self.hifigan_available = False
    
    def mel_to_audio(self, mel_np, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
        """
        Convert mel spectrogram to audio waveform.
        Uses HiFi-GAN if available, otherwise Griffin-Lim.
        """
        if self.hifigan_available:
            # HiFi-GAN expects [1, n_mels, T]
            mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                audio = self.generator(mel_tensor)
            return audio.squeeze().cpu().numpy()
        else:
            # Fallback to Griffin-Lim
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
    WARNING: Low quality - use HiFi-GAN for production.
    """
    S = np.exp(mel)
    linear_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft)
    wav = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=n_fft, n_iter=n_iter)
    return wav

# -------------------------
# PHASE 2: Psychoacoustic Masking
# -------------------------
class PsychoacousticMasking:
    """
    Implements Fletcher-Munson equal loudness curves and temporal masking
    for imperceptible perturbation generation.
    """
    def __init__(self, sr=SR, n_fft=N_FFT):
        self.sr = sr
        self.n_fft = n_fft
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.loudness_curve = self._compute_loudness_curve()
    
    def _compute_loudness_curve(self):
        """
        Simplified Fletcher-Munson equal loudness contour (ISO 226:2003).
        Returns relative threshold in dB for each frequency bin.
        """
        freqs = self.freqs
        # Simplified model: hearing threshold is higher at low/high frequencies
        # Human ear is most sensitive around 2-5 kHz
        
        threshold = np.zeros_like(freqs)
        for i, f in enumerate(freqs):
            if f < 20:
                threshold[i] = 80  # Very low frequencies
            elif f < 100:
                threshold[i] = 40 - (f - 20) / 80 * 30
            elif f < 1000:
                threshold[i] = 10 - (f - 100) / 900 * 5
            elif f < 4000:
                threshold[i] = 5  # Most sensitive range
            elif f < 8000:
                threshold[i] = 5 + (f - 4000) / 4000 * 15
            else:
                threshold[i] = 20 + (f - 8000) / 8000 * 30
        
        # Convert dB to linear scale
        return 10 ** (threshold / 20.0)
    
    def compute_masking_threshold(self, audio_spectrum):
        """
        Compute frequency-dependent masking threshold based on audio content.
        
        Args:
            audio_spectrum: STFT magnitude [freq_bins, time_frames]
        
        Returns:
            masking_threshold: Max allowed perturbation per frequency [freq_bins, time_frames]
        """
        # Temporal averaging for stability
        avg_spectrum = np.mean(audio_spectrum, axis=1, keepdims=True)
        
        # Base threshold from Fletcher-Munson
        base_threshold = self.loudness_curve.reshape(-1, 1)
        
        # Dynamic threshold: higher in loud regions (simultaneous masking)
        dynamic_threshold = avg_spectrum * 0.1  # 10% of signal energy
        
        # Combined threshold
        threshold = np.maximum(base_threshold, dynamic_threshold)
        
        # Broadcast to all time frames
        threshold = np.broadcast_to(threshold, audio_spectrum.shape)
        
        return threshold
    
    def apply_masking_to_perturbation(self, perturbation, audio_spectrum):
        """
        Constrain perturbation below perceptual threshold.
        
        Args:
            perturbation: Raw perturbation [freq_bins, time_frames]
            audio_spectrum: Original audio STFT magnitude
        
        Returns:
            masked_perturbation: Perceptually constrained perturbation
        """
        threshold = self.compute_masking_threshold(audio_spectrum)
        
        # Clip perturbation to stay below threshold
        masked = np.clip(perturbation, -threshold, threshold)
        
        return masked

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
    """
    Extract comprehensive audio features for adaptive defense selection.
    
    Returns:
        dict: Contains spectral, temporal, and energy features
    """
    features = {}
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=wav, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=wav, sr=sr))
    
    # Temporal features
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(wav))
    features['rms_energy'] = np.mean(librosa.feature.rms(y=wav))
    
    # Pitch/harmonic features
    try:
        pitches, magnitudes = librosa.piptrack(y=wav, sr=sr)
        features['pitch_mean'] = np.mean(pitches[pitches > 0])
    except:
        features['pitch_mean'] = 0.0
    
    # MFCCs (compact representation)
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    
    return features

# -------------------------
# PHASE 2: Quality Metrics
# -------------------------
def compute_snr(original, protected):
    """
    Compute Signal-to-Noise Ratio between original and protected audio.
    Higher SNR = better quality (less perturbation noise).
    """
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - protected) ** 2)
    
    if noise_power < 1e-10:
        return 100.0  # Essentially identical
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_lsd(original_mel, protected_mel):
    """
    Compute Log-Spectral Distance between mel spectrograms.
    Lower LSD = better quality.
    """
    diff = original_mel - protected_mel
    lsd = np.mean(np.sqrt(np.mean(diff ** 2, axis=0)))
    return lsd

# Try to import advanced quality metrics
try:
    from pesq import pesq as pesq_metric
    PESQ_AVAILABLE = True
except:
    PESQ_AVAILABLE = False
    print("[INFO] PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi as stoi_metric
    STOI_AVAILABLE = True
except:
    STOI_AVAILABLE = False
    print("[INFO] STOI not available. Install with: pip install pystoi")

def compute_quality_metrics(original_wav, protected_wav, sr=SR):
    """
    Compute comprehensive quality metrics.
    
    Returns:
        dict: Contains SNR, LSD, and optionally PESQ/STOI
    """
    metrics = {}
    
    # Basic metrics (always available)
    metrics['snr'] = compute_snr(original_wav, protected_wav)
    
    # Mel-domain metrics
    orig_mel = wav_to_mel(original_wav, sr=sr)
    prot_mel = wav_to_mel(protected_wav, sr=sr)
    metrics['lsd'] = compute_lsd(orig_mel, prot_mel)
    
    # Advanced metrics (if available)
    if PESQ_AVAILABLE:
        try:
            metrics['pesq'] = pesq_metric(sr, original_wav, protected_wav, 'wb')
        except:
            metrics['pesq'] = None
    
    if STOI_AVAILABLE:
        try:
            metrics['stoi'] = stoi_metric(original_wav, protected_wav, sr, extended=False)
        except:
            metrics['stoi'] = None
    
    return metrics

# -------------------------
# Helpers (unchanged)
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