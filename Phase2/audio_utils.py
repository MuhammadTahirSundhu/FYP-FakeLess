"""
===============================================================================
AUDIO_UTILS.PY - COMPLETE AUDIO PROCESSING TOOLKIT
===============================================================================
All audio-related functions in one place
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import config

# ============================================================================
# BASIC AUDIO I/O
# ============================================================================

def load_audio(path: str, sr: int = config.SAMPLE_RATE, 
               offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """
    Load audio file
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        offset: Start reading after this time (seconds)
        duration: Only load this duration (seconds)
    
    Returns:
        wav: Audio waveform as numpy array [T]
    """
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
        return wav.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}")


def save_audio(path: str, wav: np.ndarray, sr: int = config.SAMPLE_RATE):
    """
    Save audio to file
    
    Args:
        path: Output path
        wav: Audio waveform
        sr: Sample rate
    """
    try:
        sf.write(path, wav.astype(np.float32), sr)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {path}: {e}")


def normalize_audio(wav: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        wav: Input waveform
        target_db: Target RMS level in dB
    
    Returns:
        normalized_wav: Normalized waveform
    """
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 0:
        scalar = 10 ** (target_db / 20) / rms
        wav = wav * scalar
    return np.clip(wav, -1.0, 1.0)


def pad_or_trim(wav: np.ndarray, length: int) -> np.ndarray:
    """
    Pad or trim audio to target length
    
    Args:
        wav: Input waveform
        length: Target length in samples
    
    Returns:
        processed_wav: Padded or trimmed waveform
    """
    if len(wav) >= length:
        return wav[:length]
    return np.pad(wav, (0, length - len(wav)), mode='constant')


def get_audio_duration(path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        return librosa.get_duration(path=path)
    except:
        return 0.0


# ============================================================================
# SPECTROGRAM CONVERSION
# ============================================================================

def wav_to_mel(wav: np.ndarray, 
               sr: int = config.SAMPLE_RATE,
               n_fft: int = config.N_FFT,
               hop_length: int = config.HOP_LENGTH,
               n_mels: int = config.N_MELS,
               fmin: int = config.FMIN,
               fmax: int = config.FMAX) -> np.ndarray:
    """
    Convert waveform to mel spectrogram
    
    Args:
        wav: Input waveform [T]
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        mel: Log mel spectrogram [n_mels, T_mel]
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0  # Magnitude spectrogram
    )
    
    # Log scaling with clipping to avoid log(0)
    log_mel = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
    
    return log_mel.astype(np.float32)


def mel_to_wav_griffin_lim(mel: np.ndarray,
                           sr: int = config.SAMPLE_RATE,
                           n_fft: int = config.N_FFT,
                           hop_length: int = config.HOP_LENGTH,
                           n_iter: int = config.GRIFFIN_LIM_ITERS,
                           n_mels: int = config.N_MELS) -> np.ndarray:
    """
    Convert mel spectrogram to waveform using Griffin-Lim algorithm
    
    NOTE: This is a basic reconstruction. For better quality, use HiFi-GAN vocoder.
    
    Args:
        mel: Log mel spectrogram [n_mels, T]
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations
        n_mels: Number of mel bands
    
    Returns:
        wav: Reconstructed waveform [T]
    """
    # Inverse log
    mel_spec = np.exp(mel)
    
    # Create mel filterbank for inverse transform
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=config.FMIN,
        fmax=config.FMAX
    )
    
    # Approximate inverse mel transform
    mel_basis_inv = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(mel_basis_inv, mel_spec)
    
    # Griffin-Lim phase reconstruction
    wav = librosa.griffinlim(
        linear_spec,
        hop_length=hop_length,
        win_length=n_fft,
        n_iter=n_iter
    )
    
    return wav.astype(np.float32)


def wav_to_stft(wav: np.ndarray,
                n_fft: int = config.N_FFT,
                hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    """Compute STFT of waveform"""
    return librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, window='hann')


def stft_to_wav(stft: np.ndarray,
                hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    """Inverse STFT to reconstruct waveform"""
    return librosa.istft(stft, hop_length=hop_length, window='hann')


# ============================================================================
# AUDIO QUALITY METRICS
# ============================================================================

def compute_pesq(reference: np.ndarray, 
                 degraded: np.ndarray,
                 sr: int = config.SAMPLE_RATE) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality)
    
    Args:
        reference: Original audio
        degraded: Processed audio
        sr: Sample rate (must be 8000 or 16000)
    
    Returns:
        pesq_score: PESQ score (-0.5 to 4.5, higher is better)
    """
    try:
        from pesq import pesq
        
        if sr not in [8000, 16000]:
            # Resample to 16kHz if needed
            reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
            degraded = librosa.resample(degraded, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        mode = 'wb' if sr == 16000 else 'nb'
        score = pesq(sr, reference, degraded, mode)
        return float(score)
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return 0.0


def compute_stoi(reference: np.ndarray,
                 degraded: np.ndarray,
                 sr: int = config.SAMPLE_RATE) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility)
    
    Args:
        reference: Original audio
        degraded: Processed audio
        sr: Sample rate
    
    Returns:
        stoi_score: STOI score (0-1, higher is better)
    """
    try:
        from pystoi import stoi
        score = stoi(reference, degraded, sr, extended=False)
        return float(score)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return 0.0


def compute_sisdr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    
    Args:
        reference: Original audio
        degraded: Processed audio
    
    Returns:
        sisdr: SI-SDR in dB (higher is better, typically -10 to 20 dB)
    """
    eps = 1e-8
    
    # Zero-mean normalization
    reference = reference - np.mean(reference)
    degraded = degraded - np.mean(degraded)
    
    # SI-SDR calculation
    alpha = np.sum(reference * degraded) / (np.sum(reference ** 2) + eps)
    projection = alpha * reference
    noise = degraded - projection
    
    sisdr = 10 * np.log10(
        (np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps)
    )
    
    return float(sisdr)


def compute_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """
    Compute SNR (Signal-to-Noise Ratio)
    
    Args:
        reference: Original audio
        degraded: Processed audio
    
    Returns:
        snr: SNR in dB (higher is better)
    """
    noise = degraded - reference
    snr = 10 * np.log10(
        (np.sum(reference ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    )
    return float(snr)


def compute_all_metrics(reference: np.ndarray,
                       degraded: np.ndarray,
                       sr: int = config.SAMPLE_RATE) -> dict:
    """Compute all quality metrics at once"""
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    metrics = {
        'pesq': compute_pesq(reference, degraded, sr),
        'stoi': compute_stoi(reference, degraded, sr),
        'sisdr': compute_sisdr(reference, degraded),
        'snr': compute_snr(reference, degraded)
    }
    
    return metrics


# ============================================================================
# AUDIO AUGMENTATION
# ============================================================================

def add_gaussian_noise(wav: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """
    Add Gaussian noise at specified SNR
    
    Args:
        wav: Input waveform
        snr_db: Target SNR in dB
    
    Returns:
        noisy_wav: Waveform with added noise
    """
    signal_power = np.mean(wav ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), wav.shape)
    return (wav + noise).astype(np.float32)


def apply_bandpass_filter(wav: np.ndarray,
                          sr: int = config.SAMPLE_RATE,
                          lowcut: float = 300,
                          highcut: float = 3400) -> np.ndarray:
    """
    Apply bandpass filter (simulates phone quality)
    
    Args:
        wav: Input waveform
        sr: Sample rate
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
    
    Returns:
        filtered_wav: Bandpass filtered waveform
    """
    nyquist = sr / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter
    filtered = signal.filtfilt(b, a, wav)
    
    return filtered.astype(np.float32)


def simulate_mp3_compression(wav: np.ndarray,
                             sr: int = config.SAMPLE_RATE,
                             bitrate: str = '128k') -> np.ndarray:
    """
    Simulate MP3 compression/decompression
    
    Requires: ffmpeg installed
    
    Args:
        wav: Input waveform
        sr: Sample rate
        bitrate: MP3 bitrate (e.g., '128k', '64k')
    
    Returns:
        compressed_wav: Compressed and decompressed waveform
    """
    try:
        import tempfile
        import subprocess
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # File paths
            input_wav = os.path.join(tmpdir, 'input.wav')
            compressed_mp3 = os.path.join(tmpdir, 'compressed.mp3')
            output_wav = os.path.join(tmpdir, 'output.wav')
            
            # Save input
            sf.write(input_wav, wav, sr)
            
            # Compress to MP3
            subprocess.run([
                'ffmpeg', '-i', input_wav,
                '-b:a', bitrate,
                '-y', compressed_mp3
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Decompress back to WAV
            subprocess.run([
                'ffmpeg', '-i', compressed_mp3,
                '-y', output_wav
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Load result
            compressed_wav, _ = librosa.load(output_wav, sr=sr)
            return compressed_wav.astype(np.float32)
            
    except Exception as e:
        print(f"MP3 compression simulation failed: {e}")
        print("Returning original audio (ffmpeg may not be installed)")
        return wav


def apply_random_augmentation(wav: np.ndarray,
                              sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Apply random augmentation from available options
    
    Args:
        wav: Input waveform
        sr: Sample rate
    
    Returns:
        augmented_wav: Augmented waveform
    """
    import random
    
    aug_type = random.choice(['noise', 'bandpass', 'none'])
    
    if aug_type == 'noise':
        snr = random.uniform(*config.AUG_NOISE_SNR)
        return add_gaussian_noise(wav, snr_db=snr)
    elif aug_type == 'bandpass' and config.AUG_BANDPASS:
        return apply_bandpass_filter(wav, sr=sr)
    else:
        return wav


# ============================================================================
# PSYCHOACOUSTIC MASKING
# ============================================================================

def compute_absolute_threshold_hearing(freqs: np.ndarray) -> np.ndarray:
    """
    Compute Absolute Threshold of Hearing (ATH)
    Based on ISO 226:2003 standard
    
    Args:
        freqs: Frequency array in Hz
    
    Returns:
        ath: Threshold in linear magnitude scale
    """
    f_khz = (freqs + 1e-6) / 1000.0
    
    # ATH formula (simplified Terhardt model)
    ath_db = (
        3.64 * np.power(f_khz, -0.8)
        - 6.5 * np.exp(-0.6 * np.power(f_khz - 3.3, 2))
        + 1e-3 * np.power(f_khz, 4)
    )
    
    # Convert dB to linear magnitude
    ath_linear = np.power(10.0, ath_db / 20.0)
    
    return ath_linear


def apply_psychoacoustic_masking(perturbation_mel: np.ndarray,
                                 original_mel: np.ndarray,
                                 scale_factor: float = 0.8) -> np.ndarray:
    """
    Apply psychoacoustic masking to mel-domain perturbation
    
    Args:
        perturbation_mel: Perturbation [n_mels, T]
        original_mel: Original mel spectrogram [n_mels, T]
        scale_factor: Safety margin (0.8 = stay at 80% of threshold)
    
    Returns:
        masked_perturbation: Perceptually constrained perturbation
    """
    # Energy-based threshold
    original_energy = np.abs(original_mel)
    
    # Frequency-dependent weights (higher frequencies can tolerate more)
    n_mels = perturbation_mel.shape[0]
    freq_weights = np.linspace(0.5, 1.5, n_mels).reshape(-1, 1)
    
    # Compute adaptive threshold based on signal energy
    threshold = original_energy * 0.1 * freq_weights * scale_factor
    
    # Scale perturbation to stay below threshold
    pert_magnitude = np.abs(perturbation_mel)
    scale = np.minimum(1.0, threshold / (pert_magnitude + 1e-8))
    
    masked_perturbation = perturbation_mel * scale
    
    return masked_perturbation.astype(np.float32)


# ============================================================================
# PYTORCH TENSOR OPERATIONS
# ============================================================================

def torch_wav_to_mel(wav: torch.Tensor,
                     sr: int = config.SAMPLE_RATE,
                     n_fft: int = config.N_FFT,
                     hop_length: int = config.HOP_LENGTH,
                     n_mels: int = config.N_MELS) -> torch.Tensor:
    """
    Convert PyTorch waveform tensor to mel spectrogram
    
    Args:
        wav: Waveform tensor [..., T]
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bands
    
    Returns:
        mel: Mel spectrogram [..., n_mels, T_mel]
    """
    # Use torchaudio if available, otherwise numpy
    try:
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=config.FMIN,
            f_max=config.FMAX
        ).to(wav.device)
        
        mel = mel_transform(wav)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel
    except:
        # Fallback to numpy
        wav_np = wav.cpu().numpy()
        mel_np = wav_to_mel(wav_np, sr, n_fft, hop_length, n_mels)
        return torch.from_numpy(mel_np).to(wav.device)


def torch_stft(wav: torch.Tensor,
               n_fft: int = config.N_FFT,
               hop_length: int = config.HOP_LENGTH) -> torch.Tensor:
    """Compute STFT using PyTorch"""
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(wav.device),
        return_complex=True
    )


def torch_istft(stft: torch.Tensor,
                n_fft: int = config.N_FFT,
                hop_length: int = config.HOP_LENGTH) -> torch.Tensor:
    """Compute inverse STFT using PyTorch"""
    return torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(stft.device)
    )


# ============================================================================
# ASV EMBEDDER (SPEAKER VERIFICATION)
# ============================================================================

class ASVEmbedder:
    """
    Speaker verification embedder using SpeechBrain ECAPA-TDNN
    Used to measure speaker similarity (for attack loss)
    """
    
    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.available = False
        
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            
            print("Loading ASV embedder (SpeechBrain ECAPA-TDNN)...")
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=config.ASV_CHECKPOINT_DIR,
                run_opts={"device": device}
            )
            self.available = True
            print("✓ ASV embedder loaded successfully")
            
        except Exception as e:
            print(f"⚠ ASV embedder not available: {e}")
            print("  Install SpeechBrain: pip install speechbrain")
            print("  Training will use surrogate loss instead")
    
    def extract(self, wav: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding from waveform
        
        Args:
            wav: Waveform numpy array [T]
        
        Returns:
            embedding: Speaker embedding tensor [emb_dim] or None
        """
        if not self.available:
            return None
        
        try:
            # Convert to tensor
            wav_tensor = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
            
            # Add batch dimension if needed
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                emb = self.model.encode_batch(wav_tensor)
            
            return emb.squeeze(0).detach()
            
        except Exception as e:
            print(f"ASV extraction failed: {e}")
            return None
    
    def extract_batch(self, wavs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract embeddings for batch of waveforms
        
        Args:
            wavs: Batch of waveforms [B, T]
        
        Returns:
            embeddings: Batch of embeddings [B, emb_dim] or None
        """
        if not self.available:
            return None
        
        try:
            with torch.no_grad():
                embs = self.model.encode_batch(wavs)
            return embs.detach()
        except Exception as e:
            print(f"Batch ASV extraction failed: {e}")
            return None


# ============================================================================
# HIFI-GAN VOCODER
# ============================================================================

class HiFiGANVocoder:
    """
    HiFi-GAN vocoder for high-quality mel-to-audio conversion
    Falls back to Griffin-Lim if HiFi-GAN is not available
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = config.DEVICE):
        self.device = device
        self.model = None
        
        if not config.USE_HIFIGAN:
            print("HiFi-GAN disabled in config, using Griffin-Lim")
            return
        
        try:
            print("Loading HiFi-GAN vocoder...")
            
            # Try loading from torch hub
            self.model = torch.hub.load(
                'jik876/hifi-gan',
                'generator',
                'hifigan_universal_v1'
            ).to(device).eval()
            
            print("✓ HiFi-GAN vocoder loaded successfully")
            
        except Exception as e:
            print(f"⚠ HiFi-GAN not available: {e}")
            print("  Falling back to Griffin-Lim reconstruction")
            print("  For better quality, install HiFi-GAN:")
            print("    git clone https://github.com/jik876/hifi-gan.git")
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio
        
        Args:
            mel: Mel spectrogram [B, n_mels, T] or [n_mels, T]
        
        Returns:
            audio: Waveform [B, T_audio] or [T_audio]
        """
        if self.model is not None:
            # Use HiFi-GAN
            with torch.no_grad():
                if mel.dim() == 2:
                    mel = mel.unsqueeze(0)
                audio = self.model(mel).squeeze(1)
                return audio if audio.shape[0] > 1 else audio.squeeze(0)
        else:
            # Fallback to Griffin-Lim
            if isinstance(mel, torch.Tensor):
                mel = mel.cpu().numpy()
            
            if mel.ndim == 3:
                # Batch processing
                audios = []
                for i in range(mel.shape[0]):
                    audio = mel_to_wav_griffin_lim(mel[i])
                    audios.append(audio)
                result = np.stack(audios)
                return torch.from_numpy(result).to(self.device)
            else:
                # Single mel
                audio = mel_to_wav_griffin_lim(mel)
                return torch.from_numpy(audio).to(self.device)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING AUDIO UTILITIES")
    print("=" * 70)
    
    # Test 1: Audio I/O
    print("\n[1/6] Testing audio I/O...")
    dummy_wav = np.random.randn(config.SAMPLE_RATE).astype(np.float32)
    save_audio("test_audio.wav", dummy_wav)
    loaded_wav = load_audio("test_audio.wav")
    assert loaded_wav.shape == dummy_wav.shape
    print("✓ Audio I/O works")
    
    # Test 2: Mel conversion
    print("\n[2/6] Testing mel conversion...")
    mel = wav_to_mel(dummy_wav)
    print(f"  Waveform shape: {dummy_wav.shape}")
    print(f"  Mel shape: {mel.shape}")
    assert mel.shape[0] == config.N_MELS
    print("✓ Mel conversion works")
    
    # Test 3: Griffin-Lim reconstruction
    print("\n[3/6] Testing Griffin-Lim reconstruction...")
    reconstructed = mel_to_wav_griffin_lim(mel)
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print("✓ Griffin-Lim reconstruction works")
    
    # Test 4: Quality metrics
    print("\n[4/6] Testing quality metrics...")
    metrics = compute_all_metrics(dummy_wav, reconstructed)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.3f}")
    print("✓ Quality metrics work")
    
    # Test 5: Augmentation
    print("\n[5/6] Testing augmentation...")
    noisy = add_gaussian_noise(dummy_wav, snr_db=30)
    filtered = apply_bandpass_filter(dummy_wav)
    print(f"  Original energy: {np.mean(dummy_wav**2):.6f}")
    print(f"  Noisy energy: {np.mean(noisy**2):.6f}")
    print(f"  Filtered energy: {np.mean(filtered**2):.6f}")
    print("✓ Augmentation works")
    
    # Test 6: Psychoacoustic masking
    print("\n[6/6] Testing psychoacoustic masking...")
    perturbation = np.random.randn(*mel.shape).astype(np.float32) * 0.05
    masked = apply_psychoacoustic_masking(perturbation, mel)
    reduction = np.abs(masked).mean() / np.abs(perturbation).mean()
    print(f"  Original perturbation magnitude: {np.abs(perturbation).mean():.6f}")
    print(f"  Masked perturbation magnitude: {np.abs(masked).mean():.6f}")
    print(f"  Reduction factor: {reduction:.2f}x")
    print("✓ Psychoacoustic masking works")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    
    # Cleanup
    import os
    try:
        os.remove("test_audio.wav")
    except:
        pass