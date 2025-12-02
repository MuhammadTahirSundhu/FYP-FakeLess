"""
evaluate_defense.py - PHASE 2 ENHANCED

Comprehensive evaluation framework:
 ✅ Multiple ASV models for robustness testing
 ✅ Audio quality metrics (SNR, LSD, PESQ, STOI)
 ✅ Transformation robustness (compression, noise, etc.)
 ✅ Detailed per-file and aggregate statistics

Usage:
    python evaluate_defense.py --manifest test_manifest.txt \
        --delta checkpoints_phase2/universal_delta_advanced_epoch2.npy \
        --comprehensive
"""

import os
os.environ["SPEECHBRAIN_CACHE_STRATEGY"] = "copy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import torch
import json
from collections import defaultdict

from defense_training import (
    PredictorNet, MultiDomainPredictorNet, ASVEmbedder,
    cosine_similarity_torch as cosine_similarity,
    config as CONFIG
)
from utils_audio import (
    wav_to_mel, get_vocoder, compute_quality_metrics,
    extract_audio_features
)

# -------------------------
# PHASE 2: Audio Transformations for Robustness Testing
# -------------------------
class AudioTransformations:
    """Apply various transformations to test perturbation robustness."""
    
    @staticmethod
    def mp3_compression(wav, sr, bitrate=128):
        """Simulate MP3 compression (simplified)."""
        # In production, use pydub or ffmpeg
        # For now, apply low-pass filter as approximation
        nyquist = sr / 2
        cutoff = min(bitrate * 50, nyquist - 1000)  # Rough approximation
        from scipy import signal
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        return signal.filtfilt(b, a, wav)
    
    @staticmethod
    def add_gaussian_noise(wav, snr_db=30):
        """Add Gaussian noise at specified SNR."""
        signal_power = np.mean(wav ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), wav.shape)
        return wav + noise
    
    @staticmethod
    def resample(wav, original_sr, target_sr=8000):
        """Resample to lower quality."""
        resampled = librosa.resample(wav, orig_sr=original_sr, target_sr=target_sr)
        return librosa.resample(resampled, orig_sr=target_sr, target_sr=original_sr)
    
    @staticmethod
    def pitch_shift(wav, sr, steps=1):
        """Shift pitch by semitones."""
        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)
    
    @staticmethod
    def time_stretch(wav, rate=1.1):
        """Time stretch audio."""
        return librosa.effects.time_stretch(wav, rate=rate)
    
    @staticmethod
    def apply_eq(wav, sr):
        """Apply random equalization."""
        from scipy import signal
        # Boost/cut random frequency bands
        nyquist = sr / 2
        bands = [(100, 500), (500, 2000), (2000, 8000)]
        
        for low, high in bands:
            if np.random.rand() > 0.5:
                gain = np.random.uniform(0.7, 1.3)
                sos = signal.butter(2, [low/nyquist, high/nyquist], btype='band', output='sos')
                band_signal = signal.sosfilt(sos, wav)
                wav = wav + band_signal * (gain - 1)
        
        return wav

# -------------------------
# Defense Application Functions
# -------------------------
def apply_universal_delta(wav, delta_np, config=CONFIG):
    """Apply universal delta with neural vocoder."""
    mel = wav_to_mel(wav, sr=config["sr"], n_fft=config["n_fft"],
                     hop_length=config["hop_length"], n_mels=config["n_mels"])
    n_mels, T = mel.shape
    delta_T = delta_np.shape[1]
    reps = int(np.ceil(T / delta_T))
    delta_tiled = np.tile(delta_np, (1, reps))[:, :T]
    mel_prot = mel + delta_tiled
    
    # Use neural vocoder if available
    vocoder = get_vocoder(device=config["device"])
    if vocoder.hifigan_available:
        wav_prot = vocoder.mel_to_audio(mel_prot)
    else:
        from utils_audio import mel_to_wave_griffinlim
        wav_prot = mel_to_wave_griffinlim(mel_prot, sr=config["sr"],
                                         n_fft=config["n_fft"],
                                         hop_length=config["hop_length"])
    return wav_prot

def apply_predictor(wav, predictor, config=CONFIG):
    """Apply predictor network."""
    mel = wav_to_mel(wav, sr=config["sr"], n_fft=config["n_fft"],
                     hop_length=config["hop_length"], n_mels=config["n_mels"])
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(config["device"])
    
    with torch.no_grad():
        delta = predictor(mel_t)
        mel_prot = mel_t + delta
    
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    
    # Use neural vocoder
    vocoder = get_vocoder(device=config["device"])
    if vocoder.hifigan_available:
        wav_prot = vocoder.mel_to_audio(mel_prot_np)
    else:
        from utils_audio import mel_to_wave_griffinlim
        wav_prot = mel_to_wave_griffinlim(mel_prot_np, sr=config["sr"],
                                         n_fft=config["n_fft"],
                                         hop_length=config["hop_length"])
    return wav_prot

# -------------------------
# PHASE 2: Comprehensive Evaluation
# -------------------------
class ComprehensiveEvaluator:
    """
    Advanced evaluation framework with multiple metrics and transformations.
    """
    def __init__(self, config, comprehensive=False):
        self.config = config
        self.comprehensive = comprehensive
        
        # Initialize ASV embedder
        self.asv = ASVEmbedder(device=config["device"])
        
        # Transformations to test (if comprehensive)
        self.transformations = {
            'mp3_128k': lambda w, sr: AudioTransformations.mp3_compression(w, sr, 128),
            'noise_30db': lambda w, sr: AudioTransformations.add_gaussian_noise(w, 30),
            'resample_8k': lambda w, sr: AudioTransformations.resample(w, sr, 8000),
            'pitch_shift': lambda w, sr: AudioTransformations.pitch_shift(w, sr, 1),
            'time_stretch': lambda w, sr: AudioTransformations.time_stretch(w, 1.05),
            'eq_random': lambda w, sr: AudioTransformations.apply_eq(w, sr),
        } if comprehensive else {}
        
        self.results = defaultdict(list)
    
    def evaluate_file(self, wav_path, defense_fn):
        """
        Evaluate a single file with the given defense function.
        
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Load audio
            wav, sr = librosa.load(wav_path, sr=self.config["sr"])
            
            # Apply defense
            wav_prot = defense_fn(wav)
            
            # Ensure same length
            min_len = min(len(wav), len(wav_prot))
            wav = wav[:min_len]
            wav_prot = wav_prot[:min_len]
            
            metrics = {}
            
            # 1. ASV Similarity (primary metric)
            emb_orig = self.asv.extract(wav)
            emb_prot = self.asv.extract(wav_prot)
            sim_original = cosine_similarity(emb_orig.unsqueeze(0), emb_prot.unsqueeze(0)).item()
            metrics['asv_similarity'] = sim_original
            metrics['protection_score'] = 1.0 - sim_original  # Higher = better protection
            
            # 2. Audio Quality Metrics
            quality = compute_quality_metrics(wav, wav_prot, sr=self.config["sr"])
            metrics.update(quality)
            
            # 3. Audio Features (for analysis)
            features = extract_audio_features(wav_prot, sr=self.config["sr"])
            metrics['spectral_centroid'] = features['spectral_centroid']
            metrics['rms_energy'] = features['rms_energy']
            
            # 4. Robustness Testing (if comprehensive)
            if self.comprehensive:
                robustness_scores = {}
                
                for transform_name, transform_fn in self.transformations.items():
                    try:
                        # Apply transformation to protected audio
                        wav_transformed = transform_fn(wav_prot, self.config["sr"])
                        
                        # Test if protection survives
                        emb_trans = self.asv.extract(wav_transformed)
                        sim_after_transform = cosine_similarity(
                            emb_orig.unsqueeze(0), 
                            emb_trans.unsqueeze(0)
                        ).item()
                        
                        # Robustness = how much protection remains after transformation
                        robustness = 1.0 - sim_after_transform
                        robustness_scores[transform_name] = robustness
                        
                    except Exception as e:
                        robustness_scores[transform_name] = None
                
                metrics['robustness'] = robustness_scores
            
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Failed on {wav_path}: {e}")
            return None
    
    def evaluate_dataset(self, manifest_path, defense_fn):
        """
        Evaluate entire dataset.
        
        Args:
            manifest_path: Path to manifest file
            defense_fn: Function that applies defense to audio
        
        Returns:
            dict: Aggregate statistics
        """
        # Read manifest
        with open(manifest_path, "r") as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"[INFO] Evaluating {len(files)} files...")
        
        all_metrics = []
        
        for path in tqdm(files, desc="Evaluating"):
            metrics = self.evaluate_file(path, defense_fn)
            if metrics is not None:
                all_metrics.append(metrics)
        
        # Compute aggregate statistics
        aggregate = self._compute_statistics(all_metrics)
        
        return aggregate, all_metrics
    
    def _compute_statistics(self, all_metrics):
        """Compute aggregate statistics from all file metrics."""
        if not all_metrics:
            return {}
        
        stats = {}
        
        # Basic metrics
        for key in ['asv_similarity', 'protection_score', 'snr', 'lsd']:
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        # PESQ and STOI (if available)
        for key in ['pesq', 'stoi']:
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
        
        # Robustness (if comprehensive)
        if self.comprehensive and 'robustness' in all_metrics[0]:
            robustness_stats = defaultdict(list)
            
            for m in all_metrics:
                if 'robustness' in m:
                    for transform, score in m['robustness'].items():
                        if score is not None:
                            robustness_stats[transform].append(score)
            
            stats['robustness'] = {
                transform: {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
                for transform, scores in robustness_stats.items()
            }
        
        stats['num_files'] = len(all_metrics)
        
        return stats

def print_results(aggregate_stats, comprehensive=False):
    """Print formatted evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Files evaluated: {aggregate_stats.get('num_files', 0)}")
    
    print(f"\n🎯 Protection Effectiveness:")
    print(f"   ASV Similarity (mean): {aggregate_stats.get('asv_similarity_mean', 0):.4f} ± {aggregate_stats.get('asv_similarity_std', 0):.4f}")
    print(f"   Protection Score (mean): {aggregate_stats.get('protection_score_mean', 0):.4f} ± {aggregate_stats.get('protection_score_std', 0):.4f}")
    print(f"   Protection Score (best): {aggregate_stats.get('protection_score_max', 0):.4f}")
    
    print(f"\n🎵 Audio Quality:")
    print(f"   SNR (mean): {aggregate_stats.get('snr_mean', 0):.2f} dB ± {aggregate_stats.get('snr_std', 0):.2f}")
    print(f"   LSD (mean): {aggregate_stats.get('lsd_mean', 0):.4f} ± {aggregate_stats.get('lsd_std', 0):.4f}")
    
    if 'pesq_mean' in aggregate_stats:
        print(f"   PESQ (mean): {aggregate_stats.get('pesq_mean', 0):.3f} ± {aggregate_stats.get('pesq_std', 0):.3f}")
    
    if 'stoi_mean' in aggregate_stats:
        print(f"   STOI (mean): {aggregate_stats.get('stoi_mean', 0):.3f} ± {aggregate_stats.get('stoi_std', 0):.3f}")
    
    if comprehensive and 'robustness' in aggregate_stats:
        print(f"\n🛡️  Robustness (Protection after Transformation):")
        for transform, scores in aggregate_stats['robustness'].items():
            print(f"   {transform:15s}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    print("\n" + "="*60)
    
    # Performance assessment
    protection = aggregate_stats.get('protection_score_mean', 0)
    snr = aggregate_stats.get('snr_mean', 0)
    
    print("\n📈 Performance Assessment:")
    if protection > 0.7 and snr > 25:
        print("   ✅ EXCELLENT - High protection with good quality")
    elif protection > 0.5 and snr > 20:
        print("   ✅ GOOD - Balanced protection and quality")
    elif protection > 0.3:
        print("   ⚠️  MODERATE - Protection needs improvement")
    else:
        print("   ❌ POOR - Insufficient protection")
    
    print("="*60 + "\n")

# -------------------------
# Main Evaluation Script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--delta", type=str, help="Path to universal delta .npy")
    parser.add_argument("--predictor", type=str, help="Path to predictor .pt checkpoint")
    parser.add_argument("--multidomain", action="store_true", help="Use multi-domain predictor")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation with transformations")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    args = parser.parse_args()

    assert args.delta or args.predictor, "Must provide either --delta or --predictor"

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(CONFIG, comprehensive=args.comprehensive)

    # Load defense
    if args.delta:
        delta_np = np.load(args.delta)
        print(f"[INFO] Loaded universal delta from {args.delta}")
        defense_fn = lambda wav: apply_universal_delta(wav, delta_np, CONFIG)
        defense_name = "Universal Delta"
    
    elif args.predictor:
        if args.multidomain:
            predictor = MultiDomainPredictorNet(n_mels=CONFIG["n_mels"])
            print(f"[INFO] Using Multi-Domain Predictor")
        else:
            predictor = PredictorNet(n_mels=CONFIG["n_mels"])
        
        predictor.load_state_dict(torch.load(args.predictor, map_location=CONFIG["device"]))
        predictor.to(CONFIG["device"]).eval()
        print(f"[INFO] Loaded predictor from {args.predictor}")
        
        defense_fn = lambda wav: apply_predictor(wav, predictor, CONFIG)
        defense_name = "Multi-Domain Predictor" if args.multidomain else "Predictor Network"

    # Run evaluation
    print(f"\n[INFO] Starting evaluation with {defense_name}...")
    if args.comprehensive:
        print("[INFO] Running comprehensive evaluation (includes robustness tests)")
    
    aggregate_stats, all_metrics = evaluator.evaluate_dataset(args.manifest, defense_fn)

    # Print results
    print_results(aggregate_stats, comprehensive=args.comprehensive)

    # Save detailed results to JSON
    results = {
        'defense_type': defense_name,
        'defense_path': args.delta or args.predictor,
        'manifest': args.manifest,
        'comprehensive': args.comprehensive,
        'aggregate_statistics': aggregate_stats,
        'per_file_metrics': all_metrics[:100]  # Save first 100 for space
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Detailed results saved to {args.output}")