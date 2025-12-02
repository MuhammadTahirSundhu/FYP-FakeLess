# ===========================================================================
# FILE: evaluate.py
# ===========================================================================
"""
Evaluation Script
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

import config
import audio_utils
from models import MultiDomainPerturbationNetwork, load_checkpoint

class AudioEvaluator:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device or config.DEVICE)
        
        print(f"Loading model from {model_path}...")
        self.model = MultiDomainPerturbationNetwork().to(self.device)
        load_checkpoint(self.model, None, model_path)
        self.model.eval()
        
        print("Loading auxiliary models...")
        self.vocoder = audio_utils.HiFiGANVocoder(device=self.device)
        self.asv = audio_utils.ASVEmbedder(device=self.device)
        
        print("✓ Evaluator ready\n")
    
    def evaluate_file(self, audio_path, epsilon=None):
        epsilon = epsilon or config.EPSILON
        
        wav_orig = audio_utils.load_audio(audio_path)
        mel_orig = audio_utils.wav_to_mel(wav_orig)
        wav_orig_norm = audio_utils.normalize_audio(wav_orig)
        
        waveform_t = torch.from_numpy(wav_orig_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        mel_t = torch.from_numpy(mel_orig).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            protected_mel, _, _, _ = self.model.apply_perturbation(waveform_t, mel_t, epsilon)
        
        wav_prot = self.vocoder.mel_to_audio(protected_mel)
        if isinstance(wav_prot, torch.Tensor):
            wav_prot = wav_prot.cpu().numpy().squeeze()
        
        min_len = min(len(wav_orig_norm), len(wav_prot))
        wav_orig_norm = wav_orig_norm[:min_len]
        wav_prot = wav_prot[:min_len]
        
        results = audio_utils.compute_all_metrics(wav_orig_norm, wav_prot)
        
        if self.asv.available:
            emb_orig = self.asv.extract(wav_orig_norm)
            emb_prot = self.asv.extract(wav_prot)
            
            if emb_orig is not None and emb_prot is not None:
                similarity = torch.nn.functional.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_prot.unsqueeze(0)
                ).item()
                results['asv_similarity'] = similarity
                results['protection_score'] = 1.0 - similarity
        
        return results
    
    def evaluate_dataset(self, manifest_path, epsilon=None, max_files=None):
        with open(manifest_path, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        
        if max_files:
            audio_paths = audio_paths[:max_files]
        
        print(f"Evaluating {len(audio_paths)} files...\n")
        
        all_results = []
        for audio_path in tqdm(audio_paths):
            try:
                results = self.evaluate_file(audio_path, epsilon)
                all_results.append(results)
            except Exception as e:
                print(f"Error: {audio_path} - {e}")
        
        aggregated = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if key in r]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def test_robustness(self, audio_path, epsilon=None):
        epsilon = epsilon or config.EPSILON
        
        print(f"\nTesting robustness: {audio_path}\n")
        
        wav_orig = audio_utils.load_audio(audio_path)
        mel_orig = audio_utils.wav_to_mel(wav_orig)
        wav_orig_norm = audio_utils.normalize_audio(wav_orig)
        
        waveform_t = torch.from_numpy(wav_orig_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        mel_t = torch.from_numpy(mel_orig).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            protected_mel, _, _, _ = self.model.apply_perturbation(waveform_t, mel_t, epsilon)
        
        wav_prot = self.vocoder.mel_to_audio(protected_mel)
        if isinstance(wav_prot, torch.Tensor):
            wav_prot = wav_prot.cpu().numpy().squeeze()
        
        results = {}
        
        if self.asv.available:
            emb_orig = self.asv.extract(wav_orig_norm)
            
            # MP3 compression
            print("  Testing MP3 compression...")
            wav_mp3 = audio_utils.simulate_mp3_compression(wav_prot, bitrate='128k')
            emb_mp3 = self.asv.extract(wav_mp3[:len(wav_orig_norm)])
            if emb_orig is not None and emb_mp3 is not None:
                sim = torch.nn.functional.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_mp3.unsqueeze(0)
                ).item()
                results['mp3_128k'] = 1.0 - sim
            
            # Gaussian noise
            print("  Testing Gaussian noise...")
            wav_noise = audio_utils.add_gaussian_noise(wav_prot, snr_db=30)
            emb_noise = self.asv.extract(wav_noise[:len(wav_orig_norm)])
            if emb_orig is not None and emb_noise is not None:
                sim = torch.nn.functional.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_noise.unsqueeze(0)
                ).item()
                results['gaussian_noise'] = 1.0 - sim
            
            # Bandpass filter
            print("  Testing bandpass filter...")
            wav_filtered = audio_utils.apply_bandpass_filter(wav_prot)
            emb_filtered = self.asv.extract(wav_filtered[:len(wav_orig_norm)])
            if emb_orig is not None and emb_filtered is not None:
                sim = torch.nn.functional.cosine_similarity(
                    emb_orig.unsqueeze(0), emb_filtered.unsqueeze(0)
                ).item()
                results['bandpass'] = 1.0 - sim
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/best_model.pt')
    parser.add_argument('--test_manifest', default=config.TEST_MANIFEST)
    parser.add_argument('--test_file', help='Test single file')
    parser.add_argument('--robustness', action='store_true')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON)
    parser.add_argument('--max_files', type=int)
    parser.add_argument('--device', default=config.DEVICE)
    args = parser.parse_args()
    
    evaluator = AudioEvaluator(args.model, args.device)
    
    if args.test_file:
        print(f"Evaluating: {args.test_file}\n")
        results = evaluator.evaluate_file(args.test_file, args.epsilon)
        
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        for key, value in results.items():
            print(f"{key:20s}: {value:.4f}")
        print("=" * 70)
        
        if args.robustness:
            rob_results = evaluator.test_robustness(args.test_file, args.epsilon)
            print("\nROBUSTNESS RESULTS")
            print("=" * 70)
            for key, value in rob_results.items():
                print(f"{key:20s}: {value:.4f}")
            print("=" * 70)
    else:
        results = evaluator.evaluate_dataset(args.test_manifest, args.epsilon, args.max_files)
        
        print("\n" + "=" * 70)
        print("DATASET EVALUATION RESULTS")
        print("=" * 70)
        for metric, stats in results.items():
            print(f"\n{metric.upper()}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name:10s}: {value:.4f}")
        print("=" * 70)
        
        print("\nSUMMARY:")
        if 'pesq' in results:
            print(f"  Audio Quality (PESQ): {results['pesq']['mean']:.3f} ± {results['pesq']['std']:.3f}")
        if 'stoi' in results:
            print(f"  Intelligibility (STOI): {results['stoi']['mean']:.3f} ± {results['stoi']['std']:.3f}")
        if 'protection_score' in results:
            print(f"  Protection: {results['protection_score']['mean']:.2%} ± {results['protection_score']['std']:.2%}")
        print("=" * 70)

if __name__ == "__main__":
    main()