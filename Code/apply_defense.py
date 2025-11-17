"""
apply_defense.py - PHASE 2 ENHANCED

Production-ready inference script with:
 ✅ Neural vocoder integration (HiFi-GAN)
 ✅ Multi-domain predictor support
 ✅ RL agent support
 ✅ Comprehensive quality metrics
 ✅ Batch processing
 ✅ Audio comparison visualization

Usage examples:
  # Universal delta
  python apply_defense.py --input demo/sample.wav \
      --delta checkpoints_phase2/universal_delta_advanced_epoch2.npy

  # Multi-domain predictor
  python apply_defense.py --input demo/sample.wav \
      --predictor checkpoints_phase2/predictor_multidomain_epoch2.pt --multidomain

  # RL agent
  python apply_defense.py --input demo/sample.wav \
      --rl_agent checkpoints_phase2/rl_agent_epoch2.pt

  # Batch processing
  python apply_defense.py --input_dir demo/ --delta ... --batch
"""

import argparse
import numpy as np
import torch
import os
import json
from pathlib import Path

from utils_audio import (
    load_wav, save_wav, wav_to_mel, tile_delta_to_length, SR,
    get_vocoder, compute_quality_metrics, extract_audio_features
)
from defense_training import (
    PredictorNet, MultiDomainPredictorNet, RLPerturbationAgent,
    ASVEmbedder, SB_AVAILABLE, compute_rl_state
)

# -------------------------
# Defense Application Functions
# -------------------------
def apply_universal_delta_to_wav(wav_np, delta_np, use_neural_vocoder=True):
    """
    Apply universal delta perturbation to audio.
    
    Args:
        wav_np: Input waveform (numpy array)
        delta_np: Universal delta [n_mels, delta_T]
        use_neural_vocoder: Use HiFi-GAN if available
    
    Returns:
        Protected waveform
    """
    mel = wav_to_mel(wav_np)
    n_mels, T = mel.shape
    tiled = tile_delta_to_length(delta_np, T)
    mel_prot = mel + tiled
    
    # Reconstruct with best available vocoder
    if use_neural_vocoder:
        vocoder = get_vocoder(device='cpu')
        if vocoder.hifigan_available:
            wav_prot = vocoder.mel_to_audio(mel_prot)
            print("[INFO] Using HiFi-GAN vocoder (high quality)")
        else:
            from utils_audio import mel_to_wave_griffinlim
            wav_prot = mel_to_wave_griffinlim(mel_prot)
            print("[WARN] HiFi-GAN not available, using Griffin-Lim (lower quality)")
    else:
        from utils_audio import mel_to_wave_griffinlim
        wav_prot = mel_to_wave_griffinlim(mel_prot)
    
    return wav_prot

def apply_predictor_to_wav(wav_np, predictor, device='cpu', multidomain=False):
    """
    Apply predictor network to audio.
    
    Args:
        wav_np: Input waveform
        predictor: Loaded predictor model
        device: Computation device
        multidomain: Whether predictor is multi-domain
    
    Returns:
        Protected waveform
    """
    mel = wav_to_mel(wav_np)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if multidomain:
            # Multi-domain predictor can use time features
            # For now, just pass mel (time features optional)
            delta = predictor(mel_t)
        else:
            delta = predictor(mel_t)
        
        mel_prot = mel_t + delta
    
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    
    # Reconstruct with neural vocoder
    vocoder = get_vocoder(device=device)
    if vocoder.hifigan_available:
        wav_prot = vocoder.mel_to_audio(mel_prot_np)
        print("[INFO] Using HiFi-GAN vocoder")
    else:
        from utils_audio import mel_to_wave_griffinlim
        wav_prot = mel_to_wave_griffinlim(mel_prot_np)
        print("[WARN] Using Griffin-Lim vocoder")
    
    return wav_prot

def apply_rl_agent_to_wav(wav_np, rl_agent, base_predictor, device='cpu'):
    """
    Apply RL agent-guided perturbation.
    
    Args:
        wav_np: Input waveform
        rl_agent: Trained RL agent
        base_predictor: Base predictor network
        device: Computation device
    
    Returns:
        Protected waveform
    """
    # Extract features for RL state
    features = extract_audio_features(wav_np, sr=SR)
    state = compute_rl_state(features).to(device)
    
    # RL agent selects perturbation parameters
    with torch.no_grad():
        action, _ = rl_agent.select_action(state)
    
    epsilon_scale = (action[0].item() + 1.0) / 2.0  # [0, 1]
    
    # Apply perturbation with RL-selected parameters
    mel = wav_to_mel(wav_np)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        delta = base_predictor(mel_t)
        # Scale by RL action
        delta = delta * epsilon_scale * 0.02  # pgd_eps
    
    mel_prot = mel_t + delta
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    
    # Reconstruct
    vocoder = get_vocoder(device=device)
    if vocoder.hifigan_available:
        wav_prot = vocoder.mel_to_audio(mel_prot_np)
    else:
        from utils_audio import mel_to_wave_griffinlim
        wav_prot = mel_to_wave_griffinlim(mel_prot_np)
    
    return wav_prot

# -------------------------
# Batch Processing
# -------------------------
def process_batch(input_dir, output_dir, defense_fn, verbose=True):
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        defense_fn: Function to apply defense
        verbose: Print progress
    
    Returns:
        List of processed files with metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg']
    input_path = Path(input_dir)
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))
    
    if verbose:
        print(f"[INFO] Found {len(audio_files)} audio files in {input_dir}")
    
    results = []
    
    for audio_file in audio_files:
        try:
            if verbose:
                print(f"[INFO] Processing: {audio_file.name}")
            
            # Load audio
            wav = load_wav(str(audio_file))
            
            # Apply defense
            wav_prot = defense_fn(wav)
            
            # Ensure same length
            min_len = min(len(wav), len(wav_prot))
            wav = wav[:min_len]
            wav_prot = wav_prot[:min_len]
            
            # Save protected audio
            output_path = Path(output_dir) / f"protected_{audio_file.name}"
            save_wav(str(output_path), wav_prot, SR)
            
            # Compute metrics
            metrics = compute_quality_metrics(wav, wav_prot, sr=SR)
            metrics['file'] = audio_file.name
            metrics['output'] = str(output_path)
            
            results.append(metrics)
            
            if verbose:
                print(f"   ✓ SNR: {metrics['snr']:.2f} dB, LSD: {metrics['lsd']:.4f}")
        
        except Exception as e:
            print(f"[ERROR] Failed to process {audio_file.name}: {e}")
            continue
    
    return results

# -------------------------
# Visualization (Optional)
# -------------------------
def create_comparison_plot(wav_orig, wav_prot, sr, output_path):
    """
    Create side-by-side comparison plot of original and protected audio.
    """
    try:
        import matplotlib.pyplot as plt
        import librosa.display
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Waveforms
        axes[0, 0].plot(wav_orig, alpha=0.7, label='Original')
        axes[0, 0].set_title('Original Waveform')
        axes[0, 0].set_xlabel('Samples')
        axes[0, 0].set_ylabel('Amplitude')
        
        axes[0, 1].plot(wav_prot, alpha=0.7, label='Protected', color='orange')
        axes[0, 1].set_title('Protected Waveform')
        axes[0, 1].set_xlabel('Samples')
        axes[0, 1].set_ylabel('Amplitude')
        
        # Spectrograms
        mel_orig = wav_to_mel(wav_orig, sr=sr)
        mel_prot = wav_to_mel(wav_prot, sr=sr)
        
        librosa.display.specshow(mel_orig, sr=sr, hop_length=256, 
                                x_axis='time', y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title('Original Mel Spectrogram')
        
        librosa.display.specshow(mel_prot, sr=sr, hop_length=256,
                                x_axis='time', y_axis='mel', ax=axes[1, 1])
        axes[1, 1].set_title('Protected Mel Spectrogram')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"[INFO] Comparison plot saved to {output_path}")
        return True
        
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization")
        return False

# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply audio defense to protect against voice cloning")
    
    # Input/Output
    parser.add_argument("--input", help="Input audio file path")
    parser.add_argument("--input_dir", help="Input directory for batch processing")
    parser.add_argument("--outdir", default="demo_outputs", help="Output directory")
    
    # Defense methods
    parser.add_argument("--delta", help="Path to universal delta .npy")
    parser.add_argument("--predictor", help="Path to predictor .pt checkpoint")
    parser.add_argument("--rl_agent", help="Path to RL agent .pt checkpoint")
    
    # Options
    parser.add_argument("--multidomain", action="store_true", help="Use multi-domain predictor")
    parser.add_argument("--no_vocoder", action="store_true", help="Disable neural vocoder (use Griffin-Lim)")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--visualize", action="store_true", help="Create comparison plots")
    parser.add_argument("--compute_asv", action="store_true", help="Compute ASV similarity")
    
    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Must provide either --input or --input_dir")
    
    if not (args.delta or args.predictor or args.rl_agent):
        parser.error("Must provide one of: --delta, --predictor, --rl_agent")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load defense model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    defense_fn = None
    defense_name = ""
    
    if args.delta:
        delta = np.load(args.delta)
        print(f"[INFO] Loaded universal delta from {args.delta}")
        print(f"[INFO] Delta shape: {delta.shape}")
        defense_fn = lambda wav: apply_universal_delta_to_wav(
            wav, delta, use_neural_vocoder=not args.no_vocoder
        )
        defense_name = "Universal Delta"
    
    elif args.predictor:
        if args.multidomain:
            predictor = MultiDomainPredictorNet()
            print("[INFO] Loading Multi-Domain Predictor")
        else:
            predictor = PredictorNet()
            print("[INFO] Loading Simple Predictor")
        
        predictor.load_state_dict(torch.load(args.predictor, map_location=device))
        predictor.to(device).eval()
        print(f"[INFO] Loaded predictor from {args.predictor}")
        
        defense_fn = lambda wav: apply_predictor_to_wav(
            wav, predictor, device=device, multidomain=args.multidomain
        )
        defense_name = "Multi-Domain Predictor" if args.multidomain else "Predictor Network"
    
    elif args.rl_agent:
        rl_agent = RLPerturbationAgent()
        rl_agent.load_state_dict(torch.load(args.rl_agent, map_location=device))
        rl_agent.to(device).eval()
        
        # Need base predictor for RL agent
        base_predictor = PredictorNet().to(device).eval()
        print("[INFO] Loaded RL agent")
        
        defense_fn = lambda wav: apply_rl_agent_to_wav(
            wav, rl_agent, base_predictor, device=device
        )
        defense_name = "RL Adaptive Agent"

    # Process audio
    if args.batch and args.input_dir:
        # Batch processing
        print(f"\n[INFO] Starting batch processing with {defense_name}...")
        results = process_batch(args.input_dir, args.outdir, defense_fn, verbose=True)
        
        # Save batch results
        summary = {
            'defense_type': defense_name,
            'num_files': len(results),
            'results': results
        }
        
        summary_path = os.path.join(args.outdir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[INFO] Batch processing complete!")
        print(f"[INFO] Processed {len(results)} files")
        print(f"[INFO] Summary saved to {summary_path}")
        
        # Compute average metrics
        if results:
            avg_snr = np.mean([r['snr'] for r in results])
            avg_lsd = np.mean([r['lsd'] for r in results])
            print(f"\n📊 Average Quality Metrics:")
            print(f"   SNR: {avg_snr:.2f} dB")
            print(f"   LSD: {avg_lsd:.4f}")
    
    else:
        # Single file processing
        print(f"\n[INFO] Processing single file with {defense_name}...")
        
        wav = load_wav(args.input)
        print(f"[INFO] Loaded audio: {len(wav)} samples ({len(wav)/SR:.2f}s)")
        
        # Save original
        save_wav(f"{args.outdir}/original.wav", wav, SR)
        
        # Apply defense
        print("[INFO] Applying defense...")
        wav_prot = defense_fn(wav)
        
        # Ensure same length for comparison
        min_len = min(len(wav), len(wav_prot))
        wav = wav[:min_len]
        wav_prot = wav_prot[:min_len]
        
        # Save protected
        output_path = f"{args.outdir}/protected.wav"
        save_wav(output_path, wav_prot, SR)
        print(f"[INFO] Saved protected audio to {output_path}")
        
        # Compute quality metrics
        print("\n[INFO] Computing quality metrics...")
        metrics = compute_quality_metrics(wav, wav_prot, sr=SR)
        
        print("\n📊 Quality Metrics:")
        print(f"   SNR: {metrics['snr']:.2f} dB")
        print(f"   LSD: {metrics['lsd']:.4f}")
        if metrics.get('pesq'):
            print(f"   PESQ: {metrics['pesq']:.3f}")
        if metrics.get('stoi'):
            print(f"   STOI: {metrics['stoi']:.3f}")
        
        # Compute ASV similarity (if requested)
        if args.compute_asv and SB_AVAILABLE:
            print("\n[INFO] Computing ASV speaker similarity...")
            try:
                asv = ASVEmbedder(device=device)
                emb_orig = asv.extract(wav).cpu()
                emb_prot = asv.extract(wav_prot).cpu()
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_orig.unsqueeze(0), 
                    emb_prot.unsqueeze(0), 
                    dim=-1
                ).item()
                
                protection_score = 1.0 - cos_sim
                
                print(f"\n🎯 Protection Effectiveness:")
                print(f"   ASV Cosine Similarity: {cos_sim:.4f}")
                print(f"   Protection Score: {protection_score:.4f}")
                
                if protection_score > 0.7:
                    print("   ✅ EXCELLENT protection")
                elif protection_score > 0.5:
                    print("   ✅ GOOD protection")
                elif protection_score > 0.3:
                    print("   ⚠️  MODERATE protection")
                else:
                    print("   ❌ WEAK protection")
                
                metrics['asv_similarity'] = cos_sim
                metrics['protection_score'] = protection_score
                
            except Exception as e:
                print(f"[WARN] ASV similarity computation failed: {e}")
        
        # Create visualization
        if args.visualize:
            plot_path = f"{args.outdir}/comparison.png"
            create_comparison_plot(wav, wav_prot, SR, plot_path)
        
        # Save metrics
        metrics_path = f"{args.outdir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[INFO] Metrics saved to {metrics_path}")
    
    print("\n✅ Processing complete!")
    print(f"[INFO] All outputs saved to {args.outdir}/")