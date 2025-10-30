"""
apply_defense.py

Enhanced version for Phase 1 Demo:
 - Supports both Griffin-Lim and HiFi-GAN reconstruction
 - Computes PESQ/STOI quality metrics
 - Displays original/protected spectrograms
 - Optional ASV similarity via SpeechBrain ECAPA
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils_audio import (
    load_wav, save_wav, wav_to_mel,
    mel_to_wave_griffinlim, tile_delta_to_length, SR
)
from defense_training import PredictorNet, ASVEmbedder, SB_AVAILABLE

# -------------------------
# Optional: Neural Vocoder
# -------------------------
def load_hifigan(device="cpu"):
    """Load pretrained HiFi-GAN vocoder via torch.hub"""
    try:
        generator = torch.hub.load("descriptinc/melgan-neurips", "load_melgan")
        generator.to(device).eval()
        print("[INFO] HiFi-GAN vocoder loaded successfully.")
        return generator
    except Exception as e:
        print("[WARN] HiFi-GAN unavailable, falling back to Griffin-Lim:", e)
        return None

def mel_to_wave_hifigan(mel_np, generator, device="cpu"):
    """Convert mel spectrogram to waveform using HiFi-GAN."""
    mel_t = torch.from_numpy(mel_np).unsqueeze(0).to(device)
    with torch.no_grad():
        wav_t = generator.inverse(mel_t)
    wav_np = wav_t.squeeze().cpu().numpy()
    return wav_np

# -------------------------
# Apply defenses
# -------------------------
def apply_universal_delta_to_wav(wav_np, delta_np, generator=None, device='cpu'):
    mel = wav_to_mel(wav_np)
    n_mels, T = mel.shape
    tiled = tile_delta_to_length(delta_np, T)
    mel_prot = mel + tiled
    if generator is not None:
        return mel_to_wave_hifigan(mel_prot, generator, device)
    else:
        return mel_to_wave_griffinlim(mel_prot)

def apply_predictor_to_wav(wav_np, predictor_path, generator=None, device='cpu'):
    model = PredictorNet()
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.to(device).eval()
    mel = wav_to_mel(wav_np)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, n_mels, T]
    with torch.no_grad():
        delta = model(mel_t)
        mel_prot = mel_t + delta
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    if generator is not None:
        return mel_to_wave_hifigan(mel_prot_np, generator, device)
    else:
        return mel_to_wave_griffinlim(mel_prot_np)

# -------------------------
# Visualization utilities
# -------------------------
def plot_spectrograms(orig, prot, outdir):
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.imshow(wav_to_mel(orig), aspect='auto', origin='lower')
    plt.title("Original Mel Spectrogram")
    plt.subplot(2,1,2)
    plt.imshow(wav_to_mel(prot), aspect='auto', origin='lower')
    plt.title("Protected Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrogram_comparison.png"))
    plt.close()

# -------------------------
# Quality Metrics (PESQ, STOI)
# -------------------------
def compute_quality_metrics(orig, prot, sr=SR):
    try:
        from pystoi import stoi
        from pesq import pesq
        stoi_score = stoi(orig, prot, sr, extended=False)
        pesq_score = pesq(sr, orig, prot, 'wb')
        return pesq_score, stoi_score
    except Exception as e:
        print("[WARN] PESQ/STOI metrics unavailable:", e)
        return None, None

# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--delta", help="path to .npy universal delta", default=None)
    parser.add_argument("--predictor", help="path to predictor .pt", default=None)
    parser.add_argument("--outdir", default="demo_outputs")
    parser.add_argument("--use_hifigan", action="store_true", help="enable HiFi-GAN vocoder")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_hifigan(device) if args.use_hifigan else None

    wav = load_wav(args.input)
    save_wav(f"{args.outdir}/orig.wav", wav, SR)

    if args.delta:
        delta = np.load(args.delta)
        wav_prot = apply_universal_delta_to_wav(wav, delta, generator, device)
        save_wav(f"{args.outdir}/prot_universal.wav", wav_prot, SR)
    elif args.predictor:
        wav_prot = apply_predictor_to_wav(wav, args.predictor, generator, device)
        save_wav(f"{args.outdir}/prot_predictor.wav", wav_prot, SR)
    else:
        raise ValueError("Provide --delta or --predictor")

    # Visualization
    plot_spectrograms(wav, wav_prot, args.outdir)

    # Quality metrics
    pesq_score, stoi_score = compute_quality_metrics(wav, wav_prot)
    if pesq_score:
        print(f"[METRICS] PESQ: {pesq_score:.4f}, STOI: {stoi_score:.4f}")

    # ASV similarity (SpeechBrain)
    if SB_AVAILABLE:
        asv = ASVEmbedder(device=device)
        emb_orig = asv.extract(wav).cpu()
        emb_prot = asv.extract(wav_prot).cpu()
        cos = torch.nn.functional.cosine_similarity(
            emb_orig.unsqueeze(0), emb_prot.unsqueeze(0), dim=-1)
        print(f"[ASV] Cosine similarity (orig vs prot): {cos.item():.4f}")
    else:
        print("SpeechBrain not installed: skipping ASV similarity metric.")

    print(f"Saved outputs + spectrograms in: {args.outdir}")

