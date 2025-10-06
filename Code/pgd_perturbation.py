"""
pgd_perturbation.py

Per-sample PGD-style mel-domain perturbation script (standalone).
This is useful for:
 - quick experiments to demonstrate the idea (per-sample adversary)
 - producing offline per-sample deltas for strong examples

How it works (conceptually):
 - compute log-mel of input audio
 - initialize delta (same shape as mel) small
 - for K steps, update delta by minimizing total_loss that includes:
     - attack surrogate loss (placeholder)
     - quality loss (L2 on mel)
     - reg loss on delta
 - project delta to L_inf ball (clip) to keep perturbation small
 - reconstruct protected waveform via Griffin-Lim (demo; replace with HiFi-GAN for good quality)

Usage:
    python pgd_perturbation.py --input sample.wav --output protected.wav --eps 0.02 --steps 10
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_audio import load_wav, save_wav, wav_to_mel, mel_to_wave_griffinlim, SR, N_MELS

# -----------------------
# Simple surrogate attack loss (placeholder)
# -----------------------
def surrogate_attack_loss_simple(mel_prot):
    """
    PLACEHOLDER. For Phase 1 quick experiments:
      the surrogate returns mean energy in mel domain so optimization has some signal.
    Replace with real attacker loss (ASV cosine sim) when integrating SpeechBrain.
    """
    return torch.mean(mel_prot ** 2)

# -----------------------
# PGD optimizer
# -----------------------
def pgd_per_sample(input_wav_np,
                   eps=0.02,
                   steps=10,
                   alpha=None,
                   lambda_attack=1.0,
                   lambda_quality=10.0,
                   lambda_reg=0.1,
                   device='cpu'):
    """
    Input:
      input_wav_np: numpy waveform
    Returns:
      protected waveform numpy
    Notes:
      - Operates in log-mel domain
      - Uses a differentiable torch variable for delta and updates via Adam
    """
    # compute mel (numpy -> torch)
    mel_orig = wav_to_mel(input_wav_np)  # shape [n_mels, T]
    mel_torch = torch.from_numpy(mel_orig).float().to(device)

    n_mels, T = mel_orig.shape
    if alpha is None:
        alpha = eps / steps

    # delta is trainable tensor
    delta = torch.zeros((n_mels, T), requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=alpha)

    for step in range(steps):
        optimizer.zero_grad()
        mel_prot = mel_torch + delta  # protected mel (torch)
        # attack loss (placeholder) - replace with real ASV/TTS embedder loss
        attack_loss = surrogate_attack_loss_simple(mel_prot)
        quality_loss = torch.mean((mel_torch - mel_prot) ** 2)
        reg_loss = torch.sum(delta ** 2)
        total_loss = lambda_attack * attack_loss + lambda_quality * quality_loss + lambda_reg * reg_loss
        # backprop and step
        total_loss.backward()
        optimizer.step()
        # L_inf projection on delta
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
        print(f"Step {step+1}/{steps} total_loss={total_loss.item():.6f}")

    # final reconstruct
    mel_final = (mel_torch + delta).detach().cpu().numpy()
    wav_final = mel_to_wave_griffinlim(mel_final)
    # ensure same length as original (trim/pad)
    minlen = min(len(wav_final), len(input_wav_np))
    wav_final = wav_final[:minlen]
    return wav_final

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eps', type=float, default=0.02)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()

    wav = load_wav(args.input)
    prot = pgd_per_sample(wav, eps=args.eps, steps=args.steps, device='cpu')
    save_wav(args.output, prot, SR)
    print("Saved protected audio to", args.output)
