"""
apply_defense.py

Inference script for Phase 1 demo:
 - load input wav
 - apply either universal delta (.npy) or predictor checkpoint (.pt)
 - reconstruct to waveform (Griffin-Lim fallback)
 - compute ASV embedding similarity before/after using SpeechBrain ECAPA (if installed)
 - save original & protected audio for playback

Usage examples:
  python apply_defense.py --input demo/sample.wav --delta checkpoints_phase1/universal_delta_epoch25.npy
  python apply_defense.py --input demo/sample.wav --predictor checkpoints_phase1/predictor_epoch25.pt
"""

import argparse
import numpy as np
import torch
from utils_audio import load_wav, save_wav, wav_to_mel, mel_to_wave_griffinlim, tile_delta_to_length, SR
from defense_training import PredictorNet, ASVEmbedder, SB_AVAILABLE

def apply_universal_delta_to_wav(wav_np, delta_np):
    mel = wav_to_mel(wav_np)
    n_mels, T = mel.shape
    tiled = tile_delta_to_length(delta_np, T)
    mel_prot = mel + tiled
    wav_prot = mel_to_wave_griffinlim(mel_prot)
    return wav_prot

def apply_predictor_to_wav(wav_np, predictor_path, device='cpu'):
    model = PredictorNet()
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.to(device).eval()
    mel = wav_to_mel(wav_np)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, n_mels, T]
    with torch.no_grad():
        delta = model(mel_t)  # [1, n_mels, T]
        mel_prot = mel_t + delta
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    wav_prot = mel_to_wave_griffinlim(mel_prot_np)
    return wav_prot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--delta", help="path to .npy universal delta", default=None)
    parser.add_argument("--predictor", help="path to predictor .pt", default=None)
    parser.add_argument("--outdir", default="demo_outputs")
    args = parser.parse_args()
    if args.input is None:
        args.input = os.environ.get("INPUT_PATH")
    if args.delta is None:
        args.delta = os.environ.get("DELTA_PATH")
    if args.predictor is None:
        args.predictor = os.environ.get("PREDICTOR_PATH")
    if args.outdir is None:
        args.outdir = os.environ.get("OUT_DIR", args.outdir)

    # validate
    if args.input is None:
        raise ValueError("No input provided. Provide --input or set INPUT_PATH env variable.")

    import os
    os.makedirs(args.outdir, exist_ok=True)

    wav = load_wav(args.input)
    save_wav(f"{args.outdir}/orig.wav", wav, SR)

    if args.delta:
        delta = np.load(args.delta)
        wav_prot = apply_universal_delta_to_wav(wav, delta)
        save_wav(f"{args.outdir}/prot_universal.wav", wav_prot, SR)
    elif args.predictor:
        wav_prot = apply_predictor_to_wav(wav, args.predictor, device='cpu')
        save_wav(f"{args.outdir}/prot_predictor.wav", wav_prot, SR)
    else:
        raise ValueError("Provide --delta or --predictor")

    # compute ASV similarity if SpeechBrain present
    if SB_AVAILABLE:
        asv = ASVEmbedder(device='cpu')
        emb_orig = asv.extract(wav).cpu()
        emb_prot = asv.extract(wav_prot).cpu()
        cos = torch.nn.functional.cosine_similarity(emb_orig.unsqueeze(0), emb_prot.unsqueeze(0), dim=-1)
        print(f"ASV cosine similarity (orig vs prot): {cos.item():.4f}")
    else:
        print("SpeechBrain not installed: skipping ASV similarity metric.")
    print("Saved outputs in", args.outdir)
