"""
evaluate_defense.py

Batch evaluation of universal audio perturbation defense.
- Loads a manifest file (list of wav paths)
- Applies trained perturbation (universal delta or predictor)
- Computes cosine similarity before vs after (ASV embedding)
- Prints average similarity drop

Usage:
    python evaluate_defense.py --manifest data/librispeech_prepared/test-clean_manifest.txt \
        --delta checkpoints_phase1/universal_delta_epoch25.npy
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

from defense_training import (
    wav_to_mel,
    mel_to_wave_griffinlim,
    PredictorNet,
    ASVEmbedder,
    cosine_similarity_torch as cosine_similarity,
    config as CONFIG
)

def apply_universal_delta(wav, delta_np, config=CONFIG):
    """Apply universal delta tiled across mel frames."""
    mel = wav_to_mel(wav, sr=config["sr"], n_fft=config["n_fft"],
                     hop_length=config["hop_length"], n_mels=config["n_mels"])
    n_mels, T = mel.shape
    delta_T = delta_np.shape[1]
    reps = int(np.ceil(T / delta_T))
    delta_tiled = np.tile(delta_np, (1, reps))[:, :T]
    mel_prot = mel + delta_tiled
    wav_prot = mel_to_wave_griffinlim(mel_prot, sr=config["sr"],
                                      n_fft=config["n_fft"],
                                      hop_length=config["hop_length"])
    return wav_prot

def apply_predictor(wav, predictor, config=CONFIG):
    """Apply trained predictor network chunk-wise."""
    mel = wav_to_mel(wav, sr=config["sr"], n_fft=config["n_fft"],
                     hop_length=config["hop_length"], n_mels=config["n_mels"])
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(config["device"])
    with torch.no_grad():
        delta = predictor(mel_t)
        mel_prot = mel_t + delta
    mel_prot_np = mel_prot.squeeze().cpu().numpy()
    wav_prot = mel_to_wave_griffinlim(mel_prot_np, sr=config["sr"],
                                      n_fft=config["n_fft"],
                                      hop_length=config["hop_length"])
    return wav_prot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--delta", type=str, help="Path to universal delta .npy")
    parser.add_argument("--predictor", type=str, help="Path to predictor .pt checkpoint")
    args = parser.parse_args()

    assert args.delta or args.predictor, "Must provide either --delta or --predictor"

    # Init ASV model
    asv = ASVEmbedder(device=CONFIG["device"])

    # Load perturbation
    predictor = None
    delta_np = None
    if args.delta:
        delta_np = np.load(args.delta)
    elif args.predictor:
        predictor = PredictorNet(n_mels=CONFIG["n_mels"])
        predictor.load_state_dict(torch.load(args.predictor, map_location=CONFIG["device"]))
        predictor.to(CONFIG["device"]).eval()

    # Read manifest
    with open(args.manifest, "r") as f:
        files = [line.strip() for line in f.readlines()]

    sim_before = []
    sim_after = []

    print(f"[INFO] Evaluating {len(files)} files...")
    for path in tqdm(files):
        try:
            wav, sr = librosa.load(path, sr=CONFIG["sr"])
            # Original embedding
            emb_orig = asv.extract(wav)

            # Apply defense
            if delta_np is not None:
                wav_prot = apply_universal_delta(wav, delta_np, CONFIG)
            else:
                wav_prot = apply_predictor(wav, predictor, CONFIG)

            emb_prot = asv.extract(wav_prot)

            sim = cosine_similarity(emb_orig.unsqueeze(0), emb_prot.unsqueeze(0))
            sim_before.append(1.0)  # baseline (same file, similarity = ~1)
            sim_after.append(sim.item())
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
            continue

    # Results
    avg_sim = np.mean(sim_after)
    print("\n===== Evaluation Results =====")
    print(f"Files evaluated: {len(sim_after)}")
    print(f"Avg cosine similarity (after defense): {avg_sim:.4f}")
    print(f"Avg similarity drop vs baseline: {1 - avg_sim:.4f}")
    print("================================\n")
