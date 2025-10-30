"""
defense_training.py

Training script for:
 - Universal delta (mel domain)  -- fast to apply at inference
 - Predictor network (chunk-level) -- for low-latency streaming inference

Key points and design:
 - Uses SpeechBrain ECAPA-TDNN embeddings as surrogate attacker (ASV) if available.
 - Minimizes: total_loss = lambda_attack*attack_loss + lambda_quality*quality_loss + lambda_reg*reg_loss
 - Stores deltas and predictor checkpoints under `save_dir`

PHASE-1 USE:
 - Run `python defense_training.py train_universal` to train universal δ (Phase 1 main deliverable)
 - Save outputs for demo and use in apply_defense.py (inference)
"""

import os
os.environ["SPEECHBRAIN_CACHE_STRATEGY"] = "copy"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils_audio import load_wav, wav_to_mel, mel_to_wave_griffinlim, tile_delta_to_length, SR, N_MELS, N_FFT, HOP_LENGTH
# Predictor small network (Conv1D)
class PredictorNet(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(hidden, n_mels, kernel_size=1)

    def forward(self, mel_chunk):
        # mel_chunk: [B, n_mels, T]
        x = self.conv1(mel_chunk)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.conv_out(x)  # delta: [B, n_mels, T]
        return out

# ----------------------------
# Optional: SpeechBrain ECAPA embedder wrapper (surrogate attacker)
# ----------------------------
try:
    from speechbrain.inference.speaker import SpeakerRecognition

    SB_AVAILABLE = True
except Exception:
    SB_AVAILABLE = False
    # print a friendly instruction
    print("SpeechBrain not available. Install with `pip install speechbrain` to enable ASV surrogate loss.")

class ASVEmbedder:
    def __init__(self, device='cpu'):
        if not SB_AVAILABLE:
            raise RuntimeError("SpeechBrain not installed.")
        
        self.device = device
        # Load ECAPA-TDNN model for speaker embedding
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec_ecapa",
            run_opts={"device": device}
        )

    def extract(self, wav_np):
        """
        Accepts: 1D numpy array of audio samples (float32, 16kHz)
        Returns: 1D torch embedding tensor (speaker embedding)
        """
        # Ensure float tensor and move to device
        wav_tensor = torch.as_tensor(wav_np, dtype=torch.float32, device=self.device)

        # Convert to mono if multiple channels
        if wav_tensor.dim() == 2 and wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor.mean(dim=0)

        # Add batch dimension [1, time]
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Extract embedding using SpeechBrain ECAPA-TDNN
        with torch.no_grad():
            emb = self.model.encode_batch(wav_tensor)  # shape [1, emb_dim]

        # Return as flattened tensor on correct device
        return emb.squeeze(0).detach().to(self.device)
# ----------------------------
# Config (small; modify for your environment)
# ----------------------------
config = {
    "sr": SR,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "batch_size": 6,
    "epochs": 25,
    "universal_lr": 1e-2,
    "predictor_lr": 1e-4,
    "pgd_eps": 0.02,
    "lambda_attack": 1.0,
    "lambda_quality": 10.0,
    "lambda_reg": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints_phase1",
    "train_manifest": "../data/librispeech_prepared/train-clean-100_manifest.txt",
    "val_manifest": "../data/librispeech_prepared/dev-clean_manifest.txt"
}
os.makedirs(config["save_dir"], exist_ok=True)

# ----------------------------
# small helpers
# ----------------------------
def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def batch_generator(manifest, batch_size=6, shuffle=True):
    idx = list(range(len(manifest)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch = [manifest[j] for j in idx[i:i+batch_size]]
        waves = [load_wav(p, sr=config["sr"]) for p in batch]
        yield batch, waves

def cosine_similarity_torch(a, b):
    # a,b: torch tensors [dim]
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    return torch.sum(a * b)

# ----------------------------
# Universal delta training
# ----------------------------
def train_universal_delta(manifest_path, asv_embedder=None):
    manifest = load_manifest(manifest_path)
    device = config["device"]
    n_mels = config["n_mels"]
    eps = config["pgd_eps"]

    # choose delta temporal length small (e.g., 32 frames) and tile
    delta_T = 32
    delta = torch.zeros((n_mels, delta_T), dtype=torch.float32, device=device, requires_grad=True)
    opt = optim.Adam([delta], lr=config["universal_lr"])

    for epoch in range(config["epochs"]):
        pbar = tqdm(batch_generator(manifest, batch_size=config["batch_size"], shuffle=True),
                    total=math.ceil(len(manifest)/config["batch_size"]),
                    desc=f"uni-delta epoch {epoch+1}/{config['epochs']}")
        epoch_loss = 0.0
        for paths, waves in pbar:
            # mel batch and max frames
            mels = [wav_to_mel(w, sr=config["sr"], n_fft=config["n_fft"], hop_length=config["hop_length"], n_mels=n_mels) for w in waves]
            maxT = max(m.shape[1] for m in mels)
            mel_batch = []
            for m in mels:
                # pad to maxT
                pad = maxT - m.shape[1]
                mel_batch.append(np.pad(m, ((0,0),(0,pad)), mode='constant'))
            mel_batch = np.stack(mel_batch, axis=0)  # [B, n_mels, T]
            mel_batch_t = torch.from_numpy(mel_batch).float().to(device)

            # tile delta
            delta_tiled = torch.tile(delta.unsqueeze(0), (mel_batch_t.size(0),1, math.ceil(maxT/delta_T)))
            delta_tiled = delta_tiled[:,:,:maxT]

            mel_prot = mel_batch_t + delta_tiled

            # compute attack loss via asv_embedder if available (slow because of reconstruction)
            if asv_embedder is not None:
                attack_losses = []
                for b in range(mel_prot.size(0)):
                    mel_np = mel_prot[b].detach().cpu().numpy()
                    wav_recon = mel_to_wave_griffinlim(mel_np, sr=config["sr"], n_fft=config["n_fft"], hop_length=config["hop_length"])
                    # original wave for this index:
                    orig_wav = waves[b]
                    e_orig = asv_embedder.extract(orig_wav).to(device)
                    e_prot = asv_embedder.extract(wav_recon).to(device)
                    attack_losses.append(cosine_similarity_torch(e_orig, e_prot))
                # we want to minimize similarity (i.e., decrease attack success). Convert to torch scalar.
                attack_loss = torch.stack(attack_losses).mean()
            else:
                # surrogate cheap loss (mel energy) if no embedder present
                attack_loss = torch.mean(mel_prot ** 2)

            quality = torch.mean((mel_batch_t - mel_prot) ** 2)
            reg = torch.sum(delta ** 2)

            total_loss = config["lambda_attack"] * attack_loss + config["lambda_quality"] * quality + config["lambda_reg"] * reg

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -eps, eps)

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=epoch_loss)

        # save checkpoint delta
        np.save(os.path.join(config["save_dir"], f"universal_delta_epoch{epoch+1}.npy"), delta.detach().cpu().numpy())
    np.save(os.path.join("./final_delta", f"universal_delta_epoch.npy"), delta.detach().cpu().numpy())
    print("Universal delta training finished.")

# ----------------------------
# Predictor training (chunk-level)
# ----------------------------
def train_predictor(manifest_path, asv_embedder=None):
    manifest = load_manifest(manifest_path)
    device = config["device"]
    predictor = PredictorNet(n_mels=config["n_mels"]).to(device)
    opt = optim.Adam(predictor.parameters(), lr=config["predictor_lr"])

    for epoch in range(config["epochs"]):
        pbar = tqdm(batch_generator(manifest, batch_size=config["batch_size"], shuffle=True),
                    total=math.ceil(len(manifest)/config["batch_size"]),
                    desc=f"predictor epoch {epoch+1}/{config['epochs']}")
        for paths, waves in pbar:
            # build chunked mel tensors
            chunks = []
            for w in waves:
                mel = wav_to_mel(w, sr=config["sr"], n_fft=config["n_fft"], hop_length=config["hop_length"], n_mels=config["n_mels"])
                T = mel.shape[1]
                # pad to multiple of chunk length (T_chunk)
                T_chunk = 64
                pad = (T_chunk - (T % T_chunk)) % T_chunk
                mel_p = np.pad(mel, ((0,0),(0,pad)), mode='constant')
                mel_chunks = mel_p.reshape(config["n_mels"], -1, T_chunk).transpose(1,0,2)  # [n_chunks, n_mels, T_chunk]
                # sample up to 4 chunks per utterance to limit memory
                n_take = min(4, mel_chunks.shape[0])
                idxs = random.sample(range(mel_chunks.shape[0]), n_take)
                for i_idx in idxs:
                    chunks.append(torch.from_numpy(mel_chunks[i_idx]).float())

            if len(chunks) == 0:
                continue

            mel_batch = torch.stack(chunks, dim=0).to(device)  # [B, n_mels, T_chunk]
            delta_pred = predictor(mel_batch)  # [B, n_mels, T_chunk]
            mel_prot = mel_batch + delta_pred

            # for speed we don't reconstruct to waveform here; we use mel-based surrogate attack loss
            # If asv_embedder is available and you want true embedder loss, you would reconstruct each chunk to waveform
            # and run embedder.extract() — expensive in training.
            if asv_embedder is not None:
                attack_loss = torch.mean(mel_prot ** 2)  # placeholder to let training proceed
            else:
                attack_loss = torch.mean(mel_prot ** 2)

            quality_loss = torch.mean((mel_batch - mel_prot) ** 2)
            reg_loss = torch.sum(delta_pred ** 2)
            total_loss = config["lambda_attack"] * attack_loss + config["lambda_quality"] * quality_loss + config["lambda_reg"] * reg_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            pbar.set_postfix(loss=total_loss.item())

        # save predictor checkpoint
        torch.save(predictor.state_dict(), os.path.join(config["save_dir"], f"predictor_epoch{epoch+1}.pt"))
    print("Predictor training finished.")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python defense_training.py [train_universal|train_predictor]")
        sys.exit(0)
    mode = sys.argv[1]
    # attempt to create ASV embedder if SpeechBrain installed
    asv = None
    if SB_AVAILABLE:
        try:
            asv = ASVEmbedder(device=config["device"])
            print("ASV embedder loaded.")
        except Exception as e:
            print("Failed to init ASV embedder:", e)
            asv = None

    if mode == "train_universal":
        train_universal_delta(config["train_manifest"], asv_embedder=asv)
    elif mode == "train_predictor":
        train_predictor(config["train_manifest"], asv_embedder=asv)
    else:
        print("Unknown command:", mode)
