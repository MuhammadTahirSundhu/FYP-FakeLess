#!/usr/bin/env python3
"""
asv_defense_pipeline_full.py

Full three-stage pipeline (single-file):
 - create_embeddings: run real ECAPA (SpeechBrain) once over manifest and save per-file embeddings
 - train_surrogate: train a small differentiable mel->embedding surrogate
 - train_universal: train a universal mel-domain delta using surrogate gradients
 - evaluate_transfer: optional: apply delta -> HiFi-GAN -> evaluate real ECAPA transfer

Usage examples:
  python asv_defense_pipeline_full.py create_embeddings --manifest data/train_manifest.txt --out_dir embs/
  python asv_defense_pipeline_full.py train_surrogate --emb_dir embs/ --out surrogate.pt --epochs 6
  python asv_defense_pipeline_full.py train_universal --emb_dir embs/ --surrogate surrogate.pt --out_dir deltas/ --epochs 6
  python asv_defense_pipeline_full.py evaluate_transfer --delta deltas/univ_epoch6.npy --example path/to/audio.wav

Notes:
 - This file uses librosa for mel extraction (consistent with your repo).
 - SpeechBrain and HiFi-GAN are optional. If SpeechBrain is missing, you can still train surrogate if you have saved embeddings.
 - Designed to be memory/disk friendly: streaming, small batches, minimal checkpoints.
"""
import os
import math
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import soundfile as sf
# ------------------------------------------------------------------
# Config (match your utils_audio)
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
# ------------------------------------------------------------------

# --------------- audio <-> mel helpers (numpy + torch friendly) -------------
def load_wav_np(path, sr=SR):
    wav, r = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)  # 1D np array

def wav_np_to_mel_np(wav_np, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=wav_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0)
    logS = np.log(np.clip(S, 1e-9, None))
    return logS  # shape [n_mels, T]

def mel_np_to_torch(mel_np):
    return torch.from_numpy(mel_np).float()  # [n_mels, T]

# ---------------- SpeechBrain & HiFi-GAN availability --------------------

try:
    from speechbrain.pretrained import SpeakerRecognition
    SB_AVAILABLE = True
except Exception:
    SB_AVAILABLE = False

# Try to load HiFi-GAN via SpeechBrain.Pretrained if available

HI_FIN_GAN = None
try:
    from speechbrain.pretrained import Pretrained
    MODEL_ID = "speechbrain/tts-hifigan-libritts-16kHz"
    try:
        hifigan_system = Pretrained.from_hparams(source=MODEL_ID,
                                                 savedir=f"pretrained_models/{MODEL_ID.split('/')[-1]}",
                                                 run_opts={"device": "cpu"})
        HI_FIN_GAN = hifigan_system.mods.generator
        HI_FIN_GAN.eval()
    except Exception:
        HI_FIN_GAN = None
except Exception:
    HI_FIN_GAN = None
def create_embeddings(manifest_path, out_dir,
                                device="cpu",
                                batch_size=32,
                                max_duration=3.0,
                                sample_rate=SR,
                                use_torchaudio=True,
                                trim_silence=False):
    """
    Optimized embedding extraction.

    Args:
      manifest_path (str): path to newline manifest of audio file paths
      out_dir (str): directory to save per-file .pt embeddings (same format as before)
      device (str): 'cpu' or 'cuda'
      batch_size (int): number of wavs to batch for encode_batch
      max_duration (float): crop/truncate audio to this many seconds (default 3.0s)
      use_torchaudio (bool): prefer torchaudio.load if available (faster)
      trim_silence (bool): run librosa.effects.trim before cropping (optional)
    """
    # Check SpeechBrain
    if not SB_AVAILABLE:
        raise RuntimeError("SpeechBrain SpeakerRecognition not available. Install speechbrain to run create_embeddings_optimized.")

    os.makedirs(out_dir, exist_ok=True)

    # Load model once
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec_ecapa",
        run_opts={"device": device}
    )
    model.eval()

    # read manifest
    with open(manifest_path, "r") as f:
        files = [l.strip() for l in f if l.strip()]

    # helper loaders
    have_torchaudio = False
    if use_torchaudio:
        try:
            import torchaudio
            have_torchaudio = True
        except Exception:
            have_torchaudio = False

    def load_and_preprocess(path):
        # returns numpy 1D float32 waveform already resampled to sample_rate
        try:
            if have_torchaudio:
                wav_t, sr = torchaudio.load(path)  # wav_t: [channels, T]
                if wav_t.size(0) > 1:
                    wav_t = wav_t.mean(dim=0, keepdim=True)
                wav = wav_t.squeeze(0).cpu().numpy()
                if sr != sample_rate:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
            else:
                wav, sr = librosa.load(path, sr=sample_rate, mono=True)
            # optional silence trim
            if trim_silence:
                wav, _ = librosa.effects.trim(wav, top_db=20)
            # crop/truncate to max_duration
            max_len = int(sample_rate * max_duration)
            if wav.shape[0] > max_len:
                wav = wav[:max_len]
            return wav.astype(np.float32)
        except Exception as e:
            # Return None on error so main loop can skip
            print(f"[WARN] failed to load {path}: {e}")
            return None

    # Batch loop: gather waveforms and save embeddings per-file
    idx = 0
    total = len(files)
    pbar = tqdm(range(0, total, batch_size), desc="Embedding batches", unit="batch")
    for start in pbar:
        batch_paths = []
        batch_wavs = []
        # fill small batch
        for p in files[start:start+batch_size]:
            base = os.path.splitext(os.path.basename(p))[0]
            out_file = os.path.join(out_dir, base + ".pt")
            if os.path.exists(out_file):
                # already processed -> skip (resume support)
                continue
            wav = load_and_preprocess(p)
            if wav is None:
                continue
            batch_paths.append((p, out_file))
            # convert to torch and shape [1, T]
            wt = torch.from_numpy(wav).float().unsqueeze(0)
            batch_wavs.append(wt)

        if len(batch_wavs) == 0:
            continue

        # pad batch to same length (collate)
        maxT = max(w.shape[1] for w in batch_wavs)
        batched = []
        for w in batch_wavs:
            if w.shape[1] < maxT:
                pad = maxT - w.shape[1]
                w = F.pad(w, (0, pad))
            batched.append(w)
        batch_tensor = torch.cat(batched, dim=0).to(device)  # [B, T]

        # encode_batch expects input shape compatible with SpeakerRecognition
        # The SB wrapper accepts [B, T] or [B, 1, T] depending — commonly [B, T] works.
        with torch.no_grad():
            try:
                emb_batch = model.encode_batch(batch_tensor).cpu()  # [B, emb_dim]
            except Exception:
                # fallback: try unsqueezing channel dim
                emb_batch = model.encode_batch(batch_tensor.unsqueeze(1)).cpu()

        # compute mel for saving (use librosa mel; consistent with pipeline)
        for (path, save_path), emb_t, w_t in zip(batch_paths, emb_batch, batched):
            # recover numpy wave (trimmed) from w_t (on cpu)
            wav_np = w_t.squeeze(0).cpu().numpy()
            mel_np = wav_np_to_mel_np(wav_np, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            # save same format as before
            torch.save({"mel": mel_np, "embedding": emb_t.numpy(), "path": path}, save_path)

        # free memory
        del batch_tensor, emb_batch, batched
        torch.cuda.empty_cache() if device != "cpu" else None

    print("Embedding extraction finished. Saved to:", out_dir)


N_MELS = 80       # Adjust if your mel-spectrograms have different n_mels
BATCH_SIZE = 8
EPOCHS = 6
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------- Stage 2: Surrogate training ------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, emb_dir):
        self.files = sorted([str(p) for p in Path(emb_dir).glob("*.pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        mel = torch.from_numpy(data["mel"]).float()       # [n_mels, T]
        emb = torch.from_numpy(data["embedding"]).float()
        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # [192]
        emb = F.normalize(emb, dim=-1)                    # normalize embeddings
        return mel, emb

def collate_mels(batch):
    mels, embs = zip(*batch)
    maxT = max(m.shape[1] for m in mels)
    padded = []
    for m in mels:
        if m.shape[1] < maxT:
            m = F.pad(m, (0, maxT - m.shape[1]))
        padded.append(m)
    mel_batch = torch.stack(padded, dim=0)  # [B, n_mels, T]
    emb_batch = torch.stack(embs, dim=0)    # [B, emb_dim]
    return mel_batch, emb_batch

# ---------------- Collate ----------------
class SurrogateNet(nn.Module):
    def __init__(self, n_mels=N_MELS, emb_dim=192, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hidden, emb_dim)

    def forward(self, mel):
        #print("mel.shape", mel.shape)
        x = self.conv(mel)
        #print("after conv:", x.shape)
        x = x.squeeze(-1)
        #print("after squeeze:", x.shape)
        out = self.fc(x)
        #print("pred.shape", out.shape)
        out = F.normalize(out, p=2, dim=-1)
        return out

# ---------------- Surrogate Training ----------------
def train_surrogate(emb_dir, out_path, epochs=20, batch_size=8, lr=1e-2, device="cpu"):
    ds = EmbeddingDataset(emb_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mels)

    # Infer embedding dim from first file
    sample = torch.load(ds.files[0])
    emb_dim = sample["embedding"].shape[1]
    model = SurrogateNet(n_mels=sample["mel"].shape[0], emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        running_loss = 0.0
        iters = 0
        for mel, emb in loader:
            mel = mel.to(device)
            emb = emb.to(device)
            #print("mel.shape", mel.shape, "emb.shape", emb.shape)
            pred = model(mel)
            #print(pred.shape)
            # MSE on normalized embeddings
            loss = 1 - F.cosine_similarity(pred, emb).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            iters += 1

        avg_loss = running_loss / iters if iters > 0 else 0.0
        print(f"Surrogate epoch {ep+1}/{epochs} avg_loss={avg_loss:.6f}")
        torch.save(model.state_dict(), out_path + f".epoch{ep+1}.pt")  # intermediate checkpoint

    torch.save(model.state_dict(), out_path + ".final.pt")
    print("Surrogate training finished. Saved:", out_path + ".final.pt")
    return model
# ---------------- Stage 3: Universal delta training ------------------------
# --- Robust UniversalDelta + training with diagnostics ---
class UniversalDelta(nn.Module):
    def __init__(self, n_mels=N_MELS, delta_T=32, init_scale=1e-3, eps=0.02):
        """
        Small learnable universal delta parameter, expanded/tiled to match utterance length.
        - n_mels: # mel channels (80)
        - delta_T: temporal length of the base pattern (will be tiled)
        - init_scale: small random init to break symmetry
        - eps: clip magnitude in mel-domain (absolute)
        """
        super().__init__()
        self.eps = eps
        init = (torch.randn(n_mels, delta_T) * init_scale).float()
        # store parameter as shape [n_mels, delta_T] for easy inspection / saving
        self.delta = nn.Parameter(init)

    def forward(self, mel):
        """
        mel: [B, n_mels, T]
        returns: prot_mel [B, n_mels, T], tiled_delta [B, n_mels, T]
        """
        B, n_mels, T = mel.shape
        delta_T = self.delta.shape[1]
        reps = math.ceil(T / delta_T)
        # expand to shape [B, n_mels, reps*delta_T] then slice
        tiled = self.delta.unsqueeze(0).repeat(B, 1, reps)[:, :, :T]
        prot = mel + tiled
        return prot, tiled

    @torch.no_grad()
    def project(self):
        """Keep parameter inside [-eps, eps] (in-place)."""
        self.delta.data.clamp_(-self.eps, self.eps)

    def l2_norm(self):
        return torch.norm(self.delta.detach()).item()
    
# ---------- train_universal with additional differentiable losses ----------
def train_universal(
    emb_dir, surrogate_path, out_dir,
    epochs=6, batch_size=8, lr=1e-2, device="cpu",
    eps=0.02, margin=0.05,
    lambda_attack=1.0, lambda_quality=1.0, lambda_reg=1e-4,
    lambda_centroid=1e-2, lambda_band=1e-2, lambda_smooth=1e-3,
    delta_T=32,
    debug_batches=2
):
    """
    Similar signature to your prior train_universal. Added optional differentiable
    spectral/band/temporal losses while preserving surrogate-based attack flow.
    """

    # --- infer dims robustly ---
    sample = torch.load(sorted(Path(emb_dir).glob("*.pt"))[0])
    emb_np = np.asarray(sample["embedding"])
    if emb_np.ndim == 1:
        emb_dim = emb_np.shape[0]
    else:
        emb_dim = emb_np.shape[-1]

    # --- load surrogate (frozen) ---
    surrogate = SurrogateNet(n_mels=sample["mel"].shape[0], emb_dim=emb_dim).to(device)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=device))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    # --- dataset + loader (use your existing EmbeddingDataset & collate_mels) ---
    ds = EmbeddingDataset(emb_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mels, drop_last=False)

    os.makedirs(out_dir, exist_ok=True)

    # --- delta instantiation ---
    delta = UniversalDelta(n_mels=sample["mel"].shape[0], delta_T=delta_T, init_scale=1e-3, eps=eps).to(device)
    opt = torch.optim.Adam([delta.delta], lr=lr)

    # Precompute mel-bin frequency indices for centroid (differentiable)
    # We treat mel-bin index as proxy freq location; normalized to [0,1]
    n_mels = sample["mel"].shape[0]
    mel_bin_coords = torch.linspace(0.0, 1.0, steps=n_mels, device=device).view(1, n_mels, 1)  # [1, n_mels, 1]

    # small helper losses (all torch ops -> differentiable)
    def spectral_centroid_loss(mel_orig, mel_prot):
        # mel_*: [B, n_mels, T] (log-mel or mel-magnitude; we operate on them directly)
        # centroid = sum(freq * mag) / sum(mag)
        # convert to magnitude-space: if mel is log-mel, apply exp; but your pipeline uses log-mel,
        # so undo log for magnitude-like behavior. Use stable exp with clamp.
        mag_orig = torch.exp(torch.clamp(mel_orig, -20.0, 20.0))
        mag_prot = torch.exp(torch.clamp(mel_prot, -20.0, 20.0))
        num_orig = (mel_bin_coords * mag_orig).sum(dim=1)   # [B, T]
        den_orig = (mag_orig).sum(dim=1) + 1e-9            # [B, T]
        cent_orig = num_orig / den_orig                    # [B, T]

        num_prot = (mel_bin_coords * mag_prot).sum(dim=1)
        den_prot = (mag_prot).sum(dim=1) + 1e-9
        cent_prot = num_prot / den_prot

        # MSE across frames then average
        return F.mse_loss(cent_orig, cent_prot)

    def band_energy_loss(mel_orig, mel_prot):
        # energy per mel band over time: sum_t mag
        mag_orig = torch.exp(torch.clamp(mel_orig, -20.0, 20.0))
        mag_prot = torch.exp(torch.clamp(mel_prot, -20.0, 20.0))
        e_orig = mag_orig.sum(dim=2)  # [B, n_mels]
        e_prot = mag_prot.sum(dim=2)
        # normalize per-sample to prevent scale sensitivity
        e_orig = e_orig / (e_orig.sum(dim=1, keepdim=True) + 1e-9)
        e_prot = e_prot / (e_prot.sum(dim=1, keepdim=True) + 1e-9)
        return F.mse_loss(e_orig, e_prot)

    def temporal_smoothness_loss(tiled_delta):
        # penalize large frame-to-frame differences in delta (encourages smooth delta)
        # tiled_delta: [B, n_mels, T]
        diff = tiled_delta[:, :, 1:] - tiled_delta[:, :, :-1]  # [B, n_mels, T-1]
        return (diff ** 2).mean()

    # training loop
    for ep in range(epochs):
        running = 0.0
        running_attack = 0.0
        running_quality = 0.0
        running_centriod = 0.0
        running_band = 0.0
        running_smooth = 0.0
        running_reg = 0.0
        it = 0
        for batch_idx, (mel, emb) in enumerate(loader):
            mel = mel.to(device)           # [B, n_mels, T]
            emb = emb.to(device)           # [B, emb_dim] normalized in dataset

            # forward: add delta
            prot, tiled = delta(mel)       # prot [B, n_mels, T]

            # surrogate prediction (frozen) -> embedding predictions
            pred = surrogate(prot)         # [B, emb_dim]

            # --- attack loss (hinged cosine to push below margin) ---
            # cosine similarity in [-1,1]; we want cosine < margin
            cos_per_sample = F.cosine_similarity(pred, emb, dim=1)  # [B]
            # hinge: max(0, cos - margin) ; minimize hinge so cos <= margin
            attack_loss = torch.relu(cos_per_sample - margin).mean() * lambda_attack

            # --- quality loss (mel-domain MSE) ---
            quality_loss = torch.mean((mel - prot) ** 2) * lambda_quality

            # --- spectral / band / temporal losses ---
            centroid_loss = spectral_centroid_loss(mel, prot) * lambda_centroid
            band_loss = band_energy_loss(mel, prot) * lambda_band
            smooth_loss = temporal_smoothness_loss(tiled) * lambda_smooth

            # --- reg on delta param ---
            reg_loss = torch.sum(delta.delta ** 2) * lambda_reg

            total = attack_loss + quality_loss + centroid_loss + band_loss + smooth_loss + reg_loss

            opt.zero_grad()
            total.backward()

            # debug: check gradient flow for delta
            if batch_idx < debug_batches:
                if delta.delta.grad is None:
                    print("[WARN] delta.delta.grad is None -- gradients NOT flowing to delta!")
                else:
                    gnorm = delta.delta.grad.abs().mean().item()
                    print(f"[DEBUG] batch {batch_idx} shapes: mel {tuple(mel.shape)}, prot {tuple(prot.shape)}, pred {tuple(pred.shape)}, emb {tuple(emb.shape)}")
                    print(f"[DEBUG] delta.grad mean abs = {gnorm:.6e}, delta.norm = {delta.l2_norm():.6f}")
                    print(f"[DEBUG] losses attack={attack_loss.item():.6f} qual={quality_loss.item():.6e} centroid={centroid_loss.item():.6e} band={band_loss.item():.6e} smooth={smooth_loss.item():.6e} reg={reg_loss.item():.6e} total={total.item():.6f}")

            opt.step()

            # clamp to eps ball (ℓ∞)
            with torch.no_grad():
                delta.delta.data.clamp_(-eps*10, eps*10)

            running += total.item()
            running_attack += attack_loss.item()
            running_quality += quality_loss.item()
            running_centriod += centroid_loss.item()
            running_band += band_loss.item()
            running_smooth += smooth_loss.item()
            running_reg += reg_loss.item()

            it += 1
            #eps += 0.005

        avg = (running / it) if it > 0 else 0.0
        avg_attack = (running_attack / it)
        avg_quality = (running_quality/it)
        avg_centroid = (running_centriod/it)
        avg_band = (running_band/it)
        avg_smooth = (running_smooth/it)
        avg_reg = (running_reg/it)
        print(f"Delta epoch {ep+1}/{epochs} \n avg_loss={avg:.6f} avg_attack_loss= {avg_attack:.6f} avg_quality_loss = {avg_quality:.6f} avg_centroid_loss = {avg_centroid:.6f} avg_band_loss = {avg_band:.6f} avg_smooth_loss = {avg_smooth:.6f} avg_reg_loss = {avg_reg} ")


        # snapshot save
        np.save(os.path.join(out_dir, f"universal_delta_epoch{ep+1}.npy"), delta.delta.detach().cpu().numpy())
        eps += 0.001
    # final save
    np.save(os.path.join(out_dir, "universal_delta_final.npy"), delta.delta.detach().cpu().numpy())
    print("Universal delta training finished. Saved to:", out_dir)

    return delta
    #return delta

# ---------------- Evaluate transfer on real ECAPA (optional) ----------------
def mel_np_apply_delta_and_synthesize(mel_np, delta_np, hifigan=None):
    # mel_np: [n_mels, T]
    T = mel_np.shape[1]
    dT = delta_np.shape[1]
    reps = math.ceil(T / dT)
    tiled = np.tile(delta_np, (1, reps))[:, :T]
    prot = mel_np + tiled
    # if HiFi-GAN available, synthesize a waveform tensor
    if hifigan is not None:
        mel_t = torch.from_numpy(prot).float().unsqueeze(0)  # [1, n_mels, T]
        with torch.no_grad():
            wav_t = hifigan(torch.clamp(mel_t, min=-15.0, max=5.0))  # [1, 1, Twav]
        wav_np = wav_t.squeeze().cpu().numpy()
        return wav_np
    else:
        # Fallback: use Griffin-Lim (slow & low-quality) — but we keep it optional
        linear_spec = librosa.feature.inverse.mel_to_stft(np.exp(prot), sr=SR, n_fft=N_FFT)
        wav = librosa.griffinlim(linear_spec, hop_length=HOP_LENGTH, win_length=N_FFT, n_iter=32)
        return wav.astype(np.float32)

def evaluate_transfer(delta_path, examples, ecapa_device="cpu"):
    """
    delta_path: .npy with shape [n_mels, delta_T]
    examples: list of paths to audio files to evaluate
    """
    if not SB_AVAILABLE:
        raise RuntimeError("SpeechBrain not available; cannot evaluate transfer on real ECAPA.")
    delta_np = np.load(delta_path)
    from speechbrain.pretrained import SpeakerRecognition
    ecapa = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="pretrained_models/spkrec_ecapa",
                                           run_opts={"device": ecapa_device})
    for p in examples:
        wav = load_wav_np(p)
        mel = wav_np_to_mel_np(wav)
        # synth protected wav
        wav_prot = mel_np_apply_delta_and_synthesize(mel, delta_np, hifigan=HI_FIN_GAN)
        # ensure shape and torch conversion
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        wavp_t = torch.from_numpy(wav_prot).float().unsqueeze(0)
        with torch.no_grad():
            e_real = ecapa.encode_batch(wav_t).squeeze(0).cpu()
            e_prot = ecapa.encode_batch(wavp_t).squeeze(0).cpu()
        cos = F.cosine_similarity(e_real.unsqueeze(0), e_prot.unsqueeze(0)).item()
        print(f"{p}  cosine(real,prot) = {cos:.4f}")

# ------------------------------ CLI / main --------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("create_embeddings")
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--device", default="cpu")

    p2 = sub.add_parser("train_surrogate")
    p2.add_argument("--emb_dir", required=True)
    p2.add_argument("--save_path", required=True)
    p2.add_argument("--epochs", type=int, default=20)
    p2.add_argument("--batch_size", type=int, default=8)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--device", default="cpu")

    p3 = sub.add_parser("train_universal")
    p3.add_argument("--emb_dir", required=True)
    p3.add_argument("--surrogate", required=True)
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--epochs", type=int, default=20)
    p3.add_argument("--batch_size", type=int, default=8)
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--device", default="cpu")
    p3.add_argument("--eps", type=float, default=0.02)

    p4 = sub.add_parser("evaluate_transfer")
    p4.add_argument("--delta", required=True)  # .npy
    p4.add_argument("--examples", nargs="+", required=True)
    p4.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.cmd == "create_embeddings":
        create_embeddings(args.manifest, args.out_dir, device=args.device)
    elif args.cmd == "train_surrogate":
        train_surrogate(args.emb_dir, args.save_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    elif args.cmd == "train_universal":
        train_universal(args.emb_dir, args.surrogate, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, eps=args.eps)
    elif args.cmd == "evaluate_transfer":
        evaluate_transfer(args.delta, args.examples, ecapa_device=args.device)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
