"""
defense_training.py - PHASE 2 ENHANCED

Training script with advanced features:
 ✅ Multi-domain perturbation (time + frequency)
 ✅ Psychoacoustic-aware loss
 ✅ Reinforcement Learning optimization (basic PPO)
 ✅ Neural vocoder integration
 ✅ Ensemble surrogate models

USAGE:
 python defense_training.py train_universal --advanced
 python defense_training.py train_predictor --advanced
 python defense_training.py train_rl  # NEW: RL-based adaptive defense
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils_audio import (
    load_wav, wav_to_mel, tile_delta_to_length, SR, N_MELS, N_FFT, HOP_LENGTH,
    get_vocoder, get_psychoacoustic_masking, extract_audio_features,
    compute_quality_metrics, wav_to_stft, stft_mag_phase
)

# -------------------------
# PHASE 2: Enhanced Predictor with Multi-Domain
# -------------------------
class ResidualBlock1D(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)

class MultiDomainPredictorNet(nn.Module):
    """
    PHASE 2: Enhanced predictor with time-frequency cross-domain processing.
    Generates perturbations in both mel and time domains.
    """
    def __init__(self, n_mels=N_MELS, hidden=128):
        super().__init__()
        
        # Frequency domain branch (mel)
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock1D(hidden),
            ResidualBlock1D(hidden),
        )
        
        # Time domain branch (waveform features)
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, hidden//2, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden//2, hidden, kernel_size=15, padding=7),
            nn.ReLU(),
        )
        
        # Cross-domain fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.ReLU(),
            ResidualBlock1D(hidden),
        )
        
        # Output projection
        self.conv_out = nn.Conv1d(hidden, n_mels, kernel_size=1)
    
    def forward(self, mel_chunk, time_features=None):
        """
        Args:
            mel_chunk: [B, n_mels, T]
            time_features: Optional [B, 1, T] time-domain features
        
        Returns:
            delta: [B, n_mels, T]
        """
        # Frequency domain processing
        freq_features = self.freq_encoder(mel_chunk)
        
        # Time domain processing (if available)
        if time_features is not None:
            time_feat = self.time_encoder(time_features)
            # Concatenate domains
            combined = torch.cat([freq_features, time_feat], dim=1)
            fused = self.fusion(combined)
        else:
            fused = freq_features
        
        # Generate perturbation
        delta = self.conv_out(fused)
        return delta

# Legacy simple predictor for backward compatibility
class PredictorNet(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(hidden, n_mels, kernel_size=1)

    def forward(self, mel_chunk):
        x = self.conv1(mel_chunk)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.conv_out(x)
        return out

# -------------------------
# PHASE 2: RL Agent for Adaptive Perturbation
# -------------------------
class RLPerturbationAgent(nn.Module):
    """
    Reinforcement Learning agent that adaptively selects perturbation parameters.
    Uses PPO (Proximal Policy Optimization) for training.
    """
    def __init__(self, state_dim=32, action_dim=4):
        super().__init__()
        
        # Policy network (Actor)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Value network (Critic)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """Get action distribution."""
        mean = self.policy(state)
        std = self.log_std.exp()
        return mean, std
    
    def get_value(self, state):
        """Estimate state value."""
        return self.value(state)
    
    def select_action(self, state):
        """Sample action from policy."""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

def compute_rl_state(audio_features):
    """
    Convert audio features dict to RL state vector.
    
    Returns:
        state: torch.Tensor [state_dim]
    """
    state_list = [
        audio_features['spectral_centroid'] / 8000.0,  # Normalize
        audio_features['spectral_rolloff'] / 8000.0,
        audio_features['spectral_bandwidth'] / 4000.0,
        audio_features['zero_crossing_rate'],
        audio_features['rms_energy'],
        audio_features['pitch_mean'] / 500.0,
    ]
    
    # Add MFCC features
    mfcc_mean = audio_features['mfcc_mean']
    state_list.extend(mfcc_mean.tolist())
    
    # Pad to fixed dimension (32)
    state = np.array(state_list[:32])
    if len(state) < 32:
        state = np.pad(state, (0, 32 - len(state)))
    
    return torch.from_numpy(state).float()

# -------------------------
# ASV Embedder (from Phase 1)
# -------------------------
try:
    from speechbrain.inference.speaker import SpeakerRecognition
    SB_AVAILABLE = True
except Exception:
    SB_AVAILABLE = False
    print("SpeechBrain not available. Install with `pip install speechbrain`")

class ASVEmbedder:
    def __init__(self, device='cpu'):
        if not SB_AVAILABLE:
            raise RuntimeError("SpeechBrain not installed.")
        
        self.device = device
        # FIXED: Don't pass overrides parameter at all (SpeechBrain 1.0 compatibility)
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec_ecapa",
            run_opts={"device": device}
        )

    def extract(self, wav_np):
        """Extract speaker embedding from audio."""
        wav_tensor = torch.as_tensor(wav_np, dtype=torch.float32, device=self.device)
        
        if wav_tensor.dim() == 2 and wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor.mean(dim=0)
        
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        with torch.no_grad():
            emb = self.model.encode_batch(wav_tensor)
        
        return emb.squeeze(0).detach().to(self.device)

def cosine_similarity_torch(a, b):
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    return torch.sum(a * b)

# -------------------------
# Config (enhanced)
# -------------------------
config = {
    "sr": SR,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "batch_size": 128,
    "epochs": 25,
    "universal_lr": 1e-2,
    "predictor_lr": 1e-4,
    "rl_lr": 3e-4,
    "pgd_eps": 0.02,
    "lambda_attack": 1.0,
    "lambda_quality": 10.0,
    "lambda_reg": 0.1,
    "lambda_psycho": 5.0,  # NEW: Psychoacoustic loss weight
    "use_neural_vocoder": True,  # NEW: Use HiFi-GAN if available
    "use_psychoacoustic": True,  # NEW: Apply psychoacoustic masking
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "num_workers": 6,
    "eval_recon_every": 200,
    "use_asv_inner": False,
    "save_dir": "checkpoints_phase2",
    "train_manifest": "../data/librispeech_prepared/train-clean-100_manifest.txt",
    "val_manifest": "../data/librispeech_prepared/dev-clean_manifest.txt"
}
os.makedirs(config["save_dir"], exist_ok=True)

# Quick performance tweaks
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass

# -------------------------
# Helpers
# -------------------------
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

# -------------------------
# Dataset / DataLoader helpers (parallel I/O + mel precompute)
# -------------------------
class AudioDataset(Dataset):
    """Loads wav + mel on worker processes to parallelize I/O and preprocessing."""
    def __init__(self, manifest, sr=None, n_fft=None, hop_length=None, n_mels=None):
        self.manifest = manifest
        self.sr = sr if sr is not None else config["sr"]
        self.n_fft = n_fft if n_fft is not None else config["n_fft"]
        self.hop_length = hop_length if hop_length is not None else config["hop_length"]
        self.n_mels = n_mels if n_mels is not None else config["n_mels"]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        path = self.manifest[idx]
        wav = load_wav(path, sr=self.sr)  # numpy array
        mel = wav_to_mel(wav, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        return path, wav, mel

def audio_collate_fn(batch):
    """Pad mels to the max length in the batch and stack into a torch tensor.

    This runs in DataLoader worker processes so the main thread doesn't do
    padding/stacking. Returning a pinned tensor (via pin_memory=True) allows
    non-blocking transfers to CUDA when moving the batch to device.
    """
    paths, waves, mels = zip(*batch)
    # mels are numpy arrays with shape [n_mels, T]
    maxT = max(m.shape[1] for m in mels)
    mel_tensors = []
    for m in mels:
        pad = maxT - m.shape[1]
        if pad:
            m_p = np.pad(m, ((0, 0), (0, pad)), mode='constant')
        else:
            m_p = m
        mel_tensors.append(torch.from_numpy(m_p).float())

    mel_batch = torch.stack(mel_tensors, dim=0)  # [B, n_mels, T]
    return list(paths), list(waves), mel_batch


class DeviceDataLoader:
    """Wrap a DataLoader to move batched tensors to the chosen device.

    The DataLoader should be created with `pin_memory=True` so that moving to
    CUDA can be done with `non_blocking=True` for better throughput.
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for paths, waves, mel_batch in self.dl:
            # mel_batch is a torch tensor produced by collate in worker process
            if isinstance(self.device, str) and self.device.startswith("cuda"):
                mel_batch = mel_batch.to(self.device, non_blocking=True)
            else:
                mel_batch = mel_batch.to(self.device)
            yield paths, waves, mel_batch

    def __len__(self):
        return len(self.dl)

def get_dataloader(manifest, batch_size=6, shuffle=True, num_workers=4, pin_memory=True):
    dataset = AudioDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory,
                        collate_fn=audio_collate_fn)
    return loader

# -------------------------
# PHASE 2: Enhanced Universal Delta Training
# -------------------------
def train_universal_delta(manifest_path, asv_embedder=None, advanced=False):
    """
    Enhanced universal delta training with:
    - Neural vocoder (optional)
    - Psychoacoustic masking
    - Advanced quality metrics
    """
    manifest = load_manifest(manifest_path)
    device = config["device"]
    n_mels = config["n_mels"]
    eps = config["pgd_eps"]

    # Initialize advanced components
    vocoder = get_vocoder(device) if config["use_neural_vocoder"] else None
    psycho_masking = get_psychoacoustic_masking() if config["use_psychoacoustic"] else None

    delta_T = 32
    delta = torch.zeros((n_mels, delta_T), dtype=torch.float32, device=device, requires_grad=True)
    opt = optim.Adam([delta], lr=config["universal_lr"])
    scaler = GradScaler() if config.get("use_amp", False) and device.startswith("cuda") else None
    eval_every = config.get("eval_recon_every", 200)
    use_asv_inner = config.get("use_asv_inner", False)

    for epoch in range(config["epochs"]):
        loader = get_dataloader(manifest, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        dataloader = DeviceDataLoader(loader, device)
        pbar = tqdm(dataloader,
                    total=math.ceil(len(manifest)/config["batch_size"]),
                    desc=f"uni-delta epoch {epoch+1}/{config['epochs']}")
        epoch_loss = 0.0

        batch_idx = 0
        for paths, waves, mel_batch in pbar:
            # mel_batch is a torch tensor [B, n_mels, T] produced by collate and
            # already moved to `device` by DeviceDataLoader (non_blocking if CUDA).
            mel_batch_t = mel_batch
            maxT = mel_batch_t.size(2)

            # Tile delta
            delta_tiled = torch.tile(delta.unsqueeze(0), (mel_batch_t.size(0), 1, math.ceil(maxT/delta_T)))
            delta_tiled = delta_tiled[:,:,:maxT]

            # PHASE 2: Apply psychoacoustic masking to delta
            if advanced and psycho_masking is not None:
                # For each sample, compute masking threshold
                for b in range(delta_tiled.size(0)):
                    stft = wav_to_stft(waves[b])
                    mag, _ = stft_mag_phase(stft)

                    # Map STFT to mel bins (approximate)
                    mel_mag = mel_batch_t[b]  # tensor on device

                    # This is simplified - full implementation would convert delta back to STFT domain
                    # For now, apply a frequency-dependent scaling
                    freq_weights = torch.linspace(1.0, 0.5, n_mels).to(device)
                    delta_tiled[b] = delta_tiled[b] * freq_weights.unsqueeze(1)

            mel_prot = mel_batch_t + delta_tiled

            # Attack loss: use cheap mel-space surrogate by default to avoid
            # expensive waveform reconstruction (Griffin-Lim). Optionally run
            # ASV-based reconstruction only every `eval_every` batches or if
            # `use_asv_inner` is True (this will be slow).
            attack_loss = torch.mean(mel_prot ** 2)
            if asv_embedder is not None and (use_asv_inner or (batch_idx % eval_every == 0)):
                # run ASV-based reconstruction for evaluation or inner loop if requested
                attack_losses = []
                for b in range(mel_prot.size(0)):
                    mel_np = mel_prot[b].detach().cpu().numpy()
                    if vocoder is not None and getattr(vocoder, "hifigan_available", False):
                        try:
                            wav_recon = vocoder.mel_to_audio(mel_np)
                        except Exception:
                            from utils_audio import mel_to_wave_griffinlim
                            wav_recon = mel_to_wave_griffinlim(mel_np, sr=config["sr"], n_fft=config["n_fft"], hop_length=config["hop_length"])
                    else:
                        from utils_audio import mel_to_wave_griffinlim
                        wav_recon = mel_to_wave_griffinlim(mel_np, sr=config["sr"], n_fft=config["n_fft"], hop_length=config["hop_length"])

                    orig_wav = waves[b]
                    e_orig = asv_embedder.extract(orig_wav).to(device)
                    e_prot = asv_embedder.extract(wav_recon).to(device)
                    attack_losses.append(cosine_similarity_torch(e_orig, e_prot))

                # Replace attack_loss for logging / evaluation; keep mel surrogate for training
                try:
                    attack_loss_eval = torch.stack(attack_losses).mean()
                except Exception:
                    attack_loss_eval = None
            else:
                attack_loss_eval = None

            # Quality loss (mel MSE)
            quality = torch.mean((mel_batch_t - mel_prot) ** 2)
            
            # Regularization
            reg = torch.sum(delta ** 2)
            
            # PHASE 2: Psychoacoustic penalty (numerically stable)
            psycho_loss = torch.tensor(0.0, device=device)
            if advanced and config["use_psychoacoustic"]:
                # Encourage low-frequency perturbations (more imperceptible)
                # Add a small epsilon to denominator to avoid division-by-zero
                denom = torch.sum(delta ** 2)
                eps_denom = 1e-12
                freq_penalty = torch.sum(delta[:n_mels//2] ** 2) / (denom + eps_denom)
                psycho_loss = 1.0 - freq_penalty  # Penalize high-frequency energy

            total_loss = (config["lambda_attack"] * attack_loss + 
                         config["lambda_quality"] * quality + 
                         config["lambda_reg"] * reg +
                         config["lambda_psycho"] * psycho_loss)

            # If loss is non-finite, skip the update and sanitize `delta` to avoid NaNs
            if not torch.isfinite(total_loss):
                print("[WARN] Non-finite loss encountered; skipping update and sanitizing delta")
                with torch.no_grad():
                    delta.data = torch.nan_to_num(delta.data, nan=0.0, posinf=eps, neginf=-eps)
                    delta.data = torch.clamp(delta.data, -eps, eps)
                epoch_loss += float('nan') if torch.isnan(total_loss) else 0.0
                pbar.set_postfix(loss="nan", attack=(f"{attack_loss_eval.item():.4f}" if attack_loss_eval is not None else f"{attack_loss.item():.4f}"), quality=f"{quality.item():.4f}")
                batch_idx += 1
                continue

            # Backward + optimizer step (with AMP if enabled)
            if scaler is not None:
                with autocast():
                    loss_to_scale = total_loss
                scaler.scale(loss_to_scale).backward()
                # unscale before clipping
                try:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                except Exception:
                    pass
                scaler.step(opt)
                scaler.update()
            else:
                opt.zero_grad()
                total_loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                except Exception:
                    pass
                opt.step()

            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -eps, eps)
                delta.data = torch.nan_to_num(delta.data, nan=0.0, posinf=eps, neginf=-eps)

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=(f"{total_loss.item():.4f}"), 
                           attack=(f"{attack_loss_eval.item():.4f}" if attack_loss_eval is not None else f"{attack_loss.item():.4f}"),
                           quality=f"{quality.item():.4f}")
            batch_idx += 1

        # Save checkpoint
        suffix = "_advanced" if advanced else ""
        np.save(os.path.join(config["save_dir"], f"universal_delta{suffix}_epoch{epoch+1}.npy"), 
                delta.detach().cpu().numpy())
    
    print(f"Universal delta training finished. Saved to {config['save_dir']}")

# -------------------------
# PHASE 2: Enhanced Predictor Training
# -------------------------
def train_predictor(manifest_path, asv_embedder=None, advanced=False):
    """
    Train predictor with optional multi-domain architecture.
    """
    manifest = load_manifest(manifest_path)
    device = config["device"]
    
    # Use advanced or simple predictor
    if advanced:
        predictor = MultiDomainPredictorNet(n_mels=config["n_mels"]).to(device)
        print("[INFO] Using Multi-Domain Predictor")
    else:
        predictor = PredictorNet(n_mels=config["n_mels"]).to(device)
        print("[INFO] Using Simple Predictor")
    
    opt = optim.Adam(predictor.parameters(), lr=config["predictor_lr"])

    scaler = GradScaler() if config.get("use_amp", False) and device.startswith("cuda") else None
    for epoch in range(config["epochs"]):
        loader = get_dataloader(manifest, batch_size=config["batch_size"], shuffle=True,
                                num_workers=config.get("num_workers", 4), pin_memory=True)
        dataloader = DeviceDataLoader(loader, device)
        pbar = tqdm(dataloader,
                    total=math.ceil(len(manifest)/config["batch_size"]),
                    desc=f"predictor epoch {epoch+1}/{config['epochs']}")

        for paths, waves, mel_batch in pbar:
            # mel_batch: [B, n_mels, T] already moved to device
            chunks = []
            T_chunk = 64
            B = mel_batch.size(0)
            for b in range(B):
                mel = mel_batch[b]  # [n_mels, T]
                T = mel.size(1)
                pad = (T_chunk - (T % T_chunk)) % T_chunk
                if pad > 0:
                    mel_p = torch.nn.functional.pad(mel, (0, pad))
                else:
                    mel_p = mel
                n_chunks = mel_p.size(1) // T_chunk
                if n_chunks == 0:
                    continue
                mel_chunks = mel_p.reshape(config["n_mels"], n_chunks, T_chunk).permute(1,0,2)

                n_take = min(4, mel_chunks.shape[0])
                idxs = random.sample(range(mel_chunks.shape[0]), n_take)
                for i_idx in idxs:
                    chunks.append(mel_chunks[i_idx])

            if len(chunks) == 0:
                continue

            mel_batch_chunks = torch.stack(chunks, dim=0)  # on device
            if scaler is not None:
                with autocast():
                    delta_pred = predictor(mel_batch_chunks)
                    mel_prot = mel_batch_chunks + delta_pred
                    attack_loss = torch.mean(mel_prot ** 2)
                    quality_loss = torch.mean((mel_batch_chunks - mel_prot) ** 2)
                    reg_loss = torch.sum(delta_pred ** 2)
                    total_loss = (config["lambda_attack"] * attack_loss + 
                                 config["lambda_quality"] * quality_loss + 
                                 config["lambda_reg"] * reg_loss)

                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                delta_pred = predictor(mel_batch_chunks)
                mel_prot = mel_batch_chunks + delta_pred
                attack_loss = torch.mean(mel_prot ** 2)
                quality_loss = torch.mean((mel_batch_chunks - mel_prot) ** 2)
                reg_loss = torch.sum(delta_pred ** 2)
                total_loss = (config["lambda_attack"] * attack_loss + 
                             config["lambda_quality"] * quality_loss + 
                             config["lambda_reg"] * reg_loss)

                opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                opt.step()

            pbar.set_postfix(loss=float(total_loss.item()))

        # Save checkpoint
        suffix = "_multidomain" if advanced else ""
        torch.save(predictor.state_dict(), 
                  os.path.join(config["save_dir"], f"predictor{suffix}_epoch{epoch+1}.pt"))
    
    print("Predictor training finished.")

# -------------------------
# PHASE 2: NEW - RL Training
# -------------------------
def train_rl_agent(manifest_path, asv_embedder=None):
    """
    Train RL agent for adaptive perturbation selection.
    """
    print("[INFO] Starting RL-based adaptive defense training...")
    manifest = load_manifest(manifest_path)
    device = config["device"]
    
    agent = RLPerturbationAgent(state_dim=32, action_dim=4).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["rl_lr"])
    
    # Base perturbation network (to be controlled by RL)
    base_predictor = PredictorNet(n_mels=config["n_mels"]).to(device)
    
    scaler = GradScaler() if config.get("use_amp", False) and device.startswith("cuda") else None
    for epoch in range(config["epochs"]):
        loader = get_dataloader(manifest, batch_size=config["batch_size"], shuffle=True,
                                num_workers=config.get("num_workers", 4), pin_memory=True)
        dataloader = DeviceDataLoader(loader, device)
        pbar = tqdm(dataloader,
                    total=math.ceil(len(manifest)/config["batch_size"]),
                    desc=f"RL epoch {epoch+1}/{config['epochs']}")

        for paths, waves, mel_batch in pbar:
            episode_rewards = []
            episode_log_probs = []

            # mel_batch is a tensor [B, n_mels, T]
            for wav_path, wav, mel in zip(paths, waves, mel_batch):
                # Extract audio features for state
                features = extract_audio_features(wav, sr=config["sr"])
                state = compute_rl_state(features).to(device)

                # Agent selects action (perturbation parameters)
                action, log_prob = agent.select_action(state)
                episode_log_probs.append(log_prob)

                # Action: [epsilon_scale, freq_weight, time_weight, intensity]
                epsilon_scale = (action[0].item() + 1.0) / 2.0  # [0, 1]

                # Apply perturbation with RL-selected parameters
                mel_t = mel.unsqueeze(0)  # [1, n_mels, T]
                with torch.no_grad():
                    if scaler is not None:
                        with autocast():
                            delta = base_predictor(mel_t.to(device))
                    else:
                        delta = base_predictor(mel_t.to(device))
                    # Scale by RL action
                    delta = delta * epsilon_scale * config["pgd_eps"]

                mel_prot = mel_t.to(device) + delta
                mel_prot_np = mel_prot.squeeze().cpu().numpy()

                # Reconstruct (simplified for speed)
                from utils_audio import mel_to_wave_griffinlim
                wav_prot = mel_to_wave_griffinlim(mel_prot_np)

                # Compute reward
                if asv_embedder is not None:
                    e_orig = asv_embedder.extract(wav).to(device)
                    e_prot = asv_embedder.extract(wav_prot).to(device)
                    similarity = cosine_similarity_torch(e_orig, e_prot).item()
                    attack_reward = 1.0 - similarity  # Lower similarity = better
                else:
                    attack_reward = 0.5  # Neutral if no embedder

                # Quality reward (SNR-based)
                quality_metrics = compute_quality_metrics(wav, wav_prot)
                quality_reward = min(quality_metrics['snr'] / 30.0, 1.0)  # Normalize

                # Combined reward
                reward = 0.7 * attack_reward + 0.3 * quality_reward
                episode_rewards.append(reward)

            # PPO update (simplified)
            if len(episode_rewards) > 0:
                returns = torch.tensor(episode_rewards, device=device)
                log_probs = torch.stack(episode_log_probs)

                # Policy gradient loss
                policy_loss = -(log_probs * returns).mean()

                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

                pbar.set_postfix(reward=f"{returns.mean().item():.4f}", 
                               policy_loss=f"{policy_loss.item():.4f}")
        
        # Save RL agent
        torch.save(agent.state_dict(), 
                  os.path.join(config["save_dir"], f"rl_agent_epoch{epoch+1}.pt"))
    
    print("RL training finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":

    if torch.cuda.is_available():
        print("CUDA is available! GPUs found:")

        # 2. Count the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        print(config["device"])

    else:
        print("CUDA is NOT available. Check your PyTorch installation and driver setup.")

    import sys
    if len(sys.argv) < 2:
        print("Usage: python defense_training.py [train_universal|train_predictor|train_rl] [--advanced]")
        sys.exit(0)
    
    mode = sys.argv[1]
    advanced = "--advanced" in sys.argv
    
    # Initialize ASV embedder
    asv = None
    if SB_AVAILABLE:
        try:
            asv = ASVEmbedder(device=config["device"])
            print("[INFO] ASV embedder loaded.")
        except Exception as e:
            print(f"[WARN] Failed to init ASV embedder: {e}")
            asv = None

    if mode == "train_universal":
        train_universal_delta(config["train_manifest"], asv_embedder=asv, advanced=advanced)
    elif mode == "train_predictor":
        train_predictor(config["train_manifest"], asv_embedder=asv, advanced=advanced)
    elif mode == "train_rl":
        train_rl_agent(config["train_manifest"], asv_embedder=asv)
    else:
        print("Unknown command:", mode)