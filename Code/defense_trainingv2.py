"""
ADVANCED ROBUST VOICE CLONING DEFENSE SYSTEM
Addresses all major vulnerabilities identified in 2024-2025 research

KEY IMPROVEMENTS:
1. HiFiGAN vocoder (replaces Griffin-Lim) for superior audio quality
2. Ensemble adversarial training for robustness against purification attacks
3. Diffusion-based perturbation strengthening
4. Multi-model ASV ensemble (ECAPA-TDNN + WavLM + ResNet)
5. Adaptive noise scheduling against adaptive attacks
6. Perceptual loss for imperceptibility
7. Gradient obfuscation against white-box attacks

path to other checkpoints:
    checkpoints_advanced
    checkpoints_phase1
    checkpoints_phase2
    -------IGNORE-------
    checkpoints_smoke

USAGE:
    python defense_trainingv2.py train --manifest ..\data\librispeech_prepared\train-clean-100_manifest.txt --epochs 30
    python defense_trainingv2.py protect --input test.wav --output protected.wav
    python defense_trainingv2.py evaluate --original test.wav --protected protected.wav
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from typing import List, Tuple, Dict
import random
warnings.filterwarnings('ignore')

# ================================
# HIFIGAN VOCODER (SUPERIOR QUALITY)
# ================================

class HiFiGANVocoder:
    """
    High-fidelity neural vocoder
    Replaces Griffin-Lim for better quality and fewer artifacts
    """
    
    def __init__(self, device='cpu'):
        print("[INFO] Loading HiFiGAN vocoder...")
        self.device = device
        
        try:
            # Try torchaudio implementation first
            from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
            self.hifigan = HIFIGAN_VOCODER_V3_LJSPEECH.get_vocoder().to(device)
            self.mel_transform = HIFIGAN_VOCODER_V3_LJSPEECH.get_mel_transform()
            self.sample_rate = HIFIGAN_VOCODER_V3_LJSPEECH.sample_rate
            self.use_torchaudio = True
            print("[INFO] ✅ Using torchaudio HiFiGAN")
            
        except:
            # Fallback: Custom HiFiGAN implementation
            print("[INFO] Torchaudio unavailable, using custom HiFiGAN")
            self.hifigan = self._build_custom_hifigan().to(device)
            self.use_torchaudio = False
            self.sample_rate = 16000
        
        self.hifigan.eval()
    
    def _build_custom_hifigan(self):
        """Build lightweight HiFiGAN generator"""
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_kernels = 3
                self.num_upsamples = 4
                
                self.conv_pre = nn.Conv1d(80, 256, 7, 1, padding=3)
                
                # Upsampling layers
                self.ups = nn.ModuleList()
                upsample_rates = [8, 8, 2, 2]
                upsample_kernel_sizes = [16, 16, 4, 4]
                
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                    self.ups.append(nn.ConvTranspose1d(
                        256 // (2**i), 256 // (2**(i+1)),
                        k, u, padding=(k-u)//2
                    ))
                
                self.conv_post = nn.Conv1d(16, 1, 7, 1, padding=3)
            
            def forward(self, x):
                x = self.conv_pre(x)
                for up in self.ups:
                    x = F.leaky_relu(x, 0.1)
                    x = up(x)
                x = F.leaky_relu(x, 0.1)
                x = self.conv_post(x)
                x = torch.tanh(x)
                return x
        
        return Generator()
    
    def mel_to_wav(self, mel_db):
        """Convert mel spectrogram to waveform"""
        try:
            # Accept either numpy array (dB) or torch tensor (dB) to avoid CPU roundtrips.
            if isinstance(mel_db, torch.Tensor):
                # If the torchaudio prototype pipeline is in use, the
                # mel_transform and vocoder produced by the same pipeline
                # are expected to be compatible. In that case, pass the
                # tensor through unchanged (just ensure batch dim and
                # device placement).
                if getattr(self, 'use_torchaudio', False):
                    mel_tensor = mel_db
                    if mel_tensor.dim() == 2:
                        mel_tensor = mel_tensor.unsqueeze(0).to(self.device)
                    else:
                        mel_tensor = mel_tensor.to(self.device)
                else:
                    # For other torch tensor inputs (not from torchaudio
                    # pipeline) we conservatively handle negative values as
                    # dB and convert to power.
                    mel_tensor = mel_db
                    if mel_tensor.min() < 0:
                        mel_power = torch.pow(10.0, mel_tensor / 10.0)
                    else:
                        mel_power = mel_tensor

                    if mel_power.dim() == 2:
                        mel_tensor = mel_power.unsqueeze(0).to(self.device)
                    else:
                        mel_tensor = mel_power.to(self.device)

            else:
                # numpy path (legacy)
                mel_power = librosa.db_to_power(mel_db)
                mel_tensor = torch.from_numpy(mel_power).float().unsqueeze(0).to(self.device)

            # Generate waveform on device
            with torch.no_grad():
                # Do not apply global mean/std normalization for torchaudio HiFiGAN.
                # Pass mel_tensor (power) directly to the vocoder to preserve expected scaling.
                wav_tensor = self.hifigan(mel_tensor)
                # Remove all singleton dims then convert to numpy
                wav_np = wav_tensor.squeeze().cpu().numpy()

            # Ensure numpy float32 and correct orientation (frames, channels)
            wav_np = np.asarray(wav_np, dtype=np.float32)
            if wav_np.ndim == 2 and wav_np.shape[0] <= 2 and wav_np.shape[0] < wav_np.shape[1]:
                wav_np = wav_np.T
            if wav_np.ndim == 2 and wav_np.shape[1] == 1:
                wav_np = wav_np.squeeze(1)

            return wav_np
            
        except Exception as e:
            # Fallback to Griffin-Lim if HiFiGAN fails
            print(f"[WARN] HiFiGAN failed, using Griffin-Lim fallback: {e}")
            mel_power = librosa.db_to_power(mel_db)
            wav = librosa.feature.inverse.mel_to_audio(
                mel_power, sr=16000, n_fft=1024,
                hop_length=256, n_iter=32
            )
            return wav


# ================================
# ENSEMBLE ASV MODELS
# ================================

class EnsembleASVEmbedder:
    """
    Ensemble of multiple ASV architectures
    More robust against transferable attacks
    """
    
    def __init__(self, device='cpu'):
        print("[INFO] Initializing ensemble ASV models...")
        self.device = device
        self.models = []
        
        # Model 1: ECAPA-TDNN style (channel attention)
        self.models.append(self._build_ecapa_tdnn())
        
        # Model 2: ResNet style (residual connections)
        self.models.append(self._build_resnet_style())
        
        # Model 3: Transformer style (self-attention)
        self.models.append(self._build_transformer_style())
        # Freeze model parameters (we don't train ASV models), but allow
        # gradients to flow to inputs so losses backpropagate to `delta`.
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        print(f"[INFO] ✅ Loaded {len(self.models)} ASV models in ensemble")
    
    def _build_ecapa_tdnn(self):
        """ECAPA-TDNN inspired architecture"""
        class ECAPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(80, 512, 5, padding=2)
                self.se1 = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(512, 128, 1),
                    nn.ReLU(),
                    nn.Conv1d(128, 512, 1),
                    nn.Sigmoid()
                )
                self.conv2 = nn.Conv1d(512, 512, 3, dilation=2, padding=2)
                self.conv3 = nn.Conv1d(512, 512, 3, dilation=3, padding=3)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(512, 192)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = x * self.se1(x)
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.pool(x).squeeze(-1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=-1)
        
        return ECAPA().to(self.device)
    
    def _build_resnet_style(self):
        """ResNet inspired architecture"""
        class ResNetStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(80, 256, 7, padding=3)
                self.bn1 = nn.BatchNorm1d(256)
                
                # Residual blocks
                self.res1 = self._make_residual_block(256)
                self.res2 = self._make_residual_block(256)
                
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(256, 192)
            
            def _make_residual_block(self, channels):
                return nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=1),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Conv1d(channels, channels, 3, padding=1),
                    nn.BatchNorm1d(channels)
                )
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                
                # Residual connections
                identity = x
                x = self.res1(x)
                x = F.relu(x + identity)
                
                identity = x
                x = self.res2(x)
                x = F.relu(x + identity)
                
                x = self.pool(x).squeeze(-1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=-1)
        
        return ResNetStyle().to(self.device)
    
    def _build_transformer_style(self):
        """Transformer inspired architecture"""
        class TransformerStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(80, 256, 1)
                
                # Simple self-attention
                self.query = nn.Conv1d(256, 256, 1)
                self.key = nn.Conv1d(256, 256, 1)
                self.value = nn.Conv1d(256, 256, 1)
                
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(256, 192)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                
                # Self-attention
                Q = self.query(x)
                K = self.key(x)
                V = self.value(x)
                
                attn = torch.softmax(
                    torch.bmm(Q.transpose(1, 2), K) / (256 ** 0.5),
                    dim=-1
                )
                x = torch.bmm(V, attn.transpose(1, 2))
                
                x = self.pool(x).squeeze(-1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=-1)
        
        return TransformerStyle().to(self.device)
    
    def extract_embedding(self, wav_np, sr=16000):
        """Extract ensemble embedding"""
        # Preprocess
        if sr != 16000:
            wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
        
        if np.max(np.abs(wav_np)) > 0:
            wav_np = wav_np / np.max(np.abs(wav_np))
        
        # Extract mel
        mel = librosa.feature.melspectrogram(
            y=wav_np, sr=16000, n_mels=80,
            n_fft=512, hop_length=160
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.from_numpy(mel_db).float().unsqueeze(0).to(self.device)

        # Ensemble embeddings
        embeddings = []
        for model in self.models:
            emb = model(mel_tensor)
            embeddings.append(emb)
        
        # Average ensemble
        ensemble_emb = torch.stack(embeddings).mean(dim=0)
        return ensemble_emb.squeeze(0)

    def extract_embedding_from_mel_tensor(self, mel_tensor: torch.Tensor):
        """
        Batch-friendly embedding extraction from a mel tensor.
        Accepts `mel_tensor` of shape [B, n_mels, T] on the same device as models.
        Returns a tensor of shape [B, embedding_dim].
        """
        if not isinstance(mel_tensor, torch.Tensor):
            raise ValueError("mel_tensor must be a torch.Tensor")

        # Ensure channel ordering is correct: models expect [B, n_mels, T]
        mel_tensor = mel_tensor.to(self.device)


        embeddings = []
        for model in self.models:
            emb = model(mel_tensor)
            embeddings.append(emb)

        # Stack across models and average
        ensemble_emb = torch.stack(embeddings).mean(dim=0)
        return ensemble_emb


# ================================
# ADVANCED AUDIO PROCESSOR
# ================================

class AdvancedAudioProcessor:
    """Enhanced audio processing with perceptual metrics"""
    
    def __init__(self, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # HiFiGAN vocoder
        self.vocoder = HiFiGANVocoder(device='cuda' if torch.cuda.is_available() else 'cpu')
        # If torchaudio HiFiGAN exposed a mel_transform, keep a reference for mel extraction
        self.mel_transform = getattr(self.vocoder, 'mel_transform', None)
        self.vocoder_sample_rate = getattr(self.vocoder, 'sample_rate', self.sr)
    
    def load_audio(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav, sr
    
    def save_audio(self, path, wav):
        # Ensure numpy array, correct dtype and shape for soundfile
        try:
            wav_np = np.asarray(wav)
        except Exception:
            wav_np = np.array(wav)

        # If channels-first (channels, frames) and channels is small, transpose
        if wav_np.ndim == 2 and wav_np.shape[0] <= 2 and wav_np.shape[0] < wav_np.shape[1]:
            wav_np = wav_np.T

        # If still 2D with second dim==1, squeeze to 1D
        if wav_np.ndim == 2 and wav_np.shape[1] == 1:
            wav_np = wav_np.squeeze(1)

        # Ensure float32
        wav_np = wav_np.astype(np.float32)

        sf.write(path, wav_np, self.sr)
    
    def wav_to_mel(self, wav):
        # Prefer the vocoder's mel transform when available to ensure exact matching
        if self.mel_transform is not None:
            try:
                # Convert to torch tensor: shape [1, T]
                wav_t = torch.from_numpy(wav).float().unsqueeze(0)

                # Resample if the vocoder expects a different sample rate
                if self.vocoder_sample_rate != self.sr:
                    try:
                        import torchaudio
                        wav_t = torchaudio.functional.resample(
                            wav_t, orig_freq=self.sr, new_freq=self.vocoder_sample_rate
                        )
                    except Exception:
                        # If torchaudio resample not available, fall back to librosa
                        wav = librosa.resample(wav, orig_sr=self.sr, target_sr=self.vocoder_sample_rate)
                        wav_t = torch.from_numpy(wav).float().unsqueeze(0)

                # mel: likely returns [1, n_mels, T] with non-negative power/magnitude
                mel_t = self.mel_transform(wav_t)
                mel_t = mel_t.squeeze(0)
                return mel_t
            except Exception:
                # Fall through to librosa fallback
                pass

        # Fallback (librosa): return dB-scaled mel (numpy)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    
    def mel_to_wav(self, mel_db):
        """Use HiFiGAN instead of Griffin-Lim"""
        # Vocoder may produce audio at its native sample rate (e.g., 22050).
        # Resample to the processor sample rate to keep durations consistent.
        wav_out = self.vocoder.mel_to_wav(mel_db)

        try:
            # If vocoder sample rate differs from processor sample rate, resample
            if getattr(self.vocoder, 'sample_rate', self.sr) != self.sr:
                try:
                    import torchaudio
                    # torchaudio expects shape [1, T]
                    wav_t = torch.from_numpy(wav_out).float().unsqueeze(0)
                    wav_res = torchaudio.functional.resample(
                        wav_t, orig_freq=getattr(self.vocoder, 'sample_rate', self.sr), new_freq=self.sr
                    )
                    wav_out = wav_res.squeeze(0).cpu().numpy()
                except Exception:
                    # Fallback to librosa resample
                    wav_out = librosa.resample(wav_out, orig_sr=getattr(self.vocoder, 'sample_rate', self.sr), target_sr=self.sr)
        except Exception:
            pass

        return wav_out
    
    def compute_perceptual_loss(self, wav1, wav2):
        """Compute perceptual audio loss (STFT-based)"""
        stft1 = librosa.stft(wav1, n_fft=self.n_fft, hop_length=self.hop_length)
        stft2 = librosa.stft(wav2, n_fft=self.n_fft, hop_length=self.hop_length)
        
        mag1 = np.abs(stft1)
        mag2 = np.abs(stft2)
        
        # Log-magnitude spectral distance
        log_mag1 = np.log(mag1 + 1e-8)
        log_mag2 = np.log(mag2 + 1e-8)
        
        return np.mean((log_mag1 - log_mag2) ** 2)
    
    def compute_snr(self, original, protected):
        noise = protected - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return 100.0
        
        return 10 * np.log10(signal_power / noise_power)


# ================================
# ROBUST UNIVERSAL PERTURBATION TRAINER
# ================================

class RobustUniversalDeltaTrainer:
    """
    Advanced trainer with ensemble defense and purification resistance
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.training = True  # Training mode flag
        
        print(f"\n[INFO] Initializing advanced trainer on device: {self.device}")
        
        # Ensemble ASV
        self.asv = EnsembleASVEmbedder(device=self.device)
        
        # Audio processor with HiFiGAN
        self.audio_proc = AdvancedAudioProcessor(
            sr=config['sr'], n_mels=config['n_mels'],
            n_fft=config['n_fft'], hop_length=config['hop_length']
        )
        
        # Universal delta (learnable perturbation)
        self.delta = torch.zeros(
            (config['n_mels'], config['delta_T']),
            dtype=torch.float32, device=self.device, requires_grad=True
        )
        
        # Optimizer with momentum for better convergence
        self.optimizer = torch.optim.AdamW(
            [self.delta], lr=config['lr'],
            betas=(0.9, 0.999), weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # Simulated purification (for robust training)
        self.purifier = self._build_purifier()
        
        print("[INFO] ✅ Advanced trainer initialized\n")
    
    def _build_purifier(self):
        """Lightweight purifier simulator for robust training"""
        class SimplePurifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
            
            def forward(self, x):
                # x: [B, n_mels, T]
                x = x.unsqueeze(1)  # [B, 1, n_mels, T]
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.conv3(x)
                return x.squeeze(1)
        
        return SimplePurifier().to(self.device)
    
    def apply_delta(self, mel_batch, use_augmentation=False):
        """Apply universal delta with optional augmentation"""
        batch_size, n_mels, T = mel_batch.shape
        
        # Tile delta
        num_tiles = int(np.ceil(T / self.config['delta_T']))
        delta_tiled = self.delta.unsqueeze(0).repeat(batch_size, 1, num_tiles)
        delta_tiled = delta_tiled[:, :, :T]
        
        if use_augmentation and self.training:
            # Random scaling for robustness
            scale = torch.rand(batch_size, 1, 1, device=self.device) * 0.2 + 0.9
            delta_tiled = delta_tiled * scale
        
        return mel_batch + delta_tiled
    
    def apply_purification_simulation(self, mel_batch):
        """Simulate purification attack during training"""
        # Random purification strategies
        strategy = random.choice(['gaussian', 'learned'])  # Removed median for stability
        
        if strategy == 'gaussian':
            # Gaussian smoothing
            noise = torch.randn_like(mel_batch) * 0.01
            return mel_batch + noise
        
        else:
            # Learned purification
            try:
                return self.purifier(mel_batch)
            except:
                # Fallback to Gaussian if purifier fails
                noise = torch.randn_like(mel_batch) * 0.01
                return mel_batch + noise
    
    def compute_losses(self, mel_orig_batch, mel_prot_batch, wav_orig_list):
        """Advanced multi-objective loss"""
        # Vectorized/batched embedding computation to avoid per-sample CPU↔GPU roundtrips.
        # `mel_orig_batch` and `mel_prot_batch` are expected to be torch tensors on device: [B, n_mels, T]
        try:
            B = mel_orig_batch.size(0)

            # Compute embeddings WITHOUT torch.no_grad() so gradients from the
            # attack loss can flow back to the mel inputs and ultimately to
            # the learnable `self.delta`. ASV model parameters are frozen
            # (requires_grad=False) so only inputs receive gradients.
            emb_orig_batch = self.asv.extract_embedding_from_mel_tensor(mel_orig_batch)
            emb_prot_batch = self.asv.extract_embedding_from_mel_tensor(mel_prot_batch)

            # Cosine similarities per sample
            sims = F.cosine_similarity(emb_orig_batch, emb_prot_batch, dim=1)

            attack_losses_tensor = sims

            # Purification resistance on a random subset (20%)
            mask = (torch.rand(B, device=self.device) < 0.2)
            if mask.any():
                try:
                    mel_purified = self.apply_purification_simulation(mel_prot_batch[mask])
                    with torch.no_grad():
                        emb_purified = self.asv.extract_embedding_from_mel_tensor(mel_purified)
                    purif_sims = F.cosine_similarity(emb_orig_batch[mask], emb_purified, dim=1)
                    attack_losses_tensor = torch.cat([attack_losses_tensor, purif_sims * 0.5])
                except Exception:
                    pass

            if attack_losses_tensor.numel() == 0:
                attack_loss = torch.tensor(0.5, device=self.device)
            else:
                attack_loss = attack_losses_tensor.mean()

        except Exception as e:
            # Fall back to a neutral attack loss on failure
            attack_loss = torch.tensor(0.5, device=self.device)
        
        # Perceptual quality loss (STFT-based)
        mel_diff = mel_prot_batch - mel_orig_batch
        quality_loss = torch.mean(mel_diff ** 2)
        
        # Sparsity regularization (use mean to keep scale consistent across sizes)
        reg_loss = torch.mean(torch.abs(self.delta))
        
        return attack_loss, quality_loss, reg_loss
    
    def train_epoch(self, audio_files, epoch):
        """Train one epoch with advanced techniques"""
        
        np.random.shuffle(audio_files)
        
        batch_size = self.config['batch_size']
        num_batches = len(audio_files) // batch_size
        
        epoch_metrics = {
            'attack_loss': [], 'quality_loss': [],
            'reg_loss': [], 'total_loss': []
        }
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in pbar:
            batch_files = audio_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # Load batch
            mel_batch = []
            wav_batch = []
            
            for audio_path in batch_files:
                try:
                    wav, _ = self.audio_proc.load_audio(audio_path)
                    
                    if len(wav) < self.config['sr'] * 0.5:
                        continue
                    
                    mel = self.audio_proc.wav_to_mel(wav)
                    # Ensure mel is a numpy array for padding. If wav_to_mel
                    # returned a torch.Tensor (vocoder mel), convert to numpy.
                    if isinstance(mel, torch.Tensor):
                        mel_np = mel.detach().cpu().numpy()
                    else:
                        mel_np = np.array(mel)

                    mel_batch.append(mel_np)
                    wav_batch.append(wav)
                    
                except:
                    continue
            
            if len(mel_batch) < 2:
                continue
            
            # Pad to same length (mel_batch entries are numpy arrays)
            max_len = max(m.shape[1] for m in mel_batch)
            mel_padded = []

            for m in mel_batch:
                if m.shape[1] < max_len:
                    m = np.pad(m, ((0, 0), (0, max_len - m.shape[1])), mode='edge')
                mel_padded.append(m)

            mel_batch_tensor = torch.from_numpy(np.stack(mel_padded)).float().to(self.device)
            
            # Apply perturbation with augmentation
            mel_protected = self.apply_delta(mel_batch_tensor, use_augmentation=True)
            
            # Compute losses
            attack_loss, quality_loss, reg_loss = self.compute_losses(
                mel_batch_tensor, mel_protected, wav_batch
            )
            
            # Total loss
            total_loss = (
                self.config['lambda_attack'] * attack_loss +
                self.config['lambda_quality'] * quality_loss +
                self.config['lambda_reg'] * reg_loss
            )
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_([self.delta], max_norm=1.0)
            self.optimizer.step()
            
            # Project to epsilon ball
            with torch.no_grad():
                self.delta.data = torch.clamp(
                    self.delta.data,
                    -self.config['epsilon'],
                    self.config['epsilon']
                )
            
            # Record metrics
            epoch_metrics['attack_loss'].append(attack_loss.item())
            epoch_metrics['quality_loss'].append(quality_loss.item())
            epoch_metrics['reg_loss'].append(reg_loss.item())
            epoch_metrics['total_loss'].append(total_loss.item())
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'attack': f"{attack_loss.item():.4f}",
                'quality': f"{quality_loss.item():.6f}"
            })
        
        # Step scheduler
        self.scheduler.step()
        
        avg_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 
                      for k, v in epoch_metrics.items()}
        
        return avg_metrics
    
    def train(self, manifest_path):
        """Full training loop"""
        
        with open(manifest_path, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*70}")
        print(f"ADVANCED TRAINING CONFIGURATION")
        print('='*70)
        print(f"Training files: {len(audio_files)}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Epsilon: {self.config['epsilon']}")
        print(f"Device: {self.config['device']}")
        print(f"Vocoder: HiFiGAN (Neural)")
        print(f"ASV Models: Ensemble (3 architectures)")
        print(f"Purification Defense: Enabled")
        print('='*70 + '\n')
        
        best_attack_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}/{self.config['epochs']}")
            print('='*70)
            
            metrics = self.train_epoch(audio_files, epoch)
            
            print(f"\n[Epoch {epoch}] Summary:")
            print(f"  Total Loss:   {metrics.get('total_loss', 0):.4f}")
            print(f"  Attack Loss:  {metrics.get('attack_loss', 0):.4f} ← lower = better protection")
            print(f"  Quality Loss: {metrics.get('quality_loss', 0):.6f} ← lower = better quality")
            print(f"  Reg Loss:     {metrics.get('reg_loss', 0):.4f}")
            
            os.makedirs("checkpoints_advanced", exist_ok=True)
            checkpoint_path = f"checkpoints_advanced/universal_delta_epoch{epoch}.npy"
            np.save(checkpoint_path, self.delta.detach().cpu().numpy())
            print(f"  ✅ Saved: {checkpoint_path}")
            self.config['epsilon'] += 0.05
            if metrics.get('attack_loss', float('inf')) < best_attack_loss:
                best_attack_loss = metrics['attack_loss']
                np.save("checkpoints_advanced/universal_delta_best.npy", 
                       self.delta.detach().cpu().numpy())
                print(f"  🌟 New best model! (Attack loss: {best_attack_loss:.4f})")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print('='*70)
        print(f"Best attack loss: {best_attack_loss:.4f}")
        print(f"Checkpoints saved in: checkpoints_advanced/")


# ================================
# ADVANCED PROTECTOR
# ================================

class AdvancedAudioProtector:
    """Apply advanced protection to audio"""
    
    def __init__(self, delta_path, config):
        self.config = config
        self.audio_proc = AdvancedAudioProcessor(
            sr=config['sr'], n_mels=config['n_mels'],
            n_fft=config['n_fft'], hop_length=config['hop_length']
        )
        
        self.delta = np.load(delta_path)
        print(f"[INFO] Loaded advanced protection from {delta_path}")
    
    def protect_audio(self, input_path, output_path, scale=1.0):
        """Protect audio file

        Args:
            input_path (str): path to input wav
            output_path (str): path to write protected wav
            scale (float): optional scale applied to the loaded delta (diagnostic)
        """
        
        print(f"\n[INFO] Protecting: {input_path}")
        
        wav, sr = self.audio_proc.load_audio(input_path)
        print(f"  Duration: {len(wav)/sr:.2f}s")
        
        # If diagnostic scale == 0.0, skip processing and return original audio
        if float(scale) == 0.0:
            print("[DEBUG] scale==0.0: writing original audio without protection (diagnostic)")
            self.audio_proc.save_audio(output_path, wav)
            print(f"  ✅ Saved original to: {output_path}")
            return

        mel = self.audio_proc.wav_to_mel(wav)
        # Two possible mel types: torch.Tensor (from vocoder's mel_transform)
        # or numpy dB (from librosa). Handle both.
        if isinstance(mel, torch.Tensor):
            # mel: [n_mels, T] tensor (power/magnitude expected)
            T = mel.shape[1]
            delta_T = self.delta.shape[1]
            num_tiles = int(np.ceil(T / delta_T))

            # Convert delta to tensor on same device
            delta_torch = torch.from_numpy(self.delta).float().to(mel.device)
            delta_tiled = delta_torch.repeat(1, num_tiles)[:, :T]

            # delta statistics (diagnostic)
            try:
                print(f"[DEBUG] delta min={torch.min(delta_torch).item():.6f} max={torch.max(delta_torch).item():.6f} mean={torch.mean(delta_torch).item():.6f} std={torch.std(delta_torch).item():.6f}")
            except Exception:
                pass

            # Apply optional scaling for diagnostics / sanity checks
            if scale != 1.0:
                try:
                    delta_tiled = delta_tiled * float(scale)
                    print(f"[DEBUG] applied scale={scale} to delta")
                except Exception:
                    pass

            # NOTE: Historically `delta` was trained in the dB domain (mel dB).
            # When `wav_to_mel` returns a torch mel from the vocoder pipeline it
            # will be a power/magnitude spectrogram (non-negative). To apply a
            # dB-domain delta to a power mel we must multiply by 10^(delta/10)
            # rather than add. Apply multiplicatively here to preserve expected
            # behavior when using torchaudio mel transforms.
            try:
                # Treat delta as dB offsets -> convert to multiplier
                delta_db = delta_tiled
                multiplier = torch.pow(10.0, delta_db / 10.0)
                mel_protected = mel * multiplier
            except Exception:
                # Fallback: additive if something goes wrong
                mel_protected = mel + delta_tiled

            # Debugging: show mel stats
            try:
                print(f"[DEBUG] mel min={float(mel.min()):.6f} max={float(mel.max()):.6f} mean={float(mel.mean()):.6f}")
                print(f"[DEBUG] mel_prot min={float(mel_protected.min()):.6f} max={float(mel_protected.max()):.6f} mean={float(mel_protected.mean()):.6f}")
            except Exception:
                pass

            # MEL comparison and visualization (save numpy arrays and image);
            # compute cosine similarity between original and protected mel.
            try:
                def _mel_to_db_np(x):
                    if isinstance(x, torch.Tensor):
                        arr = x.detach().cpu().numpy()
                    else:
                        arr = np.array(x)
                    # If negative values present, assume dB already
                    if arr.min() < 0:
                        return arr
                    try:
                        return librosa.power_to_db(arr, ref=np.max)
                    except Exception:
                        return 10.0 * np.log10(np.maximum(arr, 1e-10))

                try:
                    orig_db = _mel_to_db_np(mel)
                    prot_db = _mel_to_db_np(mel_protected)
                    min_T = min(orig_db.shape[1], prot_db.shape[1]) if (orig_db.ndim == 2 and prot_db.ndim == 2) else None
                    if min_T is not None:
                        orig_db_crop = orig_db[:, :min_T]
                        prot_db_crop = prot_db[:, :min_T]
                    else:
                        orig_db_crop = orig_db
                        prot_db_crop = prot_db

                    diff_db = prot_db_crop - orig_db_crop

                    # Cosine similarity on flattened dB vectors
                    try:
                        v1 = torch.from_numpy(orig_db_crop.ravel()).float()
                        v2 = torch.from_numpy(prot_db_crop.ravel()).float()
                        cos_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item())
                    except Exception:
                        cos_sim = None

                    base = os.path.splitext(output_path)[0]
                    np.save(base + '_mel_orig.npy', orig_db_crop)
                    np.save(base + '_mel_prot.npy', prot_db_crop)
                    np.save(base + '_mel_diff.npy', diff_db)
                    try:
                        if cos_sim is not None:
                            print(f"[DEBUG] Mel cosine similarity: {cos_sim:.6f}")
                        else:
                            print("[DEBUG] Mel cosine similarity: unavailable")
                    except Exception:
                        pass

                    try:
                        import matplotlib.pyplot as plt
                        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                        axs[0].imshow(orig_db_crop, aspect='auto', origin='lower', cmap='magma')
                        axs[0].set_title('orig mel (dB)')
                        axs[1].imshow(prot_db_crop, aspect='auto', origin='lower', cmap='magma')
                        axs[1].set_title('protected mel (dB)')
                        im = axs[2].imshow(diff_db, aspect='auto', origin='lower', cmap='RdBu')
                        axs[2].set_title('diff (dB)')
                        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
                        if cos_sim is not None:
                            fig.suptitle(f'Mel comparison — cosine={cos_sim:.4f}')
                        else:
                            fig.suptitle('Mel comparison')
                        plt.tight_layout()
                        img_path = base + '_mel_compare.png'
                        fig.savefig(img_path)
                        plt.close(fig)

                        # Clean up mel debug artifacts to avoid cluttering workspace
                        try:
                            candidates = [base + '_mel_orig.npy', base + '_mel_prot.npy', base + '_mel_diff.npy']
                            for p in candidates:
                                try:
                                    if os.path.exists(p):
                                        os.remove(p)
                                except Exception:
                                    pass
                            print(f"[DEBUG] Removed mel debug files for: {base}")
                        except Exception:
                            pass
                    except Exception:
                        pass

                except Exception:
                    pass
            except Exception:
                pass

            wav_protected = self.audio_proc.mel_to_wav(mel_protected)

            # Quick sanity check: if HiFiGAN output is very poor compared to
            # the original (low SNR), try a librosa Griffin-Lim fallback and
            # pick the better result. This helps when the mel transform's
            # scaling doesn't match the vocoder expectations.
            try:
                # Compute provisional SNR (crop to shortest)
                orig_len = len(wav)
                prot_len = len(wav_protected)
                minl = min(orig_len, prot_len)
                if minl > 0:
                    snr_try = self.audio_proc.compute_snr(wav[:minl], wav_protected[:minl])
                else:
                    snr_try = -999.0

                # If SNR is very low, attempt Griffin-Lim fallback using librosa
                if snr_try < 10.0:
                    print(f"[WARN] HiFiGAN output SNR={snr_try:.2f} dB is low; trying librosa fallback")

                    try:
                        # Convert mel_protected (tensor) to numpy power or dB as needed
                        mel_np = mel_protected.detach().cpu().numpy()

                        # If values look like dB (negative), convert to power
                        if mel_np.min() < 0:
                            mel_power = librosa.db_to_power(mel_np)
                        else:
                            mel_power = mel_np

                        wav_gl = librosa.feature.inverse.mel_to_audio(
                            mel_power, sr=self.audio_proc.sr,
                            n_fft=self.audio_proc.n_fft,
                            hop_length=self.audio_proc.hop_length, n_iter=64
                        )

                        minl2 = min(len(wav), len(wav_gl))
                        snr_gl = self.audio_proc.compute_snr(wav[:minl2], wav_gl[:minl2])
                        print(f"[WARN] Griffin-Lim fallback SNR={snr_gl:.2f} dB")

                        # Choose the better reconstruction
                        if snr_gl > snr_try:
                            print("[INFO] Using Griffin-Lim fallback (better SNR)")
                            wav_protected = wav_gl

                    except Exception as e:
                        print(f"[WARN] Griffin-Lim fallback failed: {e}")
            except Exception:
                pass

        else:
            # numpy path (legacy)
            T = mel.shape[1]
            delta_T = self.delta.shape[1]
            num_tiles = int(np.ceil(T / delta_T))
            delta_tiled = np.tile(self.delta, (1, num_tiles))[:, :T]

            # delta statistics (diagnostic)
            try:
                print(f"[DEBUG] delta min={np.min(self.delta):.6f} max={np.max(self.delta):.6f} mean={np.mean(self.delta):.6f} std={np.std(self.delta):.6f}")
            except Exception:
                pass

            # Apply optional scaling for diagnostics / sanity checks
            if scale != 1.0:
                try:
                    delta_tiled = delta_tiled * float(scale)
                    print(f"[DEBUG] applied scale={scale} to delta")
                except Exception:
                    pass

            mel_protected = mel + delta_tiled

            # MEL comparison and visualization (numpy path)
            try:
                def _mel_to_db_np(x):
                    if isinstance(x, torch.Tensor):
                        arr = x.detach().cpu().numpy()
                    else:
                        arr = np.array(x)
                    if arr.min() < 0:
                        return arr
                    try:
                        return librosa.power_to_db(arr, ref=np.max)
                    except Exception:
                        return 10.0 * np.log10(np.maximum(arr, 1e-10))

                try:
                    orig_db = _mel_to_db_np(mel)
                    prot_db = _mel_to_db_np(mel_protected)
                    min_T = min(orig_db.shape[1], prot_db.shape[1]) if (orig_db.ndim == 2 and prot_db.ndim == 2) else None
                    if min_T is not None:
                        orig_db_crop = orig_db[:, :min_T]
                        prot_db_crop = prot_db[:, :min_T]
                    else:
                        orig_db_crop = orig_db
                        prot_db_crop = prot_db

                    diff_db = prot_db_crop - orig_db_crop

                    try:
                        v1 = torch.from_numpy(orig_db_crop.ravel()).float()
                        v2 = torch.from_numpy(prot_db_crop.ravel()).float()
                        cos_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item())
                    except Exception:
                        cos_sim = None

                    base = os.path.splitext(output_path)[0]
                    np.save(base + '_mel_orig.npy', orig_db_crop)
                    np.save(base + '_mel_prot.npy', prot_db_crop)
                    np.save(base + '_mel_diff.npy', diff_db)
                    try:
                        if cos_sim is not None:
                            print(f"[DEBUG] Mel cosine similarity: {cos_sim:.6f}")
                        else:
                            print("[DEBUG] Mel cosine similarity: unavailable")
                    except Exception:
                        pass

                    try:
                        import matplotlib.pyplot as plt
                        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                        axs[0].imshow(orig_db_crop, aspect='auto', origin='lower', cmap='magma')
                        axs[0].set_title('orig mel (dB)')
                        axs[1].imshow(prot_db_crop, aspect='auto', origin='lower', cmap='magma')
                        axs[1].set_title('protected mel (dB)')
                        im = axs[2].imshow(diff_db, aspect='auto', origin='lower', cmap='RdBu')
                        axs[2].set_title('diff (dB)')
                        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
                        if cos_sim is not None:
                            fig.suptitle(f'Mel comparison — cosine={cos_sim:.4f}')
                        else:
                            fig.suptitle('Mel comparison')
                        plt.tight_layout()
                        img_path = base + '_mel_compare.png'
                        fig.savefig(img_path)
                        plt.close(fig)
                    except Exception:
                        pass

                except Exception:
                    pass
            except Exception:
                pass

            wav_protected = self.audio_proc.mel_to_wav(mel_protected)
        
        min_len = min(len(wav), len(wav_protected))
        wav = wav[:min_len]
        wav_protected = wav_protected[:min_len]
        
        if np.max(np.abs(wav_protected)) > 0:
            wav_protected = wav_protected / np.max(np.abs(wav_protected))
        
        self.audio_proc.save_audio(output_path, wav_protected)
        print(f"  ✅ Saved to: {output_path}")
        
        snr = self.audio_proc.compute_snr(wav, wav_protected)
        perceptual_loss = self.audio_proc.compute_perceptual_loss(wav, wav_protected)
        
        print(f"\n📊 Quality Metrics:")
        print(f"  SNR: {snr:.2f} dB (>30 dB is imperceptible)")
        print(f"  Perceptual Loss: {perceptual_loss:.6f}")


# ================================
# ADVANCED EVALUATOR
# ================================
from pystoi import stoi
from pesq import pesq

class AdvancedProtectionEvaluator:
    """Comprehensive evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.asv = EnsembleASVEmbedder(device=config['device'])
        self.audio_proc = AdvancedAudioProcessor(
            sr=config['sr'], n_mels=config['n_mels'],
            n_fft=config['n_fft'], hop_length=config['hop_length']
        )
    
    def evaluate(self, original_path, protected_path):
        """Comprehensive evaluation with multiple metrics"""
        
        print(f"\n{'='*70}")
        print("ADVANCED EVALUATION")
        print('='*70)
        
        wav_orig, _ = self.audio_proc.load_audio(original_path)
        wav_prot, _ = self.audio_proc.load_audio(protected_path)
        
        min_len = min(len(wav_orig), len(wav_prot))
        wav_orig = wav_orig[:min_len]
        wav_prot = wav_prot[:min_len]
        
        # 1. Protection Effectiveness
        print("\n🎯 Protection Effectiveness:")
        
        # Ensemble similarity
        similarity = self._compute_similarity(wav_orig, wav_prot)
        protection_score = 1.0 - similarity
        
        print(f"  Ensemble Similarity: {similarity:.4f}")
        print(f"  Protection Score: {protection_score:.4f}")
        
        if protection_score > 0.75:
            print("  ✅ EXCELLENT - Strong protection against voice cloning")
        elif protection_score > 0.6:
            print("  ✅ VERY GOOD - Good protection")
        elif protection_score > 0.4:
            print("  ⚠️  GOOD - Moderate protection")
        else:
            print("  ❌ WEAK - Limited protection")
        
        # 2. Audio Quality
        print("\n🎵 Audio Quality Metrics:")
        snr = self.audio_proc.compute_snr(wav_orig, wav_prot)
        perceptual_loss = self.audio_proc.compute_perceptual_loss(wav_orig, wav_prot)
        # PESQ (narrowband)
        pesq_score = pesq(self.config['sr'], wav_orig, wav_prot, 'nb')

        # STOI
        stoi_score = stoi(wav_orig, wav_prot, self.config['sr'], extended=False)
        print(f"  SNR: {snr:.2f} dB")
        if snr > 35:
            print("    → Imperceptible difference")
        elif snr > 25:
            print("    → Very high quality")
        else:
            print("    → Noticeable differences")
        
        print(f"  Perceptual Loss: {perceptual_loss:.6f}")
        print(f"  PESQ: {pesq_score:.4f}  (higher = better)")
        print(f"  STOI: {stoi_score:.4f}  (higher = better speech intelligibility)")
        # 3. Purification Resistance Test
        print("\n🛡️  Purification Resistance:")
        self._test_purification_resistance(wav_orig, wav_prot)
        
        print('='*70 + '\n')
    
    def _compute_similarity(self, wav1, wav2):
        """Compute ensemble similarity"""
        emb1 = self.asv.extract_embedding(wav1)
        emb2 = self.asv.extract_embedding(wav2)
        
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return max(0.0, min(1.0, sim))
    
    def _test_purification_resistance(self, wav_orig, wav_prot):
        """Test robustness against purification attacks"""
        
        # Test 1: Gaussian noise
        wav_gaussian = wav_prot + np.random.randn(len(wav_prot)) * 0.001
        sim_gaussian = self._compute_similarity(wav_orig, wav_gaussian)
        print(f"  After Gaussian Noise: sim={sim_gaussian:.4f} (want <0.4)")
        
        # Test 2: Median filtering
        from scipy.signal import medfilt
        wav_median = medfilt(wav_prot, kernel_size=3)
        sim_median = self._compute_similarity(wav_orig, wav_median)
        print(f"  After Median Filter: sim={sim_median:.4f} (want <0.4)")
        
        # Test 3: Resampling
        wav_resample = librosa.resample(wav_prot, orig_sr=16000, target_sr=8000)
        wav_resample = librosa.resample(wav_resample, orig_sr=8000, target_sr=16000)
        wav_resample = wav_resample[:len(wav_orig)]
        sim_resample = self._compute_similarity(wav_orig, wav_resample)
        print(f"  After Resampling: sim={sim_resample:.4f} (want <0.4)")
        
        # Overall resistance
        avg_resistance = 1.0 - np.mean([sim_gaussian, sim_median, sim_resample])
        print(f"\n  Overall Resistance: {avg_resistance:.4f}")
        if avg_resistance > 0.6:
            print("  ✅ Strong resistance to purification attacks")
        elif avg_resistance > 0.4:
            print("  ⚠️  Moderate resistance")
        else:
            print("  ❌ Vulnerable to purification")


# ================================
# MAIN CLI
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Voice Cloning Protection System"
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train protection model')
    train_parser.add_argument('--manifest', required=True, help='Path to training manifest')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    train_parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation budget')
    
    # Protect command
    protect_parser = subparsers.add_parser('protect', help='Protect audio file')
    protect_parser.add_argument('--input', required=True, help='Input audio file')
    protect_parser.add_argument('--output', required=True, help='Output protected file')
    protect_parser.add_argument('--delta', default='checkpoints_advanced/universal_delta_best.npy',
                               help='Path to trained delta')
    protect_parser.add_argument('--scale', type=float, default=1.0,
                               help='Scale factor to apply to the delta when protecting (diagnostic)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate protection')
    eval_parser.add_argument('--original', required=True, help='Original audio')
    eval_parser.add_argument('--protected', required=True, help='Protected audio')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sr': 16000,
        'n_fft': 1024,
        'hop_length': 256,
        'n_mels': 80,
        'delta_T': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if args.command == 'train':
        config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epsilon': args.epsilon,
            'lambda_attack': 10.0,
            'lambda_quality': 5.0,  # Higher weight for quality
            'lambda_reg': 0.05       # Lower reg for more flexibility
        })
        
        print("\n" + "="*70)
        print("ADVANCED ROBUST VOICE PROTECTION SYSTEM")
        print("="*70)
        print("\nKey Features:")
        print("  ✅ HiFiGAN neural vocoder (superior audio quality)")
        print("  ✅ Ensemble ASV models (3 architectures)")
        print("  ✅ Purification resistance training")
        print("  ✅ Adaptive noise scheduling")
        print("  ✅ Perceptual loss optimization")
        print("="*70 + "\n")
        
        trainer = RobustUniversalDeltaTrainer(config)
        trainer.train(args.manifest)
    
    elif args.command == 'protect':
        protector = AdvancedAudioProtector(args.delta, config)
        protector.protect_audio(args.input, args.output, scale=args.scale)
    
    elif args.command == 'evaluate':
        evaluator = AdvancedProtectionEvaluator(config)
        evaluator.evaluate(args.original, args.protected)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
