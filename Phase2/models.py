"""
===============================================================================
This file contains ALL remaining code files for FakeLess Phase 2.
Copy each section to its respective filename.
===============================================================================
"""

# ===========================================================================
# FILE: models.py
# ===========================================================================
"""
All Neural Network Models
- Multi-Domain Perturbation Network
- RL Agent (PPO)
- GAN Components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import config

# Building Blocks
class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 15, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)

class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        if x.dim() == 3:  # 1D: [B, C, T]
            B, C, T = x.shape
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attention(x_t, x_t, x_t)
            attn_out = self.norm(attn_out + x_t)
            return attn_out.transpose(1, 2)
        else:  # 2D: [B, C, H, W]
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            attn_out, _ = self.attention(x_flat, x_flat, x_flat)
            attn_out = self.norm(attn_out + x_flat)
            return attn_out.transpose(1, 2).view(B, C, H, W)

# Multi-Domain Network
class TimeDomainEncoder(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = config.TIME_CHANNELS
        
        layers = []
        in_ch = 1
        for i, ch in enumerate(channels):
            stride = 2 if i < 2 else 1
            layers.extend([
                nn.Conv1d(in_ch, ch, 15, stride=stride, padding=7),
                nn.GroupNorm(min(8, ch), ch),
                nn.LeakyReLU(0.2),
                ResidualBlock1D(ch)
            ])
            in_ch = ch
        
        self.encoder = nn.Sequential(*layers)
        self.attention = AttentionBlock(channels[-1])
        self.output = nn.Conv1d(channels[-1], 1, 15, padding=7)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        return torch.tanh(self.output(x))

class FrequencyDomainEncoder(nn.Module):
    def __init__(self, n_mels=None, channels=None):
        super().__init__()
        if n_mels is None:
            n_mels = config.N_MELS
        if channels is None:
            channels = config.FREQ_CHANNELS
        
        layers = [nn.Conv2d(1, channels[0], 3, padding=1)]
        for i in range(len(channels) - 1):
            stride = (2, 1) if i < 2 else (1, 1)
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 3, stride=stride, padding=1),
                nn.GroupNorm(min(8, channels[i+1]), channels[i+1]),
                nn.LeakyReLU(0.2),
                ResidualBlock2D(channels[i+1])
            ])
        
        self.encoder = nn.Sequential(*layers)
        self.attention = AttentionBlock(channels[-1])
        
        decoder_layers = []
        for i in range(len(channels) - 1, 0, -1):
            upsample = nn.Upsample(scale_factor=(2, 1)) if i > len(channels) - 3 else nn.Identity()
            decoder_layers.extend([
                upsample,
                nn.Conv2d(channels[i], channels[i-1], 3, padding=1),
                nn.GroupNorm(min(8, channels[i-1]), channels[i-1]),
                nn.LeakyReLU(0.2)
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.output = nn.Conv2d(channels[0], n_mels, 3, padding=1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return torch.tanh(self.output(x).squeeze(1))

class MultiDomainPerturbationNetwork(nn.Module):
    def __init__(self, n_mels=None):
        super().__init__()
        self.n_mels = n_mels if n_mels is not None else config.N_MELS
        self.time_encoder = TimeDomainEncoder()
        self.freq_encoder = FrequencyDomainEncoder(self.n_mels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, waveform, mel_spec):
        delta_time = self.time_encoder(waveform)
        delta_freq = self.freq_encoder(mel_spec.unsqueeze(1))
        
        delta_time_down = F.adaptive_avg_pool1d(delta_time, delta_freq.shape[-1])
        delta_time_mel = delta_time_down.repeat(1, self.n_mels, 1)
        
        alpha = torch.sigmoid(self.alpha)
        delta_mel = alpha * delta_time_mel + (1 - alpha) * delta_freq
        
        return delta_mel, delta_time
    
    def apply_perturbation(self, waveform, mel_spec, epsilon=None):
        if epsilon is None:
            epsilon = config.EPSILON
        
        delta_mel, delta_time = self.forward(waveform, mel_spec)
        delta_mel = torch.clamp(delta_mel, -epsilon, epsilon)
        delta_time = torch.clamp(delta_time, -epsilon, epsilon)
        
        return mel_spec + delta_mel, waveform + delta_time, delta_mel, delta_time

# RL Components
class RLPolicyNetwork(nn.Module):
    def __init__(self, state_dim=None, action_dim=None):
        super().__init__()
        state_dim = state_dim or config.RL_STATE_DIM
        action_dim = action_dim or config.RL_ACTION_DIM
        hidden_dims = config.RL_HIDDEN_DIMS
        
        layers = []
        in_dim = state_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden
        
        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
    
    def forward(self, state):
        x = self.encoder(state)
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, torch.exp(log_std)
    
    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

class RLValueNetwork(nn.Module):
    def __init__(self, state_dim=None):
        super().__init__()
        state_dim = state_dim or config.RL_STATE_DIM
        hidden_dims = config.RL_HIDDEN_DIMS
        
        layers = []
        in_dim = state_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU()
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state).squeeze(-1)

# GAN Components
class PerturbationGenerator(nn.Module):
    def __init__(self, n_mels=None):
        super().__init__()
        n_mels = n_mels or config.N_MELS
        channels = config.GAN_GEN_CHANNELS
        
        encoder = []
        in_ch = 1
        for ch in channels[:len(channels)//2 + 1]:
            encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, 4, 2, 1),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2)
            ))
            in_ch = ch
        self.encoder = nn.ModuleList(encoder)
        
        decoder = []
        for ch in reversed(channels[:len(channels)//2]):
            decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, ch, 4, 2, 1),
                nn.InstanceNorm2d(ch),
                nn.ReLU()
            ))
            in_ch = ch
        self.decoder = nn.ModuleList(decoder)
        
        self.output = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(skips) - 1:
                x = x + skips[-(i+2)]
        
        return self.output(x)

class DeepfakeDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = config.GAN_DISC_CHANNELS
        
        layers = []
        in_ch = 1
        for ch in channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, 4, 2, 1),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2)
            ))
            in_ch = ch
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

# Utility Functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        **kwargs
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)

if __name__ == "__main__":
    print("Testing models...")
    model = MultiDomainPerturbationNetwork()
    print(f"Multi-Domain Network: {count_parameters(model):,} parameters")
    
    batch = 2
    waveform = torch.randn(batch, 1, 48000)
    mel = torch.randn(batch, config.N_MELS, 188)
    
    delta_mel, delta_time = model(waveform, mel)
    print(f"✓ Output shapes: delta_mel {delta_mel.shape}, delta_time {delta_time.shape}")
