# ===========================================================================
# FILE: dataset.py
# ===========================================================================
"""
PyTorch Dataset Classes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import config
import audio_utils

class AudioDefenseDataset(Dataset):
    def __init__(self, manifest_path, chunk_duration=None, augment=False):
        self.chunk_duration = chunk_duration or config.CHUNK_DURATION
        self.chunk_samples = int(self.chunk_duration * config.SAMPLE_RATE)
        self.augment = augment
        
        with open(manifest_path, 'r') as f:
            self.audio_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.audio_paths)} files from {manifest_path}")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        try:
            wav = audio_utils.load_audio(self.audio_paths[idx])
            
            if len(wav) > self.chunk_samples:
                start = random.randint(0, len(wav) - self.chunk_samples)
                wav = wav[start:start + self.chunk_samples]
            else:
                wav = audio_utils.pad_or_trim(wav, self.chunk_samples)
            
            if self.augment and config.USE_AUGMENTATION and random.random() < config.AUGMENTATION_PROB:
                wav = audio_utils.apply_random_augmentation(wav)
            
            mel = audio_utils.wav_to_mel(wav)
            wav = audio_utils.normalize_audio(wav)
            
            return {
                'waveform': torch.from_numpy(wav).unsqueeze(0).float(),
                'mel': torch.from_numpy(mel).float(),
                'path': self.audio_paths[idx]
            }
        except Exception as e:
            print(f"Error loading {self.audio_paths[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

def create_dataloader(manifest_path, batch_size=None, shuffle=True, augment=False):
    batch_size = batch_size or config.BATCH_SIZE
    dataset = AudioDefenseDataset(manifest_path, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )