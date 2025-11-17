# ===========================================================================
# FILE: config.py (UPDATED WITH YOUR PATHS)
# ===========================================================================
"""
Configuration for FakeLess Phase 2 - Updated for your dataset
"""

import torch
import os

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 8000
WIN_LENGTH = 1024

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
LAMBDA_ATTACK = 1.0
LAMBDA_QUALITY = 10.0
LAMBDA_REG = 0.1
LAMBDA_PSYCHO = 2.0

# ============================================================================
# PERTURBATION CONFIGURATION
# ============================================================================
EPSILON = 0.02
EPSILON_MIN = 0.01
EPSILON_MAX = 0.05
EPSILON_SCHEDULE = "fixed"

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
FREQ_CHANNELS = [64, 128, 256, 128, 64]
TIME_CHANNELS = [64, 128, 256, 128, 64]
ATTENTION_HEADS = 8
ACTIVATION = "leaky_relu"
NORM_TYPE = "group"
DROPOUT = 0.1

# ============================================================================
# RL AGENT CONFIGURATION
# ============================================================================
RL_STATE_DIM = 256
RL_ACTION_DIM = 64
RL_HIDDEN_DIMS = [512, 512, 256]
RL_GAMMA = 0.99
RL_GAE_LAMBDA = 0.95
RL_CLIP_EPSILON = 0.2
RL_EPISODES = 10000
RL_BATCH_SIZE = 64
RL_UPDATE_EPOCHS = 10

# ============================================================================
# GAN CONFIGURATION
# ============================================================================
GAN_GEN_CHANNELS = [64, 128, 256, 512, 256, 128, 64]
GAN_DISC_CHANNELS = [64, 128, 256, 512]
GAN_DISC_UPDATES = 1
GAN_GEN_UPDATES = 1
GAN_LAMBDA_ADV = 1.0
GAN_LAMBDA_FEAT = 10.0

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
CHUNK_DURATION = 3.0
MIN_DURATION = 1.0
MAX_DURATION = 10.0

USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5
AUG_NOISE_SNR = (20, 40)
AUG_BANDPASS = True

# ============================================================================
# PATHS - UPDATED FOR YOUR PROJECT STRUCTURE
# ============================================================================
# Base directories
PROJECT_ROOT = "."  # Current directory
DATA_DIR = "../data"

# Your existing LibriSpeech data
LIBRISPEECH_PREPARED_DIR = os.path.join(DATA_DIR, "librispeech_prepared")

# Manifest directory
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")

# Pretrained models
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")

# Training data manifests (will be created by preprocess_data.py)
TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train.txt")
VAL_MANIFEST = os.path.join(MANIFEST_DIR, "val.txt")
TEST_MANIFEST = os.path.join(MANIFEST_DIR, "test.txt")

# Output directories
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
OUTPUT_DIR = "protected_audio"
EVAL_DIR = "evaluation_results"

# Create necessary directories
for directory in [MANIFEST_DIR, PRETRAINED_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, EVAL_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 100
SAVE_INTERVAL = 5
EVAL_INTERVAL = 1
VERBOSE = True
USE_TENSORBOARD = True

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
QUALITY_METRICS = ["pesq", "stoi", "sisdr", "snr"]

DEEPFAKE_MODELS = [
    "tacotron2",
    "seed-vc",
    "rvc-v2",
    "gpt-sovits",
    "styletts2",
    "xtts-v2"
]

ROBUSTNESS_TESTS = [
    "mp3_128k",
    "mp3_64k",
    "aac_96k",
    "phone_codec",
    "gaussian_noise",
    "bandpass_filter",
]

# ============================================================================
# VOCODER CONFIGURATION
# ============================================================================
USE_HIFIGAN = True
HIFIGAN_CHECKPOINT = os.path.join(PRETRAINED_DIR, "hifigan/hifi-gan", "generator_v1")
GRIFFIN_LIM_ITERS = 32

# ============================================================================
# ASV CONFIGURATION
# ============================================================================
USE_ASV = False
ASV_MODEL = "ecapa-tdnn"
ASV_CHECKPOINT_DIR = os.path.join(PRETRAINED_DIR, "spkrec_ecapa")

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42
DETERMINISTIC = False

# ============================================================================
# MIXED PRECISION TRAINING
# ============================================================================
USE_AMP = True
AMP_OPT_LEVEL = "O1"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("FAKELESS PHASE 2 - CONFIGURATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epsilon (Perturbation): {EPSILON}")
    print("-" * 70)
    print("Loss Weights:")
    print(f"  Attack: {LAMBDA_ATTACK}")
    print(f"  Quality: {LAMBDA_QUALITY}")
    print(f"  Regularization: {LAMBDA_REG}")
    print(f"  Psychoacoustic: {LAMBDA_PSYCHO}")
    print("-" * 70)
    print("Data Paths:")
    print(f"  LibriSpeech: {LIBRISPEECH_PREPARED_DIR}")
    print(f"  Manifests: {MANIFEST_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print("-" * 70)
    print("Audio Settings:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  N-FFT: {N_FFT}")
    print(f"  Hop Length: {HOP_LENGTH}")
    print(f"  Mel Bands: {N_MELS}")
    print("=" * 70)

def get_audio_config():
    return {
        'sr': SAMPLE_RATE,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': N_MELS,
        'fmin': FMIN,
        'fmax': FMAX,
        'win_length': WIN_LENGTH
    }

def get_training_config():
    return {
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'lambda_attack': LAMBDA_ATTACK,
        'lambda_quality': LAMBDA_QUALITY,
        'lambda_reg': LAMBDA_REG,
        'lambda_psycho': LAMBDA_PSYCHO,
        'epsilon': EPSILON,
    }

def get_model_config():
    return {
        'freq_channels': FREQ_CHANNELS,
        'time_channels': TIME_CHANNELS,
        'attention_heads': ATTENTION_HEADS,
        'n_mels': N_MELS,
        'dropout': DROPOUT,
    }

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def use_preset(preset_name):
    """Load preset configuration"""
    global LAMBDA_ATTACK, LAMBDA_QUALITY, EPSILON, BATCH_SIZE, EPOCHS
    
    presets = {
        "quality": {
            "LAMBDA_ATTACK": 0.5,
            "LAMBDA_QUALITY": 20.0,
            "EPSILON": 0.01,
        },
        "protection": {
            "LAMBDA_ATTACK": 3.0,
            "LAMBDA_QUALITY": 5.0,
            "EPSILON": 0.04,
        },
        "balanced": {
            "LAMBDA_ATTACK": 1.0,
            "LAMBDA_QUALITY": 10.0,
            "EPSILON": 0.02,
        },
        "fast": {
            "BATCH_SIZE": 32,
            "EPOCHS": 50,
            "LAMBDA_ATTACK": 1.0,
            "LAMBDA_QUALITY": 10.0,
        }
    }
    
    if preset_name in presets:
        for key, value in presets[preset_name].items():
            globals()[key] = value
        print(f"Loaded preset: {preset_name}")
    else:
        print(f"Unknown preset: {preset_name}")

def check_dataset():
    """Check if LibriSpeech dataset exists"""
    print("\nChecking dataset availability...")
    
    if os.path.exists(LIBRISPEECH_PREPARED_DIR):
        print(f"✓ LibriSpeech data found: {LIBRISPEECH_PREPARED_DIR}")
        
        subsets = ['train-clean-100', 'dev-clean', 'test-clean']
        for subset in subsets:
            subset_path = os.path.join(LIBRISPEECH_PREPARED_DIR, subset)
            if os.path.exists(subset_path):
                print(f"  ✓ {subset}")
            else:
                print(f"  ✗ {subset} not found")
    else:
        print(f"✗ LibriSpeech data not found: {LIBRISPEECH_PREPARED_DIR}")
        print("  Please ensure your data is in the correct location")
    
    print("\nChecking manifests...")
    for manifest_name, manifest_path in [
        ('Train', TRAIN_MANIFEST),
        ('Val', VAL_MANIFEST),
        ('Test', TEST_MANIFEST)
    ]:
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                count = len([l for l in f if l.strip()])
            print(f"  ✓ {manifest_name}: {count} files")
        else:
            print(f"  ✗ {manifest_name} manifest not found")
            print(f"    Run: python preprocess_data.py --mode existing_librispeech")

if __name__ == "__main__":
    print_config()
    check_dataset()