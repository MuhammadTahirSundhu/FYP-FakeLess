# FakeLess Phase 2 - Quick Start Commands

## 🎯 Copy-Paste Ready Commands

### 1️⃣ Setup (5 minutes)

```bash
# Create environment
conda create -n fakeless python=3.10 -y
conda activate fakeless

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install librosa soundfile scipy pesq pystoi tensorboard tqdm pyyaml speechbrain resampy

# Create directories
mkdir -p data/raw data/manifests data/pretrained checkpoints logs protected_audio
```

### 2️⃣ Download Models (2 minutes)

```bash
python download_models.py --model all
```

### 3️⃣ Prepare Data

**Option A: LibriSpeech (Recommended)**
```bash
# Download and prepare (this will take 30-60 minutes)
python preprocess_data.py --mode librispeech --download --data_dir data/raw --manifest_dir data/manifests
```

**Option B: Your Own Audio**
```bash
# Create manifest from your audio directory
python preprocess_data.py --mode custom --audio_dir /path/to/your/audio --output_manifest data/manifests/all.txt

# Split into train/val/test
python preprocess_data.py --mode split --input_manifest data/manifests/all.txt
```

### 4️⃣ Train Model (2-3 days on single GPU)

```bash
# Basic training
python train_multidomain.py

# With custom settings
python train_multidomain.py --epochs 100 --batch_size 16 --lr 0.0001

# Resume from checkpoint
python train_multidomain.py --resume checkpoints/checkpoint_epoch50.pt

# Monitor training
tensorboard --logdir logs
```

### 5️⃣ Protect Audio (Instant)

```bash
# Single file
python protect_audio.py --input sample.wav --output protected.wav

# With custom strength
python protect_audio.py --input sample.wav --output protected.wav --epsilon 0.03

# Batch process directory
python protect_audio.py --input audio_folder/ --output protected_folder/
```

### 6️⃣ Evaluate Protection

```bash
# Single file evaluation
python evaluate.py --test_file sample.wav

# Dataset evaluation
python evaluate.py --test_manifest data/manifests/test.txt

# Robustness testing
python evaluate.py --test_file sample.wav --robustness
```

---

## 🔧 Configuration Quick Tweaks

Edit `config.py` for quick changes:

```python
# Increase protection strength
EPSILON = 0.03  # Default: 0.02

# Better audio quality
LAMBDA_QUALITY = 20.0  # Default: 10.0

# Stronger protection
LAMBDA_ATTACK = 2.0  # Default: 1.0

# Smaller batch if OOM
BATCH_SIZE = 8  # Default: 16
```

---

## 🐛 Common Issues - Quick Fixes

### CUDA Out of Memory
```python
# In config.py
BATCH_SIZE = 8
```

### SpeechBrain Download Fails
```bash
# It will auto-download on first use, or manually:
mkdir -p data/pretrained/spkrec_ecapa
# Model will download automatically when needed
```

### Poor Audio Quality
```bash
# Decrease epsilon
python protect_audio.py --input sample.wav --output protected.wav --epsilon 0.01
```

### Weak Protection
```bash
# Increase epsilon
python protect_audio.py --input sample.wav --output protected.wav --epsilon 0.03
```

---

## 📁 File Contents Quick Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `config.py` | All settings | Always edit this first |
| `audio_utils.py` | Audio processing | Rarely |
| `models.py` | Model architectures | For research |
| `train_multidomain.py` | Training script | Rarely |
| `protect_audio.py` | Inference | Never |
| `evaluate.py` | Evaluation | Never |
| `preprocess_data.py` | Data prep | Never |

---

## 🎓 Typical Workflow

```bash
# Day 1: Setup and data
conda create -n fakeless python=3.10 -y && conda activate fakeless
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa soundfile scipy pesq pystoi tensorboard tqdm pyyaml speechbrain
python download_models.py --model all
python preprocess_data.py --mode librispeech --download

# Day 2-4: Training
python train_multidomain.py --epochs 100
# Monitor with: tensorboard --logdir logs

# Day 5: Evaluation
python evaluate.py --test_manifest data/manifests/test.txt

# Day 6+: Production use
python protect_audio.py --input my_audio/ --output protected_audio/
```

---

## 💡 Pro Tips

### Speed Up Training
```python
# config.py
NUM_WORKERS = 8  # Use more CPU cores
BATCH_SIZE = 32  # If GPU memory allows
```

### Better Quality
```python
# config.py
LAMBDA_QUALITY = 15.0
LAMBDA_PSYCHO = 3.0
EPSILON = 0.015  # Smaller perturbation
```

### Maximum Protection
```python
# config.py
LAMBDA_ATTACK = 3.0
EPSILON = 0.04  # Larger perturbation
```

### Balanced (Recommended)
```python
# config.py (defaults)
LAMBDA_ATTACK = 1.0
LAMBDA_QUALITY = 10.0
EPSILON = 0.02
```

---

## 🧪 Quick Test Script

```bash
#!/bin/bash
# test.sh - Quick functionality test

echo "Testing audio utilities..."
python audio_utils.py

echo "Testing models..."
python models.py

echo "Testing config..."
python -c "import config; config.print_config()"

echo "Testing data preprocessing..."
python preprocess_data.py --mode custom --audio_dir test_audio/ --output_manifest test.txt

echo "All tests passed! ✓"
```

---

## 📊 Expected Timeline

| Task | Time | GPU Required |
|------|------|-------------|
| Setup | 5 min | No |
| Download models | 2 min | No |
| Download LibriSpeech | 30 min | No |
| Training (100 epochs) | 2-3 days | Yes |
| Protect single file | 1 sec | Optional |
| Evaluate file | 5 sec | Optional |

---

## 🚀 One-Liner Commands

```bash
# Full setup
conda create -n fakeless python=3.10 -y && conda activate fakeless && pip install torch torchvision torchaudio librosa soundfile scipy pesq pystoi tensorboard tqdm pyyaml speechbrain --index-url https://download.pytorch.org/whl/cu118

# Download everything
python download_models.py --model all && python preprocess_data.py --mode librispeech --download

# Train and monitor
python train_multidomain.py & tensorboard --logdir logs

# Protect and evaluate
python protect_audio.py --input sample.wav --output protected.wav && python evaluate.py --test_file protected.wav
```

---

## 📝 Checklist

- [ ] Environment created and activated
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Data prepared (manifests created)
- [ ] Training completed
- [ ] Model evaluated
- [ ] Audio protected successfully

---

## 🆘 Emergency Contacts

- **Repository Issues**: Open GitHub issue
- **Email**: 22i-0871@nu.edu.pk
- **Documentation**: See README.md

---

**Remember**: Start with defaults, train, evaluate, then optimize! 🎯