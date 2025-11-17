# FakeLess Phase 2: Audio Deepfake Prevention

**Simplified, Production-Ready Implementation**

Protect your voice from unauthorized AI cloning using imperceptible adversarial perturbations.

---

## 📁 Project Structure

```
fakeless_phase2/
├── config.py                  # All configuration settings
├── audio_utils.py            # Audio processing utilities
├── models.py                 # All model architectures
├── preprocess_data.py        # Data preparation
├── train_multidomain.py      # Train multi-domain network
├── train_rl.py              # Train RL agent (optional)
├── train_gan.py             # Train GAN defense (optional)
├── protect_audio.py          # Protect audio files (inference)
├── evaluate.py               # Comprehensive evaluation
├── download_models.py        # Download pretrained models
├── requirements.txt          # Dependencies
└── README.md                 # This file

data/
├── raw/                      # Raw audio files
├── manifests/               # Train/val/test splits
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── pretrained/              # Downloaded models

checkpoints/                  # Saved model checkpoints
logs/                        # TensorBoard logs
protected_audio/             # Protected audio outputs
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Installation

```bash
# Create environment
conda create -n fakeless python=3.10
conda activate fakeless

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Models

```bash
python download_models.py --model all
```

### 3. Prepare Data

**Option A: Use LibriSpeech (recommended for training)**
```bash
python preprocess_data.py --mode librispeech --download --data_dir data/raw --manifest_dir data/manifests
```

**Option B: Use your own audio**
```bash
python preprocess_data.py --mode custom --audio_dir /path/to/your/audio --output_manifest data/manifests/all.txt
python preprocess_data.py --mode split --input_manifest data/manifests/all.txt
```

### 4. Train Model

```bash
python train_multidomain.py --train_manifest data/manifests/train.txt --val_manifest data/manifests/val.txt --epochs 100
```

### 5. Protect Audio

```bash
# Single file
python protect_audio.py --input sample.wav --output protected.wav --model checkpoints/best_model.pt

# Entire directory
python protect_audio.py --input audio_dir/ --output protected_dir/ --model checkpoints/best_model.pt
```

---

## 📚 Detailed Usage

### Configuration

Edit `config.py` to customize settings:

```python
# Audio settings
SAMPLE_RATE = 16000
N_MELS = 80
EPSILON = 0.02  # Perturbation strength

# Training settings
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

# Loss weights
LAMBDA_ATTACK = 1.0
LAMBDA_QUALITY = 10.0
LAMBDA_REG = 0.1
LAMBDA_PSYCHO = 2.0
```

### Training

**Basic Training:**
```bash
python train_multidomain.py
```

**Advanced Options:**
```bash
python train_multidomain.py \
    --train_manifest data/manifests/train.txt \
    --val_manifest data/manifests/val.txt \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --device cuda
```

**Resume Training:**
```bash
python train_multidomain.py --resume checkpoints/checkpoint_epoch50.pt
```

**Monitor Training:**
```bash
tensorboard --logdir logs
```

### Inference

**Protect Single File:**
```python
from protect_audio import AudioProtector

protector = AudioProtector('checkpoints/best_model.pt')
protector.protect_file('input.wav', 'output.wav', epsilon=0.02)
```

**Command Line:**
```bash
# Default settings
python protect_audio.py --input sample.wav --output protected.wav

# Custom perturbation strength
python protect_audio.py --input sample.wav --output protected.wav --epsilon 0.03

# Batch processing
python protect_audio.py --input audio_folder/ --output protected_folder/
```

### Evaluation

**Evaluate Single File:**
```bash
python evaluate.py --test_file sample.wav --model checkpoints/best_model.pt
```

**Evaluate Dataset:**
```bash
python evaluate.py --test_manifest data/manifests/test.txt --model checkpoints/best_model.pt
```

**Test Robustness:**
```bash
python evaluate.py --test_file sample.wav --robustness
```

**Metrics Reported:**
- **PESQ**: Perceptual quality (higher is better, max ~4.5)
- **STOI**: Speech intelligibility (0-1, higher is better)
- **SI-SDR**: Signal distortion ratio in dB
- **ASV Similarity**: Speaker similarity (lower means better protection)
- **Protection Score**: 1 - ASV Similarity (higher is better)

---

## 🎯 Expected Results

### After Training (100 epochs on LibriSpeech)

| Metric | Target | Description |
|--------|--------|-------------|
| **PESQ** | > 4.0 | Audio quality preserved |
| **STOI** | > 0.95 | Speech intelligibility maintained |
| **Protection Score** | > 85% | Effective against voice cloning |
| **Processing Time** | < 2s | For 30-second audio (GPU) |

### Perturbation Strength Guidelines

| Epsilon | Quality | Protection | Use Case |
|---------|---------|------------|----------|
| 0.01 | Excellent | Moderate | High-quality recordings |
| 0.02 | Very Good | Good | Recommended default |
| 0.03 | Good | Strong | Maximum protection |
| 0.05 | Fair | Very Strong | Last resort |

---

## 🔬 Advanced Features

### 1. Reinforcement Learning Agent (Optional)

Train adaptive perturbation optimizer:

```bash
python train_rl.py --pretrained_base checkpoints/best_model.pt --episodes 10000
```

### 2. GAN-Based Defense (Optional)

Train GAN for robust perturbations:

```bash
python train_gan.py --epochs 50
```

### 3. Custom Architecture

Modify `models.py` to experiment with different architectures:

```python
# In models.py
class CustomPerturbationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
```

### 4. Psychoacoustic Masking

Enhance imperceptibility by tuning in `config.py`:

```python
LAMBDA_PSYCHO = 5.0  # Increase for stronger masking
```

---

## 📊 Monitoring & Debugging

### TensorBoard Visualization

```bash
tensorboard --logdir logs --port 6006
```

View:
- Training/validation losses
- Attack loss (lower is better)
- Quality loss (should stay low)
- Learning rate schedule

### Common Issues

**1. CUDA Out of Memory**
```python
# In config.py
BATCH_SIZE = 8  # Reduce from 16
```

**2. Poor Audio Quality**
```python
# Increase quality weight
LAMBDA_QUALITY = 20.0  # Increase from 10.0
```

**3. Weak Protection**
```python
# Increase attack weight
LAMBDA_ATTACK = 2.0  # Increase from 1.0
EPSILON = 0.03  # Increase from 0.02
```

**4. HiFi-GAN Not Available**
```python
# Will automatically fallback to Griffin-Lim
# Quality will be lower but functional
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test audio utilities
python audio_utils.py

# Test models
python models.py

# Test configuration
python -c "import config; config.print_config()"
```

### Integration Test

```bash
# Quick end-to-end test
python protect_audio.py --input test.wav --output test_protected.wav
python evaluate.py --test_file test_protected.wav
```

---

## 📈 Performance Optimization

### For Training

```python
# In config.py
NUM_WORKERS = 8  # Increase for faster data loading
BATCH_SIZE = 32  # If you have enough GPU memory
```

### For Inference

```python
# Use GPU
python protect_audio.py --input sample.wav --output protected.wav --device cuda

# Batch processing (faster)
python protect_audio.py --input folder/ --output protected_folder/
```

---

## 🔧 Troubleshooting

### Model Not Learning

1. Check data quality: `python audio_utils.py`
2. Verify manifests are correct
3. Reduce learning rate: `--lr 0.00005`
4. Check loss weights in `config.py`

### Poor Audio Quality

1. Increase `LAMBDA_QUALITY` in `config.py`
2. Decrease `EPSILON` value
3. Use HiFi-GAN vocoder (better than Griffin-Lim)

### Low Protection Score

1. Increase `LAMBDA_ATTACK` in `config.py`
2. Train for more epochs
3. Increase `EPSILON` value
4. Use ensemble of multiple models

---

## 📖 Citation

```bibtex
@software{fakeless2025,
  title={FakeLess: Audio Deepfake Prevention via Multi-Domain Adversarial Perturbations},
  author={Haider, Muhammad Nabeed and Sundhu, Muhammad Tahir and Siddiqui, Sameed Ahmed},
  year={2025},
  institution={National University of Computer and Emerging Sciences}
}
```

---

## 👥 Team

**Muhammad Nabeed Haider** (22I-0871)  
**Muhammad Tahir Sundhu** (22I-0821)  
**Sameed Ahmed Siddiqui** (22I-8223)

**Supervisor:** Dr. Syed Qaiser Ali Shah  
**Co-Supervisor:** Mr. Muhammad Almas Khan

**Department of Computer Science**  
National University of Computer and Emerging Sciences  
Islamabad, Pakistan

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- SpeechBrain team for ECAPA-TDNN models
- HiFi-GAN authors for high-quality vocoder
- LibriSpeech dataset creators
- Research community for foundational work on adversarial audio

---

## 🚀 Next Steps

1. **Train your model** on LibriSpeech (2-3 days on single GPU)
2. **Evaluate performance** on test set
3. **Protect your audio** files
4. **Test against deepfake models** (TTS, voice conversion)
5. **Fine-tune** hyperparameters for your use case
6. **Deploy** for production use

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: 22i-0871@nu.edu.pk

**Good luck with your project!** 🎉