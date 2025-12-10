"""
===============================================================================
AUTOMATED SETUP SCRIPT FOR EXISTING LIBRISPEECH DATA
===============================================================================
Run this once to set up everything automatically

Usage:
    python setup_existing_data.py
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

def check_dependencies():
    """Check if required packages are installed"""
    print_header("STEP 1: Checking Dependencies")
    
    required_packages = [
        'torch',
        'librosa',
        'soundfile',
        'numpy',
        'scipy',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print("\n⚠ Missing packages detected!")
        print("Install with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install librosa soundfile scipy tqdm numpy pyyaml")
        return False
    
    print("\n✓ All core dependencies installed!")
    return True

def check_dataset():
    """Check if LibriSpeech data exists"""
    print_header("STEP 2: Checking LibriSpeech Dataset")
    
    # Try multiple possible locations
    possible_paths = [
        "../data/librispeech_prepared",
        "data/librispeech_prepared",
        "../librispeech_prepared",
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("✗ LibriSpeech prepared data not found!")
        print("  Looked in:")
        for path in possible_paths:
            print(f"    - {os.path.abspath(path)}")
        return None
    
    print(f"✓ Found LibriSpeech data: {os.path.abspath(data_path)}")
    
    # Check subsets
    subsets = ['train-clean-100', 'dev-clean', 'test-clean']
    found_subsets = []
    
    for subset in subsets:
        subset_path = os.path.join(data_path, subset)
        if os.path.exists(subset_path):
            # Count audio files
            audio_files = list(Path(subset_path).rglob('*.flac'))
            audio_files.extend(Path(subset_path).rglob('*.wav'))
            print(f"  ✓ {subset}: {len(audio_files)} files")
            found_subsets.append(subset)
        else:
            print(f"  ✗ {subset}: NOT FOUND")
    
    if len(found_subsets) == 0:
        print("\n✗ No valid subsets found!")
        return None
    
    return data_path

def create_directories():
    """Create necessary directories"""
    print_header("STEP 3: Creating Directories")
    
    directories = [
        'data/manifests',
        'data/pretrained',
        'checkpoints',
        'logs',
        'protected_audio',
        'evaluation_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        abs_path = os.path.abspath(directory)
        print(f"✓ {directory}")
    
    print("\n✓ All directories created!")

def create_manifests(data_path):
    """Create manifest files from existing data"""
    print_header("STEP 4: Creating Manifest Files")
    
    manifest_dir = 'data/manifests'
    os.makedirs(manifest_dir, exist_ok=True)
    
    subsets = {
        'train.txt': 'train-clean-100',
        'val.txt': 'dev-clean',
        'test.txt': 'test-clean'
    }
    
    for manifest_name, subset_name in subsets.items():
        subset_path = Path(data_path) / subset_name
        
        if not subset_path.exists():
            print(f"⚠ Skipping {subset_name} (not found)")
            continue
        
        print(f"\nProcessing {subset_name}...")
        
        # Find audio files
        audio_files = list(subset_path.rglob('*.flac'))
        audio_files.extend(subset_path.rglob('*.wav'))
        
        if not audio_files:
            print(f"  ✗ No audio files found!")
            continue
        
        # Write manifest
        manifest_path = os.path.join(manifest_dir, manifest_name)
        with open(manifest_path, 'w') as f:
            for audio_file in sorted(audio_files):
                f.write(f"{str(audio_file.absolute())}\n")
        
        print(f"  ✓ Created {manifest_name}: {len(audio_files)} files")
        print(f"  Path: {os.path.abspath(manifest_path)}")
    
    print("\n✓ Manifest files created successfully!")

def verify_setup():
    """Verify the setup is complete"""
    print_header("STEP 5: Verifying Setup")
    
    checks = []
    
    # Check manifests
    manifest_dir = 'data/manifests'
    for manifest in ['train.txt', 'val.txt', 'test.txt']:
        manifest_path = os.path.join(manifest_dir, manifest)
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                count = len([l for l in f if l.strip()])
            print(f"✓ {manifest}: {count} files")
            checks.append(True)
        else:
            print(f"✗ {manifest}: NOT FOUND")
            checks.append(False)
    
    # Check Python files
    required_files = [
        'config.py',
        'audio_utils.py',
        'models.py',
        'dataset.py',
        'train_multidomain.py',
        'protect_audio.py',
        'evaluate.py'
    ]
    
    print("\nChecking Python files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
            checks.append(True)
        else:
            print(f"✗ {file}: NOT FOUND")
            checks.append(False)
    
    # Check CUDA
    print("\nChecking CUDA:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_name}")
            checks.append(True)
        else:
            print("⚠ CUDA not available (will use CPU - slower)")
            checks.append(True)  # Not critical
    except:
        print("✗ PyTorch not installed properly")
        checks.append(False)
    
    return all(checks)

def print_next_steps():
    """Print next steps"""
    print_header("🎉 SETUP COMPLETE!")
    
    print("""
Your FakeLess Phase 2 is ready to use!

NEXT STEPS:

1. Download pretrained models (recommended):
   python download_models.py --model all

2. Test configuration:
   python config.py

3. Test data loading:
   python -c "from dataset import create_dataloader; print('✓ Data loading works!')"

4. Start training:
   python train_multidomain.py

5. Monitor training:
   tensorboard --logdir logs
   (Open browser: http://localhost:6006)

QUICK COMMANDS:

# Basic training
python train_multidomain.py

# Training with custom settings
python train_multidomain.py --epochs 50 --batch_size 8

# Protect audio (after training)
python protect_audio.py --input sample.wav --output protected.wav

# Evaluate
python evaluate.py --test_file protected.wav

TIPS:
- Training takes 2-3 days on a single GPU
- Checkpoints save every 5 epochs
- Best model saves automatically
- Reduce batch_size if GPU memory issues

Good luck! 🚀
    """)

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            FakeLess Phase 2 - Automated Setup                 ║
    ║                                                               ║
    ║        Setup for Existing LibriSpeech Prepared Data           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies first!")
        print("See SETUP.md for installation instructions")
        return False
    
    # Step 2: Find dataset
    data_path = check_dataset()
    if not data_path:
        print("\n✗ Setup failed: LibriSpeech data not found!")
        print("Please ensure your data is in one of these locations:")
        print("  - ../data/librispeech_prepared")
        print("  - data/librispeech_prepared")
        return False
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Create manifests
    create_manifests(data_path)
    
    # Step 5: Verify
    if not verify_setup():
        print("\n⚠ Setup incomplete - some checks failed!")
        print("Please review the errors above")
        return False
    
    # Success!
    print_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)