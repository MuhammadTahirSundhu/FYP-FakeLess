"""
test_install.py

Comprehensive test to verify Phase 2 installation and setup.
Run this before starting training to ensure everything is configured correctly.

Usage:
    python test_install.py
"""

import sys
import os

def test_imports():
    """Test if all required dependencies are installed."""
    print("="*60)
    print("FakeLess Phase 2 - Installation Verification")
    print("="*60)
    print("\n[1/4] Testing Core Dependencies...\n")
    
    errors = []
    warnings = []
    
    # Core dependencies
    try:
        import librosa
        print(f"✅ librosa (version {librosa.__version__})")
    except ImportError as e:
        errors.append(("librosa", str(e)))
        print("❌ librosa - NOT INSTALLED")
    
    try:
        import soundfile
        print(f"✅ soundfile (version {soundfile.__version__})")
    except ImportError as e:
        errors.append(("soundfile", str(e)))
        print("❌ soundfile - NOT INSTALLED")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_info = f"CUDA available: {cuda_available}"
        if cuda_available:
            device_info += f" (Device: {torch.cuda.get_device_name(0)})"
        print(f"✅ torch (version {torch.__version__}, {device_info})")
    except ImportError as e:
        errors.append(("torch", str(e)))
        print("❌ torch - NOT INSTALLED")
    
    try:
        import numpy
        print(f"✅ numpy (version {numpy.__version__})")
    except ImportError as e:
        errors.append(("numpy", str(e)))
        print("❌ numpy - NOT INSTALLED")
    
    try:
        import scipy
        print(f"✅ scipy (version {scipy.__version__})")
    except ImportError as e:
        errors.append(("scipy", str(e)))
        print("❌ scipy - NOT INSTALLED")
    
    try:
        import tqdm
        print(f"✅ tqdm (version {tqdm.__version__})")
    except ImportError as e:
        errors.append(("tqdm", str(e)))
        print("❌ tqdm - NOT INSTALLED")
    
    try:
        import speechbrain
        print(f"✅ speechbrain (version {speechbrain.__version__})")
    except ImportError as e:
        errors.append(("speechbrain", str(e)))
        print("❌ speechbrain - NOT INSTALLED")
    
    # Optional dependencies
    print("\n[2/4] Testing Optional Dependencies...\n")
    
    try:
        import pesq
        print("✅ pesq (PESQ quality metric available)")
    except ImportError:
        warnings.append("pesq - Optional: install with 'pip install pesq'")
        print("⚠️  pesq - not installed (optional)")
    
    try:
        import pystoi
        print("✅ pystoi (STOI quality metric available)")
    except ImportError:
        warnings.append("pystoi - Optional: install with 'pip install pystoi'")
        print("⚠️  pystoi - not installed (optional)")
    
    try:
        import matplotlib
        print(f"✅ matplotlib (visualization available)")
    except ImportError:
        warnings.append("matplotlib - Optional: install with 'pip install matplotlib'")
        print("⚠️  matplotlib - not installed (optional)")
    
    return errors, warnings

def test_file_structure():
    """Test if required files and directories exist."""
    print("\n[3/4] Testing File Structure...\n")
    
    required_files = [
        "utils_audio.py",
        "defense_training.py",
        "evaluate_defense.py",
        "apply_defense.py"
    ]
    
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            missing_files.append(filename)
    
    # Check directories (will be auto-created, so just inform)
    print("\n   Directories (will be auto-created if missing):")
    dirs = ["checkpoints_phase2", "demo_outputs", "pretrained_models"]
    for dirname in dirs:
        exists = os.path.exists(dirname)
        status = "exists" if exists else "will be created"
        print(f"   📁 {dirname} - {status}")
    
    return missing_files

def test_imports_from_files():
    """Test if our custom modules can be imported."""
    print("\n[4/4] Testing Custom Module Imports...\n")
    
    errors = []
    
    try:
        from utils_audio import (
            load_wav, save_wav, wav_to_mel, mel_to_wave_griffinlim,
            get_vocoder, get_psychoacoustic_masking, compute_quality_metrics
        )
        print("✅ utils_audio - All functions imported successfully")
    except Exception as e:
        print(f"❌ utils_audio - Import failed: {e}")
        errors.append(("utils_audio", str(e)))
    
    try:
        from defense_training import (
            PredictorNet, MultiDomainPredictorNet, RLPerturbationAgent,
            ASVEmbedder, train_universal_delta, train_predictor
        )
        print("✅ defense_training - All classes/functions imported successfully")
    except Exception as e:
        print(f"❌ defense_training - Import failed: {e}")
        errors.append(("defense_training", str(e)))
    
    try:
        import evaluate_defense
        print("✅ evaluate_defense - Module imported successfully")
    except Exception as e:
        print(f"❌ evaluate_defense - Import failed: {e}")
        errors.append(("evaluate_defense", str(e)))
    
    try:
        import apply_defense
        print("✅ apply_defense - Module imported successfully")
    except Exception as e:
        print(f"❌ apply_defense - Import failed: {e}")
        errors.append(("apply_defense", str(e)))
    
    return errors

def print_summary(dep_errors, warnings, file_errors, import_errors):
    """Print final summary and recommendations."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_errors = len(dep_errors) + len(file_errors) + len(import_errors)
    
    if total_errors == 0:
        print("\n✅ SUCCESS! Your Phase 2 environment is ready!")
        print("\n📋 Next Steps:")
        print("   1. Prepare your data manifests")
        print("   2. Run: python defense_training.py train_universal --advanced")
        print("   3. Evaluate: python evaluate_defense.py --manifest test.txt --delta ...")
        
        if warnings:
            print("\n⚠️  Optional packages not installed (recommended but not required):")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("\n🚀 You're ready to start Phase 2 training!")
        return True
    
    else:
        print("\n❌ ERRORS FOUND - Please fix the following issues:\n")
        
        if dep_errors:
            print("Missing Dependencies:")
            for pkg, error in dep_errors:
                print(f"   ❌ {pkg}")
            print("\n   Fix with:")
            missing_pkgs = " ".join([pkg for pkg, _ in dep_errors])
            print(f"   pip install {missing_pkgs}")
        
        if file_errors:
            print("\nMissing Files:")
            for filename in file_errors:
                print(f"   ❌ {filename}")
            print("\n   These files should be in your Code/ directory.")
        
        if import_errors:
            print("\nModule Import Errors:")
            for module, error in import_errors:
                print(f"   ❌ {module}: {error}")
            print("\n   Check that all files are in the same directory.")
        
        return False

def run_quick_test():
    """Run a quick functional test if everything is installed."""
    print("\n" + "="*60)
    print("QUICK FUNCTIONAL TEST")
    print("="*60)
    
    try:
        import numpy as np
        import torch
        from utils_audio import wav_to_mel, mel_to_wave_griffinlim
        
        print("\n[Test] Creating dummy audio and processing...")
        
        # Create 1 second of dummy audio
        dummy_wav = np.random.randn(16000).astype(np.float32) * 0.1
        
        # Convert to mel
        mel = wav_to_mel(dummy_wav)
        print(f"✅ Mel spectrogram shape: {mel.shape}")
        
        # Convert back to wav
        reconstructed = mel_to_wave_griffinlim(mel)
        print(f"✅ Reconstructed audio shape: {reconstructed.shape}")
        
        # Test neural network
        from defense_training import PredictorNet
        model = PredictorNet()
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        output = model(mel_tensor)
        print(f"✅ PredictorNet output shape: {output.shape}")
        
        print("\n✅ All functional tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functional test failed: {e}")
        print("   This might indicate a configuration issue.")
        return False

def main():
    """Main test runner."""
    # Test dependencies
    dep_errors, warnings = test_imports()
    
    # Test file structure
    file_errors = test_file_structure()
    
    # Test module imports
    import_errors = test_imports_from_files()
    
    # Print summary
    success = print_summary(dep_errors, warnings, file_errors, import_errors)
    
    # Run functional test if basic checks passed
    if success and not dep_errors and not file_errors:
        functional_success = run_quick_test()
        success = success and functional_success
    
    print("\n" + "="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)