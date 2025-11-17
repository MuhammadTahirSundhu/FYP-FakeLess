# ===========================================================================
# FILE: preprocess_data.py
# ===========================================================================
"""
Data Preprocessing Script
"""

import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import config

def create_manifest(audio_dir, output_file, extensions=['.wav', '.flac', '.mp3'], min_duration=1.0):
    import audio_utils
    
    print(f"Scanning {audio_dir} for audio files...")
    audio_dir = Path(audio_dir)
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} files, filtering by duration...")
    
    valid_files = []
    for audio_file in tqdm(audio_files):
        try:
            duration = audio_utils.get_audio_duration(str(audio_file))
            if duration >= min_duration:
                valid_files.append(audio_file)
        except:
            pass
    
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        for audio_file in valid_files:
            f.write(f"{str(audio_file)}\n")
    
    print(f"✓ Created manifest: {output_file} ({len(valid_files)} files)")
    return len(valid_files)

def split_manifest(input_manifest, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with open(input_manifest, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    random.shuffle(files)
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)
    
    splits = {
        'train.txt': files[:n_train],
        'val.txt': files[n_train:n_train + n_val],
        'test.txt': files[n_train + n_val:]
    }
    
    manifest_dir = Path(input_manifest).parent
    for filename, file_list in splits.items():
        with open(manifest_dir / filename, 'w') as f:
            f.writelines([f"{p}\n" for p in file_list])
        print(f"✓ {filename}: {len(file_list)} files")

def download_librispeech(output_dir, subsets=['train-clean-100', 'dev-clean', 'test-clean']):
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://www.openslr.org/resources/12"
    
    for subset in subsets:
        tar_file = f"{subset}.tar.gz"
        url = f"{base_url}/{tar_file}"
        print(f"\nDownloading {subset}...")
        
        try:
            subprocess.run(['wget', '-P', output_dir, url], check=True)
            print(f"Extracting {subset}...")
            subprocess.run(['tar', '-xzf', f"{output_dir}/{tar_file}", '-C', output_dir], check=True)
            os.remove(f"{output_dir}/{tar_file}")
            print(f"✓ {subset} ready")
        except:
            print(f"✗ Failed: {subset}. Download manually from {url}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['custom', 'librispeech', 'split'])
    parser.add_argument('--audio_dir', type=str)
    parser.add_argument('--output_manifest', type=str)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--data_dir', type=str, default=config.RAW_DATA_DIR)
    parser.add_argument('--manifest_dir', type=str, default=config.MANIFEST_DIR)
    parser.add_argument('--input_manifest', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()
    
    if args.mode == 'custom':
        create_manifest(args.audio_dir, args.output_manifest)
    elif args.mode == 'librispeech':
        if args.download:
            download_librispeech(args.data_dir)
        librispeech_dir = Path(args.data_dir) / 'LibriSpeech'
        for subset_name, manifest_name in [('train-clean-100', 'train.txt'),
                                           ('dev-clean', 'val.txt'),
                                           ('test-clean', 'test.txt')]:
            subset_dir = librispeech_dir / subset_name
            if subset_dir.exists():
                create_manifest(subset_dir, Path(args.manifest_dir) / manifest_name)
    elif args.mode == 'split':
        split_manifest(args.input_manifest, args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main()