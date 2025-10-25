"""
prepare_librispeech.py

Prepares LibriSpeech dataset for FYP Phase-1:
- Extracts tar.gz archives from a raw download folder
- Converts to 16 kHz mono WAV
- Creates manifest files for train/dev/test subsets

Usage:
    python prepare_librispeech.py --root /path/to/raw_downloads --output data/librispeech_prepared
Initially we are using on libreSpeech but we will make our custome dataset as well

"""

import os
import argparse
import tarfile
import torchaudio
from glob import glob
from tqdm import tqdm

def extract_archives(root, output):
    """Extract all .tar.gz files in root into output dir."""
    archives = glob(os.path.join(root, "*.tar.gz"))
    extracted_dirs = []
    for arch in archives:
        target_dir = os.path.join(output, os.path.basename(arch).replace(".tar.gz", ""))
        if not os.path.exists(target_dir):
            print(f"[INFO] Extracting {arch} to {output} ...")
            with tarfile.open(arch, "r:gz") as tar:
                tar.extractall(output)
        else:
            print(f"[SKIP] Already extracted: {arch}")
        print(target_dir)
        extracted_dirs.append(target_dir)
    return extracted_dirs

def convert_to_wav(input_path, output_path, sr=16000):
    """Convert audio to mono 16kHz WAV."""
    if os.path.exists(output_path):
        return
    wav, in_sr = torchaudio.load(input_path)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    if wav.shape[0] > 1:  # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav, sr)

def create_manifest(wav_files, manifest_path):
    """Write manifest listing all wav file paths."""
    with open(manifest_path, "w") as f:
        for w in wav_files:
            f.write(os.path.abspath(w) + "\n")
    print(f"[INFO] Wrote manifest: {manifest_path} ({len(wav_files)} files)")

def process_subset(subset_dir, subset_name, out_dir, sr=16000):
    """Convert FLAC to WAV and create manifest for one subset."""
    print(f"[INFO] Processing {subset_name} ...")
    wav_files = []
    flac_files = glob(os.path.join(subset_dir, "**/*.flac"), recursive=True)
    for flac in tqdm(flac_files):
        rel_path = os.path.relpath(flac, subset_dir)
        out_path = os.path.join(out_dir, subset_name, rel_path).replace(".flac", ".wav")
        convert_to_wav(flac, out_path, sr)
        wav_files.append(out_path)

    manifest_path = os.path.join(out_dir, f"{subset_name}_manifest.txt")
    create_manifest(wav_files, manifest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Folder containing raw .tar.gz files")
    parser.add_argument("--output", type=str, default="data/librispeech_prepared", help="Output dir for WAV + manifests")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Step 1: extract into output folder
    extracted_dirs = extract_archives(args.root, args.output)

    # Step 2: process required subsets only
    subsets = ["train-clean-100", "dev-clean", "test-clean"]
    for s in subsets:
        subset_dir = os.path.join(args.output, "LibriSpeech", s)
        if os.path.exists(subset_dir):
            process_subset(subset_dir, s, args.output, sr=16000)
        else:
            print(f"[WARN] Subset {s} not found in extracted dirs. Did you download it?")
    
    print("[DONE] LibriSpeech preprocessing complete.")
