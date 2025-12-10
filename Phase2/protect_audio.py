# ===========================================================================
# FILE: protect_audio.py
# ===========================================================================
"""
Audio Protection Script (Inference)
"""

import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

import config
import audio_utils
from models import MultiDomainPerturbationNetwork, load_checkpoint

class AudioProtector:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device or config.DEVICE)
        
        print(f"Loading model from {model_path}...")
        self.model = MultiDomainPerturbationNetwork().to(self.device)
        load_checkpoint(self.model, None, model_path)
        self.model.eval()
        
        print("Loading vocoder...")
        self.vocoder = audio_utils.HiFiGANVocoder(device=self.device)
        
        print("✓ Audio protector ready\n")
    
    def protect_file(self, input_path, output_path, epsilon=None):
        epsilon = epsilon or config.EPSILON
        
        print(f"Processing: {input_path}")
        
        wav = audio_utils.load_audio(input_path)
        mel = audio_utils.wav_to_mel(wav)
        wav = audio_utils.normalize_audio(wav)
        
        waveform_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(self.device)
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            protected_mel, _, _, _ = self.model.apply_perturbation(waveform_t, mel_t, epsilon)
        
        protected_audio = self.vocoder.mel_to_audio(protected_mel)
        
        if isinstance(protected_audio, torch.Tensor):
            protected_audio = protected_audio.cpu().numpy().squeeze()
        
        os.makedirs(Path(output_path).parent, exist_ok=True)
        audio_utils.save_audio(output_path, protected_audio)
        
        print(f"✓ Saved: {output_path}\n")
        return protected_audio
    
    def protect_directory(self, input_dir, output_dir, epsilon=None):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = list(input_dir.rglob('*.wav'))
        audio_files.extend(input_dir.rglob('*.flac'))
        audio_files.extend(input_dir.rglob('*.mp3'))
        
        print(f"Found {len(audio_files)} audio files\n")
        
        for audio_file in tqdm(audio_files, desc="Protecting audio"):
            rel_path = audio_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.wav')
            
            try:
                self.protect_file(str(audio_file), str(output_path), epsilon)
            except Exception as e:
                print(f"✗ Failed: {audio_file} - {e}")

def main():
    parser = argparse.ArgumentParser(description="Protect audio from deepfakes")
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output file or directory')
    parser.add_argument('--model', default='checkpoints/best_model.pt', help='Model checkpoint')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON, help='Perturbation strength')
    parser.add_argument('--device', default=config.DEVICE)
    args = parser.parse_args()
    
    protector = AudioProtector(args.model, args.device)
    
    if os.path.isfile(args.input):
        protector.protect_file(args.input, args.output, args.epsilon)
    elif os.path.isdir(args.input):
        protector.protect_directory(args.input, args.output, args.epsilon)
    else:
        print(f"Error: {args.input} not found")

if __name__ == "__main__":
    main()