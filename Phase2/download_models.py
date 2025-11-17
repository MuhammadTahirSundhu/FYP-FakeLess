# ===========================================================================
# FILE: download_models.py
# ===========================================================================
"""
Download Pretrained Models
"""

import os
import torch
import config

def download_hifigan():
    print("Downloading HiFi-GAN vocoder...")
    try:
        model = torch.hub.load('jik876/hifi-gan', 'generator', 'hifigan_universal_v1')
        os.makedirs(os.path.dirname(config.HIFIGAN_CHECKPOINT), exist_ok=True)
        torch.save(model.state_dict(), config.HIFIGAN_CHECKPOINT)
        print("✓ HiFi-GAN downloaded")
    except Exception as e:
        print(f"✗ Failed: {e}")

def download_speechbrain():
    print("Downloading SpeechBrain ECAPA-TDNN...")
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=config.ASV_CHECKPOINT_DIR,
            fetching_strategy="LocalStrategy"  # Use file copying instead of symlinks
        )
        print("✓ SpeechBrain model downloaded")
    except Exception as e:
        print(f"✗ Failed: {e}")
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['hifigan', 'speechbrain', 'all'], default='all')
    args = parser.parse_args()
    
    if args.model in ['hifigan', 'all']:
        download_hifigan()
    if args.model in ['speechbrain', 'all']:
        download_speechbrain()

if __name__ == "__main__":
    main()
