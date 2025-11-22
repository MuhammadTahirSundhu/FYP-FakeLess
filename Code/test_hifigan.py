"""
Test HiFiGAN vs Griffin-Lim reconstruction

Usage:
    python test_hifigan.py --input test.wav

Outputs saved to `demo_outputs/`:
    - orig.wav (copied original)
    - mel_librosa.npy (librosa mel dB)
    - mel_vocoder.npy (vocoder mel tensor if available)
    - recon_griffin.wav
    - recon_hifigan.wav

Prints SNRs for both reconstructions.
"""
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch

from pathlib import Path

# Import components from the main file
try:
    from defense_trainingv2 import AdvancedAudioProcessor, HiFiGANVocoder
except Exception as e:
    # If import fails, we still try to use librosa-only path
    AdvancedAudioProcessor = None
    HiFiGANVocoder = None
    print(f"[WARN] Could not import from defense_trainingv2: {e}")


def compute_snr(orig, recon):
    noise = recon - orig
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--outdir', default='hifigan_test_outputs')
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load original
    wav, sr = librosa.load(str(inp), sr=16000)
    sf.write(outdir / 'orig.wav', wav.astype(np.float32), 16000)

    # Librosa mel (power -> dB)
    mel_power = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, n_fft=1024, hop_length=256, power=2.0)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    np.save(outdir / 'mel_librosa_db.npy', mel_db)
    print(f"librosa mel_db: shape={mel_db.shape} min={mel_db.min():.3f} max={mel_db.max():.3f} mean={mel_db.mean():.3f}")

    # Griffin-Lim reconstruction from librosa mel
    try:
        wav_gl = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=1024, hop_length=256, n_iter=64)
        sf.write(outdir / 'recon_griffin.wav', wav_gl.astype(np.float32), sr)
        snr_gl = compute_snr(wav[:min(len(wav), len(wav_gl))], wav_gl[:min(len(wav), len(wav_gl))])
        print(f"Griffin-Lim recon saved: recon_griffin.wav  SNR={snr_gl:.2f} dB")
    except Exception as e:
        print(f"[WARN] Griffin-Lim reconstruction failed: {e}")
        wav_gl = None

    # If AdvancedAudioProcessor is available, try vocoder mel and HiFiGAN
    wav_hifi = None
    try:
        if AdvancedAudioProcessor is not None:
            proc = AdvancedAudioProcessor(sr=16000, n_fft=1024, hop_length=256, n_mels=80)
            mel_v = proc.wav_to_mel(wav)
            # Save mel_v
            if isinstance(mel_v, torch.Tensor):
                mel_v_np = mel_v.detach().cpu().numpy()
            else:
                mel_v_np = np.array(mel_v)
            np.save(outdir / 'mel_vocoder.npy', mel_v_np)
            print(f"vocoder mel: shape={mel_v_np.shape} min={mel_v_np.min():.3f} max={mel_v_np.max():.3f} mean={mel_v_np.mean():.3f}")

            # Try HiFiGAN reconstruction via proc.mel_to_wav
            try:
                # Try several variants to discover the mel format expected by HiFiGAN
                def try_variant(mel_input, name):
                    try:
                        wav_try = proc.mel_to_wav(mel_input)
                        outname = outdir / f'recon_hifigan_{name}.wav'
                        sf.write(outname, np.asarray(wav_try, dtype=np.float32), sr)
                        snr_try = compute_snr(wav[:min(len(wav), len(wav_try))], wav_try[:min(len(wav), len(wav_try))])
                        print(f"HiFiGAN recon saved: {outname.name}  SNR={snr_try:.2f} dB")
                    except Exception as e:
                        print(f"[WARN] HiFiGAN ({name}) failed: {e}")

                # Variant A: pass mel as returned (tensor or numpy)
                try_variant(mel_v, 'as_is')

                # Variant B: if mel looks like dB (negative values), convert to power
                try:
                    if isinstance(mel_v, torch.Tensor):
                        mel_np = mel_v.detach().cpu().numpy()
                    else:
                        mel_np = np.array(mel_v)

                    if mel_np.min() < 0:
                        # convert dB -> power
                        mel_power = librosa.db_to_power(mel_np)
                        try_variant(mel_power, 'db_to_power')

                        # Variant C: normalize by max (make max 0 dB) then convert
                        mel_db_norm = mel_np - mel_np.max()
                        mel_power_norm = librosa.db_to_power(mel_db_norm)
                        try_variant(mel_power_norm, 'db_norm_to_power')
                    else:
                        # If already non-negative, try a direct power input
                        try_variant(mel_np, 'direct_power')
                except Exception as e:
                    print(f"[WARN] mel conversion trials failed: {e}")
            except Exception as e:
                print(f"[WARN] HiFiGAN mel_to_wav failed: {e}")

        else:
            print('[WARN] AdvancedAudioProcessor not available; skipping HiFiGAN test')
    except Exception as e:
        print(f"[WARN] vocoder path failed: {e}")

    print('\nFiles saved to', outdir)

if __name__ == '__main__':
    main()
