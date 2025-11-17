"""Build small SavedModels/TFLite models for:
 - wav -> log-mel (mel_extractor)
 - log-mel -> wav (mel_to_wave) using Griffin-Lim fallback

These are small helper models intended to be converted to TFLite and run from Flutter.

Usage (PowerShell):
 python Code\build_mel_models.py --out_dir saved_models --export_tflite --mobile_assets mobile_app_fakeless/assets/models

Note: TFLite conversion may fail on some ops (pinv, stft) depending on your TF/TFLite build.
If conversion fails, the script will save SavedModels and print guidance.
"""

import os
import argparse
import math
import tensorflow as tf
import numpy as np


class MelExtractor(tf.Module):
    def __init__(self, sr=16000, n_fft=1024, hop=256, n_mels=80):
        super().__init__()
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_mels = int(n_mels)
        self.fft_bins = self.n_fft // 2 + 1
        # Precompute mel matrix as float32
        # Use positional args for compatibility with older TF versions
        mel_mat = tf.signal.linear_to_mel_weight_matrix(
            self.n_mels,
            self.fft_bins,
            self.sr,
            0.0,
            float(self.sr // 2),
        )
        self.mel_mat = tf.cast(mel_mat, tf.float32)  # shape [fft_bins, n_mels]

    @tf.function(input_signature=[tf.TensorSpec([1, None], tf.float32, name="waveform")])
    def __call__(self, waveform):
        # waveform: [1, N]
        waveform = tf.reshape(waveform, [-1])
        # stft -> [T, fft_bins]
        stft = tf.signal.stft(waveform, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft, window_fn=tf.signal.hann_window)
        mag = tf.abs(stft)
        # mel: [T, n_mels] = mag @ mel_mat
        mel = tf.matmul(mag, self.mel_mat)
        # log mel: transpose to [n_mels, T], add batch dim -> [1, n_mels, T]
        log_mel = tf.math.log(tf.maximum(mel, 1e-6))
        log_mel = tf.transpose(log_mel)  # [n_mels, T]
        out = tf.expand_dims(log_mel, 0)
        return out


class MelToWave(tf.Module):
    def __init__(self, sr=16000, n_fft=1024, hop=256, n_mels=80, n_iters=32):
        super().__init__()
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_mels = int(n_mels)
        self.n_iters = int(n_iters)
        self.fft_bins = self.n_fft // 2 + 1
        # Use positional args for compatibility with older TF versions
        mel_mat = tf.signal.linear_to_mel_weight_matrix(
            self.n_mels,
            self.fft_bins,
            self.sr,
            0.0,
            float(self.sr // 2),
        )
        # mel_mat: [fft_bins, n_mels]
        self.mel_mat = tf.cast(mel_mat, tf.float32)
        # pinv: [n_mels, fft_bins]
        try:
            self.pinv = tf.linalg.pinv(self.mel_mat)
        except Exception:
            # fallback to transpose if pinv unsupported
            self.pinv = tf.transpose(self.mel_mat)

    @tf.function(input_signature=[tf.TensorSpec([1, None, None], tf.float32, name="log_mel")])
    def __call__(self, log_mel):
        # log_mel: [1, n_mels, T]
        log_mel = tf.squeeze(log_mel, axis=0)  # [n_mels, T]
        mel_T = tf.transpose(log_mel)  # [T, n_mels]
        mag_est = tf.matmul(mel_T, self.pinv)  # [T, fft_bins]
        # Griffin-Lim iterative phase reconstruction
        # initialize zero phase
        phase = tf.zeros_like(mag_est, dtype=tf.float32)
        mag = tf.cast(mag_est, tf.complex64)

        def loop_body(i, phase, _):
            complex_spec = tf.cast(mag_est, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
            waveform = tf.signal.inverse_stft(complex_spec, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft, window_fn=tf.signal.hann_window)
            stft_recon = tf.signal.stft(waveform, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft, window_fn=tf.signal.hann_window)
            phase = tf.math.angle(stft_recon)
            return i + 1, phase, waveform

        i = tf.constant(0)
        recon = tf.zeros([1], dtype=tf.float32)
        cond = lambda i, phase, recon: tf.less(i, self.n_iters)
        _, phase, recon = tf.while_loop(cond, loop_body, loop_vars=[i, phase, recon], shape_invariants=[i.get_shape(), tf.TensorShape([None, self.fft_bins]), tf.TensorShape([None])], maximum_iterations=self.n_iters)

        recon = tf.reshape(recon, [1, -1])
        return recon


def save_savedmodel(module, out_dir, name):
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    tf.saved_model.save(module, path, signatures={'serving_default': module.__call__.get_concrete_function()})
    print(f"SavedModel written to {path}")
    return path


def convert_to_tflite(saved_model_dir, tflite_path):
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        open(tflite_path, 'wb').write(tflite_model)
        print(f"Wrote TFLite to {tflite_path}")
        return True
    except Exception as e:
        print("TFLite conversion failed:", e)
        print("SavedModel remains at", saved_model_dir)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='saved_models', help='directory to write SavedModels')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--n_iters', type=int, default=32, help='Griffin-Lim iterations for mel->wav')
    parser.add_argument('--export_tflite', action='store_true', help='Also convert SavedModels to TFLite and copy to mobile assets')
    parser.add_argument('--mobile_assets', default='mobile_app_fakeless/assets/models', help='mobile assets dir to write tflite files')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Building mel_extractor SavedModel...')
    mel_ex = MelExtractor(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels)
    mel_ex_dir = save_savedmodel(mel_ex, args.out_dir, 'mel_extractor')

    print('Building mel_to_wave SavedModel (Griffin-Lim)...')
    m2w = MelToWave(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels, n_iters=args.n_iters)
    m2w_dir = save_savedmodel(m2w, args.out_dir, 'mel_to_wave')

    if args.export_tflite:
        print('Converting mel_extractor to TFLite...')
        ok1 = convert_to_tflite(mel_ex_dir, os.path.join(args.mobile_assets, 'mel_extractor.tflite'))
        print('Converting mel_to_wave to TFLite...')
        ok2 = convert_to_tflite(m2w_dir, os.path.join(args.mobile_assets, 'mel_to_wave.tflite'))
        if not (ok1 and ok2):
            print('\nOne or more TFLite conversions failed. This commonly happens when TFLite does not support some TF ops (e.g. pinv or certain tf.signal ops) in your TF/TFLite build.')
            print('If conversion fails, you can still use the SavedModel with TensorFlow (desktop) or consider building a mel-extractor with simpler ops or using a native DSP plugin in Flutter.')


if __name__ == '__main__':
    main()
