"""Download TensorSpeech MB-MelGAN pretrained and convert to SavedModel/TFLite.

This script uses tensorflow_tts TFAutoModel.from_pretrained to load the model
and wraps it so the SavedModel signature accepts mel spectrograms in the
shape [1, n_mels, T] (same layout as our mel_extractor), transposes to
[1, T, n_mels], runs the vocoder, and returns waveform [1, N].

Usage:
 python Code\convert_melgan_to_tflite.py --hf_id tensorspeech/tts-mb_melgan-ljspeech-en --out_dir saved_models --export_tflite --mobile_assets mobile_app_fakeless/assets/models
"""

import os
import argparse
import tensorflow as tf


def build_wrapper(model, n_mels=80):
    class VocoderWrapper(tf.Module):
        def __init__(self, model, n_mels):
            super().__init__()
            self.model = model
            self.n_mels = n_mels

        @tf.function(input_signature=[tf.TensorSpec([1, None, None], tf.float32, name="log_mel")])
        def __call__(self, log_mel):
            # Accept [1, n_mels, T], convert to [1, T, n_mels]
            mel = tf.transpose(tf.squeeze(log_mel, axis=0))
            mel = tf.expand_dims(mel, 0)  # [1, T, n_mels]
            # call model.inference if available, otherwise call the model directly
            if hasattr(self.model, 'inference'):
                out = self.model.inference(mel)
            else:
                # assume Keras model or callable
                out = self.model(mel)
            # many TF-TTS models return [1, N, 1] or [1, N]
            wav = out[0]
            # ensure shape [1, N]
            wav = tf.reshape(wav, [1, -1])
            return wav

    return VocoderWrapper(model, n_mels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_id', default='tensorspeech/tts-mb_melgan-ljspeech-en', help='HuggingFace id for pretrained melgan')
    parser.add_argument('--out_dir', default='saved_models/vocoder', help='directory to write SavedModel')
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--filename', default=None, help='specific filename in the HF repo to download (optional)')
    parser.add_argument('--export_tflite', action='store_true')
    parser.add_argument('--mobile_assets', default='mobile_app_fakeless/assets/models')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Attempting to download vocoder artifacts from Hugging Face repo:', args.hf_id)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception:
        raise RuntimeError('huggingface_hub not installed. Please pip install huggingface-hub and retry')

    # Helper: try to find a likely vocoder file in the repo
    candidates = []
    if args.filename:
        candidates = [args.filename]
    else:
        # common filenames that might be present in HF repo
        candidates = [
            'vocoder.tflite', 'vocoder_melgan.tflite', 'model.tflite',
            'generator.onnx', 'model.onnx', 'melgan.onnx',
            'saved_model.tar.gz', 'saved_model.zip', 'saved_model',
            'mb_melgan.pth', 'generator.pth', 'model.pth'
        ]

    repo_files = []
    try:
        repo_files = list_repo_files(args.hf_id)
    except Exception:
        # if listing fails, we'll still try to download candidates directly
        repo_files = []

    found = None
    for c in candidates:
        if c in repo_files or args.filename:
            try:
                print('Trying to download', c)
                local_path = hf_hub_download(repo_id=args.hf_id, filename=c)
                found = local_path
                print('Downloaded', local_path)
                break
            except Exception:
                # try next
                found = None
                continue

    if found is None:
        # try wildcard search in repo_files
        for f in repo_files:
            if any(f.lower().endswith(ext) for ext in ('.tflite', '.onnx', '.pth', '.pt', '.h5', '.zip', '.tar.gz')):
                try:
                    print('Attempting to download', f)
                    local_path = hf_hub_download(repo_id=args.hf_id, filename=f)
                    found = local_path
                    print('Downloaded', local_path)
                    break
                except Exception:
                    continue

    if found is None:
        raise RuntimeError('Could not find a suitable vocoder artifact in the repo. Provide --filename or upload a TFLite/ONNX/SavedModel in the HF repo.')

    # If the file is already a .tflite, copy into mobile assets
    fname = os.path.basename(found)
    ext = os.path.splitext(fname)[1].lower()
    if ext == '.tflite':
        dest = os.path.join(args.mobile_assets, fname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        import shutil
        shutil.copy(found, dest)
        print('Copied TFLite vocoder to', dest)
        return

    # If ONNX, use existing conversion utilities from convert_predictor_to_tflite
    if ext == '.onnx':
        try:
            from convert_predictor_to_tflite import onnx_to_savedmodel, savedmodel_to_tflite
        except Exception:
            raise RuntimeError('Conversion utilities not available (convert_predictor_to_tflite).')

        tmp_saved = os.path.join(args.out_dir, 'from_onnx_savedmodel')
        os.makedirs(tmp_saved, exist_ok=True)
        print('Converting ONNX -> SavedModel...')
        onnx_to_savedmodel(found, tmp_saved)
        if args.export_tflite:
            tflite_path = os.path.join(args.mobile_assets, 'vocoder_melgan.tflite')
            os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
            savedmodel_to_tflite(tmp_saved, tflite_path)
        return

    # If it's a Keras .h5 model, try loading and wrapping it
    if ext == '.h5' or fname.lower().endswith('.h5'):
        try:
            print('Loading Keras .h5 model...')
            kmodel = tf.keras.models.load_model(found, compile=False)
        except Exception as e:
            raise RuntimeError(f'Failed to load .h5 Keras model: {e}')

        wrapper = build_wrapper(kmodel, n_mels=args.n_mels)
        print('Saving SavedModel to', args.out_dir)
        tf.saved_model.save(wrapper, args.out_dir, signatures={'serving_default': wrapper.__call__.get_concrete_function()})
        if args.export_tflite:
            tflite_path = os.path.join(args.mobile_assets, 'vocoder_melgan.tflite')
            os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
            try:
                converter = tf.lite.TFLiteConverter.from_saved_model(args.out_dir)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                tflite_model = converter.convert()
                open(tflite_path, 'wb').write(tflite_model)
                print('Wrote TFLite to', tflite_path)
            except Exception as e:
                print('TFLite conversion failed:', e)
                print('SavedModel available at', args.out_dir)
        return

    # If it's a PyTorch checkpoint (.pt/.pth), attempt torch->onnx->tf conversion
    if ext in ('.pt', '.pth'):
        try:
            import torch
        except Exception:
            raise RuntimeError('PyTorch not installed; install torch to convert .pt vocoder checkpoints')

        # Without a generator class this is hard; ask user to provide an ONNX or SavedModel instead
        raise RuntimeError('Downloaded a PyTorch checkpoint. This script cannot reliably convert arbitrary PyTorch vocoder checkpoints without the generator architecture. Provide an ONNX or SavedModel artifact in the HF repo or upload a TFLite directly.')

    # If it's an archive, attempt to extract and look for SavedModel dir or ONNX inside
    if ext in ('.zip', '.gz', '.tar') or fname.endswith('.tar.gz'):
        import shutil, tempfile
        tmpd = tempfile.mkdtemp(prefix='vocoder_unpack_')
        print('Extracting archive to', tmpd)
        try:
            shutil.unpack_archive(found, tmpd)
        except Exception as e:
            print('Failed to extract archive:', e)
            raise

        # search for saved_model or onnx/tflite inside
        candidate = None
        for root, dirs, files in os.walk(tmpd):
            for f in files:
                if f.endswith('.tflite') or f.endswith('.onnx') or f.endswith('.pt') or f.endswith('.pth'):
                    candidate = os.path.join(root, f)
                    break
            if candidate:
                break

        if not candidate:
            # maybe the archive already contains SavedModel directory
            if 'saved_model.pb' in [f for _, _, files in os.walk(tmpd) for f in files]:
                candidate = tmpd

        if not candidate:
            raise RuntimeError('No suitable model artifact found inside archive')

        # recurse by calling main-like behavior
        print('Found inner artifact:', candidate)
        # simple handling: if tflite -> copy; if onnx -> convert; else bail
        inner_ext = os.path.splitext(candidate)[1].lower()
        if inner_ext == '.tflite':
            shutil.copy(candidate, os.path.join(args.mobile_assets, os.path.basename(candidate)))
            print('Copied inner tflite to mobile assets')
            return
        if inner_ext == '.onnx':
            from convert_predictor_to_tflite import onnx_to_savedmodel, savedmodel_to_tflite
            tmp_saved2 = os.path.join(args.out_dir, 'from_archive_savedmodel')
            onnx_to_savedmodel(candidate, tmp_saved2)
            if args.export_tflite:
                savedmodel_to_tflite(tmp_saved2, os.path.join(args.mobile_assets, 'vocoder_melgan.tflite'))
            return

    raise RuntimeError('Unsupported artifact type: ' + ext)


if __name__ == '__main__':
    main()
