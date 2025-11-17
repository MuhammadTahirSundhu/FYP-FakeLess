"""
convert_predictor_to_tflite.py

Converts the PyTorch `PredictorNet` checkpoint (.pt) to a TFLite model.
Workflow (best-effort):
 1. Load PredictorNet class from `defense_training.py`.
 2. Create an example input tensor with shape [1, N_MELS, T_chunk].
 3. Export the model to ONNX.
 4. Convert ONNX -> TensorFlow SavedModel using onnx-tf.
 5. Convert SavedModel -> TFLite using tflite converter.

Usage:
  python convert_predictor_to_tflite.py --checkpoint checkpoints_phase1/predictor_epoch25.pt --out assets/models/predictor.tflite

Notes and troubleshooting:
 - Requires onnx, onnx-tf and tensorflow installed. See `requirements.txt` in assets/python.
 - If ONNX -> TF conversion fails, try different opset versions (11,12,13).
 - If final TFLite model is too big, enable post-training quantization by passing --quantize.
 - This script assumes `defense_training.py` and `utils_audio.py` are importable from this working directory.

"""

import argparse
import os
import sys
import tempfile

# Insert repository root so defense_training can be imported if this script is run from assets/python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from defense_training import PredictorNet
    from utils_audio import N_MELS
except Exception as e:
    print("Failed to import project modules. Make sure you run this script from the project (assets/python) folder and that defense_training.py and utils_audio.py are available.")
    raise


def export_onnx(pt_path, onnx_path, n_mels, t_chunk=64, opset=11):
    import torch
    model = PredictorNet(n_mels=n_mels)
    state = torch.load(pt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    example = torch.randn(1, n_mels, t_chunk, dtype=torch.float32)

    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=['mel_in'],
        output_names=['delta_out'],
        dynamic_axes={'mel_in': {2: 'T'}, 'delta_out': {2: 'T'}},
        opset_version=opset,
    )
    print(f"ONNX exported to {onnx_path} (opset={opset})")


def onnx_to_savedmodel(onnx_path, saved_model_dir):
    try:
        import onnx
        from onnx_tf.backend import prepare
        onx = onnx.load(onnx_path)
        tf_rep = prepare(onx)
        tf_rep.export_graph(saved_model_dir)
        print(f"SavedModel exported to {saved_model_dir}")
    except Exception as e:
        print("ONNX->SavedModel conversion failed:", e)
        raise


def savedmodel_to_tflite(saved_model_dir, tflite_path, quantize=False):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For integer-only quantization you may need a representative dataset function.
    tflite_model = converter.convert()
    # ensure output directory exists
    out_dir = os.path.dirname(tflite_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model written to {tflite_path}")


def delta_npy_to_savedmodel(npy_path, saved_model_dir, n_mels_expected=None):
    """Create a small SavedModel that returns the universal delta tiled to the
    input time length. The SavedModel signature accepts a mel spectrogram input
    of shape [1, n_mels, T] and outputs a tensor [1, n_mels, T] containing the
    tiled delta.
    """
    import tensorflow as tf
    import numpy as np

    delta = np.load(npy_path)
    if delta.ndim != 2:
        raise ValueError(f"Expected delta with shape [n_mels, T_delta], got {delta.shape}")
    n_mels, delta_T = delta.shape
    if n_mels_expected is not None and n_mels != n_mels_expected:
        print(f"Warning: delta n_mels ({n_mels}) != expected N_MELS ({n_mels_expected})")

    class DeltaModule(tf.Module):
        def __init__(self, delta_np):
            super().__init__()
            # store as tf constant [n_mels, delta_T]
            self.delta = tf.constant(delta_np.astype(np.float32))

        @tf.function(input_signature=[tf.TensorSpec([1, n_mels, None], tf.float32)])
        def __call__(self, mel_in):
            # mel_in: [1, n_mels, T]
            T = tf.shape(mel_in)[2]
            delta = self.delta  # [n_mels, delta_T]
            # make [1, n_mels, delta_T]
            delta_3 = tf.expand_dims(delta, axis=0)
            # compute how many repeats needed along time
            reps = tf.math.floordiv(T + delta_T - 1, delta_T)
            # tile along time axis
            tiled = tf.tile(delta_3, [1, 1, reps])
            out = tiled[:, :, :T]
            return out

    mod = DeltaModule(delta)
    # save the model
    tf.saved_model.save(mod, saved_model_dir, signatures=mod.__call__.get_concrete_function())
    print(f"SavedModel (delta) exported to {saved_model_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='output .tflite path (e.g. assets/models/predictor.tflite)')
    p.add_argument('--t_chunk', type=int, default=64)
    p.add_argument('--opset', type=int, default=11)
    p.add_argument('--quantize', action='store_true')
    args = p.parse_args()

    checkpoint_path = args.checkpoint
    out_path = args.out
    t_chunk = args.t_chunk

    n_mels = N_MELS

    tmpdir = tempfile.mkdtemp(prefix='convert_tmp_')
    onnx_path = os.path.join(tmpdir, 'predictor.onnx')
    saved_model_dir = os.path.join(tmpdir, 'saved_model')

    # dispatch based on file extension
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == '.npy':
        # convert delta numpy -> SavedModel -> TFLite
        try:
            delta_npy_to_savedmodel(checkpoint_path, saved_model_dir, n_mels_expected=n_mels)
        except Exception as e:
            print('Delta .npy -> SavedModel failed:', e)
            sys.exit(2)

        try:
            savedmodel_to_tflite(saved_model_dir, out_path, quantize=args.quantize)
        except Exception as e:
            print('SavedModel->TFLite failed:', e)
            sys.exit(3)

        print('Delta conversion finished successfully.')
    elif ext in ('.pt', '.pth'):
        # convert predictor .pt -> ONNX -> SavedModel -> TFLite
        try:
            export_onnx(checkpoint_path, onnx_path, n_mels, t_chunk=t_chunk, opset=args.opset)
        except Exception as e:
            print('Failed to export ONNX (is this a predictor .pt?):', e)
            print('Try a different opset with --opset 12 or 13')
            sys.exit(4)

        try:
            onnx_to_savedmodel(onnx_path, saved_model_dir)
        except Exception as e:
            print('ONNX->SavedModel failed:', e)
            sys.exit(5)

        try:
            savedmodel_to_tflite(saved_model_dir, out_path, quantize=args.quantize)
        except Exception as e:
            print('SavedModel->TFLite failed:', e)
            sys.exit(6)

        print('Predictor conversion finished successfully.')
    else:
        print(f'Unsupported checkpoint extension: {ext}. Provide a .pt predictor or .npy universal delta.')
        sys.exit(7)


if __name__ == '__main__':
    main()
