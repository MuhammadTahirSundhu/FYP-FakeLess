"""Convert a PyTorch MelGAN checkpoint to ONNX -> SavedModel -> TFLite.

This is a conversion scaffold you can run locally. Many public MelGAN
checkpoints provide only weights; to export to ONNX we need the generator
architecture. This script provides the plumbing and a small template where
you should supply or import the generator class used by the checkpoint.

Usage (example):
  # install required packages first (run locally)
  pip install torch onnx onnxruntime onnx-tf huggingface-hub safetensors tensorflow

  # then run (example using HF repo "amupd/melgan")
  python Code\convert_pytorch_melgan.py --hf_id amupd/melgan --filename pytorch_model.bin --out_dir saved_models/vocoder --onnx_path build/melgan.onnx --tflite_path mobile_app_fakeless/assets/models/vocoder_melgan.tflite --n_mels 80

Notes:
 - You must provide a working PyTorch generator class matching the checkpoint
   architecture. Edit `load_generator_from_state_dict()` below to return a
   ready-to-export nn.Module with loaded weights.
 - The script will export ONNX with a dynamic time axis (T) and then call
   the repository's ONNX->SavedModel->TFLite helpers (if available).
 - If ONNX->TFLite conversion fails due to unsupported ops, try different
   opset versions or convert using a TF Select (flex) build.
"""

import os
import argparse
import sys


def download_from_hf(hf_id, filename=None):
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:
        raise RuntimeError('Please install huggingface-hub: pip install huggingface-hub') from e

    if filename:
        local = hf_hub_download(repo_id=hf_id, filename=filename)
        return local

    # try listing repo files and pick a likely checkpoint
    files = list_repo_files(hf_id)
    candidates = [f for f in files if f.endswith(('.pt', '.pth', '.bin', '.safetensors'))]
    if not candidates:
        raise RuntimeError('No checkpoint-like files found in the repo; provide --filename')
    # pick first
    return hf_hub_download(repo_id=hf_id, filename=candidates[0])


def load_generator_from_state_dict(state_dict, n_mels=80):
    """TODO: Implement this function for your checkpoint.

    You must return a torch.nn.Module instance (generator) with loaded weights.
    Example skeleton (you must replace with the actual architecture):

    import torch.nn as nn

    class DummyGen(nn.Module):
        def __init__(self, n_mels):
            super().__init__()
            # ... define layers
        def forward(self, x):
            # x: [B, n_mels, T]
            return x

    gen = DummyGen(n_mels)
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen

    If your checkpoint stores a nested state (e.g. {'generator': {...}}) extract
    the generator sub-dict before passing it here.
    """
    raise NotImplementedError('Please implement load_generator_from_state_dict() to construct your generator architecture and load weights')


def export_onnx(model, onnx_path, n_mels=80, opset=11):
    import torch
    model.eval()
    # example input: [1, n_mels, T] where T is dynamic; use a small T for tracing
    example_T = 16
    example = torch.randn(1, n_mels, example_T, dtype=torch.float32)

    dynamic_axes = {
        'mel_in': {2: 'T'},
        'wav_out': {1: 'N'}
    }

    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=['mel_in'],
        output_names=['wav_out'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    print('Exported ONNX to', onnx_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf_id', default=None, help='Hugging Face repo id (optional)')
    p.add_argument('--filename', default=None, help='specific file in HF repo to download (optional)')
    p.add_argument('--local_path', default=None, help='local checkpoint path (if you already downloaded)')
    p.add_argument('--out_dir', default='saved_models/vocoder', help='directory for saved intermediates')
    p.add_argument('--onnx_path', default='build/melgan.onnx', help='where to write ONNX')
    p.add_argument('--tflite_path', default='mobile_app_fakeless/assets/models/vocoder_melgan.tflite', help='where to write final tflite')
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--opset', type=int, default=11)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = args.local_path
    if ckpt_path is None:
        if args.hf_id is None:
            raise RuntimeError('Provide either --local_path or --hf_id (and optional --filename)')
        print('Downloading checkpoint from', args.hf_id)
        ckpt_path = download_from_hf(args.hf_id, filename=args.filename)
        print('Downloaded to', ckpt_path)

    print('Loading checkpoint (torch.load)')
    try:
        import torch
        sd = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        # try safetensors
        try:
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(ckpt_path)
        except Exception:
            raise RuntimeError('Failed to load checkpoint via torch or safetensors') from e

    # If the checkpoint is a dict with nested generator, try common keys
    if isinstance(sd, dict) and 'generator' in sd:
        gen_sd = sd['generator']
    elif isinstance(sd, dict) and 'state_dict' in sd and 'generator' in sd['state_dict']:
        gen_sd = sd['state_dict']['generator']
    else:
        gen_sd = sd

    print('Preparing generator from state dict — you may need to edit the script to match architecture')
    try:
        gen = load_generator_from_state_dict(gen_sd, n_mels=args.n_mels)
    except NotImplementedError as e:
        print(e)
        print('\nThis script requires you to implement `load_generator_from_state_dict()` to construct the generator architecture and load the weights.\n')
        print('I created a scaffold. Please edit the function and re-run the script locally.\n')
        sys.exit(2)

    # Export to ONNX
    print('Exporting generator to ONNX...')
    export_onnx(gen, args.onnx_path, n_mels=args.n_mels, opset=args.opset)

    # Convert ONNX -> SavedModel -> TFLite using existing helpers if available
    try:
        from convert_predictor_to_tflite import onnx_to_savedmodel, savedmodel_to_tflite
        saved_model_dir = os.path.join(args.out_dir, 'vocoder_saved_model')
        print('Converting ONNX -> SavedModel...')
        onnx_to_savedmodel(args.onnx_path, saved_model_dir)
        print('Converting SavedModel -> TFLite...')
        savedmodel_to_tflite(saved_model_dir, args.tflite_path)
        print('Wrote final TFLite to', args.tflite_path)
    except Exception as e:
        print('ONNX->TFLite conversion failed or helpers not found:', e)
        print('ONNX file is at', args.onnx_path, ' — you can convert it manually.')


if __name__ == '__main__':
    main()
