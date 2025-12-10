"""
Smoke test to verify `delta` is trainable and updates after an optimizer step.

Usage:
    python smoke_train_delta.py

This script:
 - Instantiates `RobustUniversalDeltaTrainer` with a tiny config
 - Synthesizes a 1s sine waveform
 - Converts to mel, applies `delta`, attempts to compute losses and do one optimizer step
 - If the real loss path fails, performs a fallback toy update on `delta`
 - Prints delta stats before and after and writes `checkpoints_smoke/smoke_delta_after.npy`
"""
import os
import numpy as np
import torch

try:
    from defense_trainingv2 import RobustUniversalDeltaTrainer
except Exception as e:
    print(f"[ERROR] Could not import RobustUniversalDeltaTrainer: {e}")
    raise


def stats(x: np.ndarray):
    return f"min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}"


def main():
    cfg = {
        'sr': 16000,
        'n_fft': 1024,
        'hop_length': 256,
        'n_mels': 80,
        'delta_T': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-3,
        'epochs': 1,
        'batch_size': 1,
        # Loss weights (match defaults from training script if present)
        # Increased attack weight and reduced quality weight for smoke test
        'lambda_attack': 10.0,
        'lambda_quality': 2.0,
        'lambda_reg': 0.05,
    }

    print(f"[INFO] Using device: {cfg['device']}")
    trainer = RobustUniversalDeltaTrainer(cfg)

    # Print initial delta stats
    delta_np_before = trainer.delta.detach().cpu().numpy()
    print("[INFO] Initial delta stats:", stats(delta_np_before))

    # Synthesize a 1s sine wave at 220 Hz
    sr = cfg['sr']
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    wav = 0.05 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    # Convert to mel using the trainer's audio processor
    mel = trainer.audio_proc.wav_to_mel(wav)

    # Normalize/convert mel to torch tensor of shape [B, n_mels, T]
    if isinstance(mel, torch.Tensor):
        mel_t = mel.detach()
        if mel_t.dim() == 2:
            mel_t = mel_t.unsqueeze(0)
    else:
        mel_np = np.array(mel, dtype=np.float32)
        if mel_np.ndim == 2:
            mel_np = mel_np[np.newaxis, ...]
        mel_t = torch.from_numpy(mel_np).float()

    mel_t = mel_t.to(trainer.device)

    # Apply delta (no augmentation)
    mel_prot = trainer.apply_delta(mel_t.clone(), use_augmentation=False)

    # Run multiple smoke training steps to confirm training behaves as expected
    steps = 20
    lam_att = cfg.get('lambda_attack', 1.0)
    lam_qual = cfg.get('lambda_quality', 1.0)
    lam_reg = cfg.get('lambda_reg', 0.0)

    out_dir = 'checkpoints_smoke'
    os.makedirs(out_dir, exist_ok=True)

    # CSV metrics log
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    with open(metrics_path, 'w') as mf:
        mf.write('step,total,attack,quality,reg,grad_norm,delta_norm\n')

    print(f"[INFO] Running {steps} training steps (smoke)...")
    for step in range(1, steps + 1):
        try:
            mel_prot = trainer.apply_delta(mel_t.clone(), use_augmentation=False)

            attack_loss, quality_loss, reg_loss = trainer.compute_losses(mel_t, mel_prot, [wav])
            total_loss = lam_att * attack_loss + lam_qual * quality_loss + lam_reg * reg_loss

            trainer.optimizer.zero_grad()
            total_loss.backward()

            grad_norm = trainer.delta.grad.norm().item() if trainer.delta.grad is not None else 0.0
            trainer.optimizer.step()

            # Project to epsilon if available in config
            eps = cfg.get('epsilon', None)
            if eps is not None:
                with torch.no_grad():
                    trainer.delta.data.clamp_(-eps, eps)

            delta_norm = trainer.delta.detach().norm().item()
            print(f"[STEP {step:02d}] total={total_loss.item():.6f} attack={attack_loss.item():.6f} quality={quality_loss.item():.6f} reg={reg_loss.item():.6f} grad_norm={grad_norm:.6e} delta_norm={delta_norm:.6e}")

            # Append metrics to CSV
            with open(metrics_path, 'a') as mf:
                mf.write(f"{step},{total_loss.item():.6f},{attack_loss.item():.6f},{quality_loss.item():.6f},{reg_loss.item():.6f},{grad_norm:.6e},{delta_norm:.6e}\n")

            # Save intermediate snapshot every 5 steps
            if step % 5 == 0:
                np.save(os.path.join(out_dir, f'smoke_delta_step{step}.npy'), trainer.delta.detach().cpu().numpy())

        except Exception as e:
            print(f"[WARN] step {step} failed: {e}; applying toy L2 update")
            trainer.optimizer.zero_grad()
            toy_loss = torch.mean(trainer.delta ** 2)
            toy_loss.backward()
            if trainer.delta.grad is not None:
                print(f"[INFO] delta.grad stats (toy): min={trainer.delta.grad.min():.6f}, max={trainer.delta.grad.max():.6f}")
            trainer.optimizer.step()

    # Print delta stats after one step
    delta_np_after = trainer.delta.detach().cpu().numpy()
    print("[INFO] Delta stats after one optimizer step:", stats(delta_np_after))

    # Save the delta snapshot
    out_dir = 'checkpoints_smoke'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'smoke_delta_after.npy')
    np.save(out_path, delta_np_after)
    print(f"[INFO] Saved delta snapshot to: {out_path}")


if __name__ == '__main__':
    main()
