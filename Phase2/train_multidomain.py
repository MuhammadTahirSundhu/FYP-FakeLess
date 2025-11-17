# ===========================================================================
# FILE: train_multidomain.py
# ===========================================================================
"""
Main Training Script for Multi-Domain Network
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import audio_utils
from models import MultiDomainPerturbationNetwork, save_checkpoint, load_checkpoint, count_parameters
from dataset import create_dataloader

def train_epoch(model, dataloader, optimizer, epoch, writer, device, asv_embedder, vocoder):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    
    for batch_idx, batch in enumerate(pbar):
        waveform = batch['waveform'].to(device)
        mel_spec = batch['mel'].to(device)
        
        # Forward
        delta_mel, delta_time = model(waveform, mel_spec)
        protected_mel = mel_spec + delta_mel
        protected_wave = waveform + delta_time
        
        # Attack loss
        if asv_embedder and asv_embedder.available:
            with torch.no_grad():
                if vocoder and vocoder.model:
                    protected_audio = vocoder.mel_to_audio(protected_mel)
                else:
                    protected_audio = protected_wave.squeeze(1)
                
                orig_emb = asv_embedder.extract_batch(waveform.squeeze(1))
                prot_emb = asv_embedder.extract_batch(protected_audio)
            
            if orig_emb is not None and prot_emb is not None:
                attack_loss = F.cosine_similarity(orig_emb, prot_emb).mean()
            else:
                attack_loss = -torch.mean(delta_mel ** 2)
        else:
            attack_loss = -torch.mean(delta_mel ** 2)
        
        # Quality loss
        quality_loss = F.mse_loss(protected_mel, mel_spec)
        
        # Regularization
        reg_loss = torch.mean(delta_mel ** 2) + torch.mean(delta_time ** 2)
        
        # Psychoacoustic loss
        if config.LAMBDA_PSYCHO > 0:
            original_energy = torch.abs(mel_spec)
            threshold = original_energy * 0.1
            excess = F.relu(torch.abs(delta_mel) - threshold)
            psycho_loss = torch.mean(excess ** 2)
        else:
            psycho_loss = 0.0
        
        # Total loss
        loss = (config.LAMBDA_ATTACK * attack_loss +
                config.LAMBDA_QUALITY * quality_loss +
                config.LAMBDA_REG * reg_loss +
                config.LAMBDA_PSYCHO * psycho_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'attack': f'{attack_loss.item():.4f}',
            'quality': f'{quality_loss.item():.4f}'
        })
        
        # Logging
        if batch_idx % config.LOG_INTERVAL == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/loss', loss.item(), step)
            writer.add_scalar('Train/attack', attack_loss.item(), step)
            writer.add_scalar('Train/quality', quality_loss.item(), step)
            writer.add_scalar('Train/reg', reg_loss.item(), step)
            if isinstance(psycho_loss, torch.Tensor):
                writer.add_scalar('Train/psycho', psycho_loss.item(), step)
    
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validating"):
        waveform = batch['waveform'].to(device)
        mel_spec = batch['mel'].to(device)
        
        delta_mel, _ = model(waveform, mel_spec)
        protected_mel = mel_spec + delta_mel
        
        loss = F.mse_loss(protected_mel, mel_spec)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_manifest', type=str, default=config.TRAIN_MANIFEST)
    parser.add_argument('--val_manifest', type=str, default=config.VAL_MANIFEST)
    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument('--log_dir', type=str, default=config.LOG_DIR)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(config.DEVICE)
    config.set_seed()
    
    print("=" * 70)
    print("FAKELESS PHASE 2 - TRAINING")
    print("=" * 70)
    config.print_config()
    
    # Data
    print("\nLoading data...")
    train_loader = create_dataloader(args.train_manifest, args.batch_size, shuffle=True, augment=True)
    val_loader = create_dataloader(args.val_manifest, args.batch_size, shuffle=False, augment=False)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    
    # Model
    print("\nInitializing model...")
    model = MultiDomainPerturbationNetwork().to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Auxiliary models
    print("\nLoading auxiliary models...")
    asv_embedder = audio_utils.ASVEmbedder(device)
    vocoder = audio_utils.HiFiGANVocoder(device=device)
    
    # Training
    writer = SummaryWriter(args.log_dir)
    best_val_loss = float('inf')
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, writer, device, asv_embedder, vocoder)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        writer.add_scalar('Epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch, path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    writer.close()
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Best val loss: {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()