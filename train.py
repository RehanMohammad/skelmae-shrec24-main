import math
import random
import numpy as np
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model.sgcn import SGCNModel
from dataset.IPN import IPNDataset as PretrainingDataset, FinetuningDataset 
import os
import os.path as opt
import sys
sys.path.append('./model')
from model.vit import ViT
from model.mae import MAE
from utils.visualize import plot_temporal_heatmap, plot_joint_importance
from model.temporal_attention import AdaptiveTemporal
import warnings
warnings.filterwarnings(
    "ignore",                    # or "once" to show it only once
    category=FutureWarning,
    module=r"timm(\.models)?\.layers"
)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_mae_epoch(epoch, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'[Epoch {epoch}] Training MAE')
    
    for batch in pbar:
        x = batch['Sequence'].to(device)
        A = batch['A_temporal'].to(device).float()
        
        optimizer.zero_grad()
        output = model(x, A)
        loss = output['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / len(pbar)})
    
    return total_loss / len(dataloader)

def validate_mae(model, dataloader, device):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc='Validating MAE')
    
    with torch.no_grad():
        for batch in pbar:
            x = batch['Sequence'].to(device)
            A = batch['A_temporal'].to(device).float()
            
            output = model(x, A)
            total_loss += output['loss'].item()
            pbar.set_postfix({'val_loss': total_loss / len(pbar)})
    
    return total_loss / len(dataloader)

def train_finetune_epoch(epoch, mae, sgcn, train_loader, optimizer, criterion, device):
    mae.eval()        # freeze MAE or keep it eval if only used for edges
    for p in mae.parameters(): p.requires_grad = False

    sgcn.train()
    total_loss = correct = total = 0
    pbar = tqdm(train_loader, desc=f'[Epoch {epoch}] Finetuning')

    for batch in pbar:
        x = batch['Sequence'].to(device)                # [B, 9, T, V]
        y = batch['Label'].to(device)
        A_temporal = batch['A_temporal'].to(device).float()
        A_spatial  = batch['A_spatial'].to(device).float()
        identity   = [A_spatial, A_temporal]

        graph  = x[:, :3].permute(0, 2, 3, 1)                                 
        pred, _, _, _ = sgcn(graph, identity)

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        pbar.set_postfix({'loss': total_loss/len(pbar), 'acc': f'{100.*correct/total:.2f}%'})

    return total_loss/len(train_loader), correct/total

def validate_finetune(mae, sgcn, val_loader, criterion, device):
    mae.eval()
    sgcn.eval()
    total_loss = correct = total = 0
    pbar = tqdm(val_loader, desc='Validation')

    with torch.no_grad():
        for batch in pbar:
            x = batch['Sequence'].to(device)            # [B, 9, T, V]
            y = batch['Label'].to(device)
            A_temporal = batch['A_temporal'].to(device).float()
            A_spatial  = batch['A_spatial'].to(device).float()
            identity   = [A_spatial, A_temporal]

            graph = x[:, :3].permute(0, 2, 3, 1)                        
            pred, _, _, _ = sgcn(graph, identity)

            loss = criterion(pred, y)
            total_loss += loss.item()
            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            pbar.set_postfix({'val_loss': total_loss/len(pbar), 'val_acc': f'{100.*correct/total:.2f}%'})

    return total_loss/len(val_loader), correct/total


def visualize_attention(model, sample, epoch):
    """Save attention visualizations for a sample"""
    model.eval()
    with torch.no_grad():
        x = sample['Sequence'].unsqueeze(0).to(next(model.parameters()).device)
        A = sample['A_temporal'].unsqueeze(0).to(next(model.parameters()).device)
        
        output = model(x, A)
        dynamic_A = output['dynamic_A'].cpu()
        
        # Visualize for first head
        plot_temporal_heatmap(dynamic_A[0], frame_idx=0, head_idx=0)
        plt.savefig(f'attention_epoch{epoch}.png')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg_path', default='ip_configs.yaml', help='Path to the config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.cfg_path)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    
    # Print config
    print('\nCONFIGURATION:')
    print(OmegaConf.to_yaml(cfg))
    
    # Create save directory
    save_dir = opt.join(cfg.save_folder_path, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    weights_dir = opt.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Phase 1: MAE Pretraining
    print('\n' + '='*20, 'PHASE 1: MAE PRETRAINING', '='*20)
    temporal_att = AdaptiveTemporal(dim=3, heads=cfg.mae.num_heads)

    # Create datasets
    train_set = PretrainingDataset(
    data_dir=cfg.data.data_dir,
    ann_file=cfg.data.train_ann,
    max_seq=cfg.mae.sequence_length,
    normalize=cfg.data.normalize,
    temporal_edge_model=temporal_att,    
    edge_threshold=0.5
	)
    val_set = PretrainingDataset(
    data_dir=cfg.data.data_dir,
    ann_file=cfg.data.val_ann,
    max_seq=cfg.mae.sequence_length,
    normalize=cfg.data.normalize,
    temporal_edge_model=temporal_att,   
    edge_threshold=0.5
	)
		
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.mae.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.mae.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize MAE
    encoder = ViT(
    num_nodes=cfg.mae.num_joints * cfg.mae.sequence_length,
    node_dim=3,                    # <-- was 9; Fourier embed expects 3
    num_classes=cfg.mae.coords_dim,
    dim=cfg.mae.encoder_embed_dim,
    depth=cfg.mae.encoder_depth,
    heads=cfg.mae.num_heads,
    mlp_dim=cfg.mae.mlp_dim,
    pool='cls',
    dropout=0.0,
    emb_dropout=0.0
     )

    mae = MAE(
        encoder=encoder,
        decoder_dim=cfg.mae.decoder_dim,
        masking_ratio=cfg.mae.masking_ratio,
        decoder_depth=cfg.mae.decoder_depth,
        decoder_heads=cfg.mae.num_heads
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(mae.parameters(), lr=cfg.mae.lr, weight_decay=cfg.mae.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.mae.num_epochs)
    
    # Train MAE
    best_val_loss = float('inf')
    for epoch in range(1, cfg.mae.num_epochs + 1):
        train_loss = train_mae_epoch(epoch, mae, train_loader, optimizer, device)
        val_loss = validate_mae(mae, val_loader, device)
        scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': mae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, opt.join(weights_dir, 'best_mae.pth'))
        
        # Visualize attention for a sample
        if epoch % 5 == 0:
            sample = val_set[0]
            visualize_attention(mae, sample, epoch)
    
    # Phase 2: SGCN Finetuning
    print('\n' + '='*20, 'PHASE 2: SGCN FINETUNING', '='*20)
    
    # Load best MAE
    mae_checkpoint = torch.load(opt.join(weights_dir, 'best_mae.pth'))
    mae.load_state_dict(mae_checkpoint['model_state_dict'])
    mae.to(device)
    
    train_set = FinetuningDataset(
    data_dir=cfg.data.data_dir,
    ann_file=cfg.data.train_ann,
    max_seq=cfg.sgcn.max_seq_len,
    normalize=cfg.data.normalize,
    temporal_edge_model=temporal_att,  
    edge_threshold=0.6
     )
    val_set = FinetuningDataset(
    data_dir=cfg.data.data_dir,
    ann_file=cfg.data.val_ann,
    max_seq=cfg.sgcn.max_seq_len,
    normalize=cfg.data.normalize,
    temporal_edge_model=temporal_att,  
    edge_threshold=0.6
    )


    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.sgcn.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.sgcn.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize SGCN
    sgcn = SGCNModel(cfg.sgcn).to(device)
    
    # Training setup
    optimizer = optim.AdamW(
        list(sgcn.parameters()) + list(mae.parameters()),  # Update both models
        lr=cfg.sgcn.lr,
        weight_decay=cfg.sgcn.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.sgcn.num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Finetune
    best_val_acc = 0
    for epoch in range(1, cfg.sgcn.num_epochs + 1):
        train_loss, train_acc = train_finetune_epoch(epoch, mae, sgcn, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_finetune(mae, sgcn, val_loader, criterion, device)
        scheduler.step()
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'sgcn_state_dict': sgcn.state_dict(),
                'mae_state_dict': mae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, opt.join(weights_dir, 'best_sgcn.pth'))
        
        # Visualize joint importance
        if epoch % 10 == 0:
            plot_joint_importance(sgcn)
            plt.savefig(f'joint_importance_epoch{epoch}.png')
            plt.close()
    
    print(f'\nTraining complete! Best validation accuracy: {best_val_acc*100:.2f}%')