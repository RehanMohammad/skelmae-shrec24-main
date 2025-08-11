import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from model.temporal_attention import AdaptiveTemporal
from vit import Transformer
from dataset.IPN import build_temporal_edges

class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim,
        masking_ratio,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        n_nodes, encoder_dim = encoder.pos_embedding.shape[-2:]
        
        # Enhanced reconstruction heads
        self.to_position = nn.Linear(decoder_dim, 3)  # x,y,z
        self.to_velocity = nn.Linear(decoder_dim, 3)  # vx,vy,vz
        self.to_accel = nn.Linear(decoder_dim, 3)    # ax,ay,az
        
        # Adaptive temporal module
        self.adaptive_temp = AdaptiveTemporal(
            dim=encoder_dim,
            heads=decoder_heads
        )
        
        # Edge decoder
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 1),
        )
        
        # Decoder components
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        
        self.n_nodes = n_nodes
        self.encoder_dim = encoder_dim

    def _generate_edge_mask(self, A: torch.Tensor):
        B, N, _ = A.shape
        total = N * N
        k = int(self.masking_ratio * total)

        flat = A.view(B, -1)
        perm = torch.rand(B, total, device=A.device).argsort(dim=-1)
        mask_pos = perm[:, :k]

        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, mask_pos, True)
        return mask.view(B, N, N)
    
    def forward(self, x, A_temporal=None):
        # x: [B, 9, T, V]  (xyz + vel + acc)
        device = x.device
        B, C, T, V = x.shape
        coords_dim = 3

        # 1) Use only xyz for MAE targets/inputs
        target = x[:, :coords_dim]                                   # [B,3,T,V]

        # 2) Flatten to tokens [B, N, 3] where N = T*V
        nodes_xyz = target.permute(0, 2, 3, 1).reshape(B, T * V, coords_dim)

        # 3) Build big adjacency A_big: [B, N, N]
        #    - block-diagonal identity (within-frame)
        #    - optional inter-frame edges from A_temporal (t -> t+1)
        N = T * V
        A_big = torch.eye(N, device=device, dtype=nodes_xyz.dtype).unsqueeze(0).repeat(B, 1, 1)
        if A_temporal is not None:
            if A_temporal.dim() == 3:               # [T, V, V]
                A_temporal = A_temporal.unsqueeze(0).expand(B, -1, -1, -1)
            assert A_temporal.shape == (B, T, V, V)
            for t in range(T - 1):
                src = slice(t * V,     (t + 1) * V)
                dst = slice((t + 1) * V, (t + 2) * V)
                A_big[:, src, dst] = A_temporal[:, t].to(nodes_xyz.dtype)

        # 4) Token embedding with Fourier features → [B, N, D] then add learned pos emb
        emb = self.encoder.to_node_embedding(nodes_xyz)              # usually [B, D, N]
        if emb.dim() == 3 and emb.shape[1] == self.encoder.pos_embedding.shape[-1]:
            emb = emb.transpose(1, 2)                                # ensure [B, N, D]

        # pos_embedding shape: [1, N, D] (initialized with num_nodes = T*V)
        emb = emb + self.encoder.pos_embedding

        # 5) Encode with Transformer using A_big as an attention mask
        z = self.encoder.transformer(emb, A_big)                     # [B, N, D_enc]

        # 6) Map to decoder dim and reconstruct xyz per token, then reshape back to [B,3,T,V]
        dec = self.enc_to_dec(z)                                     # [B, N, D_dec]
        pos_hat = self.to_position(dec)                              # [B, N, 3]
        recon = pos_hat.transpose(1, 2).reshape(B, 3, T, V)          # [B,3,T,V]

        # 7) Random joint mask per frame (same as before)
        num_mask = max(1, int(round(V * self.masking_ratio)))
        mask_list = []
        for _ in range(B):
            m = torch.zeros(T, V, dtype=torch.bool, device=device)
            for t in range(T):
                idx = torch.randperm(V, device=device)[:num_mask]
                m[t, idx] = True
            mask_list.append(m)
        mask = torch.stack(mask_list, dim=0)                         # [B,T,V]

        # 8) MAE loss on masked tokens only (xyz channels)
        mask_expand = mask.unsqueeze(1).expand(-1, coords_dim, -1, -1)  # [B,3,T,V]
        diff = (recon - target) * mask_expand
        denom = mask_expand.sum().clamp_min(1.0)
        recon_loss = (diff ** 2).sum() / denom

        return {
            'loss': recon_loss,
            'recon': recon,
            'mask': mask,
            'dynamic_A': A_temporal if A_temporal is not None else torch.zeros(B, T, V, V, device=device),
            'components': {'recon': recon_loss.detach(), 'edge': torch.tensor(0.0, device=device)}
        }


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def inference(self, x, a):
        device = x.device
        B, C, T, V = x.shape
        N = T * V
        
        with torch.no_grad():
            x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
            emb = self.encoder.to_node_embedding(x_flat)
            
            if emb.dim() == 3 and emb.shape[1] == self.encoder.pos_embedding.shape[-1]:
                emb = emb.permute(0, 2, 1)
                
            # Handle adjacency if provided
            if a is not None and a.dim() == 4:
                A_big = torch.zeros((B, N, N), device=device)
                for t in range(T-1):
                    src_slice = slice(t*V, (t+1)*V)
                    tgt_slice = slice((t+1)*V, (t+2)*V)
                    A_big[:, src_slice, tgt_slice] = a[:, t]
                a = A_big
            
            encoded = self.encoder.transformer(emb, a)
            encoded = self.enc_to_dec(encoded)
        
        return encoded.reshape(B, T, V, -1)