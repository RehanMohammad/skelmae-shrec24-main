import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from dataset.IPN import _TEMPORAL_RULES


from vit import Transformer


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim,
        masking_strategy,
        masking_ratio,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        assert masking_strategy in ['random', 'random-hand']
        
        self.masking_ratio = masking_ratio
        self.masking_strategy = masking_strategy
        
        # extract some hyperparameters and functions from mae_encoder
        self.encoder = encoder        
        n_nodes, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.node_to_emb = encoder.to_node_embedding

        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 1),
        )

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_edge = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
                
        self.decoder_pos_emb = nn.Embedding(n_nodes, decoder_dim)
        self.to_pred = nn.Linear(decoder_dim, encoder.node_dim)
        
        self.unm_enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        
        self.n_nodes = n_nodes
        self.encoder_dim = encoder_dim
        
    def change_masking_strategy(self, strategy):
        assert strategy in ['random', 'one-hand'], "masking strategy must be in {'none', 'random', 'fingers', 'tips', 'connex', 'mixte'}"
        self.masking_strategy = strategy
        
    def change_masking_ratio(self, ratio):
        assert 0 < ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = ratio

    # def _generate_mask(self, tokens, strategy, adj_matrix):
        
    #     device = tokens.device
    #     # batch, n_tokens, token_dim = tokens.shape
    #     batch, frames ,n_tokens, token_dim = tokens.shape
        
    #     if strategy == 'random':
    #         num_masked = int(self.masking_ratio * n_tokens)
    #         rand_indices = torch.rand(batch, n_tokens, device=device).argsort(dim=-1)
    #         masked_indices, unmasked_indices = (
    #             rand_indices[:, :num_masked],
    #             rand_indices[:, num_masked:],
    #         )
            
    #     elif strategy == 'random-hand':
    #         n_tokens_per_hand = n_tokens // 2
    #         num_masked = int(self.masking_ratio * n_tokens_per_hand)
    #         rand_indices_left_hand = torch.rand(batch, n_tokens_per_hand, device=device).argsort(dim=-1)
    #         rand_indices_right_hand = 14 + torch.rand(batch, n_tokens_per_hand, device=device).argsort(dim=-1)
            
    #         masked_indices_left_hand, unmasked_indices_left_hand = (
    #             rand_indices_left_hand[:, :num_masked],
    #             rand_indices_left_hand[:, num_masked:],
    #         )
            
    #         masked_indices_right_hand, unmasked_indices_right_hand = (
    #             rand_indices_right_hand[:, :num_masked],
    #             rand_indices_right_hand[:, num_masked:],
    #         )
            
    #         masked_indices = torch.cat((masked_indices_left_hand, masked_indices_right_hand), dim=-1)
    #         unmasked_indices = torch.cat((unmasked_indices_left_hand, unmasked_indices_right_hand), dim=-1)
    #         num_masked *= 2
            
    #     else:
    #         raise NotImplementedError
        
    #     print(" num_masked", num_masked)
    #     print(" masked_indices.shape", masked_indices.shape)
    #     print(" masked_indices", masked_indices)
    #     print(" unmasked_indices.shape", unmasked_indices.shape)
    #     print(" unmasked_indices", unmasked_indices)

    #     return num_masked, masked_indices, unmasked_indices
        
    # def _generate_mask(self, tokens, strategy, adj_matrix):
        
    #     device = tokens.device
    #     batch, frames ,n_tokens, token_dim = tokens.shape   #64, 28, 256  #64,4,21,256
        
    #     _TOTAL = n_tokens*n_tokens
    #     # n_tokens_per_hand = n_tokens // 2    #14
    #     num_masked = int(self.masking_ratio * _TOTAL) #0.7 * 14 = 9   0.7*83
    #     rand_indices = torch.rand( batch, frames,_TOTAL, device=device).argsort(dim=-1)
    #     masked_indices, unmasked_indices = (
    #         rand_indices[..., :num_masked],
    #         rand_indices[..., num_masked:],
    #     )
            
               
    #     print(" num_masked", num_masked)
    #     print(" masked_indices.shape", masked_indices.shape)
    #     print(" masked_indices", masked_indices)
    #     print(" unmasked_indices.shape", unmasked_indices.shape)
    #     print(" unmasked_indices", unmasked_indices)

    #     return num_masked, masked_indices, unmasked_indices

    def _generate_edge_mask(self, A: torch.Tensor):
        # always receives A as (B, N, N)
        B, N, _ = A.shape
        total = N * N
        k = int(self.masking_ratio * total)

        flat = A.view(B, -1)                       # (B, N*N)
        perm = torch.rand(B, total, device=A.device).argsort(dim=-1)
        mask_pos = perm[:, :k]                     # (B, k)

        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, mask_pos, True)           # set those k positions to True
        return mask.view(B, N, N)
    
    def forward(self, x: torch.Tensor, A: torch.Tensor):
        B, C, T, V = x.shape
        print("B",B)
        print("C",C)
        print("T",T)
        print("V",V)
        N = T * V

        # ———————————————
        # if the adjacency came in as (B, T, V, V),
        # stitch it into a single (B, N, N) graph first
        # ———————————————
        
        if A.dim() == 4:
            # A is (B, T, V, V)  →  we want (B, T*V, T*V)
            A_big = A.new_zeros((B, N, N))
            for t in range(T-1):
                # copy the t→t+1 block
                A_big[:, t*V:(t+1)*V, (t+1)*V:(t+2)*V] = A[:, t]
            A = A_big
        # now A is guaranteed to be (B, N, N)

        # — your existing 3-D logic resumes —
        # flatten your node features
        x_flat = x.permute(0,2,3,1).reshape(B, N, C)
        # mask some edges
        edge_mask = self._generate_edge_mask(A)
        A_masked = A.clone()
        A_masked[edge_mask] = 0

        # encode
        emb = self.encoder.to_node_embedding(x_flat)            # (B, N, D)
        if emb.dim() == 3 and emb.shape[1] == self.encoder.pos_embedding.shape[-1]:
            # we think emb is (B, D, N)
            emb = emb.permute(0, 2, 1) 
        enc = self.encoder.transformer(emb, A_masked)           # (B, N, D)
        dec = self.enc_to_dec(enc)                              # (B, N, D_dec)

        # reconstruct edges and compute loss...
        Zi = dec.unsqueeze(2).expand(B,N,N,-1)
        Zj = dec.unsqueeze(1).expand(B,N,N,-1)
        feats = torch.cat([Zi, Zj], dim=-1)         # (B, N, N, 2*dec_dim)
        pred_edges = self.edge_decoder(feats).squeeze(-1)  # (B, N, N)
        loss = F.mse_loss(pred_edges[edge_mask], A[edge_mask].float())

        return pred_edges, loss

        
    # def forward(self, x, a):
    #     device = x.device
        
    #     # get patches
    #     # print(" x.shape", x.shape)
    #     # batch, n_nodes, node_dim = x.shape  # shrec24
    #     batch,node_dim, frame, n_nodes = x.shape #IPN

    #     print("batch", batch)
    #     print("node_dim", node_dim)
    #     print("frame", frame)
    #     print("n_nodes", n_nodes)
    #     if x.ndim==4:
    #         x=x.reshape(batch, node_dim, frame* n_nodes)
    #         x=x.permute(0,2,1)
    #     # print(" x.shape: ", x.shape)

    #     # patch to mae_encoder tokens and add positions
    #     tokens = self.node_to_emb(x)
    #     tokens = tokens.reshape(batch,frame,n_nodes,-1)
    #     print(" tokens.shape:", tokens.shape)
    #     # tokens = tokens.permute(0, 2, 1)
    #     tokens = tokens + self.encoder.pos_embedding

    #     # print("masking_strategy", self.masking_strategy)
    #     #generate mask
    #     num_masked, masked_indices, unmasked_indices = self._generate_mask(tokens, self.masking_strategy, a)
    #     # get the unmasked tokens to be encoded
    #     # rows = masked_indices // n_nodes       # shape (B, F, num_masked)
    #     # cols = masked_indices %  n_nodes
    #     # batch_range = torch.arange(batch, device=device)[:, None]
    #     # frame_range = torch.arange(frame, device=device)[:, None]
    #     # print("batch_range.shape",batch_range.shape)
    #     # print("frame_range.shape",frame_range.shape)
    #     # print(' a.shape', a.shape)
    #     # unmasked_temporal_edges = a[batch_range, frame, rows,cols]
    #     # print("unmasked_temporal_edges.shape", unmasked_temporal_edges.shape)

    #     B, F, V, _ = a.shape
    #     K  = masked_indices.shape[-1]
    #     KU = unmasked_indices.shape[-1]

    #     # 1) Flat→(row,col)
    #     rows_M = masked_indices   // V           # (B, F, K)
    #     cols_M = masked_indices   %  V           # (B, F, K)
    #     rows_U = unmasked_indices // V           # (B, F, KU)
    #     cols_U = unmasked_indices %  V           # (B, F, KU)

    #     # 2) Batch & frame grids
    #     batch_idx = (torch.arange(B, device=a.device)
    #                 .view(B,1,1)
    #                 .expand(B,F,K))            # (B, F, K)
    #     frame_idx = (torch.arange(F, device=a.device)
    #                 .view(1,F,1)
    #                 .expand(B,F,K))            # (B, F, K)

    #     batch_idx_U = (torch.arange(B, device=a.device)
    #                 .view(B,1,1)
    #                 .expand(B,F,KU))          # (B, F, KU)
    #     frame_idx_U = (torch.arange(F, device=a.device)
    #                 .view(1,F,1)
    #                 .expand(B,F,KU))          # (B, F, KU)

    #     # 3) Gather values
    #     masked_edge_vals   = a[batch_idx,   frame_idx,   rows_M, cols_M]   # (B, F, K)
    #     unmasked_edge_vals = a[batch_idx_U, frame_idx_U, rows_U, cols_U]  # (B, F, KU)
    #     print("masked_edge_vals.shape", masked_edge_vals.shape)
    #     print("unmasked_edge_vals.shape", unmasked_edge_vals.shape)

    #     unmasked_a = torch.empty(B, F, KU, KU, device=a.device, dtype=a.dtype)

    #     for b in range(B):
    #         for f in range(F):
    #             idx = unmasked_indices[b, f]    # shape (KU,)
    #             submat = a[b, f][idx][:, idx]    # shape (KU, KU)
    #             unmasked_a[b, f] = submat
    #     ##############################################################
    #     # unmasked_a = torch.empty((*unmasked_indices.shape, unmasked_indices.shape[-1]), device=device)
    #     # print("*unmasked_indices.shape",*unmasked_indices.shape)
    #     # print("unmasked_a.shape", unmasked_a.shape)
    #     # for i, idx in enumerate(unmasked_indices):
    #     #     print("idx",idx)
    #     #     print("idx.shape",idx.shape)
    #     #     unmasked_a[i] = a[i,  idx]
    #         # print("i", i)
    #         # print("idx", idx)
    #         # print("idx.view(-1, 1)", idx.view(-1, 1))
    #         # print("a[i, idx.view(-1, 1), idx]", a[i, idx.view(-1, 1), idx])
    #     # print("unmasked_a.shape",unmasked_a.shape )
    #     # print("unmasked_tokens.shape",unmasked_tokens.shape )
              
    #     encoded_edges = self.encoder.transformer(unmasked_edge_vals, unmasked_a)
    #     # project mae_encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
    #     decoder_edges = self.enc_to_dec(encoded_edges)

    #     # reapply decoder position embedding to unmasked tokens
    #     unmasked_decoder_edges = decoder_edges + self.decoder_pos_emb(unmasked_indices)

    #     # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
    #     mask_edges = repeat(self.mask_edge, "d -> b n d", b=batch, n=num_masked)
    #     mask_edges = mask_edges + self.decoder_pos_emb(masked_indices)

    #     # concat the masked tokens to the decoder tokens and attend with decoder
    #     decoder_edges = torch.cat((mask_edges, decoder_edges), dim=1)
    #     decoder_edges = self.decoder(decoder_edges, a)
        
    #     # splice out the mask tokens and project to coordinates
    #     mask_edges = decoder_edges[batch_idx, masked_indices]
    #     masked_pred = self.to_pred(mask_edges)
        
    #     # calculate reconstruction loss
    #     recon_loss = F.mse_loss(masked_pred, masked_edge_vals)

    #     return decoder_edges, masked_edge_vals, masked_indices, masked_pred, recon_loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def inference(self, x, a):
        device = x.device
        len_x = len(x.shape)
        if len_x == 4:
            b, t, n, d = x.shape
            x = rearrange(x, 'b t n d -> (b t) n d') 
            if a is not None:
                a = a.repeat(t, 1, 1)

        with torch.no_grad():
            # get patches
            batch, n_nodes, node_dim = x.shape

            # patch to mae_encoder tokens and add positions
            tokens = self.node_to_emb(x)
            tokens = tokens.permute(0, 2, 1)
            tokens = tokens + self.encoder.pos_embedding

            encoded_tokens = self.encoder.transformer(tokens, a)
            encoded_tokens = self.enc_to_dec(encoded_tokens)
        
        if len_x == 4:
            encoded_tokens = rearrange(encoded_tokens, '(b t) n d -> b t n d', b=b, t=t)
        
        return encoded_tokens