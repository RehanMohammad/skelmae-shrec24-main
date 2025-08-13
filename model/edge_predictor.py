# model/edge_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class EdgePredictor(nn.Module):
    """
    Inputs:
      x: [B, 3, T, V]  (xyz only)
    Outputs:
      {
        'logits_full': [B, T, V, V]  (last frame is zeros; no t->t+1 there),
        'logits':      [B, T-1, V, V]
      }
    """
    def __init__(self, d_model=128, hid=128, dropout=0.1):
        super().__init__()
        # Two encoders: one for joints at t, one for joints at t+1
        self.enc_t   = MLP(3, hid, d_model, dropout=dropout)
        self.enc_tp1 = MLP(3, hid, d_model, dropout=dropout)
        self.scale   = d_model ** -0.5  # temperature-like scaling for dot-product

        # Optional bias that can learn to encourage/discourage self-edges
        self.self_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(1) == 3, f"Expected [B,3,T,V], got {x.shape}"
        B, C, T, V = x.shape
        if T < 2:
            raise ValueError("T must be >= 2 to form t->t+1 pairs")

        # [B,3,T-1,V] and [B,3,T-1,V]
        x_t   = x[:, :, :-1, :]      # joints at time t
        x_tp1 = x[:, :,  1:, :]      # joints at time t+1

        # -> [B,T-1,V,3]
        x_t   = x_t.permute(0, 2, 3, 1).contiguous()
        x_tp1 = x_tp1.permute(0, 2, 3, 1).contiguous()

        # embed to [B,T-1,V,D]
        q = self.enc_t(x_t)
        k = self.enc_tp1(x_tp1)

        # pairwise dot-product per frame: [B,T-1,V,V]
        logits = torch.einsum('btid,btjd->btij', q, k) * self.scale

        # add a learnable bias to the diagonal (self-edges), if desired
        eye = torch.eye(V, device=logits.device).view(1, 1, V, V)
        logits = logits + self.self_bias * eye

        # pad to [B,T,V,V] so it matches the teacher shape in your dataset
        logits_full = torch.zeros((B, T, V, V), device=logits.device, dtype=logits.dtype)
        logits_full[:, :-1] = logits

        return {'logits_full': logits_full, 'logits': logits}
