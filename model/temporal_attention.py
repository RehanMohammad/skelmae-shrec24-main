import torch
import torch.nn as nn

class AdaptiveTemporal(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim * heads)
        self.to_k = nn.Linear(dim, dim * heads)
        
    def forward(self, current, next_frame):
        # current/next: [B, V, C]
        q = self.to_q(current).view(*current.shape[:2], self.heads, -1)
        k = self.to_k(next_frame).view(*next_frame.shape[:2], self.heads, -1)
        
        # Compute attention scores
        attn = torch.einsum('bvhd,bwhd->bvhw', q, k) * self.scale
        return attn.softmax(dim=-1)  # [B, V, heads, V]