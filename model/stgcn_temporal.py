import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# ----------------------------------------
# Utility: build spatio‐temporal adjacency
# ----------------------------------------
def build_batch_temporal_adj(a4: torch.Tensor) -> torch.Tensor:
    """
    a4: (B, T, V, V) boolean adjacency between t→t+1
    Returns A_big: (B, N, N) with N=T*V
    """
    B, T, V, _ = a4.shape
    N = T * V
    A = torch.zeros(B, N, N, device=a4.device, dtype=a4.dtype)
    for b in range(B):
        for t in range(T - 1):
            # place the V×V adjacency of frame t into block [t→t+1]
            A[b,
              t*V:(t+1)*V,
              (t+1)*V:(t+2)*V] = a4[b, t]
    return A

# ----------------------------------------
# A small Temporal GCN that takes runtime A
# ----------------------------------------
class TemporalGCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 use_local_bn: bool = False):
        super().__init__()
        self.in_c  = in_channels
        self.out_c = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_local_bn = use_local_bn

        # 1×1 conv over “(time×nodes)” dimension
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size//2, 0),
                              stride=(stride, 1),
                              bias=False)
        # batchnorm
        # we’ll defer creating local‐bn until we see V
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.V = None

        # initialize conv
        n = out_channels
        for k in self.conv.kernel_size:
            n *= k
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, 1, N)  where N = T*V
        A: (B, N, N)     the spatio-temporal adjacency per batch
        """
        B, C, _, N = x.shape

        if self.V is None:
            self.V = N
            if self.use_local_bn:
                # local bn over (out_c * V)
                self.bn1d = nn.BatchNorm1d(self.out_c * self.V)
        # 1) graph‐product
        x_flat = x.reshape(B * C, N)
        x_g = torch.matmul(x_flat, A.transpose(1, 2))  # (B*C, N)
        x_g = x_g.view(B, C, 1, N)

        # 2) conv + bn + relu
        y = self.conv(x_g)
        y = self.bn2d(y)
        return self.relu(y)

# ----------------------------------------
# ST‐GCN with injected TemporalGCN
# ----------------------------------------
class STGCN(nn.Module):
    def __init__(self,
                 channel: int,
                 num_class: int,
                 window_size: int,
                 num_point: int,
                 num_person: int = 1,
                 use_data_bn: bool = False,
                 graph_args: dict = None,
                 mask_learning: bool = False,
                 use_local_bn: bool = False,
                 multiscale: bool = False,
                 temporal_kernel_size: int = 9,
                 dropout: float = 0.5):
        super().__init__()

        # --- data batchnorm over (C*V*M) if requested ---
        self.use_data_bn = use_data_bn
        self.M_dim_bn   = True
        if self.use_data_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)

        # --- the temporal GCN over flattened (T,V) graph edges ---
        self.temporal_gcn = TemporalGCN(
            in_channels=channel,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            use_local_bn=False,
        )

        # --- your existing static spatial graph conv layers ---
        # For simplicity, we'll re-use your unit_gcn and Unit2D
        from model.stgcn import unit_gcn, Unit2D, TCN_GCN_unit, Graph

        # build the static spatial adjacency
        self.graph = Graph(**(graph_args or {}))
        A_spatial = torch.from_numpy(self.graph.A).float()
        self.A_spatial = nn.Parameter(A_spatial, requires_grad=False)

        # initial GCN+TCN
        self.gcn0 = unit_gcn(channel,
                             channel,
                             self.A_spatial,
                             use_local_bn=use_local_bn,
                             mask_learning=mask_learning)
        self.tcn0 = Unit2D(channel, channel, kernel_size=temporal_kernel_size, dropout=dropout)

        # backbone
        unit = TCN_GCN_unit
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, self.A_spatial,
                 kernel_size=temporal_kernel_size,
                 stride=stride,
                 use_local_bn=use_local_bn,
                 mask_learning=mask_learning,
                 dropout=dropout)
            for in_c, out_c, stride in [
                (channel, channel, 1),
                (channel, channel, 1),
                (channel, channel, 1),
                (channel, channel*2, 2),
                (channel*2, channel*2, 1),
                (channel*2, channel*2, 1),
                (channel*2, channel*4, 2),
                (channel*4, channel*4, 1),
                (channel*4, channel*4, 1),
            ]
        ])

        # head
        self.person_bn = nn.BatchNorm1d(self.backbone[-1].out_channels)
        self.gap_size = math.ceil(window_size / 4)  # after two stride-2 layers
        self.fcn = nn.Conv1d(self.backbone[-1].out_channels,
                             num_class,
                             kernel_size=1)
        self.num_class = num_class

    def forward(self, x: torch.Tensor, A_temporal: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V, M)
        A_temporal: (N, T, V, V) boolean adjacency between t→t+1
        """
        N, C, T, V, M = x.shape

        # ——— data bn if desired ———
        if self.use_data_bn:
            # collapse to (N*M, C, T, V)
            if self.M_dim_bn:
                x = x.permute(0,4,3,1,2).contiguous().view(N, M*V*C, T)
            else:
                x = x.permute(0,4,3,1,2).contiguous().view(N*M, V*C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0,1,3,4,2).contiguous().view(N*M, C, T, V)
        else:
            x = x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V)

        # ——— flatten (T,V) into N=T*V and build A_big ———
        x_flat = x.reshape(N*M, C, 1, T*V)                   # (N*M, C,1,N)
        A_big  = build_batch_temporal_adj(A_temporal.view(N*M, T, V, V))  # (N*M, N, N)

        # ——— temporal GCN ———
        x_temp = self.temporal_gcn(x_flat, A_big)             # (N*M, C,1,N)
        x = x_temp.reshape(N*M, C, T, V)                      # back to (N*M,C,T,V)

        # ——— spatial GCN+TCN backbone ———
        x = self.gcn0(x)
        x = self.tcn0(x)
        for layer in self.backbone:
            x = layer(x)

        # ——— global pooling & classification ———
        x = F.avg_pool2d(x, kernel_size=(1, V))              # (N*M, C', T, 1)
        x = x.view(N, M, x.size(1), x.size(2))               # (N, M, C', T)
        x = x.mean(1)                                        # pool persons → (N,C',T)
        x = F.avg_pool1d(x, kernel_size=x.size(2))           # (N,C',1)
        x = self.fcn(x)                                      # (N,num_class,1)
        return x.view(N, self.num_class)
