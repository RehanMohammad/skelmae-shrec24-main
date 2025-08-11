# utils/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_temporal_heatmap(attn, frame_idx=0, head_idx=0, title=None, savepath=None):
    """
    Accepts:
      - [B, V, heads, V]  (multi-head attention between joints)
      - [B, T, V, V]      (temporal adjacency per frame)
      - [T, V, V]
      - [V, V]
    and renders a 2D [V, V] heatmap.
    """
    if isinstance(attn, torch.Tensor):
        A = attn.detach().cpu()
    else:
        import numpy as np
        A = torch.tensor(attn)

    # Normalize to 2D [V, V]
    if A.dim() == 4:
        B, D1, D2, D3 = A.shape
        if D1 == D3 and D2 <= 16:
            # assume [B, V, heads, V]
            M = A[0, :, min(head_idx, D2-1), :]
        elif D2 == D3:
            # assume [B, T, V, V]
            M = A[0, min(frame_idx, D1-1), :, :]
        else:
            raise ValueError(f"Unrecognized 4D shape for attention/adjacency: {tuple(A.shape)}")
    elif A.dim() == 3:
        T, V1, V2 = A.shape
        if V1 == V2:
            # [T, V, V]
            M = A[min(frame_idx, T-1), :, :]
        else:
            raise ValueError(f"Unrecognized 3D shape: {tuple(A.shape)}")
    elif A.dim() == 2:
        M = A
    else:
        raise ValueError(f"Expected 2D/3D/4D, got {A.dim()}D with shape {tuple(A.shape)}")

    M = M.numpy() if isinstance(M, torch.Tensor) else M

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(M,
                     xticklabels=range(M.shape[1]),
                     yticklabels=range(M.shape[0]))
    ax.set_xlabel("Targets")
    ax.set_ylabel("Sources")
    if title:
        ax.set_title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
