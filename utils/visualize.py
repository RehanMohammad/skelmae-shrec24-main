# utils/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_temporal_heatmap(attn, frame_idx=0, head_idx=0, title=None, savepath=None):
    """
    Accepts:
      - [B, V, heads, V]  (multi-head joint→joint attention)
      - [B, T, V, V]      (temporal adjacency per frame)
      - [T, V, V]
      - [V, V]
    and renders a 2D [V, V] heatmap with robust defaults.
    """
    # ---- move to CPU numpy
    A = attn.detach().cpu() if isinstance(attn, torch.Tensor) else torch.as_tensor(attn)
    M = None

    # ---- normalize to 2D [V, V]
    if A.dim() == 4:
        B, D1, D2, D3 = A.shape
        if D1 == D3 and D2 <= 64:               # [B, V, heads, V]
            M = A[0, :, min(head_idx, D2-1), :]
        elif D2 == D3:                          # [B, T, V, V]
            M = A[0, min(frame_idx, D1-1), :, :]
        else:
            raise ValueError(f"Unrecognized 4D shape: {tuple(A.shape)}")
    elif A.dim() == 3:
        T, V1, V2 = A.shape
        if V1 == V2:                            # [T, V, V]
            M = A[min(frame_idx, T-1), :, :]
        else:
            raise ValueError(f"Unrecognized 3D shape: {tuple(A.shape)}")
    elif A.dim() == 2:
        M = A
    else:
        raise ValueError(f"Expected 2D/3D/4D, got {A.dim()}D with shape {tuple(A.shape)}")

    M = M.numpy().astype(np.float32)

    # ---- handle NaNs and constant matrices
    nan_mask = np.isnan(M)
    if nan_mask.any():
        M = np.where(nan_mask, 0.0, M)  # seaborn can also accept mask=nan_mask

    vmin, vmax = np.nanquantile(M, 0.02), np.nanquantile(M, 0.98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallbacks for binary / constant matrices
        uniq = np.unique(M[~nan_mask]) if (~nan_mask).any() else np.array([0.0])
        if set(uniq.tolist()).issubset({0.0, 1.0}):
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(M)), float(np.nanmax(M) + 1e-6)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        M, cmap="viridis", vmin=vmin, vmax=vmax, square=True,
        xticklabels=range(M.shape[1]), yticklabels=range(M.shape[0]),
        cbar=True
    )
    ax.set_xlabel("Targets")
    ax.set_ylabel("Sources")
    if title:
        ax.set_title(title)

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")  # tighter & sharper export
        plt.close()
    else:
        plt.show()
