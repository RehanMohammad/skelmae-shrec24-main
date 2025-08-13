# utils/temporal_teacher.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def velocity_similarity_teacher(
    x_bctv: torch.Tensor,
    tau: float = 0.5,
    topk: int | None = 4,
    remove_self: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build teacher distributions P[t, i, j] from velocity similarity.

    Inputs
    ------
    x_bctv : Tensor
        Either [C, T, V]  (single sample)  or  [B, C, T, V] (batch).
        We'll use the first 3 channels as xyz.
    tau : float
        Temperature before softmax (smaller => sharper).
    topk : int | None
        Keep top-k per row before softmax. None keeps all.
    remove_self : bool
        Zero the diagonal (i -> i).
    eps : float
        Small constant for norms.

    Returns
    -------
    P : Tensor
        If input was [C,T,V]  ->  [T, V, V]  (last frame all zeros)
        If input was [B,C,T,V]->  [B, T, V, V]
    """
    # ensure [B,C,T,V]
    if x_bctv.dim() == 3:           # [C,T,V]
        x_bctv = x_bctv.unsqueeze(0)
        squeeze_back = True
    elif x_bctv.dim() == 4:         # [B,C,T,V]
        squeeze_back = False
    else:
        raise ValueError(f"x_bctv must be [C,T,V] or [B,C,T,V], got {tuple(x_bctv.shape)}")

    B, C, T, V = x_bctv.shape
    xyz = x_bctv[:, :3]                      # [B,3,T,V]

    # velocity v_t(j) = x_{t+1}(j) - x_t(j)
    v = xyz[:, :, 1:, :] - xyz[:, :, :-1, :] # [B,3,T-1,V]

    # row-normalize (per joint) for cosine
    # v_norm: [B,1,T-1,V]
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(eps)
    v_hat  = v / v_norm

    # cosine similarity per time step:
    # S_t(i,j) = max(0, <v_hat_t(i), v_hat_t(j)>)
    # (compare velocities at the same t; this yields edges for t->t+1)
    # einsum over channel dimension
    S = torch.einsum('bcti,bctj->btij', v_hat, v_hat).clamp_min(0.0)   # [B,T-1,V,V]

    if remove_self:
        eye = torch.eye(V, device=S.device, dtype=S.dtype)[None, None] # [1,1,V,V]
        S = S * (1.0 - eye)

    # (optional) top-k keep before softmax
    if topk is not None and topk < V:
        # zero everything except topk per row
        vals, idxs = torch.topk(S, k=topk, dim=-1)
        mask = torch.zeros_like(S, dtype=torch.bool)
        mask.scatter_(-1, idxs, True)
        S = S.masked_fill(~mask, float('-inf'))

    # temperature + row-wise softmax
    P = F.softmax(S / max(tau, eps), dim=-1)                       # [B,T-1,V,V]

    # pad one all-zero slice for the last (no t+1)
    pad = torch.zeros((B, 1, V, V), device=P.device, dtype=P.dtype)
    P_full = torch.cat([P, pad], dim=1)                            # [B,T,V,V]

    if squeeze_back:
        P_full = P_full.squeeze(0)                                 # [T,V,V]

    return P_full
