# utils/visualize_edges.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# MediaPipe 21-joint skeleton (0=wrist)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]

def _choose_t(T, t=None):
    # choose a usable frame index so that t+1 exists
    if t is None:
        t = max(0, min(T-2, (T//2) - 1))
    t = int(np.clip(t, 0, T-2))
    return t

def _frame_xy(x_bctv, b, t):
    """
    x_bctv: [B, 3, T, V]  (normalized coords are fine)
    Returns (xy_t [V,2], xy_tp1 [V,2]) normalized and horizontally separated.
    """
    with torch.no_grad():
        xy_t   = x_bctv[b, :2, t,   :].T.detach().cpu().numpy()  # [V,2]
        xy_tp1 = x_bctv[b, :2, t+1, :].T.detach().cpu().numpy()  # [V,2]

    # normalize across both frames so relative scale is preserved
    all_xy = np.vstack([xy_t, xy_tp1])
    mn = all_xy.min(axis=0)
    mx = all_xy.max(axis=0)
    span = (mx - mn)
    span[span < 1e-6] = 1.0
    xy_t   = (xy_t   - mn) / span
    xy_tp1 = (xy_tp1 - mn) / span

    # horizontal separation so the two hands don’t overlap
    sep = 1.4
    xy_tp1 = xy_tp1 + np.array([sep, 0.0])

    # small margins
    xy_t   = xy_t * 0.9 + 0.05
    xy_tp1 = xy_tp1 * 0.9 + 0.05

    return xy_t, xy_tp1, sep

def _collect_topk_edges(P_tij, k=4, remove_self=True):
    """
    P_tij: [V,V] probabilities (row-wise distributions from joint i@t to joints j@(t+1))
    Returns list of (i, j, w)
    """
    V = P_tij.shape[-1]
    P = P_tij.clone()
    if remove_self:
        diag = torch.eye(V, device=P.device, dtype=P.dtype)
        P = P * (1.0 - diag)

    edges = []
    vals, idxs = torch.topk(P, k=min(k, V), dim=-1)
    for i in range(V):
        for r in range(idxs.shape[1]):
            j = int(idxs[i, r].item())
            w = float(vals[i, r].item())
            if w > 0:
                edges.append((i, j, w))
    return edges

def _edge_style(weights, w):
    """
    Map weight -> style
    strong (top third): solid, thick
    medium: dashed, mid
    weak: dotted, thin
    """
    eps = 1e-8
    wmin, wmax = float(np.min(weights)), float(np.max(weights))
    # guard for degenerate cases
    if wmax - wmin < 1e-8:
        tier = "strong"
        width = 2.8
        color = cm.viridis(0.95)
        return tier, width, color, '-'

    q1, q2 = np.quantile(weights, [1/3, 2/3])
    if w <= q1:
        tier = "weak";   ls = ':';  base = 0.6
    elif w <= q2:
        tier = "medium"; ls = '--'; base = 0.75
    else:
        tier = "strong"; ls = '-';  base = 0.9

    # continuous width & color within tier
    norm = (w - wmin) / (wmax - wmin + eps)
    width = 0.8 + 3.2 * norm
    color = cm.viridis(0.35 + 0.6 * norm)
    return tier, width, color, ls

def _draw_hand_skeleton(ax, xy, color='#B0B0B0', lw=1.0, alpha=0.6):
    for i, j in HAND_CONNECTIONS:
        ax.plot([xy[i,0], xy[j,0]], [xy[i,1], xy[j,1]],
                color=color, lw=lw, alpha=alpha, zorder=1)

def _draw_panel(ax, title, xy_t, xy_tp1, P_tij, k=4):
    """
    Draw one panel (either teacher or student) for frame t->t+1.
    """
    V = P_tij.shape[-1]

    # context skeletons
    _draw_hand_skeleton(ax, xy_t,   color='#C8C8C8', lw=1.0, alpha=0.7)
    _draw_hand_skeleton(ax, xy_tp1, color='#C8C8C8', lw=1.0, alpha=0.7)

    # nodes
    ax.scatter(xy_t[:,0],   xy_t[:,1],   s=55, c='#2B6CB0', zorder=3)  # blue
    ax.scatter(xy_tp1[:,0], xy_tp1[:,1], s=55, c='#DD6B20', zorder=3)  # orange

    # labels
    for i in range(V):
        ax.text(xy_t[i,0],   xy_t[i,1],   f"{i}", fontsize=8, ha='center', va='center',
                color='white', zorder=4)
        ax.text(xy_tp1[i,0], xy_tp1[i,1], f"{i}", fontsize=8, ha='center', va='center',
                color='white', zorder=4)

    # edges (top-k per row)
    edges = _collect_topk_edges(P_tij, k=k, remove_self=True)
    if len(edges) > 0:
        weights = np.array([w for _,_,w in edges], dtype=float)
        for i, j, w in edges:
            tier, width, color, ls = _edge_style(weights, w)
            ax.plot([xy_t[i,0], xy_tp1[j,0]], [xy_t[i,1], xy_tp1[j,1]],
                    ls=ls, lw=width, color=color, alpha=0.95, zorder=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 2.5)   # accommodates the separation between hands
    ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')

def visualize_teacher_student_edges(
    x_bctv: torch.Tensor,
    teacher_btij: torch.Tensor,   # [B, T,   V, V]    (probs)
    student_btij: torch.Tensor,   # [B, T-1, V, V]    (probs or logits OK if softmaxed before)
    epoch: int,
    save_dir: str,
    sample_idx: int = 0,
    frame_t: int | None = None,
    k: int = 4,
    fname_prefix: str = "teacher_student"
) -> str:
    """
    Make a side-by-side figure:
      Left  = teacher edges (t->t+1)
      Right = student edges (t->t+1) at current epoch
    Returns: path to saved image.
    """
    os.makedirs(save_dir, exist_ok=True)

    B, _, T, V = x_bctv.shape
    t = _choose_t(T, frame_t)

    # positions
    xy_t, xy_tp1, _ = _frame_xy(x_bctv, b=sample_idx, t=t)

    # teacher slice [V,V]
    P_teach = teacher_btij[sample_idx, t].detach()
    # student slice (note: student has T-1 time steps)
    P_stud  = student_btij[sample_idx, t].detach()

    # if student_btij are logits, softmax them
    if P_stud.ndim == 2 and (P_stud.min() < 0 or P_stud.max() > 1.0):
        P_stud = torch.softmax(P_stud, dim=-1)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _draw_panel(axes[0], f"Teacher  (t→t+1)", xy_t, xy_tp1, P_teach, k=k)
    _draw_panel(axes[1], f"Student  (t→t+1) — epoch {epoch}", xy_t, xy_tp1, P_stud,  k=k)

    out_path = os.path.join(save_dir, f"{fname_prefix}_epoch{epoch:03d}_t{t:02d}.png")
    fig.suptitle(f"Temporal edges (t→t+1) — sample {sample_idx}, t={t}", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
