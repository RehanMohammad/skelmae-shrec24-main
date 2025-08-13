# train_edge_pretext.py
import os
import os.path as opt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.IPN import IPNDataset
from model.edge_predictor import EdgePredictor
from utils.visualize_edges import visualize_teacher_student_edges


# =========================
# Hyperparameters (tweak here)
# =========================
EPOCHS          = 50
BATCH_SIZE      = 32
MAX_SEQ         = 80
LR              = 1e-3
WD              = 5e-2
D_MODEL         = 192          # capacity ++
D_HID           = 192
DROPOUT         = 0.05

# Student/teacher temperatures & sparsity
TAU_STUDENT     = 0.5          # sharper student
TAU_TEACHER     = 0.3          # sharpen teacher (applied to probs)
TOPK_TEACHER    = 4            # sparsify teacher rows

# Masking & regularization
MOTION_Q        = 0.20         # keep rows whose motion >= 20th percentile
LAMBDA_ENT      = 1e-3         # tiny entropy penalty on student

# LR schedule
WARMUP_EPOCHS   = 5
EARLY_PATIENCE  = 8

SAVE_DIR        = "./experiments/edge_pretext"
VIZ_DIR         = "./experiments/edge_pretext/viz_edges"


# =========================
# Utils
# =========================
def _align_student_teacher(logits, P_full):
    """
    Align time dims for metrics:
      logits:  [B, T-1, V, V]  (student)
      P_full:  [B, T,   V, V]  (teacher; last frame zeros)
    Returns:  (logits_use, P_use) with shape [B, T-1, V, V]
    """
    Bt = logits.shape[1]
    Pt = P_full.shape[1]
    if Bt == Pt:
        return logits[:, :-1], P_full[:, :-1]
    if Bt == Pt - 1:
        return logits, P_full[:, :-1]
    raise ValueError(f"Time mismatch: student={Bt}, teacher={Pt} (expected Bt==Pt or Bt==Pt-1)")


def compute_motion_mask(x_bctv, q=MOTION_Q):
    """
    x_bctv: [B, 3, T, V]
    Returns mask M: [B, T-1, V] where row supervised (moving enough).
    Keeps rows whose |v_t(i)| is above the q-quantile of motion magnitudes.
    """
    v = x_bctv[:, :, 1:, :] - x_bctv[:, :, :-1, :]  # [B, 3, T-1, V]
    mag = torch.linalg.vector_norm(v, dim=1)        # [B, T-1, V]
    flat = mag.detach().flatten()
    try:
        thresh = torch.quantile(flat, q)
    except TypeError:
        # older torch uses 'interpolation'
        thresh = torch.quantile(flat, q, interpolation="linear")
    return (mag >= thresh)


def refine_teacher_probs(P_full, tau=TAU_TEACHER, topk=TOPK_TEACHER, remove_self=False, eps=1e-8):
    """
    Row-wise sharpen + optional top-k sparsify on teacher probabilities.
    P_full: [B, T, V, V] (last frame zeros)
    Returns P_ref: same shape.
    """
    P = P_full.clone()

    # Optionally remove self loops (if they exist)
    if remove_self:
        B, T, V, _ = P.shape
        eye = torch.eye(V, device=P.device, dtype=P.dtype).view(1, 1, V, V)
        P = P * (1.0 - eye)

    # Top-k on probabilities: keep largest-k per row, renormalize
    if topk is not None:
        # mask rows that actually have supervision (sum>0)
        rowsum = P.sum(-1, keepdim=True)                         # [B,T,V,1]
        valid = (rowsum > 0).float()

        vals, idx = torch.topk(P, k=min(topk, P.shape[-1]), dim=-1)
        kth = vals[..., -1:].expand_as(P)                        # threshold per row
        keep = (P >= kth).float() * valid

        P = P * keep
        rowsum = P.sum(-1, keepdim=True).clamp_min_(eps)
        P = P / rowsum

    # Temperature sharpening on probs: P^alpha / sum P^alpha  where alpha = 1/tau
    if tau is not None and tau > 0 and abs(tau - 1.0) > 1e-6:
        alpha = 1.0 / tau
        P = P.clamp_min(eps).pow(alpha)
        rowsum = P.sum(-1, keepdim=True).clamp_min_(eps)
        P = P / rowsum

    return P


def ce_H_kl(logits, P_full):
    """
    Returns scalar CE, H(P), KL averaged over valid rows (no masking for reporting).
    """
    logits_use, P = _align_student_teacher(logits, P_full)  # [B, T-1, V, V]
    mask = (P.sum(dim=-1) > 0)                              # [B, T-1, V]
    logQ = F.log_softmax(logits_use, dim=-1)
    CE = (-(P * logQ).sum(dim=-1))[mask].mean()
    P_safe = P.clamp_min(1e-9)
    H = (-(P * P_safe.log()).sum(dim=-1))[mask].mean()
    KL = CE - H
    return CE.item(), H.item(), KL.item()


def topk_agreement(P_full, logits, k=4):
    """
    Proportion of top-k indices that overlap per row (averaged).
    """
    logits_use, P = _align_student_teacher(logits, P_full)
    Q = F.softmax(logits_use, dim=-1)
    topP = P.topk(k, dim=-1).indices
    topQ = Q.topk(k, dim=-1).indices
    overlap = (topQ[..., None, :] == topP[..., :, None]).any(-1).float().mean().item()
    return overlap


def rowwise_kl_loss(logits_btij, teacher_btij, tau_student=TAU_STUDENT,
                    mask=None, lambda_ent=LAMBDA_ENT):
    """
    logits_btij:  [B, T-1, V, V]  (unnormalized)
    teacher_btij:[B, T-1, V, V]  (probabilities; rows sum to 1)
    """
    log_q = F.log_softmax(logits_btij / tau_student, dim=-1)
    p = teacher_btij.clamp_min(1e-8)
    kl = (p * (p.log() - log_q)).sum(dim=-1)  # [B, T-1, V]

    if mask is not None:
        kl = kl[mask]

    loss = kl.mean()

    if lambda_ent > 0.0:
        q = log_q.exp()
        Hq = (-(q * log_q).sum(dim=-1))       # [B, T-1, V]
        if mask is not None:
            Hq = Hq[mask]
        loss = loss + lambda_ent * Hq.mean()

    return loss


# =========================
# Train script
# =========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- datasets -----
    train_set = IPNDataset(
        data_dir="D:\\Dataset\\IPN\\ipn\\landmarks",
        ann_file="D:\\Dataset\\IPN\\ipn\\annotations\\Annot_TrainList_splitted.txt",
        max_seq=MAX_SEQ, normalize=True
    )
    val_set = IPNDataset(
        data_dir="D:\\Dataset\\IPN\\ipn\\landmarks",
        ann_file="D:\\Dataset\\IPN\\ipn\\annotations\\Annot_ValidList_splitted.txt",
        max_seq=MAX_SEQ, normalize=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ----- model / optim -----
    model  = EdgePredictor(d_model=D_MODEL, hid=D_HID, dropout=DROPOUT).to(device)
    optimz = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # cosine with warmup
    warmup = torch.optim.lr_scheduler.LinearLR(optimz, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimz, T_max=EPOCHS - WARMUP_EPOCHS)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        optimz, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    best_val = float('inf')
    no_improve = 0

    def run_epoch(loader, train: bool):
        model.train(mode=train)
        total_loss = 0.0

        ce_vals, H_vals, kl_vals, agr_vals = [], [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in tqdm(loader, desc="train" if train else "val"):
                x = batch['Sequence'].to(device).float()                 # [B,3,T,V]
                P_full_raw = batch['A_temporal_teacher'].to(device).float()  # [B,T,V,V]

                # on-the-fly teacher refinement (sharpen + top-k)
                P_full = refine_teacher_probs(P_full_raw,
                                              tau=TAU_TEACHER,
                                              topk=TOPK_TEACHER,
                                              remove_self=False)

                P = P_full[:, :-1]                                       # [B, T-1, V, V]
                logits = model(x)['logits']                               # [B, T-1, V, V]

                # motion-based row mask
                mask = compute_motion_mask(x, q=MOTION_Q)                 # [B, T-1, V]

                loss = rowwise_kl_loss(logits, P,
                                       tau_student=TAU_STUDENT,
                                       mask=mask,
                                       lambda_ent=LAMBDA_ENT)

                if train:
                    optimz.zero_grad(set_to_none=True)
                    loss.backward()
                    optimz.step()
                else:
                    # diagnostics (masked)
                    logQ = F.log_softmax(logits, dim=-1)
                    CE = (-(P * logQ).sum(dim=-1))
                    CE = CE[mask].mean()

                    P_safe = P.clamp_min(1e-9)
                    H = (-(P * P_safe.log()).sum(dim=-1))
                    H = H[mask].mean()
                    KL = CE - H

                    # top-k agreement on masked rows
                    Q = logQ.exp()
                    topP = P.topk(4, dim=-1).indices
                    topQ = Q.topk(4, dim=-1).indices
                    agree = (topQ[..., None, :] == topP[..., :, None]).any(-1).float()
                    agree = agree[mask].mean()

                    ce_vals.append(CE.item()); H_vals.append(H.item())
                    kl_vals.append(KL.item()); agr_vals.append(agree.item())

                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        metrics = None
        if not train and ce_vals:
            metrics = {
                'CE': sum(ce_vals)/len(ce_vals),
                'H':  sum(H_vals)/len(H_vals),
                'KL': sum(kl_vals)/len(kl_vals),
                'topk@4': sum(agr_vals)/len(agr_vals),
            }
        return avg_loss, metrics

    # ----- train loop -----
    for ep in range(1, EPOCHS + 1):
        tr_loss, _       = run_epoch(train_loader, True)
        val_loss, vstats = run_epoch(val_loader,   False)
        sched.step()

        if vstats is not None:
            print(f"[{ep:03d}/{EPOCHS}] train {tr_loss:.4f} | val {val_loss:.4f} "
                  f"| CE={vstats['CE']:.4f} H(P)={vstats['H']:.4f} "
                  f"KL={vstats['KL']:.4f} topk@4={vstats['topk@4']:.3f}")
        else:
            print(f"[{ep:03d}/{EPOCHS}] train {tr_loss:.4f} | val {val_loss:.4f}")

        # save + early stop
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({'epoch': ep,
                        'model_state': model.state_dict(),
                        'best_val': best_val},
                       opt.join(SAVE_DIR, "best_edge_predictor.pth"))
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE:
                print(f"Early stop at epoch {ep} (no improvement for {EARLY_PATIENCE} epochs).")
                break

        # --- visualize every 5 epochs ---
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                x = batch['Sequence'].to(device).float()                 # [B,3,T,V]

                # sharpen/sparsify the same way you do for training
                P_full_raw = batch['A_temporal_teacher'].to(device).float()  # [B,T,V,V]
                P_full     = refine_teacher_probs(P_full_raw, tau=TAU_TEACHER, topk=TOPK_TEACHER)

                logits = model(x)['logits']                              # [B,T-1,V,V]
                probs  = torch.softmax(logits, dim=-1)

                img_path = visualize_teacher_student_edges(
                    x_bctv=x,
                    teacher_btij=P_full,          # [B,T,V,V]
                    student_btij=probs,           # [B,T-1,V,V], already softmaxed
                    epoch=ep,
                    save_dir=VIZ_DIR,
                    sample_idx=0,
                    frame_t=None,                 # middle frame by default
                    k=4
                )
            print(f"[viz] saved {img_path}")

if __name__ == "__main__":
    main()
