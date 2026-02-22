# bc_train.py
import os
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import itertools
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import argparse

try:
    from .skill_helpers import get_unique_skills
except ImportError:
    # Fallback for when run as a script
    from skill_helpers import get_unique_skills

# ------------------------------
# Static Config
# ------------------------------
OBS_DIM = 512
NUM_BUTTONS = 20
NUM_CAM = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Config we can change
# ------------------------------
BATCH_SIZE = 1024
LR = 3e-4
EPOCHS = 100  # max epochs; early stopping may end sooner
VAL_SPLIT = 0.1
SEED = 42
EARLY_STOP = True  # set False to disable
PATIENCE = 20  # stop if no improvement for this many epochs
MIN_DELTA = 0.0  # minimum improvement in val loss to reset patience


# Set seed for reproducibility
def set_seed(seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

# ------------------------------
# Camera value grid (given)
# ------------------------------

# -10.          -5.80948313  -3.21536913  -1.60949864  -0.61539427
#    0.           0.61539427   1.60949864   3.21536913   5.80948313
#   10.
CAM_VALUES = np.array(
    [
        -10.0,
        -5.80948313,
        -3.21536913,
        -1.60949864,
        -0.61539427,
        0.0,
        0.61539427,
        1.60949864,
        3.21536913,
        5.80948313,
        10.0,
    ],
    dtype=np.float32,
)


def float_values_to_indices(v):
    """
    Map float camera values to nearest index in CAM_VALUES (robust to tiny float noise).
    v: shape (N,)
    return: int64 indices in [0,10]
    """
    v = np.asarray(v, dtype=np.float32)
    idx = np.argmin(np.abs(v[:, None] - CAM_VALUES[None, :]), axis=1)
    return idx.astype(np.int64)


# ------------------------------
# Data utilities
# ------------------------------
def preprocess_actions(A):
    """
    A: np.ndarray (N,22)
      cols 0..19 -> buttons in {0,1}
      col 20 -> yaw from CAM_VALUES
      col 21 -> pitch from CAM_VALUES
    returns:
      buttons: torch.float32 [N,20]
      yaw_idx: torch.int64    [N]
      pitch_idx: torch.int64  [N]
    """
    assert A.shape[1] == 22, f"Expected (N,22), got {A.shape}"
    buttons = torch.tensor(A[:, :NUM_BUTTONS], dtype=torch.float32)
    yaw_idx = torch.tensor(float_values_to_indices(A[:, 20]), dtype=torch.long)
    pitch_idx = torch.tensor(float_values_to_indices(A[:, 21]), dtype=torch.long)
    return buttons, yaw_idx, pitch_idx


class BCDataset(Dataset):
    def __init__(self, obs_np, act_np):
        """
        obs_np: (N,512) float
        act_np: (N,22)
        """
        assert obs_np.shape[0] == act_np.shape[0]
        assert obs_np.shape[1] == OBS_DIM
        self.obs = torch.tensor(obs_np, dtype=torch.float32)
        self.btn, self.yaw, self.pitch = preprocess_actions(act_np)
        # Optional: standardize features (here: z-score per feature)
        self.mean = self.obs.mean(dim=0, keepdim=True)
        self.std = self.obs.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.obs = (self.obs - self.mean) / self.std

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.btn[idx], self.yaw[idx], self.pitch[idx]


def compute_pos_weight(buttons):
    """
    buttons: torch.float32 [N,20]
    returns pos_weight tensor [20] for BCEWithLogitsLoss
    pos_weight = (neg / pos)
    """
    with torch.no_grad():
        pos = buttons.sum(dim=0)  # [20]
        neg = buttons.shape[0] - pos
        pos = torch.clamp(pos, min=1.0)  # avoid div by zero
        pos_weight = (neg / pos).clamp(max=1e6)
    return pos_weight


# ------------------------------
# Model
# ------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=512, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.btn_head = nn.Linear(hidden, NUM_BUTTONS)  # Bernoulli logits
        self.yaw_head = nn.Linear(hidden, NUM_CAM)  # Categorical logits
        self.pitch_head = nn.Linear(hidden, NUM_CAM)

        # small init for stability
        nn.init.xavier_uniform_(self.btn_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.yaw_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.pitch_head.weight, gain=0.01)

    def forward(self, obs):
        h = self.backbone(obs)
        return self.btn_head(h), self.yaw_head(h), self.pitch_head(h)


# ------------------------------
# Loss & metrics
# ------------------------------
@dataclass
class LossWeights:
    lam_btn: float = 1.0
    lam_cam: float = 1.0  # applies to both yaw and pitch (sum)


def bc_loss(
    btn_logits,
    btn_targets,
    yaw_logits,
    yaw_idx,
    pitch_logits,
    pitch_idx,
    pos_weight=None,
    w: LossWeights = LossWeights(),
    ce_label_smoothing: float = 0.0,
):
    if pos_weight is None:
        bce = F.binary_cross_entropy_with_logits(
            btn_logits, btn_targets, reduction="mean"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            btn_logits,
            btn_targets,
            reduction="mean",
            pos_weight=pos_weight.to(btn_logits.device),
        )
    ce_yaw = F.cross_entropy(yaw_logits, yaw_idx, label_smoothing=ce_label_smoothing)
    ce_pitch = F.cross_entropy(pitch_logits, pitch_idx, label_smoothing=ce_label_smoothing)
    return w.lam_btn * bce + w.lam_cam * (ce_yaw + ce_pitch), {
        "bce": bce.item(),
        "ce_yaw": ce_yaw.item(),
        "ce_pitch": ce_pitch.item(),
    }


@torch.no_grad()
def eval_metrics(
    btn_logits, btn_targets, yaw_logits, yaw_idx, pitch_logits, pitch_idx, thresh=0.5
):
    # Buttons: Hamming F1-ish components and exact-match combo rate
    probs = torch.sigmoid(btn_logits)
    preds = (probs > thresh).float()
    tp = (preds.eq(1) & btn_targets.eq(1)).sum().item()
    fp = (preds.eq(1) & btn_targets.eq(0)).sum().item()
    fn = (preds.eq(0) & btn_targets.eq(1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Exact combo match (buttons only)
    combo_exact = (preds.eq(btn_targets).all(dim=1)).float().mean().item()

    # Camera accuracy
    yaw_acc = (yaw_logits.argmax(dim=-1) == yaw_idx).float().mean().item()
    pitch_acc = (pitch_logits.argmax(dim=-1) == pitch_idx).float().mean().item()

    return {
        "btn_precision": precision,
        "btn_recall": recall,
        "btn_f1": f1,
        "btn_combo_exact": combo_exact,
        "yaw_acc": yaw_acc,
        "pitch_acc": pitch_acc,
    }


@torch.no_grad()
def calibrate_thresholds(model, val_loader):
    model.eval()
    all_logits, all_targets = [], []
    for obs, btn_t, _, _ in val_loader:
        obs = obs.to(DEVICE)
        btn_l, _, _ = model(obs)
        all_logits.append(btn_l.cpu())
        all_targets.append(btn_t)
    logits = torch.cat(all_logits, 0)  # [Nv, 20]
    targets = torch.cat(all_targets, 0)  # [Nv, 20]

    thrs = torch.zeros(20)
    for i in range(20):
        y = targets[:, i].numpy()
        p = torch.sigmoid(logits[:, i]).numpy()
        # search thresholds on quantiles (fast & good enough)
        cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 50)))
        best_f1, best_t = 0.0, 0.5
        for t in cand:
            pred = (p >= t).astype(np.float32)
            tp = (pred * y).sum()
            fp = (pred * (1 - y)).sum()
            fn = ((1 - pred) * y).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thrs[i] = best_t
    return thrs


# ------------------------------
# Training
# ------------------------------
def train_epoch(
    model,
    loader,
    optimizer,
    pos_weight=None,
    grad_clip_norm: float = 0.0,
    btn_label_smoothing: float = 0.0,
    ce_label_smoothing: float = 0.0,
    gaussian_noise_std: float = 0.0,
):
    model.train()
    total_loss = 0.0
    for obs, btn_t, yaw_t, pitch_t in loader:
        obs = obs.to(DEVICE)
        if gaussian_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * gaussian_noise_std
        btn_t = btn_t.to(DEVICE)
        if btn_label_smoothing > 0.0:
            btn_t = btn_t * (1.0 - btn_label_smoothing) + 0.5 * btn_label_smoothing
        yaw_t = yaw_t.to(DEVICE)
        pitch_t = pitch_t.to(DEVICE)

        btn_l, yaw_l, pitch_l = model(obs)
        loss, _ = bc_loss(
            btn_l,
            btn_t,
            yaw_l,
            yaw_t,
            pitch_l,
            pitch_t,
            pos_weight,
            ce_label_smoothing=ce_label_smoothing,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * obs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, pos_weight=None, thresh=0.5, ce_label_smoothing: float = 0.0):
    model.eval()
    total_loss = 0.0
    agg = {
        "btn_precision": 0,
        "btn_recall": 0,
        "btn_f1": 0,
        "btn_combo_exact": 0,
        "yaw_acc": 0,
        "pitch_acc": 0,
    }
    n = 0
    for obs, btn_t, yaw_t, pitch_t in loader:
        bsz = obs.size(0)
        n += bsz
        obs = obs.to(DEVICE)
        btn_t = btn_t.to(DEVICE)
        yaw_t = yaw_t.to(DEVICE)
        pitch_t = pitch_t.to(DEVICE)

        btn_l, yaw_l, pitch_l = model(obs)
        loss, _ = bc_loss(
            btn_l, btn_t, yaw_l, yaw_t, pitch_l, pitch_t, pos_weight, ce_label_smoothing=ce_label_smoothing
        )
        total_loss += loss.item() * bsz
        m = eval_metrics(btn_l, btn_t, yaw_l, yaw_t, pitch_l, pitch_t, thresh=thresh)
        for k in agg:
            agg[k] += m[k] * bsz

    metrics = {k: v / n for k, v in agg.items()}
    metrics["loss"] = total_loss / n
    return metrics


@torch.no_grad()
def eval_epoch_with_thresholds(model, loader, per_button_thr, pos_weight=None, ce_label_smoothing: float = 0.0):
    model.eval()
    total_loss = 0.0
    agg = {
        "btn_precision": 0,
        "btn_recall": 0,
        "btn_f1": 0,
        "btn_combo_exact": 0,
        "yaw_acc": 0,
        "pitch_acc": 0,
    }
    n = 0
    for obs, btn_t, yaw_t, pitch_t in loader:
        bsz = obs.size(0)
        n += bsz
        obs = obs.to(DEVICE)
        btn_t = btn_t.to(DEVICE)
        yaw_t = yaw_t.to(DEVICE)
        pitch_t = pitch_t.to(DEVICE)

        btn_l, yaw_l, pitch_l = model(obs)

        # compute loss (same as train/val)
        loss, _ = bc_loss(
            btn_l, btn_t, yaw_l, yaw_t, pitch_l, pitch_t, pos_weight, ce_label_smoothing=ce_label_smoothing
        )
        total_loss += loss.item() * bsz

        # apply calibrated thresholds for metrics
        probs = torch.sigmoid(btn_l)
        preds = (probs >= per_button_thr.to(probs.device)).float()

        tp = (preds.eq(1) & btn_t.eq(1)).sum().item()
        fp = (preds.eq(1) & btn_t.eq(0)).sum().item()
        fn = (preds.eq(0) & btn_t.eq(1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        combo_exact = (preds.eq(btn_t).all(dim=1)).float().mean().item()
        yaw_acc = (yaw_l.argmax(dim=-1) == yaw_t).float().mean().item()
        pitch_acc = (pitch_l.argmax(dim=-1) == pitch_t).float().mean().item()

        agg["btn_precision"] += precision * bsz
        agg["btn_recall"] += recall * bsz
        agg["btn_f1"] += f1 * bsz
        agg["btn_combo_exact"] += combo_exact * bsz
        agg["yaw_acc"] += yaw_acc * bsz
        agg["pitch_acc"] += pitch_acc * bsz

    metrics = {k: v / n for k, v in agg.items()}
    metrics["loss"] = total_loss / n
    return metrics


# ------------------------------
# Decode to (B,22) action vectors
# ------------------------------
@torch.no_grad()
def decode_actions(button_logits, yaw_logits, pitch_logits, per_button_thr, topk=None):
    """
    Returns a torch.FloatTensor (B,22) with buttons in {0,1} and camera in float values.
    If topk is set (e.g., 2), clamps to at most k pressed buttons.
    """
    probs = torch.sigmoid(button_logits)  # [B,20]
    btn = (probs >= per_button_thr.to(probs.device)).float()

    # if topk is None:
    #     btn = (probs > threshold).float()
    # else:
    #     btn = torch.zeros_like(probs)
    #     topk_idx = probs.topk(k=topk, dim=-1).indices
    #     btn.scatter_(1, topk_idx, 1.0)

    yaw_idx = yaw_logits.argmax(dim=-1)  # [B]
    pitch_idx = pitch_logits.argmax(dim=-1)  # [B]
    cam_vals = torch.tensor(
        CAM_VALUES, dtype=torch.float32, device=button_logits.device
    )
    yaw_vals = cam_vals[yaw_idx]
    pitch_vals = cam_vals[pitch_idx]

    B = btn.size(0)
    act = torch.zeros(B, 22, device=button_logits.device)
    act[:, :NUM_BUTTONS] = btn
    act[:, 20] = yaw_vals
    act[:, 21] = pitch_vals
    return act


# ------------------------------
# Main
# ------------------------------
@dataclass
class HyperParams:
    batch_size: int = BATCH_SIZE
    lr: float = LR
    weight_decay: float = 1e-4
    epochs: int = EPOCHS
    early_stop: bool = EARLY_STOP
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    hidden: int = 512
    dropout: float = 0.1
    lam_btn: float = 1.0
    lam_cam: float = 1.0
    use_pos_weight: bool = True
    scheduler: str = "none"  # one of: none, cosine, plateau
    grad_clip_norm: float = 0.0
    gaussian_noise_std: float = 0.0
    btn_label_smoothing: float = 0.0
    ce_label_smoothing: float = 0.0
    swa: bool = False
    swa_start_epoch: int = 999999  # if <= epochs, SWA kicks in near end


def train_eval_bc_once(
    obs_np,
    act_np,
    skill_name: str,
    save_path: str,
    h: HyperParams,
    objective: str = "loss",  # or btn_f1
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    os.makedirs(save_path, exist_ok=True)

    dataset = BCDataset(obs_np, act_np)

    # Train/val split
    val_len = int(len(dataset) * VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds, batch_size=h.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=h.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Pos-weight computed on TRAIN ONLY
    train_buttons = torch.stack(
        [train_ds[i][1] for i in range(len(train_ds))], dim=0
    )  # [train_N,20]
    pos_weight = compute_pos_weight(train_buttons) if h.use_pos_weight else None

    model = PolicyNet(hidden=h.hidden, dropout=h.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h.lr, weight_decay=h.weight_decay)

    # scheduler setup
    if h.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=h.epochs)
    elif h.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, h.patience // 3))
    else:
        scheduler = None

    # SWA setup
    swa_model = None
    swa_scheduler = None
    if h.swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        # attach a separate annealing LR for SWA phase if using cosine/none
        swa_scheduler = SWALR(optimizer, swa_lr=h.lr * 0.5)

    best_val = math.inf
    best_epoch = 0
    no_improve = 0
    best_metrics_record = None
    loss_weights = LossWeights(lam_btn=h.lam_btn, lam_cam=h.lam_cam)
    for epoch in range(1, h.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            pos_weight,
            grad_clip_norm=h.grad_clip_norm,
            btn_label_smoothing=h.btn_label_smoothing,
            ce_label_smoothing=h.ce_label_smoothing,
            gaussian_noise_std=h.gaussian_noise_std,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            pos_weight,
            thresh=0.5,
            ce_label_smoothing=h.ce_label_smoothing,
        )

        if verbose:
            print(
                f"[{epoch:03d}] train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"F1={val_metrics['btn_f1']:.3f} | "
                f"ComboExact={val_metrics['btn_combo_exact']:.3f} | "
                f"YawAcc={val_metrics['yaw_acc']:.3f} | PitchAcc={val_metrics['pitch_acc']:.3f}"
            )

        # Check improvement w.r.t. best validation loss
        current_objective = val_metrics["loss"] if objective == "loss" else -val_metrics.get("btn_f1", 0.0)
        best_objective = best_val if objective == "loss" else - (best_metrics_record.get("btn_f1", 0.0) if best_metrics_record else -math.inf)
        if current_objective < (best_objective - h.min_delta):
            if objective == "loss":
                best_val = val_metrics["loss"]
            best_metrics_record = val_metrics.copy()
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": dataset.mean,
                    "std": dataset.std,
                    "pos_weight": pos_weight,
                },
                save_path + f"/bc_model_{skill_name}.pt",
            )
            if verbose:
                if objective == "loss":
                    print(
                        f"  New best val (loss): {best_val:.4f} (epoch {epoch:03d}) — model checkpoint saved."
                    )
                else:
                    print(
                        f"  New best val (F1): {val_metrics['btn_f1']:.4f} (epoch {epoch:03d}) — model checkpoint saved."
                    )
        else:
            no_improve += 1
            if h.early_stop and no_improve >= h.patience:
                if verbose:
                    if objective == "loss":
                        print(
                            f"Early stopping at epoch {epoch:03d} — no val improvement for {h.patience} epochs. Best at epoch {best_epoch:03d}."
                        )
                break

        # SWA update near the end
        if h.swa and epoch >= h.swa_start_epoch and swa_model is not None:
            swa_model.update_parameters(model)
            if swa_scheduler is not None:
                swa_scheduler.step()
        else:
            if scheduler is not None:
                if h.scheduler == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

    if verbose:
        print(
            f"Saved best model to {save_path}, best epoch={best_epoch:03d}"
        )

    # Load best checkpoint before calibration and final eval
    ckpt_path = save_path + f"/bc_model_{skill_name}.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model"])  # use best weights for calibration
        if verbose:
            print(f"Loaded best model from {ckpt_path} for threshold calibration.")

    # If SWA, optionally use swa_model for evaluation
    model_for_eval = model
    if h.swa and swa_model is not None and best_epoch >= h.swa_start_epoch:
        # update BN statistics on train data
        try:
            from torch.optim.swa_utils import update_bn
            update_bn(train_loader, swa_model, device=DEVICE)
        except Exception:
            pass
        model_for_eval = swa_model

    # --- AFTER training: calibrate thresholds, re-evaluate, and decode ---
    if verbose:
        print("Calibrating per-button thresholds on validation set...")
    per_button_thr = calibrate_thresholds(model_for_eval, val_loader)
    if verbose:
        print("Per-button thresholds:", per_button_thr)
    torch.save(per_button_thr, save_path + f"/thresholds_{skill_name}.pt")

    # Optional: see calibrated metrics
    cal_metrics = eval_epoch_with_thresholds(
        model_for_eval, val_loader, per_button_thr, pos_weight, ce_label_smoothing=h.ce_label_smoothing
    )
    if verbose:
        print(
            f"Calibrated val | loss={cal_metrics['loss']:.4f} | "
            f"F1={cal_metrics['btn_f1']:.3f} | ComboExact={cal_metrics['btn_combo_exact']:.3f} | "
            f"YawAcc={cal_metrics['yaw_acc']:.3f} | PitchAcc={cal_metrics['pitch_acc']:.3f}"
        )

    # Inference example
    model_for_eval.eval()
    obs, btn_t, yaw_t, pitch_t = next(iter(val_loader))
    obs = obs.to(DEVICE)
    with torch.no_grad():
        btn_l, yaw_l, pitch_l = model_for_eval(obs)
        act_decoded = decode_actions(
            btn_l, yaw_l, pitch_l, per_button_thr=per_button_thr
        )
    if verbose:
        print("Decoded actions sample:", act_decoded[:2].cpu().numpy())

    return best_metrics_record if best_metrics_record is not None else {}, cal_metrics


def train_eval_bc(
    obs_np, act_np, skill_name="minecraft_bc", save_path: str = "Data/test"
):
    h = HyperParams()
    return train_eval_bc_once(obs_np, act_np, skill_name, save_path, h)


def sweep_hyperparams(
    obs_np,
    act_np,
    skill_name: str,
    save_dir: str,
    objective: str = "loss",
    max_trials: int = None,
):
    os.makedirs(save_dir, exist_ok=True)

    # Search space (adjust as needed)
    search_space = {
        "batch_size": [512, 1024, 2048],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-4, 5e-4],
        "hidden": [512, 768, 1024],
        "dropout": [0.0, 0.1, 0.2],
        "scheduler": ["none", "cosine", "plateau"],
        "grad_clip_norm": [0.0, 1.0],
        "gaussian_noise_std": [0.0, 0.01, 0.05],
        "btn_label_smoothing": [0.0, 0.05],
        "ce_label_smoothing": [0.0, 0.05],
        "use_pos_weight": [True, False],
        "swa": [False, True],
    }

    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    if max_trials is not None:
        combos = combos[:max_trials]

    results_csv = os.path.join(save_dir, f"sweep_{skill_name}.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = keys + ["objective", "val_loss", "btn_f1", "btn_combo_exact", "yaw_acc", "pitch_acc"]
        writer.writerow(header)

        best_score = math.inf if objective == "loss" else -math.inf
        best_combo = None

        for i, values in enumerate(combos, start=1):
            h_kwargs = {k: v for k, v in zip(keys, values)}
            h = HyperParams(**h_kwargs)
            # Shorten epochs for sweep but keep early stop
            h.epochs = min(h.epochs, 60)
            if h.swa:
                h.swa_start_epoch = max(5, h.epochs - 10)

            trial_dir = os.path.join(save_dir, f"trial_{i:03d}")
            os.makedirs(trial_dir, exist_ok=True)
            print(f"\n=== Trial {i}/{len(combos)}: {h_kwargs} ===")
            val_metrics, cal_metrics = train_eval_bc_once(
                obs_np, act_np, skill_name, trial_dir, h, objective=objective, verbose=False
            )

            metric_row = [h_kwargs[k] for k in keys] + [objective, cal_metrics.get("loss", math.inf), cal_metrics.get("btn_f1", 0.0), cal_metrics.get("btn_combo_exact", 0.0), cal_metrics.get("yaw_acc", 0.0), cal_metrics.get("pitch_acc", 0.0)]
            writer.writerow(metric_row)
            f.flush()

            score = cal_metrics.get("loss", math.inf) if objective == "loss" else cal_metrics.get("btn_f1", 0.0)
            is_better = (score < best_score) if objective == "loss" else (score > best_score)
            if is_better:
                best_score = score
                best_combo = (h_kwargs, cal_metrics, trial_dir)

    if best_combo is not None:
        print(f"Best combo: {best_combo[0]} -> metrics: {best_combo[1]}")
        print(f"Best model path: {best_combo[2]}")
    else:
        print("No best combo found (unexpected)")


# -------------------------
def get_bc_data_by_episode(
    dir_, files, skill, feature_name="features", skill_dir_name="groundTruth"
):
    """
    Loads raw images per-episode (NHWC float32 in [0,1]) and actions.
    Splits each episode into skill vs other frames via your
    """
    episodes = []
    for file in files:
        with open(os.path.join(dir_, skill_dir_name, file), "r") as f:
            lines = f.read().splitlines()  # len = T

        img_path = os.path.join(dir_, feature_name, file + ".npy")
        act_path = os.path.join(dir_, "raw_actions", file + ".npy")

        images = np.load(img_path)  # [T, H, W, 3] float32 in [0,1]
        actions = np.load(act_path)  # [T]

        if len(lines) != len(actions):
            lines.append(lines[-1])

        if len(lines) != len(images) or len(images) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: labels={len(lines)} images={len(images)} actions={len(actions)}"
            )

        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        ep = dict(
            episode_id=file,
            skill_states=images[skill_mask],  # [Ns, H, W, 3]
            skill_actions=actions[skill_mask],
            other_states=images[other_mask],
            other_actions=actions[other_mask],
            images=images,
            actions=actions,
            skill_mask=skill_mask,
        )
        episodes.append(ep)
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet policy for a specific skill"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped",
        help="Dataset root",
    )
    parser.add_argument(
        "--skills_name", type=str, default="groundTruth", help="Dataset root"
    )
    parser.add_argument(
        "--save_dir", type=str, default="bc_groundTruth", help="Dataset root"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run hyperparameter sweep instead of single training"
    )
    parser.add_argument(
        "--objective", type=str, default="loss", choices=["loss", "btn_f1"], help="Sweep objective"
    )
    parser.add_argument(
        "--max_trials", type=int, default=None, help="Limit number of sweep trials"
    )
    args = parser.parse_args()

    dir_ = args.dir
    skills_dir = os.path.join(dir_, args.skills_name)
    files = os.listdir(skills_dir)
    save_dir = dir_ + "/" + args.save_dir

    files = os.listdir(skills_dir)
    all_skills = get_unique_skills(skills_dir, files)
    print(f"Found skills: {all_skills}")
    # skill = args.skill
    #
    for skill in all_skills:
        episodes = get_bc_data_by_episode(
            dir_, files, skill, skill_dir_name=args.skills_name
        )

        print(f"Loaded {len(episodes)} episodes for skill '{skill}'")

        all_acts = []
        all_obs = []

        for ep in episodes:
            all_acts.append(ep["skill_actions"])
            all_obs.append(ep["skill_states"])

        obs_np = np.concatenate(all_obs, axis=0)
        act_np = np.concatenate(all_acts, axis=0)
        print(f"Loaded data: obs {obs_np.shape}, act {act_np.shape}")

        if args.sweep:
            skill_save_dir = os.path.join(save_dir, f"sweep_{skill}")
            sweep_hyperparams(obs_np, act_np, skill_name=skill, save_dir=skill_save_dir, objective=args.objective, max_trials=args.max_trials)
        else:
            train_eval_bc(obs_np, act_np, skill_name=skill, save_path=save_dir)
