import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from .skill_helpers import get_unique_skills
    from .minecraft_bc import (
        OBS_DIM,
        NUM_BUTTONS,
        NUM_CAM,
        DEVICE,
        VAL_SPLIT,
        SEED,
        preprocess_actions,
        bc_loss,
        compute_pos_weight,
    )
except ImportError:
    # Fallback for when run as a script
    from skill_helpers import get_unique_skills
    from minecraft_bc import (
        OBS_DIM,
        NUM_BUTTONS,
        NUM_CAM,
        DEVICE,
        VAL_SPLIT,
        SEED,
        preprocess_actions,
        bc_loss,
        compute_pos_weight,
    )


# ------------------------------
# Recurrent Policy (GRU) Head
# ------------------------------


class RecurrentPolicyNet(nn.Module):
    """
    GRU-based BC policy.

    Expects MineCLIP / feature embeddings as observations with shape:
      obs: (batch, seq_len, OBS_DIM)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 512, rnn_layers: int = 1, dropout: float = 0.1):
        super().__init__()

        # 1. Feature extractor (simple MLP on per-frame embeddings)
        self.feat_extract = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Recurrent layer (GRU)
        #    batch_first=True => input (B, T, hidden)
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # 3. Action heads (per time step)
        self.btn_head = nn.Linear(hidden, NUM_BUTTONS)
        self.yaw_head = nn.Linear(hidden, NUM_CAM)
        self.pitch_head = nn.Linear(hidden, NUM_CAM)

    def forward(self, obs: torch.Tensor, h_in: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        obs : (B, T, OBS_DIM) float32
        h_in : optional initial hidden state (num_layers, B, hidden)

        Returns
        -------
        btn_logits : (B, T, NUM_BUTTONS)
        yaw_logits : (B, T, NUM_CAM)
        pitch_logits : (B, T, NUM_CAM)
        h_out : (num_layers, B, hidden)
        """

        # Feature encoding on each frame
        # We apply the MLP per-frame by flattening the sequence dimension, then reshape.
        B, T, D = obs.shape
        x = obs.reshape(B * T, D)
        x = self.feat_extract(x)
        x = x.reshape(B, T, -1)

        # GRU over time
        out, h_out = self.rnn(x, h_in)

        # Decode actions at each time step
        btn_logits = self.btn_head(out)
        yaw_logits = self.yaw_head(out)
        pitch_logits = self.pitch_head(out)
        return btn_logits, yaw_logits, pitch_logits, h_out


# ------------------------------
# Sequence Dataset
# ------------------------------


class SequenceBCDataset(Dataset):
    """
    Builds sliding windows of (states, actions) from per-episode dicts.

    Each sample:
      obs:  (seq_len, OBS_DIM)
      btn:  (seq_len, NUM_BUTTONS)
      yaw:  (seq_len,)
      pitch:(seq_len,)

    IMPORTANT: The windows respect the environment frame skip.
      If skip=8, then consecutive elements in a sequence are 8 ticks apart.
    """

    def __init__(self, episodes: List[Dict[str, Any]], seq_len: int = 32, skip: int = 1):
        self.seq_len = seq_len
        self.skip = max(1, int(skip))
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        # First, compute the maximum feasible sequence length for this skip.
        # We may need to shorten seq_len if episodes are too short.
        max_T = 0
        for ep in episodes:
            states = ep["skill_states"]
            max_T = max(max_T, states.shape[0])

        if max_T <= 1:
            raise ValueError("Episodes are too short to build any sequences.")

        max_seq_len = 1 + (max_T - 1) // self.skip  # largest seq_len s.t. required_len <= max_T
        if max_seq_len < 2:
            raise ValueError("Episodes are too short for the given skip.")

        if self.seq_len > max_seq_len:
            # Reduce to the largest feasible seq_len while keeping the desired skip.
            print(
                f"[SequenceBCDataset] Requested seq_len={self.seq_len} is too large "
                f"for skip={self.skip} and max_T={max_T}; using seq_len={max_seq_len} instead."
            )
            self.seq_len = int(max_seq_len)

        # Required raw length to obtain seq_len frames with step==skip
        required_len = (self.seq_len - 1) * self.skip + 1

        # Collect all skill states to compute global normalization
        all_states = []
        for ep in episodes:
            states = ep["skill_states"]  # expected shape (T, OBS_DIM)
            if states.shape[0] >= required_len:
                all_states.append(states.astype(np.float32))

        if not all_states:
            raise ValueError(
                "No episodes with enough frames for the requested seq_len and skip."
            )

        stacked = np.concatenate(all_states, axis=0)  # (N, OBS_DIM)
        self.mean = torch.tensor(stacked.mean(axis=0, keepdims=True), dtype=torch.float32)
        self.std = torch.tensor(
            np.clip(stacked.std(axis=0, keepdims=True), 1e-6, None),
            dtype=torch.float32,
        )

        # Build sliding windows for each episode, respecting frame skip
        stride = max(1, self.seq_len // 2)
        for ep in episodes:
            states = ep["skill_states"].astype(np.float32)  # (T, OBS_DIM)
            actions = ep["skill_actions"]  # (T, 22)

            T = states.shape[0]

            if T < required_len:
                continue

            # Slide over raw frames, then subsample every `skip` steps
            for t in range(0, T - required_len + 1, stride):
                s_window = states[t : t + required_len : self.skip]
                a_window = actions[t : t + required_len : self.skip]

                # Safety check in case of off-by-one
                if len(s_window) == self.seq_len and len(a_window) == self.seq_len:
                    self.samples.append((s_window, a_window))

        if not self.samples:
            raise ValueError("SequenceBCDataset has no samples after windowing.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        obs_np, act_np = self.samples[idx]

        # Normalize observations with dataset-wide stats
        obs = torch.tensor(obs_np, dtype=torch.float32)
        obs = (obs - self.mean) / self.std

        # Existing helper: converts (N,22) to button/yaw/pitch targets
        btn, yaw, pitch = preprocess_actions(act_np)
        return obs, btn, yaw, pitch


# ------------------------------
# Training Utilities (RNN)
# ------------------------------


@dataclass
class GRUHyperParams:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    hidden: int = 512
    dropout: float = 0.1
    rnn_layers: int = 1
    grad_clip_norm: float = 0.0
    ce_label_smoothing: float = 0.0
    use_pos_weight: bool = True
    seq_len: int = 32


def train_epoch_rnn(
    model: RecurrentPolicyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[torch.Tensor] = None,
    grad_clip_norm: float = 0.0,
    ce_label_smoothing: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for obs, btn_t, yaw_t, pitch_t in loader:
        # obs: (B, T, D)
        obs = obs.to(DEVICE)
        btn_t = btn_t.to(DEVICE)  # (B, T, NUM_BUTTONS)
        yaw_t = yaw_t.to(DEVICE)  # (B, T)
        pitch_t = pitch_t.to(DEVICE)  # (B, T)

        # Forward over entire sequence
        btn_l, yaw_l, pitch_l, _ = model(obs, h_in=None)

        # Flatten time + batch for loss
        B, T, _ = btn_l.shape
        btn_l_flat = btn_l.reshape(B * T, NUM_BUTTONS)
        yaw_l_flat = yaw_l.reshape(B * T, NUM_CAM)
        pitch_l_flat = pitch_l.reshape(B * T, NUM_CAM)

        btn_t_flat = btn_t.reshape(B * T, NUM_BUTTONS)
        yaw_t_flat = yaw_t.reshape(B * T)
        pitch_t_flat = pitch_t.reshape(B * T)

        loss, _ = bc_loss(
            btn_l_flat,
            btn_t_flat,
            yaw_l_flat,
            yaw_t_flat,
            pitch_l_flat,
            pitch_t_flat,
            pos_weight=pos_weight,
            ce_label_smoothing=ce_label_smoothing,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * B * T
        total_count += B * T

    return total_loss / max(1, total_count)


@torch.no_grad()
def eval_epoch_rnn(
    model: RecurrentPolicyNet,
    loader: DataLoader,
    pos_weight: Optional[torch.Tensor] = None,
    ce_label_smoothing: float = 0.0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for obs, btn_t, yaw_t, pitch_t in loader:
        obs = obs.to(DEVICE)
        btn_t = btn_t.to(DEVICE)
        yaw_t = yaw_t.to(DEVICE)
        pitch_t = pitch_t.to(DEVICE)

        btn_l, yaw_l, pitch_l, _ = model(obs, h_in=None)

        B, T, _ = btn_l.shape
        btn_l_flat = btn_l.reshape(B * T, NUM_BUTTONS)
        yaw_l_flat = yaw_l.reshape(B * T, NUM_CAM)
        pitch_l_flat = pitch_l.reshape(B * T, NUM_CAM)

        btn_t_flat = btn_t.reshape(B * T, NUM_BUTTONS)
        yaw_t_flat = yaw_t.reshape(B * T)
        pitch_t_flat = pitch_t.reshape(B * T)

        loss, _ = bc_loss(
            btn_l_flat,
            btn_t_flat,
            yaw_l_flat,
            yaw_t_flat,
            pitch_l_flat,
            pitch_t_flat,
            pos_weight=pos_weight,
            ce_label_smoothing=ce_label_smoothing,
        )

        total_loss += loss.item() * B * T
        total_count += B * T

    return total_loss / max(1, total_count)


@torch.no_grad()
def calibrate_thresholds_rnn(
    model: RecurrentPolicyNet, loader: DataLoader, device: str = DEVICE
) -> torch.Tensor:
    """
    Calibrate a per-button decision threshold based on validation F1.

    This mirrors the calibration used in the non-GRU BC, but operates on
    sequence batches: we flatten (B, T, NUM_BUTTONS) -> (B*T, NUM_BUTTONS).
    """
    model.eval()
    all_logits = []
    all_targets = []

    for obs, btn_t, _, _ in loader:
        obs = obs.to(device)
        # Reset hidden state per batch; validation sequences are independent
        btn_logits, _, _, _ = model(obs, h_in=None)

        # Flatten time + batch
        B, T, _ = btn_logits.shape
        all_logits.append(btn_logits.reshape(B * T, NUM_BUTTONS).cpu())
        all_targets.append(btn_t.reshape(B * T, NUM_BUTTONS))

    if not all_logits:
        # Extremely small dataset / empty loader; fall back to default 0.5 thresholds.
        print(
            "[calibrate_thresholds_rnn] No data available; "
            "using default threshold=0.5 for all buttons."
        )
        return torch.full((NUM_BUTTONS,), 0.5, dtype=torch.float32, device=device)

    logits = torch.cat(all_logits, dim=0)  # (N, NUM_BUTTONS)
    targets = torch.cat(all_targets, dim=0)  # (N, NUM_BUTTONS)

    thrs = torch.zeros(NUM_BUTTONS, dtype=torch.float32)
    for i in range(NUM_BUTTONS):
        y = targets[:, i].numpy()
        p = torch.sigmoid(logits[:, i]).numpy()

        # Fast quantile-based search over candidate thresholds
        if y.sum() == 0:
            # Degenerate case: button never pressed; keep default 0.5
            thrs[i] = 0.5
            continue

        candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 50)))
        best_f1, best_t = 0.0, 0.5

        for t in candidates:
            pred = (p >= t).astype(float)
            tp = (pred * y).sum()
            fp = (pred * (1 - y)).sum()
            fn = ((1 - pred) * y).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thrs[i] = best_t

    return thrs


def _compute_sequence_pos_weight(train_ds) -> Optional[torch.Tensor]:
    """
    Compute pos_weight for BCE from a sequence dataset by flattening all time steps.
    """
    all_buttons = []
    for i in range(len(train_ds)):
        _, btn_seq, _, _ = train_ds[i]  # (T, NUM_BUTTONS)
        all_buttons.append(btn_seq.reshape(-1, NUM_BUTTONS))
    if not all_buttons:
        return None
    buttons = torch.cat(all_buttons, dim=0)  # (N_steps, NUM_BUTTONS)
    return compute_pos_weight(buttons)


def train_eval_gru_bc_once(
    episodes: List[Dict[str, Any]],
    skill_name: str,
    save_path: str,
    h: GRUHyperParams,
    env_skip: int = 8,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Train a GRU-based BC model for a single skill using sequential windows.
    """
    os.makedirs(save_path, exist_ok=True)

    # Build sequence dataset that matches the environment frame skip.
    dataset = SequenceBCDataset(episodes, seq_len=h.seq_len, skip=env_skip)

    # If episodes were too short, the dataset may have reduced seq_len.
    # Keep hyperparams metadata in sync for checkpointing / inspection.
    if h.seq_len != dataset.seq_len:
        if verbose:
            print(
                f"[GRU][{skill_name}] Adjusting seq_len from {h.seq_len} to "
                f"{dataset.seq_len} based on available episode lengths."
            )
        h.seq_len = dataset.seq_len

    # Train/val split at sequence level.
    # Handle tiny datasets gracefully:
    #   - n_seq == 1: use that single sequence for both training and calibration.
    #   - n_seq >= 2: standard train/val split with at least 1 sample in each.
    n_seq = len(dataset)
    if n_seq == 1:
        train_ds = dataset
        val_ds = None
    else:
        val_len = max(1, int(n_seq * VAL_SPLIT))
        train_len = n_seq - val_len
        if train_len <= 0:
            train_len = 1
            val_len = n_seq - train_len
        train_ds, val_ds = random_split(
            dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=h.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=h.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    pos_weight = _compute_sequence_pos_weight(train_ds) if h.use_pos_weight else None

    model = RecurrentPolicyNet(
        obs_dim=OBS_DIM,
        hidden=h.hidden,
        rnn_layers=h.rnn_layers,
        dropout=h.dropout,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=h.lr, weight_decay=h.weight_decay
    )

    best_val = math.inf
    best_epoch = 0
    no_improve = 0
    patience = 10

    for epoch in range(1, h.epochs + 1):
        train_loss = train_epoch_rnn(
            model,
            train_loader,
            optimizer,
            pos_weight=pos_weight,
            grad_clip_norm=h.grad_clip_norm,
            ce_label_smoothing=h.ce_label_smoothing,
        )

        if val_loader is not None:
            val_loss = eval_epoch_rnn(
                model,
                val_loader,
                pos_weight=pos_weight,
                ce_label_smoothing=h.ce_label_smoothing,
            )
        else:
            # No explicit validation set (tiny dataset); use train loss as proxy.
            val_loss = train_loss

        if verbose:
            print(
                f"[GRU][{skill_name}][{epoch:03d}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0

            ckpt = {
                "model": model.state_dict(),
                "mean": dataset.mean,
                "std": dataset.std,
                "pos_weight": pos_weight,
                "hyperparams": h.__dict__,
                "seq_len": h.seq_len,
            }
            torch.save(
                ckpt,
                os.path.join(save_path, f"bc_gru_model_{skill_name}.pt"),
            )
            if verbose:
                print(
                    f"  New best val_loss={best_val:.4f} at epoch {epoch:03d} "
                    f"— checkpoint saved."
                )
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch:03d} "
                        f"(no improvement for {patience} epochs). "
                        f"Best epoch={best_epoch:03d}, best_val={best_val:.4f}."
                )
                break

    # ------------------------------
    # Threshold calibration (per-button)
    # ------------------------------
    if verbose:
        print(f"Calibrating thresholds for {skill_name}...")

    best_ckpt_path = os.path.join(save_path, f"bc_gru_model_{skill_name}.pt")
    checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])

    # Prefer validation set for calibration; if not available (tiny dataset), fall back to train set.
    calib_loader = val_loader if val_loader is not None else train_loader
    thresholds = calibrate_thresholds_rnn(model, calib_loader, device=DEVICE)
    if verbose:
        print(f"Thresholds for {skill_name}: {thresholds}")

    torch.save(thresholds, os.path.join(save_path, f"thresholds_{skill_name}.pt"))

    return {"best_val_loss": best_val, "best_epoch": best_epoch}


def train_eval_gru_bc_for_all_skills(
    dir_: str,
    skills_dir_name: str,
    save_root: str,
    seq_len: int = 32,
):
    """
    High-level entry point:
      - discovers skills
      - builds per-skill episodes via existing get_bc_data_by_episode
      - trains a GRU BC model per skill
    """
    try:
        from .minecraft_bc import get_bc_data_by_episode  # local import to avoid cycles
    except ImportError:
        from minecraft_bc import get_bc_data_by_episode  # local import to avoid cycles

    skills_dir = os.path.join(dir_, skills_dir_name)
    files = os.listdir(skills_dir)

    all_skills = get_unique_skills(skills_dir, files)
    print(f"[GRU] Found skills: {sorted(all_skills)}")

    save_dir = os.path.join(dir_, save_root)
    os.makedirs(save_dir, exist_ok=True)

    h = GRUHyperParams(seq_len=seq_len)

    # Collect per-skill metrics for summary printing at the end
    summary: Dict[str, Dict[str, float]] = {}

    for skill in sorted(all_skills):
        print(f"\n=== [GRU] Training skill '{skill}' ===")
        episodes = get_bc_data_by_episode(
            dir_, files, skill, skill_dir_name=skills_dir_name
        )
        print(f"[GRU] Loaded {len(episodes)} episodes for skill '{skill}'")

        metrics = train_eval_gru_bc_once(
            episodes=episodes,
            skill_name=skill,
            save_path=save_dir,
            h=h,
            verbose=True,
        )
        summary[skill] = metrics
        print(
            f"[GRU] Skill '{skill}' done. "
            f"best_val_loss={metrics['best_val_loss']:.4f} "
            f"at epoch={metrics['best_epoch']}"
        )

    # ------------------------------
    # Final summary across all skills
    # ------------------------------
    if summary:
        print("\n[GRU] Training summary across all skills:")
        print(f"{'Skill':30s} | {'Best Val Loss':>12s} | {'Best Epoch':>10s}")
        print("-" * 60)
        for skill in sorted(summary.keys()):
            m = summary[skill]
            print(
                f"{skill:30s} | {m['best_val_loss']:12.4f} | {int(m['best_epoch']):10d}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GRU-based BC policies on MineCLIP embeddings with frame skip."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped",
        help="Dataset root (same as minecraft_bc).",
    )
    parser.add_argument(
        "--skills_name",
        type=str,
        default="asot_skills",
        help="Skill labels directory name under --dir (same as minecraft_bc).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="bc_gru_asot",
        help="Subdirectory under --dir where GRU models will be saved.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Sequence length (time window) used for GRU training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size of sequences.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs.",
    )
    args = parser.parse_args()

    # Allow quick overrides of a few hyperparams via CLI
    GRUHyperParams.batch_size = args.batch_size  # type: ignore[attr-defined]
    GRUHyperParams.epochs = args.epochs  # type: ignore[attr-defined]

    train_eval_gru_bc_for_all_skills(
        dir_=args.dir,
        skills_dir_name=args.skills_name,
        save_root=args.save_dir,
        seq_len=args.seq_len,
    )


