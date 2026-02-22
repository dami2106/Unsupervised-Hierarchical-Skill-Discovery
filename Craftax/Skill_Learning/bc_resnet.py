import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from skill_helpers import *
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

# -------------------------
# Data utilities
# -------------------------
def get_bc_images_by_episode(dir_, files, skill, image_dir_name='top_down_obs', skill_dir_name='groundTruth'):
    """
    Loads raw images per-episode (NHWC float32 in [0,1]) and actions.
    Splits each episode into skill vs other frames via your
    """
    episodes = []
    for file in files:
        with open(os.path.join(dir_, skill_dir_name, file), 'r') as f:
            lines = f.read().splitlines()  # len = T

        img_path = os.path.join(dir_, image_dir_name, file + '.npy')
        act_path = os.path.join(dir_, 'actions', file + '.npy')

        images  = np.load(img_path)   # [T, H, W, 3] float32 in [0,1]
        actions = np.load(act_path)   # [T]

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
            skill_states=images[skill_mask],     # [Ns, H, W, 3]
            skill_actions=actions[skill_mask],
            other_states=images[other_mask],
            other_actions=actions[other_mask],
            images=images,
            actions=actions,
            skill_mask=skill_mask
        )
        episodes.append(ep)
    return episodes


def compute_channel_mean_std(X):
    """
    X: numpy array [N, H, W, 3], float32 in [0,1]
    Returns per-channel mean/std as tuples of floats.
    (Kept for convenience; not used when imagenet_norm=True)
    """
    N, H, W, C = X.shape
    n_pixels = N * H * W
    flat = X.reshape(-1, C).astype(np.float64)
    chan_sum = flat.sum(axis=0)
    chan_sqsum = np.square(flat).sum(axis=0)
    mean = chan_sum / n_pixels
    var = chan_sqsum / n_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


# -------------------------
# Normalization
# -------------------------
class ImageNormalizer:
    """
    Normalizes CHW images using given per-channel mean/std (float32).
    Expects inputs in [0,1].
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        return (x - self.mean) / self.std


# -------------------------
# Dataset (no augmentation; resize only)
# -------------------------
class ImageBCDataset(Dataset):
    """
    Returns (img, action) where:
      - img: torch.float32 [3, 256, 256] normalized
      - action: torch.long
    """
    def __init__(self, X, y, normalizer):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.norm = normalizer

    def __len__(self):
        return self.X.shape[0]

    def _to_chw(self, img):
        # NHWC -> CHW
        return torch.from_numpy(np.transpose(img, (2,0,1))).float()

    def _resize(self, x, target=256):
        # bilinear resize, keep full board
        x = x.unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, idx):
        img = self._to_chw(self.X[idx])          # [3,H,W] in [0,1]
        img = self._resize(img, target=256)
        img = self.norm(img)
        y = torch.tensor(self.y[idx]).long()
        return img, y


# -------------------------
# Model wrapper (ResNet-18/34)
# -------------------------
class ResNetPolicy(nn.Module):
    def __init__(self, n_actions=16, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet34(weights=weights)
        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet34'")

        # Keep standard stride/downsampling; no random crops, so detail is preserved by 256x input
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_actions)

    def forward(self, x):
        return self.backbone(x)


# -------------------------
# Training script
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResNet policy for a specific skill")
    parser.add_argument("--skill", type=str, default="wood", help="Skill to train")
    parser.add_argument("--dir", type=str, default="Traces/stone_pick_static", help="Dataset root")
    parser.add_argument("--skills_name", type=str, default="groundTruth", help="Dataset root")
    parser.add_argument("--save_dir", type=str, default="bc_checkpoints", help="Dataset root")
    parser.add_argument("--image_dir_name", type=str, default="top_down_obs", help="Subdir with per-episode .npy images")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--use_sampler", action="store_true", help="Use WeightedRandomSampler (else shuffle)")
    parser.add_argument("--imagenet_norm", action="store_true", help="Use ImageNet mean/std (recommended for pretrained)")
    args = parser.parse_args()

    dir_ = args.dir
    skills_dir = os.path.join(dir_, args.skills_name)
    files = os.listdir(skills_dir)
    # unique_skills = get_unique_skills(skills_dir, files)
    skill = args.skill

    print(f"===== TRAINING SKILL {skill} (ResNet, pretrained) =====")
    episodes = get_bc_images_by_episode(dir_, files, skill, image_dir_name=args.image_dir_name, skill_dir_name=args.skills_name)

    # Splits by episode
    rng = np.random.default_rng(0)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n = len(idx)
    train_idx = idx[: int(0.8*n)]
    val_idx   = idx[int(0.8*n): int(0.9*n)]
    test_idx  = idx[int(0.9*n):]

    train_eps = [episodes[i] for i in train_idx]
    val_eps   = [episodes[i] for i in val_idx]
    test_eps  = [episodes[i] for i in test_idx]

    def bc_flatten_split_images(episode_dicts, use_skill=True):
        X, y = [], []
        s_key = 'skill_states' if use_skill else 'other_states'
        a_key = 'skill_actions' if use_skill else 'other_actions'
        for ep in episode_dicts:
            X.append(ep[s_key])
            y.append(ep[a_key])
        if len(X) == 0:
            return np.empty((0,274,274,3), dtype=np.float32), np.empty((0,), dtype=int)
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    X_tr, y_tr = bc_flatten_split_images(train_eps, use_skill=True)
    X_va, y_va = bc_flatten_split_images(val_eps,   use_skill=True)
    X_te, y_te = bc_flatten_split_images(test_eps,  use_skill=True)

    print(X_tr.shape, y_tr.shape)
    print(X_va.shape, y_va.shape)
    print(X_te.shape, y_te.shape)

    # Normalization: ImageNet stats recommended for pretrained weights
    if args.imagenet_norm:
        mean_tr = (0.485, 0.456, 0.406)
        std_tr  = (0.229, 0.224, 0.225)
        print("Using ImageNet normalization:", mean_tr, std_tr)
    else:
        mean_tr, std_tr = compute_channel_mean_std(X_tr)
        print("Using dataset normalization:", mean_tr, std_tr)

    normalizer = ImageNormalizer(mean_tr, std_tr)

    # Datasets / loaders (NO augmentation)
    train_ds = ImageBCDataset(X_tr, y_tr, normalizer=normalizer)
    val_ds   = ImageBCDataset(X_va, y_va, normalizer=normalizer)
    test_ds  = ImageBCDataset(X_te, y_te, normalizer=normalizer)

    n_actions = 16
    counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
    inv = np.zeros_like(counts); obs = counts > 0
    inv[obs] = 1.0 / counts[obs]
    sample_w = inv[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    use_mps = torch.backends.mps.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if use_mps else 'cpu'))
    pin_mem = not use_mps

    if args.use_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                                  pin_memory=pin_mem, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=pin_mem, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, pin_memory=pin_mem, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, pin_memory=pin_mem, num_workers=4)

    print("Using device:", device)

    # Model: ResNet-18/34 pretrained
    model = ResNetPolicy(n_actions=n_actions, backbone=args.backbone, pretrained=True).to(device)

    # Loss/optimizer/scheduler (no augments; same schedule logic)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=4, threshold=1e-3,
        threshold_mode='rel', cooldown=2, min_lr=1e-6, verbose=True
    )

    # Train
    best_val = float('inf')
    es_patience = 15
    bad = 0


    ckpt_dir = os.path.join(dir_, args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{skill}_policy_{args.backbone}_pt.pt')

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            tot, correct = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                tot += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
            val_loss = tot / len(val_ds)
            val_acc = correct / len(val_ds)

        sched.step(val_loss)
        cur_lr = opt.param_groups[0]['lr']
        print(f"epoch {epoch:03d} | lr {cur_lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

        # Early stopping + checkpoint
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': best_state,
                'mean': mean_tr,
                'std': std_tr,
                'n_actions': n_actions,
                'skill': skill,
                'arch': f'ResNetPolicy_{args.backbone}_pretrained',
                'epoch': epoch,
                'val_loss': best_val,
                'imagenet_norm': args.imagenet_norm,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= es_patience:
                break

    # Load best
    model.load_state_dict(best_state)
    model.to(device)
    print(f"Loaded best model. Checkpoint saved at: {ckpt_path}")

    # Test
    model.eval()
    with torch.no_grad():
        tot, correct = 0.0, 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        print(f"TEST  NLL {tot/len(test_ds):.4f} | ACC {correct/len(test_ds):.3f}")


if __name__ == "__main__":
    main()
