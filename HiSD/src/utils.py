import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import wandb

from metrics import pred_to_gt_match, filter_exclusions
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
# import matplotlib.pyplot as plt
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

def plot_segmentation_gt(gt, pred, mask, gt_uniq=None, pred_to_gt=None, exclude_cls=None, name=''):
    colors = {}

    pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy(), exclude_cls)
    pred_orig = pred_.copy()
    
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())

    pred_to_gt_dict = {pr_lab: gt_lab for pr_lab, gt_lab in zip(pred_opt, gt_opt)}
    pred_ = np.vectorize(pred_to_gt_dict.get)(pred_)

    n_frames = len(pred_)
    if gt_uniq is None:
        gt_uniq = np.unique(gt_)
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    gt_uniq = np.unique(gt_uniq)
    n_class = len(gt_uniq)

    # Colormap and hatching
    cmap_base = plt.get_cmap('tab20')
    cmap_extra = plt.get_cmap('tab20b')
    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', '.', '*']
    use_hatching = n_class > 40

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = ((0, 0, 0), None)
        else:
            base_color = (
                cmap_base(i / 20) if i < 20 else
                cmap_extra((i - 20) / 20)
            )
            hatch = hatch_patterns[i // 40 % len(hatch_patterns)] if use_hatching else None
            colors[label] = (base_color, hatch)

    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=45, pad=20)

    # --- GT plot ---
    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel('GT', fontsize=45, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    gt_segment_boundaries = np.where(gt_[1:] != gt_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

    for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
        label = gt_[start]
        color, hatch = colors.get(label, ((0, 0, 0), None))
        ax.axvspan(start / n_frames, end / n_frames, facecolor=color, hatch=hatch, alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    # --- Pred plot ---
    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel('Ours', fontsize=45, rotation=0, labelpad=60, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred_orig[1:] != pred_orig[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_orig)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        orig_label = pred_orig[start]
        mapped_label = pred_to_gt_dict.get(orig_label, orig_label)
        color, hatch = colors.get(mapped_label, ((0, 0, 0), None))
        ax.axvspan(start / n_frames, end / n_frames, facecolor=color, hatch=hatch, alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_segmentation(pred, mask, name=''):
    colors = {}
    cmap = plt.get_cmap('tab20')
    uniq = np.unique(pred[mask].cpu().numpy())
    n_frames = len(pred)

    # add colors for predictions which do not match to a gt class

    for i, label in enumerate(uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / len(uniq))

    fig = plt.figure(figsize=(16, 2))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred[mask].cpu().numpy()[1:] - pred[mask].cpu().numpy()[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred[mask].cpu().numpy()[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_matrix(mat, gt=None, colorbar=True, title=None, figsize=(10, 5), ylabel=None, xlabel=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot1 = ax.matshow(mat)
    if gt is not None: # plot gt segment boundaries
        gt_change = np.where((np.diff(gt) != 0))[0] + 1
        for ch in gt_change:
            ax.axvline(ch, color='red')
    if colorbar:
        plt.colorbar(plot1, ax=ax)
    if title:
        ax.set_title(f'{title}')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.tick_params(axis='x', labelsize=24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=36)
        ax.tick_params(axis='y', labelsize=24)
    ax.set_aspect('auto')
    fig.tight_layout()
    return fig
def save_skill_ordering(skills, fname, out_dir="skill_orderings"):
    """
    Save the skill ordering (skills at each time step) to a text file.
    Args:
        skills: 1D numpy array or list of skill indices (length = num_frames)
        fname:  Name of the episode/file (used for naming the output file)
        out_dir: Directory to save the text files
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_skills.txt")
    with open(out_path, "w") as f:
        for skill in skills:
            f.write(f"{skill}\n")


def save_skill_segments(skills, fname, out_dir="skill_segments"):
    """
    Save skill segments as a dictionary mapping skill_id to list of active indices.
    Args:
        skills: 1D numpy array or list of skill indices (length = num_frames)
        fname:  Name of the episode/file (used for naming the output file)
        out_dir: Directory to save the text files
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_segments.txt")
    
    # Convert to numpy array for easier processing
    if hasattr(skills, 'cpu'):
        skills = skills.cpu().numpy()
    skills = np.array(skills)
    
    # Find segment boundaries
    skill_changes = np.where(np.diff(skills) != 0)[0] + 1
    segment_boundaries = np.concatenate(([0], skill_changes, [len(skills)]))
    
    # Group consecutive indices by skill
    skill_segments = {}
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        skill_id = skills[start_idx]
        
        if skill_id not in skill_segments:
            skill_segments[skill_id] = []
        
        # Add the range of indices for this skill segment
        skill_segments[skill_id].append(list(range(start_idx, end_idx)))
    
    # Write to file
    with open(out_path, "w") as f:
        f.write("skill_segments:\n")
        for skill_id in sorted(skill_segments.keys()):
            f.write(f"skill_{skill_id}:\n")
            for segment in skill_segments[skill_id]:
                f.write(f"  - {segment}\n")
            f.write("\n")
    
    return skill_segments


def save_matching_mapping(pred_to_gt, out_dir="mapping"):
    """
    Save the Hungarian matching (predicted to GT mapping) to a text file.
    Args:
        pred_to_gt: dict or list of (pred_label, gt_label) pairs
        out_dir: Directory to save the mapping file (default: 'mapping' next to 'skill_orderings')
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mapping.txt")
    with open(out_path, "w") as f:
        if isinstance(pred_to_gt, dict):
            items = pred_to_gt.items()
        else:
            items = pred_to_gt
        for pred_label, gt_label in items:
            f.write(f"{pred_label} {gt_label}\n")