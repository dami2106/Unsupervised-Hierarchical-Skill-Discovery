import os
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.svm import SVC  # use SVC(probability=True); LinearSVC has no predict_proba
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from joblib import dump
import pandas as pd

from skill_helpers import *  # assumes build_endability_dataset, get_unique_skills etc.

# ----------------------------
# Global config
# ----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

parser = argparse.ArgumentParser(description="Train PU-learning models for skill endability.")
parser.add_argument('--dir', type=str, default='Traces/stone_pick_static', help='Base directory for data')
parser.add_argument('--skills_dirname', type=str, default='asot_predicted', help='Subdirectory for skills')
parser.add_argument('--features_name', type=str, default='pca_features_650', help='Feature set name')
parser.add_argument('--old_data_mode', action='store_true', help='Use old data mode')
parser.add_argument('--save_dir', type=str, default='pu_end_models', help='Directory to save models and results (default: <dir>/pu_start_models_asot)')
args = parser.parse_args()

dir_ = args.dir
skills_dirname = args.skills_dirname
features_name = args.features_name
old_data_mode = args.old_data_mode

if old_data_mode:
    print("[WARN] Using OLD DATA MODE (for compatibility with older datasets)")

models_dir = os.path.join(dir_, args.save_dir )
os.makedirs(models_dir, exist_ok=True)
skills_dir = os.path.join(dir_, skills_dirname)
files = os.listdir(skills_dir)

# ----------------------------
# PU builder
# ----------------------------
def make_pu_clf(
    method: str = "elkanoto",         # "elkanoto" | "bagging"
    C: float = 10.0,
    kernel: str = "rbf",              # "linear" or "rbf"
    gamma: str | float = "scale",     # ignored by linear kernel (safe to pass)
    hold_out_ratio: float = 0.2,      # used by Elkanoto
    n_estimators: int = 15,           # used by BaggingPuClassifier
    seed: int = SEED,
):
    """
    Builds a PU-learning estimator that exposes predict_proba.
    Base classifier: SVC(probability=True). Standardized inputs.
    """
    base = make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=seed)
    )

    if method.lower() == "bagging":
        # Bagging over subsamples of unlabeled examples
        pu_estimator = BaggingPuClassifier(base_estimator=base, n_estimators=n_estimators, random_state=seed)
    else:
        # Classic Elkan–Noto PU
        pu_estimator = ElkanotoPuClassifier(estimator=base, hold_out_ratio=hold_out_ratio, random_state=seed)

    return pu_estimator

# ----------------------------
# Threshold selection helpers
# ----------------------------
def best_threshold_from_pr(y_true, p_scores):
    prec, rec, thr = precision_recall_curve(y_true, p_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)

    if len(thr) == 0:
        # degenerate (all scores same); fall back
        best_idx = int(np.nanargmax(f1s))
        return 0.5, float(f1s[best_idx])

    # thresholds correspond to points 1..n in (prec, rec)
    valid = f1s[1:]
    best_idx = int(np.nanargmax(valid)) + 1  # shift back into full f1s indexing
    return float(thr[best_idx - 1]), float(f1s[best_idx])

def safe_n_splits(y, groups, requested=5):
    """
    Ensure we don't request more CV folds than the number of groups
    or than the minority-class count.
    """
    n_groups = len(np.unique(groups))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    # For StratifiedGroupKFold feasibility: at least one positive per fold
    upper_by_class = max(1, min(pos, neg))
    return max(2, min(requested, n_groups, upper_by_class))

# ----------------------------
# Group-aware CV: model selection by PR-AUC (Average Precision)
# ----------------------------
def cv_score_for_params(X, y, groups, *, method, C, kernel, gamma,
                        hold_out_ratio=0.2, n_estimators=15, seed=SEED, requested_splits=5):
    n_splits = safe_n_splits(y, groups, requested_splits)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ap_scores = []
    for tr, va in gkf.split(X, y, groups):
        pu = make_pu_clf(method=method, C=C, kernel=kernel, gamma=gamma,
                         hold_out_ratio=hold_out_ratio, n_estimators=n_estimators, seed=seed)
        pu.fit(X[tr], y[tr])
        proba = pu.predict_proba(X[va])[:, 1]
        ap_scores.append(average_precision_score(y[va], proba))
    return float(np.mean(ap_scores)) if len(ap_scores) else float("nan")

# Search space includes linear kernel (Req #5)
METHODS = ["elkanoto"]
Cs      = [10]
KERNELS = ["rbf"]
GAMMAS  = ["scale"]   # only used for rbf (safe to pass for linear)
BAG_N   = [10, 25]     # for bagging
HOLDOUT = [0.2]        # for elkanoto

def pick_best_hparams(X, y, groups, seed=SEED):
    best = None  # (ap, params_dict)
    for method in METHODS:
        for C in Cs:
            for kernel in KERNELS:
                gamma_list = GAMMAS if kernel == "rbf" else ["scale"]
                for gamma in gamma_list:
                    if method == "bagging":
                        for n_estimators in BAG_N:
                            ap = cv_score_for_params(
                                X, y, groups, method=method, C=C, kernel=kernel, gamma=gamma,
                                n_estimators=n_estimators, seed=seed
                            )
                            params = dict(method=method, C=C, kernel=kernel, gamma=gamma,
                                          n_estimators=n_estimators)
                            if (best is None) or (ap > best[0]):
                                best = (ap, params)
                    else:
                        for hold_out_ratio in HOLDOUT:
                            ap = cv_score_for_params(
                                X, y, groups, method=method, C=C, kernel=kernel, gamma=gamma,
                                hold_out_ratio=hold_out_ratio, seed=seed
                            )
                            params = dict(method=method, C=C, kernel=kernel, gamma=gamma,
                                          hold_out_ratio=hold_out_ratio)
                            if (best is None) or (ap > best[0]):
                                best = (ap, params)
    # Fallback if search failed (shouldn't happen)
    if best is None:
        best = (float("nan"), dict(method="elkanoto", C=10.0, kernel="rbf", gamma="scale", hold_out_ratio=0.2))
    return best  # (best_ap, best_params)

# ----------------------------
# Stable threshold via grouped CV (median of fold-wise best F1 thresholds)
# ----------------------------
def cv_threshold(X, y, groups, params, seed=SEED, requested_splits=5):
    n_splits = safe_n_splits(y, groups, requested_splits)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    thrs = []
    val_f1s = []
    for tr, va in gkf.split(X, y, groups):
        pu = make_pu_clf(**params, seed=seed)
        pu.fit(X[tr], y[tr])
        proba = pu.predict_proba(X[va])[:, 1]
        thr, f1 = best_threshold_from_pr(y[va], proba)
        thrs.append(thr)
        val_f1s.append(f1)
    if len(thrs) == 0:
        return 0.5, float("nan")
    return float(np.median(thrs)), float(np.nanmean(val_f1s))

# ----------------------------
# Train final PU on the full training split (per skill)
# ----------------------------
def fit_final_pu(X, y, *, params, seed=SEED):
    pu = make_pu_clf(**params, seed=seed)
    pu.fit(X, y)
    return pu

# ----------------------------
# Main training / evaluation loop
# ----------------------------
results = {}
skills = get_unique_skills(skills_dir, files)

for skill in skills:
    # Load dataset
    X, y, groups = build_endability_dataset(dir_, skill, files, features_dirname=features_name, old_data_mode=old_data_mode, skills_dir=skills_dirname)

    # Shuffle for reproducibility
    rng_np = np.random.RandomState(SEED)
    perm = rng_np.permutation(len(X))
    X, y, groups = X[perm], y[perm], groups[perm]

    # Grouped Train/Test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    print(f"Skill: {skill}")
    print("train balance:", np.bincount(y_train))
    print("test  balance:",  np.bincount(y_test))

    # (1) Group-aware HParam search with PR-AUC
    best_ap, best_params = pick_best_hparams(X_train, y_train, groups_train, seed=SEED)
    print(f"[{skill}] Best AP={best_ap:.4f} with params={best_params}")

    # (2) Stable threshold via CV (median of per-fold best F1)
    thr_cv, mean_val_f1 = cv_threshold(X_train, y_train, groups_train, best_params, seed=SEED)
    print(f"[{skill}] CV-median threshold={thr_cv:.3f} (mean val F1≈{mean_val_f1:.3f})")

    # Fit final PU on all training data using best params
    clf = fit_final_pu(X_train, y_train, params=best_params, seed=SEED)

    # Inspect probabilities & evaluate on held-out test using the CV threshold
    proba_test = clf.predict_proba(X_test)[:, 1]
    print("min/max prob:", float(proba_test.min()), float(proba_test.max()))
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds_t = (proba_test >= t).astype(int)
        print(t, int(preds_t.sum()))
    print("Chosen threshold (CV median of F1-best):", thr_cv)

    y_pred = (proba_test >= thr_cv).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results[skill] = {
        "threshold": float(thr_cv),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "val_f1": float(mean_val_f1),
        "best_ap": float(best_ap),
        "best_params": best_params
    }

    # Persist model + metadata
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
    try:
        dump(clf, model_path)
        with open(meta_path, 'w') as f:
            json.dump({
                'skill': skill,
                'threshold': float(thr_cv),
                'cv_mean_val_f1': float(mean_val_f1),
                'best_ap': float(best_ap),
                'best_params': best_params,
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1': float(f1),
                'n_train_pos': int((y_train == 1).sum()),
                'n_train_unl': int((y_train == 0).sum()),  # unlabeled in PU terms
                'n_test_pos': int((y_test == 1).sum()),
                'n_test_unl': int((y_test == 0).sum()),
                'seed': SEED,
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save model/metadata for skill {skill}: {e}")

# ----------------------------
# Aggregated reporting
# ----------------------------
rows = []
tot_tn = tot_fp = tot_fn = tot_tp = 0

for skill, res in results.items():
    cm = res["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    tot_tn += int(tn); tot_fp += int(fp); tot_fn += int(fn); tot_tp += int(tp)

    support_pos = int(tp + fn)
    support_unl = int(tn + fp)
    support_all = support_pos + support_unl
    acc = (tp + tn) / support_all if support_all else float("nan")

    rows.append({
        "skill": skill,
        "pos_support": support_pos,
        "unl_support": support_unl,
        "threshold": res["threshold"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "accuracy": acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

# Overall (micro) metrics
overall_support = tot_tp + tot_fp + tot_fn + tot_tn
overall_precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) else 0.0
overall_recall    = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) else 0.0
overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
              if (overall_precision + overall_recall) > 0 else 0.0)
overall_accuracy  = (tot_tp + tot_tn) / overall_support if overall_support else float("nan")

# Macro (mean across skills)
macro_precision = float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
macro_recall    = float(np.mean([r["recall"]    for r in rows])) if rows else float("nan")
macro_f1        = float(np.mean([r["f1"]        for r in rows])) if rows else float("nan")
macro_accuracy  = float(np.mean([r["accuracy"]  for r in rows])) if rows else float("nan")

# Pretty print
print("\n" + "="*80)
print("PER-SKILL METRICS (sorted by F1 desc)")
print("="*80)

df = pd.DataFrame(rows)
df = df.sort_values("f1", ascending=False)
disp_cols = ["skill", "pos_support", "unl_support", "threshold",
             "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
for c in ["threshold", "precision", "recall", "f1", "accuracy"]:
    df[c] = df[c].astype(float).round(3)
print(df[disp_cols].to_string(index=False))

# Save per-skill metrics table & overall summary
metrics_csv = os.path.join(models_dir, 'per_skill_metrics.csv')
metrics_json = os.path.join(models_dir, 'summary_metrics.json')
try:
    df.to_csv(metrics_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({
            'overall': {
                'support': int(overall_support),
                'tp': int(tot_tp), 'fp': int(tot_fp), 'fn': int(tot_fn), 'tn': int(tot_tn),
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(overall_f1),
                'accuracy': float(overall_accuracy)
            },
            'macro': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1),
                'accuracy': float(macro_accuracy)
            }
        }, f, indent=2)
except Exception as e:
    print(f"[WARN] Failed to save aggregate metrics: {e}")

print("\n" + "="*80)
print("OVERALL (MICRO) METRICS — pooled over all skills")
print("="*80)
print(f"Support (all skills): {overall_support}")
print(f"TP={tot_tp}  FP={tot_fp}  FN={tot_fn}  TN={tot_tn}")
print(f"Precision: {overall_precision:.3f}  Recall: {overall_recall:.3f}  F1: {overall_f1:.3f}  Accuracy: {overall_accuracy:.3f}")

print("\n" + "="*80)
print("MACRO AVERAGES — mean of per-skill metrics")
print("="*80)
print(f"Precision: {macro_precision:.3f}  Recall: {macro_recall:.3f}  F1: {macro_f1:.3f}  Accuracy: {macro_accuracy:.3f}")
print("="*80 + "\n")
