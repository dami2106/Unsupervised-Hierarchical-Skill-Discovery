import os
import numpy as np
import argparse
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import joblib
import json
import torch
from typing import List, Dict, Iterable, Any
import os
from pathlib import Path
# --------------------------------------------------------------------------------------
# Existing helper functions above
# --------------------------------------------------------------------------------------


def get_fake_acts_shift_gt(gt):

    if not gt:
        return None
    
    changes = []
    for i in range(1, len(gt)):
        if gt[i] != gt[i-1]:
            changes.append(i)
    changes.append(len(gt))
    changes = np.array(changes) 
    changes = changes - 1

    actions = np.zeros(len(gt) + 1)
    actions[changes] = 5

    shifted_gt = gt.copy()
    shifted_gt.insert(0, gt[0])
    
    return actions, shifted_gt

def get_unique_skills(dir_, files):
    unique_skills = set()
    for file in files:
        with open(os.path.join(dir_, file), 'r') as f:
            lines = f.read().splitlines()
        unique_skills.update(lines)
    return unique_skills

def segment_edges(lst, mode):
    if not lst:
        return []

    if mode not in {"start", "end"}:
        raise ValueError("mode must be 'start' or 'end'")

    edges = []
    seg_start = lst[0]

    for i in range(1, len(lst) + 1):
        if i == len(lst) or lst[i] != lst[i - 1] + 1:
            # segment ended at lst[i-1]
            if mode == "start":
                edges.append(seg_start)
            else:  # mode == "end"
                edges.append(lst[i - 1])
            # prepare for next segment
            if i < len(lst):
                seg_start = lst[i]

    return edges

def make_skill_segments(acts, truth):

    segment_actions = [i for i in range(5, 17)]

    def _segments(cum_ends: List[int]) -> List[List[int]]:
        # cum_ends contains 1-based inclusive end-index positions
        bounds = [-1] + cum_ends
        return [list(range(lo + 1, hi + 1)) for lo, hi in zip(bounds[:-1], bounds[1:])]

    # 1-based inclusive segment ends where acts[i] is a boundary action
    cum_ends = [i + 1 for i, a in enumerate(acts) if a in segment_actions]

    idx_groups = _segments(cum_ends)

    # print("IDX GROUPS", idx_groups) 

    #Make sure idx_groups[-1][-1] is len(acts)-1
    if idx_groups[-1][-1] != len(acts)-1:
        idx_groups[-1].append(len(acts)-1)

    skill_segments: Dict[str, List[List[int]]] = {}
    for seg in idx_groups:
        labels = {truth[idx] for idx in seg}
        if len(labels) != 1:
            raise ValueError(f"Mixed labels in segment {seg}: {labels}")
        label = labels.pop()
        skill_segments.setdefault(label, []).append(seg)

    return skill_segments

# [s1, s2, s3],   [s4, s5],    [s6, s7, s8, s9, s10, s11, s12]
# skill_1         skill_2      skill_1 is used twice (s6 - s9, s10 - s11)

# for skill_1, 
# start_states     = [s1, s6]
# end_states       = [s3, s9]
# all_skill_states = [s1, s2, s3,   s6, s7, s8, s9]

# negative_end_skill = [s1, s2, s6, s7, s8]


# ------
# negative_end_all = [s1, s2, s4, s5, s6, s7, s8]
# all_other_states = [s4, s5]

def get_start_end_states(dir_, skill, features_dirname='pca_features'):
    dir_ = Path(dir_)

    start_states = []
    end_states = []
    all_skill_states = []

    negative_end_skill = []
    all_other_states = []   # states from skills != `skill`

    gt_dir = dir_ / 'groundTruth'
    act_dir = dir_ / 'actions'
    feat_dir = dir_ / features_dirname

    files = os.listdir(gt_dir)

    for file in files:
        with open(gt_dir / file, 'r') as f:
            truths = f.read().splitlines()

        feats = np.load(feat_dir / f'{file}.npy')      # shape: [T, D]
        actions = np.load(act_dir / f'{file}.npy')     # shape: [T] or [T, ...]

        # dict: { skill_name: [ [idxs...], [idxs...], ... ] }
        segs_for_skill = make_skill_segments(actions, truths)

        for skill_name, segs in segs_for_skill.items():
            for seg in segs:
                # seg is a list of time indices for this segment
                if skill_name == skill:
                    # starts/ends for the target skill
                    start_states.append(feats[seg[0]])
                    end_states.append(feats[seg[-1]])

                    # all states for the target skill
                    all_skill_states.extend(feats[seg])

                    # "negative end" within the skill = everything except the segment's last state
                    if len(seg) > 1:
                        negative_end_skill.extend(feats[seg[:-1]])
                else:
                    # all states from other skills
                    all_other_states.extend(feats[seg])

    # negative_end_all = negative_end_skill plus all states from other skills
    negative_end_all = list(negative_end_skill) + list(all_other_states)

    return (
        np.asarray(start_states),
        np.asarray(end_states),
        np.asarray(all_skill_states),
        np.asarray(negative_end_skill),
        np.asarray(negative_end_all),
        np.asarray(all_other_states),
    )

def evaluate_ocsvm(X_pos_train, X_pos_val, X_neg_val, nu, gamma):
    # Scale + fit on positive-only training split
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                        OneClassSVM(kernel="rbf", nu=nu, gamma=gamma))
    clf.fit(X_pos_train)

    # Predict on validation (pos + neg)
    X_val = np.vstack([X_pos_val, X_neg_val])
    y_true = np.hstack([np.ones(len(X_pos_val), dtype=int),
                        np.zeros(len(X_neg_val), dtype=int)])
    y_pred = clf.predict(X_val)
    y_pred = (y_pred == 1).astype(int)  # map {+1,-1} -> {1,0}

    # Use F1 of the positive class as selection metric
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    return f1, clf


def choose_skill_given_state(state, start_models):
    best_skill = None
    best_score = -np.inf

    for skill, model in start_models.items():
        score = model.decision_function(state.reshape(1, -1))
        if score > best_score:
            best_score = score
            best_skill = skill

    return best_skill, best_score


# def check_if_end_skill(state, end_model_skill):
  
#     score = end_model_skill.decision_function(state.reshape(1, -1))

#     return score

def check_if_end_skill(state, end_model_skill, threshold=0.0):
    """
    Returns (is_end, score) where:
      - is_end = True if state is an inlier to the end-model
      - score  = decision_function value (higher = more inlier)
    """
    score = float(end_model_skill.decision_function(state.reshape(1, -1)))
    return (score > threshold), score


# --------------------------------------------------------------------------------------
# NEW: Start-model loading & inference utilities
# --------------------------------------------------------------------------------------

def load_models(models_dir):
    """Load all per-skill start models and their thresholds.

    Expects files of the form:
      {skill}_clf.joblib      - the calibrated sklearn pipeline model
      {skill}_meta.json       - contains at least a 'threshold' field

    Returns
    -------
    dict: skill -> { 'model': model, 'threshold': float, 'meta': meta_dict }
    """
    models = {}
    # if not os.path.isdir(models_dir):
    #     raise FileNotFoundError(f"Models directory not found: {models_dir}")

    for fname in os.listdir(models_dir):
        if not fname.endswith('_clf.joblib'):
            continue
        skill = fname[:-10]  # strip '_clf.joblib'
        model_path = os.path.join(models_dir, fname)
        meta_path = os.path.join(models_dir, f"{skill}_meta.json")

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"[WARN] Could not load model {model_path}: {e}")
            continue

        threshold = 0.5
        meta = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                threshold = float(meta.get('threshold', 0.5))
            except Exception as e:
                print(f"[WARN] Could not read meta for {skill}: {e}")

        models[skill] = {
            'model': model,
            'threshold': threshold,
            'meta': meta
        }
    return models


def classify_state_with_start_models(state_vec, start_models, strategy="prob", return_all=False):
    """Classify a single state into one of the start skill models.

    Parameters
    ----------
    state_vec : array-like, shape (d,)
        Feature vector (same representation used during training, e.g. PCA features).
    start_models : dict
        Output of load_start_models(). Each value has keys 'model', 'threshold'.
    strategy : {'prob', 'margin'}
        prob   -> choose skill with highest P(positive)
        margin -> choose skill with largest (P(positive) - threshold)
    return_all : bool
        If True, also return detailed per-skill scores/decisions.

    Returns
    -------
    best_skill : str or None
    best_score : float (probability or margin depending on strategy)
    details (optional) : list of dict per skill
    """
    if len(state_vec.shape) != 1:
        state_vec = state_vec.reshape(-1)

    best_skill = None
    best_score = -1e9
    details = []

    for skill, bundle in start_models.items():
        model = bundle['model']
        thr = bundle['threshold']
        try:
            proba = float(model.predict_proba(state_vec.reshape(1, -1))[0, 1])
        except AttributeError:
            # Fallback if model has no predict_proba (shouldn't happen with CalibratedClassifierCV)
            pred = model.predict(state_vec.reshape(1, -1))[0]
            proba = float(pred)
        margin = proba - thr
        score = proba if strategy == 'prob' else margin

        details.append({
            'skill': skill,
            'prob': proba,
            'threshold': thr,
            'margin': margin,
            'passes_threshold': proba >= thr
        })

        if score > best_score:
            best_score = score
            best_skill = skill

    if return_all:
        return sorted(details, key=lambda d: d['prob'], reverse=True)
    return best_skill, best_score


def filter_skills_passing_threshold(details):
    """Given the details list from classify_state_with_start_models(return_all=True),
    return list of skills whose prob >= threshold sorted by descending prob."""
    passing = [d for d in details if d['passes_threshold']]
    return sorted(passing, key=lambda d: d['prob'], reverse=True)



def check_end_state(state_vec, end_models, skill, strategy="prob", return_all=False):
    """Decide if a single state is an END state for a given skill.

    Parameters
    ----------
    state_vec : array-like, shape (d,)
        Feature vector (same representation used during training).
    end_models : dict
        Mapping {skill_name: {'model': fitted_estimator, 'threshold': float}}.
        Typically created by your end-model training/saving code.
    skill : str
        The skill whose end-state model should be used.
    strategy : {'prob', 'margin'}, default='prob'
        'prob'   -> return the model's P(end | state) as the score.
        'margin' -> return (P(end | state) - threshold) as the score.
    return_all : bool, default=False
        If True, return a details dict instead of (is_end, score).

    Returns
    -------
    is_end : bool
        Whether the state is predicted to be an end state for `skill`
        using the stored threshold.
    score : float
        Either probability or margin depending on `strategy`.
    details (optional) : dict
        {'skill', 'prob', 'threshold', 'margin', 'passes_threshold'}
    """
    if hasattr(state_vec, "shape") and len(state_vec.shape) != 1:
        state_vec = state_vec.reshape(-1)

    if skill not in end_models:
        raise ValueError(f"No end-state model found for skill '{skill}'")

    bundle = end_models[skill]
    model = bundle.get('model', None)
    thr = float(bundle.get('threshold', 0.5))

    if model is None:
        raise ValueError(f"End-state model bundle for '{skill}' is missing 'model'")

    # Get probability of 'end' class
    try:
        proba = float(model.predict_proba(state_vec.reshape(1, -1))[0, 1])
    except AttributeError:
        # Fallback if the model lacks predict_proba (should be rare with CalibratedClassifierCV)
        pred = model.predict(state_vec.reshape(1, -1))[0]
        proba = float(pred)

    margin = proba - thr
    is_end = proba >= thr
    score = proba if strategy == "prob" else margin

    details = {
        'skill': skill,
        'prob': proba,
        'threshold': thr,
        'margin': margin,
        'passes_threshold': is_end
    }

    if return_all:
        return details
    return is_end, score

def get_bc_data_by_episode(dir_, files, skill, feature_name='pca_features'):
    """
    Returns a list[dict] with one dict per episode (file).
    Each dict keeps states/actions for the requested `skill` and for 'other'.
    Also includes full episode arrays and a boolean skill mask for flexibility.
    """
    episodes = []

    for file in files:
        # Load per-episode artifacts
        with open(os.path.join(dir_, 'groundTruth', file), 'r') as f:
            lines = f.read().splitlines()  # len = T

        pca_path = os.path.join(dir_, feature_name, file + '.npy')
        act_path = os.path.join(dir_, 'actions', file + '.npy')

        states = np.load(pca_path)        # shape [T, d]
        actions = np.load(act_path)       # shape [T]

        # Sanity checks
        if len(lines) != len(states) or len(states) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: "
                f"labels={len(lines)} states={len(states)} actions={len(actions)}"
            )

        # Boolean mask for frames belonging to the skill
        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        # Slice once, keep per-episode arrays
        ep = dict(
            episode_id=file,
            # requested slices
            skill_states=states[skill_mask],
            skill_actions=actions[skill_mask],
            other_states=states[other_mask],
            other_actions=actions[other_mask],
            # optional full episode (handy for sequence models)
            states=states,
            actions=actions,
            skill_mask=skill_mask
        )
        episodes.append(ep)

    return episodes

def bc_flatten_split(episode_dicts, use_skill=True):
    """
    Converts a list of per-episode dicts into (X, y) step-level arrays.
    If use_skill=True, uses only frames matching the `skill`; otherwise uses 'other'.
    """
    X, y = [], []
    s_key = 'skill_states' if use_skill else 'other_states'
    a_key = 'skill_actions' if use_skill else 'other_actions'
    for ep in episode_dicts:
        X.append(ep[s_key])
        y.append(ep[a_key])
    if len(X) == 0:
        return np.empty((0,)), np.empty((0,), dtype=int)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def compute_class_weights(y, n_classes):
    # inverse frequency -> normalize to mean=1
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv *= (counts.mean() * 1.0) / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def build_startability_dataset(dir_: str, skill: str, files, features_dirname='pca_features_512', old_data_mode = False, skills_dir='groundTruth'):
    """
    Returns X, y, groups where groups[i] is the episode id (filename) for X[i].
    Positives: all states of `skill` except each segment's last state.
    Negatives: end states of `skill` + all states from other skills.
    """
    # # reuse your existing segmentation logic
    # (start_states, end_states, all_skill_states,
    #  negative_end_skill, negative_end_all, all_other_states) = get_start_end_states(
    #     dir_, skill, features_dirname=features_dirname
    # )

    # We also need to rebuild groups, so iterate files again and mirror the exact
    # positive/negative selection while recording episode ids per frame.
    pos, neg, grp = [], [], []

    gt_dir   = Path(dir_) / skills_dir
    act_dir  = Path(dir_) / 'actions'
    feat_dir = Path(dir_) / features_dirname

    for file in files:
        with open(gt_dir / file, 'r') as f:
            truths = f.read().splitlines()
        
        feats   = np.load(feat_dir / f'{file}.npy')

        if old_data_mode:
            acts, truths = get_fake_acts_shift_gt(truths)
        else:
            acts = np.load(act_dir  / f'{file}.npy')

        # print(file)
        # print(feats.shape)
        # print(acts.shape)
        # print(len(truths))
        # print(acts)
        # print(truths)
        segs_for_skill = make_skill_segments(acts, truths)
        # print(segs_for_skill)
        


        for skill_name, segs in segs_for_skill.items():
            for seg in segs:
                if skill_name == skill:
                    # positives: all but last index
                    if len(seg) > 1:
                        pos.append(feats[seg[:-1]])
                        grp.append(np.array([file] * (len(seg) - 1)))
                    # negatives (end state of this skill)
                    neg.append(feats[[seg[-1]]])
                else:
                    # negatives: all frames from other skills
                    neg.append(feats[seg])

     

    X_pos = np.concatenate(pos, axis=0) if len(pos) else np.empty((0, feats.shape[1]))
    X_neg = np.concatenate(neg, axis=0) if len(neg) else np.empty((0, feats.shape[1]))
    g_pos = np.concatenate(grp, axis=0) if len(grp) else np.empty((0,), dtype=object)

    # We still need groups for negatives. Reconstruct them similarly:
    neg_groups = []
    for file in files:
        with open(gt_dir / file, 'r') as f:
            truths = f.read().splitlines()
        feats   = np.load(feat_dir / f'{file}.npy')

        if old_data_mode:
            acts, truths = get_fake_acts_shift_gt(truths)
        else:
            acts    = np.load(act_dir  / f'{file}.npy')

        segs_for_skill = make_skill_segments(acts, truths)
        for skill_name, segs in segs_for_skill.items():
            for seg in segs:
                if skill_name == skill:
                    neg_groups.append(np.array([file]))              # end state
                else:
                    neg_groups.append(np.array([file] * len(seg)))   # other-skill frames
    g_neg = np.concatenate(neg_groups, axis=0) if len(neg_groups) else np.empty((0,), dtype=object)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos), dtype=int),
                   np.zeros(len(X_neg), dtype=int)])
    groups = np.concatenate([g_pos, g_neg], axis=0)

    return X, y, groups

def build_endability_dataset(dir_: str, skill: str, files, features_dirname='pca_features_512', old_data_mode = False, skills_dir='groundTruth'):
    """
    Build (X, y, groups) for END-state prediction of `skill`.
    Positives: last frame of each `skill` segment (end_states)
    Negatives: all frames except those last frames of the target skill
               i.e., negative_end_all = (skill frames except last) + (all other-skill frames)
    groups: episode id (filename) per frame.
    """
    gt_dir   = os.path.join(dir_, skills_dir)
    act_dir  = os.path.join(dir_, 'actions')
    feat_dir = os.path.join(dir_, features_dirname)

    X_pos, X_neg = [], []
    groups_pos, groups_neg = [], []

    for file in files:
        with open(os.path.join(gt_dir, file), 'r') as f:
            truths = f.read().splitlines()
        feats   = np.load(os.path.join(feat_dir, f'{file}.npy'))

        if old_data_mode:
            acts, truths = get_fake_acts_shift_gt(truths)
        else:
            acts = np.load(os.path.join(act_dir,  f'{file}.npy'))

        segs_for_skill = make_skill_segments(acts, truths) 

        # First pass: collect target skill segments' ends as positives,
        # and the target skill's "non-end" frames as part of negatives.
        if skill in segs_for_skill:
            for seg in segs_for_skill[skill]:
                # positive = last frame of the segment
                X_pos.append(feats[seg[-1]][None, :]); groups_pos.append(file)
                # negative within-skill = all but last
                if len(seg) > 1:
                    X_neg.append(feats[seg[:-1]]); groups_neg.extend([file] * (len(seg) - 1))

        # Second pass: all frames from other skills are negatives
        for other_skill, segs in segs_for_skill.items():
            if other_skill == skill:
                continue
            for seg in segs:
                X_neg.append(feats[seg]); groups_neg.extend([file] * len(seg))

    if len(X_pos) == 0:
        X_pos = np.empty((0, feats.shape[1]))
    else:
        X_pos = np.concatenate(X_pos, axis=0)

    if len(X_neg) == 0:
        X_neg = np.empty((0, feats.shape[1]))
    else:
        X_neg = np.concatenate(X_neg, axis=0)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(len(X_pos), dtype=int),
        np.zeros(len(X_neg), dtype=int)
    ])
    groups = np.array(groups_pos + groups_neg, dtype=object)
    return X, y, groups

from joblib import load as joblib_load

def load_pu_models(models_dir: str):
    """
    Load (skill, clf, threshold, meta) tuples from <models_dir>.
    Expects files: <skill>_clf.joblib and <skill>_meta.json
    Returns: list[dict] with keys: skill, clf, thr, meta
    """
    models = []
    for fname in os.listdir(models_dir):
        if not fname.endswith("_meta.json"):
            continue
        skill = fname[:-10]  # strip "_meta.json"
        meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
        model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
        if not os.path.exists(model_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            thr = float(meta["threshold"])
            clf = joblib_load(model_path)
            models.append({"skill": skill, "clf": clf, "thr": thr, "meta": meta})
        except Exception as e:
            print(f"[WARN] Skipping {skill}: {e}")
    return models


def applicable_pu_start_models(models, state, *, return_details=False, eps=0.0):
    """
    Given a list from load_pu_models(...) and a state feature vector (shape [d] or [1,d]),
    return/print skills whose probability >= threshold (+eps).
    - return_details=True returns a list of dicts with scores/margins
    - eps lets you demand a small margin above threshold (e.g., eps=0.02).
    """
    # Accept 1D or 2D input
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1,d].")

    rows = []
    for m in models:
        prob = float(m["clf"].predict_proba(X)[:, 1][0])
        thr  = float(m["thr"])
        margin = prob - thr
        is_applicable = prob >= (thr + eps)
        rows.append({
            "skill": m["skill"],
            "prob": prob,
            "thr": thr,
            "margin": margin,
            "applicable": is_applicable
        })

    # Sort by confidence margin (best first)
    rows.sort(key=lambda r: r["margin"], reverse=True)

    # Print list of applicable models
    applicable = [r for r in rows if r["applicable"]]
    if applicable:
        print("Applicable models (prob ≥ threshold):")
        for r in applicable:
            print(f"  - {r['skill']}: p={r['prob']:.3f}  thr={r['thr']:.3f}  margin={r['margin']:.3f}")
    else:
        print("No applicable models for this state.")

    return rows if return_details else [r["skill"] for r in applicable]

from joblib import load as joblib_load

def end_state_probability_pu(model, state) -> float:
    """
    Given a fitted PU skill model (clf) and a single state feature vector,
    return P(end-state | state) as a float in [0,1].
    - model: the fitted PU model you trained (the thing you dumped with joblib)
    - state: shape [d] or [1, d]
    """
    x = np.asarray(state)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        pass
    else:
        raise ValueError("`state` must be a single vector of shape [d] or [1, d].")
    proba = model.predict_proba(x)[:, 1][0]
    return float(proba)

def end_state_prob_pu(models_dir: str, skill: str, state) -> dict:
    """
    Load a specific skill model + its saved threshold from disk and score a state.
    Returns a dict with {prob, threshold, is_end, margin}.
    - models_dir: e.g., 'Craftax/Traces/stone_pickaxe_easy/pu_end_models'
    - skill: the skill name used in filenames '<skill>_clf.joblib' and '<skill>_meta.json'
    - state: shape [d] or [1, d]
    """
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file:  {meta_path}")

    clf = joblib_load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    thr = float(meta["threshold"])

    p = end_state_probability_pu(clf, state)
    margin = p - thr
    return {
        "prob": p,
        "threshold": thr,
        "is_end": bool(p >= thr),
        "margin": margin
    }