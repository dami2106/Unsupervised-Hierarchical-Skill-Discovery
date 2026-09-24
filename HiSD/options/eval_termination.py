"""
Offline evaluation of option termination rules (component C) on any segmentation.

Episodes are split by episode into train/test.  From the *train* episodes'
segmentation (``--skills-dir``: groundTruth, a run's predicted_skills, ...) we fit
  - per-skill Poisson-Gamma duration models (hazard + 95th-percentile cap),
  - per-skill nnPU end-state models (calibrated), and
  - the paper's Elkan-Noto PU end models (SVM-RBF, F1-tuned threshold) as baseline.
Every *test* segment is then replayed as an option execution starting at its first
frame, and we score when each rule terminates it, using the exact distribution of the
stopping time tau under the stochastic rules:

  exact   P(tau == L)     within1  P(|tau - L| <= 1)     mae  E|tau - L|

Rules: horizon (fixed H), pu_horizon (old HiSD: Elkan-Noto threshold, else H),
duration, and for each state model m in {nnpu, en} (en = Elkan-Noto's calibrated
posterior P(s=1|x)/c): state_m, noisy_or_m, bayes_m.  Stochastic rules are scored
under the exact stopping-time distribution ('sample') and with the deterministic
decision beta >= 0.5 ('thr').

--state-noise sigma adds N(0, sigma^2 * var) noise to the (standardised) features at
train and test time: it degrades the state models towards the Minecraft regime
(PU-End F1 ~0.1) to test graceful degradation.  Frame-level end-state micro-F1 (the
paper's PU-End metric) is reported for both PU models.  If --gt-dir is given, test
segments/lengths come from ground truth instead.

    python options/eval_termination.py --dataset ../Craftax/Traces/wsws_static --skills-dir groundTruth
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duration import extract_segments, fit_skill_durations  # noqa: E402
from nnpu import NNPUClassifier  # noqa: E402
from termination import combine_bayes  # noqa: E402


def load_labels(folder, names):
    out = []
    for n in names:
        p = Path(folder) / n
        if not p.exists():
            p = Path(folder) / f'{n}_skills.txt'  # predicted_skills naming
        out.append(p.read_text().split())
    return out


def segments_with_start(labels):
    segs, t = [], 0
    for lab, ln in extract_segments(labels):
        segs.append((lab, t, ln))
        t += ln
    return segs


def pu_data(labels_list, feats_list, skill):
    P, U = [], []
    for labels, feats in zip(labels_list, feats_list):
        for lab, s, ln in segments_with_start(labels):
            if lab != skill:
                continue
            P.append(feats[s + ln - 1])
            U.extend(feats[s:s + ln - 1])
    return np.array(P), np.array(U) if U else np.zeros((0, feats_list[0].shape[1]))


def fit_elkan_noto(P, U, seed=0):
    """The paper's PU-End model: Elkan-Noto over SVM-RBF, threshold maximising train F1."""
    from pulearn import ElkanotoPuClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import precision_recall_curve
    X = np.vstack([P, U])
    y = np.r_[np.ones(len(P)), -np.ones(len(U))]
    base = make_pipeline(StandardScaler(), SVC(C=10., kernel='rbf', gamma='scale', probability=True, random_state=seed))
    clf = ElkanotoPuClassifier(estimator=base, hold_out_ratio=0.2, random_state=seed)
    clf.fit(X, y)
    prob = clf.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve((y > 0).astype(int), prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    thr_best = float(thr[np.nanargmax(f1[1:])]) if len(thr) else 0.5
    return clf, thr_best


def stop_time_distribution(betas, cap):
    """P(tau = t), t = 1..len(betas), given per-step termination probabilities."""
    p, surv = np.zeros(len(betas)), 1.
    for i, b in enumerate(betas):
        t = i + 1
        b = 1. if (cap is not None and t >= cap) or i == len(betas) - 1 else b
        p[i] = surv * b
        surv *= (1. - b)
    return p


def score(p_tau, L):
    t = np.arange(1, len(p_tau) + 1)
    return {'exact': float(p_tau[L - 1]) if L <= len(p_tau) else 0.,
            'within1': float(p_tau[np.abs(t - L) <= 1].sum()),
            'mae': float((p_tau * np.abs(t - L)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--skills-dir', type=Path, default=None, help='segmentation labels (default <dataset>/groundTruth)')
    ap.add_argument('--gt-dir', type=Path, default=None, help='evaluate test segments against these labels')
    ap.add_argument('--features-name', default='pca_features')
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--cap-quantile', type=float, default=0.95)
    ap.add_argument('--test-frac', type=float, default=0.3)
    ap.add_argument('--state-noise', type=float, default=0.)
    ap.add_argument('--no-elkan-noto', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    skills_dir = args.skills_dir or args.dataset / 'groundTruth'
    names = sorted(os.listdir(args.dataset / 'groundTruth'))
    labels = load_labels(skills_dir, names)
    eval_labels = load_labels(args.gt_dir, names) if args.gt_dir else labels
    feats = [np.load(args.dataset / args.features_name / f'{n}.npy') for n in names]
    rng = np.random.default_rng(args.seed)
    if args.state_noise > 0:
        sd = np.concatenate(feats).std(0, keepdims=True)
        feats = [f + args.state_noise * sd * rng.standard_normal(f.shape) for f in feats]
    perm = rng.permutation(len(names))
    n_test = int(round(args.test_frac * len(names)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    tr_lab, tr_feat = [labels[i] for i in train_idx], [feats[i] for i in train_idx]

    durations, _ = fit_skill_durations(tr_lab, t_max=max(len(l) for l in labels) + 1)
    caps = {k: m.percentile(args.cap_quantile) for k, m in durations.items()}
    state_models = {'nnpu': {}, 'en': {}}  # skill -> (prob_fn, train prior)
    en_thr = {}
    for skill in durations:
        P, U = pu_data(tr_lab, tr_feat, skill)
        if len(P) < 2 or len(U) < 2:
            continue
        prior = len(P) / (len(P) + len(U))
        m = NNPUClassifier(seed=args.seed).fit(P, U, prior).calibrate(P, U)
        state_models['nnpu'][skill] = (lambda X, m=m: m.predict_proba(X)[:, 1], prior)
        if not args.no_elkan_noto:
            clf, thr = fit_elkan_noto(P, U, args.seed)
            state_models['en'][skill] = (lambda X, c=clf: np.clip(c.predict_proba(X)[:, 1], 0., 1.), prior)
            en_thr[skill] = (clf, thr)

    res = defaultdict(lambda: defaultdict(list))
    frame_true, frame_pred = [], defaultdict(list)
    for i in test_idx:
        seg_lab, ev_lab, f = labels[i], eval_labels[i], feats[i]
        for lab, s, L in segments_with_start(ev_lab):
            skill = lab if lab in durations else max(set(seg_lab[s:s + L]), key=seg_lab[s:s + L].count)
            if skill not in durations:
                continue
            T = len(ev_lab) - s  # the option can run until the episode ends
            window = f[s:s + T]
            dur, cap = durations[skill], caps[skill]
            b_dur = np.array([dur.hazard(t) for t in range(1, T + 1)])
            betas = {'duration': b_dur}
            for mname, bank in state_models.items():
                if skill not in bank:
                    continue
                fn, prior = bank[skill]
                b_state = fn(window)
                betas[f'state_{mname}'] = b_state
                betas[f'noisy_or_{mname}'] = 1 - (1 - b_dur) * (1 - b_state)
                betas[f'bayes_{mname}'] = combine_bayes(b_dur, b_state, prior)
                frame_pred[mname].extend((b_state[:L] >= 0.5).astype(int))
            dists = {'horizon': stop_time_distribution(np.zeros(T), args.horizon)}
            for r, b in betas.items():
                dists[f'{r}/sample'] = stop_time_distribution(b, cap)
                dists[f'{r}/thr'] = stop_time_distribution((b >= 0.5).astype(float), cap)
            if skill in en_thr:
                clf, thr = en_thr[skill]
                fires = (clf.predict_proba(window)[:, 1] >= thr).astype(float)
                dists['pu_horizon'] = stop_time_distribution(fires, args.horizon)
                frame_pred['en_thr'].extend(fires[:L].astype(int))
            for r, p in dists.items():
                for k, v in score(p, L).items():
                    res[r][k].append(v)
            y = np.zeros(L, dtype=int)
            y[-1] = 1
            frame_true.extend(y)

    summary = {'rules': {r: {k: round(float(np.mean(v)), 4) for k, v in m.items()} for r, m in sorted(res.items())}}
    summary['end_state_micro_f1'] = {m: round(float(f1_score(frame_true, v)), 4) for m, v in frame_pred.items()}
    summary['durations'] = {str(k): {'mean': round(m.mean, 2), 'cap': caps[k]} for k, m in durations.items()}
    summary['n_test_segments'] = len(res['horizon']['exact'])
    summary['config'] = {k: str(v) for k, v in vars(args).items()}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
