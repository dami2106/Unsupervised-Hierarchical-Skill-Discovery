"""
Fit per-skill Bayesian duration models from any segmentation and save them as JSON
for the option environments (Craftax/option_helpers.py: attach_duration_models).

    python options/fit_durations.py --skills-dir ../Craftax/Traces/wsws_static/groundTruth \
        --out ../Craftax/Traces/wsws_static/duration_models_gt.json

The skills directory may hold ground-truth label files or a run's predicted_skills
(``*_skills.txt``); labels are one per line / whitespace separated.  --frameskip
rescales segment lengths to environment steps (e.g. 8 for Minecraft).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duration import fit_skill_durations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skills-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--a0', type=float, default=1.0)
    ap.add_argument('--b0', type=float, default=0.1)
    ap.add_argument('--t-max', type=int, default=512)
    ap.add_argument('--frameskip', type=int, default=1)
    ap.add_argument('--exclude-last', action='store_true', help='drop right-censored final segments')
    args = ap.parse_args()

    seqs = []
    for name in sorted(os.listdir(args.skills_dir)):
        if name.endswith('_segments.txt'):
            continue
        labels = (args.skills_dir / name).read_text().split()
        seqs.append([x for x in labels for _ in range(args.frameskip)])
    models, lengths = fit_skill_durations(seqs, a0=args.a0, b0=args.b0, t_max=args.t_max,
                                          exclude_last=args.exclude_last)
    out = {str(k): {'pmf': m.pmf.tolist(), 'mean': m.mean, 'p95': m.percentile(0.95),
                    'n_segments': len(lengths[k])} for k, m in models.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    for k, v in out.items():
        print(f"{k}: n={v['n_segments']} mean={v['mean']:.2f} p95={v['p95']}")


if __name__ == '__main__':
    main()
