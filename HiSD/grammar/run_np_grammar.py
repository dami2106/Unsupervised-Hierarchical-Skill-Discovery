"""
Component B driver: nonparametric grammar over a segmentation's skill strings.

Reads per-episode skill files (``<dir>/<skill_folder>/*``) exactly like
sequitur/sequitur.py, fits the PY noisy-channel grammar, and writes HiSD-format
hierarchy output so downstream option code is unchanged:

  <out>/tree_<i>.json, grammar.txt, structure_metrics.json, unique_tree_count.txt
  <out>/np_metrics.json                       (MAP/posterior tree counts, perplexity, MDL)
  <out>/posterior_samples/sample_<s>/tree_<i>.json   (hierarchies drawn from the posterior)

    python grammar/run_np_grammar.py --input-dir runs/.../version_0 --skill-folder predicted_skills \
        --out-dir runs/.../version_0/hierarchy_data/np_predicted_hierarchy
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sequitur'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_grammar import PYNoisyChannelGrammar, sequitur_grammar, hierarchy_metrics  # noqa: E402
from structure_metrics import compute_structure_metrics  # noqa: E402


def read_sequences(input_dir, skill_folder):
    folder = Path(input_dir) / skill_folder
    names = sorted(os.listdir(folder))
    seqs = []
    for name in names:
        items = (folder / name).read_text().strip().split()
        collapsed = [x for i, x in enumerate(items) if i == 0 or x != items[i - 1]]
        seqs.append(tuple(collapsed))
    return names, seqs


def write_hierarchy(out_dir, strings, names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar, trees = sequitur_grammar(strings)
    with (out_dir / 'grammar.txt').open('w') as f:
        for head, rhs in grammar.items():
            f.write(f"{head}: {rhs}\n")
    metrics = {}
    for i, tree in enumerate(trees):
        (out_dir / f'tree_{i}.json').write_text(json.dumps(tree, indent=2))
        metrics[f'tree_{i}'] = compute_structure_metrics(tree)
    (out_dir / 'structure_metrics.json').write_text(json.dumps(metrics, indent=2))
    n_unique = len({json.dumps(t, sort_keys=True) for t in trees})
    (out_dir / 'unique_tree_count.txt').write_text(
        f"Number of unique trees: {n_unique} (out of {len(trees)} total)\n")
    if names is not None:
        (out_dir / 'episodes.json').write_text(json.dumps(names, indent=2))
    avg = {k: float(np.mean([m[k] for m in metrics.values()])) for k in next(iter(metrics.values()))}
    avg['unique_trees'] = n_unique
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--skill-folder', type=str, default='predicted_skills')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--method', choices=['gibbs', 'bmm'], default='gibbs')
    p.add_argument('--no-channel', action='store_true', help='disable the noisy-channel terminal model')
    p.add_argument('--py-d', type=float, default=0.5)
    p.add_argument('--py-theta', type=float, default=1.0)
    p.add_argument('--p-noise', type=float, default=0.05, help='initial channel sub/ins/del rate')
    p.add_argument('--n-iter', type=int, default=60)
    p.add_argument('--burn-in', type=int, default=20)
    p.add_argument('--max-len-diff', type=int, default=None)
    p.add_argument('--n-posterior-trees', type=int, default=5, help='posterior samples written as hierarchies')
    p.add_argument('--heldout-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    names, seqs = read_sequences(args.input_dir, args.skill_folder)
    kw = dict(d=args.py_d, theta=args.py_theta, channel=not args.no_channel, p_noise=args.p_noise,
              max_len_diff=args.max_len_diff, seed=args.seed)
    fit_kw = dict(method=args.method, n_iter=args.n_iter, burn_in=args.burn_in)

    # held-out perplexity: fit on a train split, score the rest
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(seqs))
    n_test = int(round(args.heldout_frac * len(seqs)))
    test = [seqs[i] for i in perm[:n_test]]
    train = [seqs[i] for i in perm[n_test:]]
    heldout = hierarchy_metrics(PYNoisyChannelGrammar(**kw).fit(train, **fit_kw), heldout=test) if n_test else {}

    model = PYNoisyChannelGrammar(**kw).fit(seqs, **fit_kw)
    metrics = hierarchy_metrics(model)
    for k in ['np_heldout_perplexity', 'seq_heldout_perplexity']:
        if k in heldout:
            metrics[k] = heldout[k]

    metrics['np_structure'] = write_hierarchy(args.out_dir, model.denoised(), names)
    metrics['seq_structure'] = write_hierarchy(args.out_dir / 'sequitur_baseline', list(seqs), names)
    samples = model.samples[-args.n_posterior_trees:] if model.samples else []
    for s, state in enumerate(samples):
        write_hierarchy(args.out_dir / 'posterior_samples' / f'sample_{s}', model.denoised(state))
    metrics['config'] = vars(args) | {'input_dir': str(args.input_dir), 'out_dir': str(args.out_dir)}
    (args.out_dir / 'np_metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'config'}, indent=2, default=str))


if __name__ == '__main__':
    main()
