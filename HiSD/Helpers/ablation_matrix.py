"""
Offline ablation matrix over components A (segmenter), B (grammar), C (termination).

Given one fixed-K ASOT run and one DP-marginal (K-free) ASOT run (each a Lightning
log dir containing predicted_skills/, produced with --log --visualize), this fills
the 2^3 table of the evaluation plan with every metric family that does not need
RL rollouts:

  config   segmenter  grammar  options     seg. metrics   hierarchy metrics   termination metrics
  Old      fixed-K    Seq      PU/horizon
  A        DP         Seq      PU/horizon
  A+B      DP         NP       PU/horizon
  A+C      DP         Seq      Dur+Cal
  A+B+C    DP         NP       Dur+Cal
  B+C      fixed-K    NP       Dur+Cal
  B        fixed-K    NP       PU/horizon
  C        fixed-K    Seq      Dur+Cal

Segmentation metrics are read from each run's metrics; hierarchy metrics come from
grammar/run_np_grammar.py (NP) and its Sequitur baseline; termination metrics from
options/eval_termination.py, scored against ground-truth segment ends.

    python Helpers/ablation_matrix.py --dataset ../Craftax/Traces/stone_pick \
        --fixed-run <dir> --dp-run <dir> --out runs/bnp_eval/stone_pick/ablation
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent

CONFIGS = [  # name, segmenter, grammar, options
    ('Old HiSD', 'fixed', 'seq', 'pu'),
    ('A', 'dp', 'seq', 'pu'),
    ('A+B', 'dp', 'np', 'pu'),
    ('A+C', 'dp', 'seq', 'dur'),
    ('A+B+C', 'dp', 'np', 'dur'),
    ('B+C', 'fixed', 'np', 'dur'),
    ('B-only', 'fixed', 'np', 'pu'),
    ('C-only', 'fixed', 'seq', 'dur'),
]


def run(cmd):
    res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
        raise RuntimeError(' '.join(map(str, cmd)))
    return res.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--fixed-run', type=Path, required=True)
    ap.add_argument('--dp-run', type=Path, required=True)
    ap.add_argument('--seg-metrics', type=Path, default=None,
                    help='optional CSV with columns segmenter,<metric> (e.g. from bnp_sweep) to include')
    ap.add_argument('--features-name', default='pca_features')
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--state-noise', type=float, default=0.)
    ap.add_argument('--grammar-method', default='gibbs')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {'fixed': args.fixed_run, 'dp': args.dp_run}
    hier, term = {}, {}
    gt = run([sys.executable, 'grammar/run_np_grammar.py', '--input-dir', str(args.dataset),
              '--skill-folder', 'groundTruth', '--out-dir', str(args.out / 'hier_gt'),
              '--method', args.grammar_method])
    gt_metrics = json.loads((args.out / 'hier_gt' / 'np_metrics.json').read_text())
    for seg, rdir in runs.items():
        run([sys.executable, 'grammar/run_np_grammar.py', '--input-dir', str(rdir), '--skill-folder',
             'predicted_skills', '--out-dir', str(args.out / f'hier_{seg}'), '--method', args.grammar_method])
        hier[seg] = json.loads((args.out / f'hier_{seg}' / 'np_metrics.json').read_text())
        out = args.out / f'term_{seg}.json'
        run([sys.executable, 'options/eval_termination.py', '--dataset', str(args.dataset),
             '--skills-dir', str(rdir / 'predicted_skills'), '--gt-dir', str(args.dataset / 'groundTruth'),
             '--features-name', args.features_name, '--horizon', str(args.horizon),
             '--state-noise', str(args.state_noise), '--out', str(out)])
        term[seg] = json.loads(out.read_text())

    rows = []
    for name, seg, gram, opt in CONFIGS:
        h, t = hier[seg], term[seg]['rules']
        s = h['np_structure'] if gram == 'np' else h['seq_structure']
        row = {'config': name, 'segmenter': seg, 'grammar': gram, 'options': opt,
               'unique_trees': s['unique_trees'],
               'unique_types': h['np_map_unique_trees'] if gram == 'np' else h['seq_unique_trees'],
               'tree_size': round(s['size'], 2), 'max_branching': round(s['max_branching'], 2),
               'heldout_ppl': round(h['np_heldout_perplexity'] if gram == 'np' else h['seq_heldout_perplexity'], 3),
               'mdl_bits': round(h['np_mdl_bits'] if gram == 'np' else h['seq_mdl_bits'], 1)}
        rule = 'pu_horizon' if opt == 'pu' else 'bayes_en/thr'
        if rule not in t:
            rule = 'duration/thr'
        row.update({f'term_{k}': round(v, 3) for k, v in t[rule].items()})
        row['term_rule'] = rule
        rows.append(row)
    df = pd.DataFrame(rows)
    if args.seg_metrics:
        seg_df = pd.read_csv(args.seg_metrics)
        df = df.merge(seg_df, on='segmenter', how='left')
    gt_s = gt_metrics['seq_structure']
    df.loc[len(df)] = {'config': 'Ground truth', 'unique_trees': gt_s['unique_trees'],
                       'unique_types': gt_metrics['seq_unique_trees'], 'tree_size': round(gt_s['size'], 2),
                       'max_branching': round(gt_s['max_branching'], 2)}
    df.to_csv(args.out / 'ablation_matrix.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
