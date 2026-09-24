"""
Phase-1 (component A) evaluation runner.

  tune  : Optuna search of the shared ASOT hyperparameters for the *baseline*
          (uniform q, K = ground truth).  The best config is frozen and reused by
          every configuration, so the K-free method never gets a larger budget.
  sweep : fixed-K ASOT at K in --ks and DP-marginal ASOT at K_max in --kmaxes, over
          --seeds, with the frozen config.  Writes per-run CSV + a summary with the
          K-sensitivity (metric range across K / K_max).

    python Helpers/bnp_sweep.py tune  --dataset ../Craftax/Traces/wsws_random --k-gt 2 --trials 40
    python Helpers/bnp_sweep.py sweep --dataset ../Craftax/Traces/wsws_random --ks 2 3 4 6 --kmaxes 4 8 --seeds 0 1 2 3 4
"""
import argparse
import itertools
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_RE = re.compile(r"(test_\w+)\s*│\s*([-\d\.eE]+)")
FLAGS = ['ub-frames', 'ub-actions', 'std-feats']


def run_train(params, extra, env=None):
    cli = []
    for k, v in {**params, **extra}.items():
        if k in FLAGS:
            if v:
                cli.append(f'--{k}')
        elif isinstance(v, (list, tuple)):
            cli.append(f'--{k} ' + ' '.join(map(str, v)))
        elif v is not None:
            cli.append(f'--{k} {v}')
    cmd = 'python src/train.py ' + ' '.join(cli)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', **(env or {}))
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    metrics = {k: float(v) for k, v in METRIC_RE.findall(res.stdout)}
    if not metrics:
        print('FAILED:', cmd, '\n', res.stderr[-2000:])
    return metrics


def suggest(trial):
    return {
        'alpha-train': trial.suggest_float('alpha-train', 0.01, 1, step=0.01),
        'alpha-eval': trial.suggest_float('alpha-eval', 0.01, 1, step=0.01),
        'lambda-frames-train': trial.suggest_float('lambda-frames-train', 0.01, 0.1, step=0.01),
        'lambda-actions-train': trial.suggest_float('lambda-actions-train', 0.01, 0.1, step=0.01),
        'lambda-frames-eval': trial.suggest_float('lambda-frames-eval', 0.01, 0.1, step=0.01),
        'lambda-actions-eval': trial.suggest_float('lambda-actions-eval', 0.01, 0.1, step=0.01),
        'eps-train': trial.suggest_float('eps-train', 0.001, 0.5, step=0.001),
        'eps-eval': trial.suggest_float('eps-eval', 0.001, 0.5, step=0.001),
        'radius-gw': trial.suggest_float('radius-gw', 0.001, 0.1, step=0.001),
        'learning-rate': trial.suggest_categorical('learning-rate', [1e-5, 1e-4, 1e-3, 1e-2]),
        'weight-decay': trial.suggest_categorical('weight-decay', [1e-6, 1e-4, 1e-3, 1e-2]),
        'n-epochs': trial.suggest_int('n-epochs', 5, 30, step=5),
        'ub-frames': trial.suggest_categorical('ub-frames', [True, False]),
        'ub-actions': trial.suggest_categorical('ub-actions', [True, False]),
        'std-feats': trial.suggest_categorical('std-feats', [True, False]),
        'rho': trial.suggest_float('rho', 0.0, 0.3, step=0.001),
        'n-frames': trial.suggest_int('n-frames', 10, 80, step=2),
    }


ENQUEUE = [
    {'alpha-train': 0.11, 'alpha-eval': 0.14, 'lambda-frames-train': 0.1, 'lambda-actions-train': 0.1,
     'lambda-frames-eval': 0.03, 'lambda-actions-eval': 0.1, 'eps-train': 0.022, 'eps-eval': 0.382,
     'radius-gw': 0.098, 'learning-rate': 1e-5, 'weight-decay': 1e-3, 'n-epochs': 30, 'ub-frames': False,
     'ub-actions': False, 'std-feats': True, 'rho': 0.182, 'n-frames': 80},
    {'alpha-train': 0.3, 'alpha-eval': 0.3, 'lambda-frames-train': 0.05, 'lambda-actions-train': 0.05,
     'lambda-frames-eval': 0.05, 'lambda-actions-eval': 0.01, 'eps-train': 0.07, 'eps-eval': 0.04,
     'radius-gw': 0.04, 'learning-rate': 1e-3, 'weight-decay': 1e-4, 'n-epochs': 20, 'ub-frames': False,
     'ub-actions': False, 'std-feats': True, 'rho': 0.1, 'n-frames': 40},
]


def base_extra(args):
    return {'dataset': args.dataset, 'feature-name': args.feature_name, 'layers': args.layers,
            'batch-size': args.batch_size, 'val-freq': 1000}


def tune(args):
    import optuna
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    for seed_cfg in ENQUEUE:  # known-reasonable starting points (the paper's stone-pickaxe config, ASOT defaults)
        study.enqueue_trial(seed_cfg)

    def objective(trial):
        m = run_train(suggest(trial), {**base_extra(args), 'n-clusters': args.k_gt, 'seed': 0})
        return m.get('test_miou_full', 0.)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)
    study.trials_dataframe().to_csv(out / 'tune_trials.csv', index=False)
    best = dict(study.best_params)
    (out / 'best_params.json').write_text(json.dumps(best, indent=2))
    print('best', study.best_value, best)


def sweep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(args.params).read_text()) if args.params else {}
    jobs = []
    for k, seed in itertools.product(args.ks, args.seeds):
        jobs.append(({'config': 'fixedK', 'K': k, 'seed': seed},
                     {'n-clusters': k, 'marginal': 'uniform', 'seed': seed}))
    for (kmax, gamma, usage), seed in itertools.product(
            itertools.product(args.kmaxes, args.gammas, args.usages), args.seeds):
        extra = {'n-clusters': kmax, 'marginal': 'dp', 'dp-gamma': gamma, 'dp-usage': usage,
                 'dp-warmup': args.dp_warmup, 'dp-warmup-frac': args.dp_warmup_frac, 'dp-decay': args.dp_decay,
                 'dp-merge-delta': args.dp_merge_delta, 'seed': seed}
        if args.gamma_prior:
            extra['dp-gamma-prior'] = args.gamma_prior
        jobs.append(({'config': f'dp_g{gamma}_{usage}', 'K': kmax, 'seed': seed}, extra))

    def work(job):
        tag, extra = job
        m = run_train(params, {**base_extra(args), **extra})
        row = {**tag, **m}
        print(row, flush=True)
        return row

    with ThreadPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(work, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'sweep_runs.csv', index=False)
    cols = [c for c in ['test_miou_full', 'test_mof_full', 'test_f1_full', 'test_miou_per', 'test_nmi',
                        'test_k_hat', 'test_k_used', 'test_k_err'] if c in df]
    summary = df.groupby(['config', 'K'])[cols].agg(['mean', 'std']).round(3)
    summary.to_csv(out / 'sweep_summary.csv')
    means = df.groupby(['config', 'K'])['test_miou_full'].mean().reset_index()
    sens = means.groupby('config')['test_miou_full'].agg(lambda x: x.max() - x.min()).rename('miou_range_over_K')
    sens.to_csv(out / 'k_sensitivity.csv')
    print(summary.to_string())
    print(sens.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['tune', 'sweep'])
    p.add_argument('--dataset', required=True)
    p.add_argument('--feature-name', default='pca_features')
    p.add_argument('--layers', nargs='+', type=int, default=[256, 300, 40])
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--out', default='runs/bnp_eval')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--k-gt', type=int, default=2)
    p.add_argument('--trials', type=int, default=40)
    p.add_argument('--params', type=str, default=None, help='frozen best_params.json from tune')
    p.add_argument('--ks', nargs='+', type=int, default=[2, 3, 4])
    p.add_argument('--kmaxes', nargs='+', type=int, default=[4, 8])
    p.add_argument('--gammas', nargs='+', type=float, default=[1.0])
    p.add_argument('--usages', nargs='+', default=['plan'])
    p.add_argument('--gamma-prior', nargs=2, type=float, default=None)
    p.add_argument('--dp-warmup', type=int, default=None)
    p.add_argument('--dp-warmup-frac', type=float, default=0.5)
    p.add_argument('--dp-merge-delta', type=float, default=0.1)
    p.add_argument('--dp-decay', type=float, default=0.99)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    tune(args) if args.mode == 'tune' else sweep(args)


if __name__ == '__main__':
    main()
