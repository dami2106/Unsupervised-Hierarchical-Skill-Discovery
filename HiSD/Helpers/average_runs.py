import pandas as pd
import subprocess
import csv
import re
import os

TASK_NAME = "minecraft"
DATASET_SIZE = "minecraft_cobblestone_mapped"
CLUSTER_SIZE = "14"

SEEDS = [10, 42]

df = pd.read_csv(f'Traces/{TASK_NAME}/{DATASET_SIZE}/best.csv')

def build_cli(row, seed):
    args = [
        f"--alpha-eval {row['params_alpha-eval']}",
        f"--alpha-train {row['params_alpha-train']}",
        f"--batch-size {row['params_batch-size']}",
        f"--eps-eval {row['params_eps-eval']}",
        f"--eps-train {row['params_eps-train']}",
        f"--lambda-actions-eval {row['params_lambda-actions-eval']}",
        f"--lambda-actions-train {row['params_lambda-actions-train']}",
        f"--lambda-frames-eval {row['params_lambda-frames-eval']}",
        f"--lambda-frames-train {row['params_lambda-frames-train']}",
        f"--learning-rate {row['params_learning-rate']}",
        f"--n-epochs {row['params_n-epochs']}",
        f"--n-frames {row['params_n-frames']}",
        f"--radius-gw {row['params_radius-gw']}",
        f"--rho {row['params_rho']}",
        f"--weight-decay {row['params_weight-decay']}",
        f"--dataset {TASK_NAME}/{DATASET_SIZE}",
        "--feature-name features",
        "--save-directory runs",
        f"--run {TASK_NAME}_{DATASET_SIZE}_{row['number']}_seed{seed}",
        "--val-freq 5",
        "--layers 512 256 40",
        f"--seed {seed}",
        "--visualize",
        "--log",
        f"--n-clusters {CLUSTER_SIZE}",
    ]
    if row['params_ub-frames']: args.append("--ub-frames")
    if row['params_ub-actions']: args.append("--ub-actions")
    if row['params_std-feats']: args.append("--std-feats")
    return " ".join(args)

csv_filename = f"{TASK_NAME}_{DATASET_SIZE}_AVERAGE_RUNS.csv"
headers_written = False
headers = ["run_num", "seed"]

for idx, row in df.iterrows():
    print(f"Trying config number {row['number']} with expected value {row['value']}")
    for i, seed in enumerate(SEEDS):
        cli = build_cli(row, seed)
        result = subprocess.run(f"python src/train.py {cli}", shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running config {row['number']} with seed {seed}: {result.stderr}")
            continue


        pattern = r"\b(test_\w+)\b[^\d\-]*([0-9]*\.?[0-9]+)"
        matches = re.findall(pattern, result.stdout)
        metrics = {metric: float(value) for metric, value in matches}
        if not headers_written:
            headers = ["run_num", "seed"] + list(metrics.keys())
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            headers_written = True
        row_data = [i + 1, seed] + [metrics.get(h, "") for h in headers[2:]]
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
