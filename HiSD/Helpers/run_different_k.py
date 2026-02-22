import pandas as pd
import subprocess
import csv
import re

DATASET_SIZE = "pixels_big"

runs = {
    # "wsws_static": [3, 4], #Optimal is 2 
    # "wsws_random": [3, 4], #Optimal is 2
    # "mixed_static": [3, 4, 6, 7], #Optimal is 5
    "stone_pick_static": [3, 4, 6, 7],  # Optimal is 5
}

def build_cli(row, task_name, cluster_size):
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
        f"--dataset {task_name}/{task_name}_{DATASET_SIZE}",
        "--feature-name pca_features",
        "--save-directory runs",
        f"--run {task_name}_{DATASET_SIZE}_{row['number']}_k{cluster_size}",
        "--val-freq 5",
        "--layers 650 300 40",
        "--seed 0",
        "--visualize",
        "--log",
        f"--n-clusters {cluster_size}",
    ]
    if row['params_ub-frames']: args.append("--ub-frames")
    if row['params_ub-actions']: args.append("--ub-actions")
    if row['params_std-feats']: args.append("--std-feats")
    return " ".join(args)

pattern = r"\b(test_\w+)\b[^\d\-]*([0-9]*\.?[0-9]+)"

all_results = []

for task_name, cluster_sizes in runs.items():
    # Load and sort CSV for each task
    df = pd.read_csv(f'Traces/{task_name}/{task_name}_{DATASET_SIZE}/best.csv')
    for cluster_k in cluster_sizes:
        row = df.iloc[0]  # Use the best row (or change as needed)
        cli = build_cli(row, task_name, cluster_k)
        print(f"Running {task_name} with cluster_k={cluster_k}")
        result = subprocess.run(f"python src/train.py {cli}", shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running command for {task_name} cluster_k={cluster_k}: {result.stderr}")
            continue

        matches = re.findall(pattern, result.stdout)
        metrics = {metric: float(value) for metric, value in matches}
        all_results.append({
            "task_name": task_name,
            "cluster_k": cluster_k,
            "f1_per": metrics.get("test_f1_per", 0),
            "f1_full": metrics.get("test_f1_full", 0),
            "miou_per": metrics.get("test_miou_per", 0),
            "miou_full": metrics.get("test_miou_full", 0),
            "mof_per": metrics.get("test_mof_per", 0),
            "mof_full": metrics.get("test_mof_full", 0),
        })

out_csv = f"all_tasks_{DATASET_SIZE}.csv"
df_out = pd.DataFrame(all_results)
df_out.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}")
