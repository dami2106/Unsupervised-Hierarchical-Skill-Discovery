import pandas as pd
import subprocess
import csv
import re

TASK = "minecraft"
DATASET = "cobblestone_all"
CLUSTERS = 44

# Load and sort CSV
df = pd.read_csv(f'Traces/{TASK}/{TASK}_{DATASET}/optuna_results.csv')
df = df.sort_values(by='value', ascending=False)

def build_cli(row):
    # Build CLI string from row
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
        f"--dataset {TASK}/{TASK}_{DATASET}",
        "--feature-name features",
        "--save-directory runs",
        f"--run {TASK}_{DATASET}_{row['number']}",
        "--val-freq 5",
        "--layers 512 256 40",
        "--seed 0",
        "--visualize",
        "--log",
        f"--n-clusters {CLUSTERS}",
    ]
    # Add boolean flags
    if row['params_ub-frames']: args.append("--ub-frames")
    if row['params_ub-actions']: args.append("--ub-actions")
    if row['params_std-feats']: args.append("--std-feats")
    return " ".join(args)

def is_within_5pct(expected, actual):
    return abs(actual - expected) <= 0.05 * expected

output_csv = f"{TASK}_{DATASET}_rerun_results.csv"
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["number", "expected_value", "actual_value"])

    found_consistent = False

    for idx, row in df.iterrows():
        cli = build_cli(row)
        print(f"Trying config number {row['number']} with expected value {row['value']}")
        result = subprocess.run(f"python src/train.py {cli}", shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running command for config {row['number']}: {result.stderr}")
            writer.writerow([row['number'], row['value'], -1])
            continue

        

        pattern = r"\b(test_\w+)\b[^\d\-]*([0-9]*\.?[0-9]+)"
        print(result.stdout)
        matches = re.findall(pattern, result.stdout)
        print(matches)
        metrics = {metric: float(value) for metric, value in matches}

        test_miou_full = metrics.get("test_miou_full", 0)
        test_miou_per = metrics.get("test_miou_per", 0)
        actual_value = float((0.8 * test_miou_full) + (0.2 * test_miou_per))
        print(f"Actual value: {actual_value}")
        writer.writerow([row['number'], row['value'], actual_value])

        if is_within_5pct(row['value'], actual_value) and not found_consistent:
            print(f"Match found for config {row['number']}: {actual_value}")
            # Run 2 more times for consistency
            consistent_values = [actual_value]
            for i in range(4):
                print(f"Repeat run {i+2} for config {row['number']}")
                repeat_result = subprocess.run(f"python src/train.py {cli}", shell=True, capture_output=True, text=True)
                if repeat_result.returncode != 0:
                    print(f"Error on repeat run {i+2}: {repeat_result.stderr}")
                    repeat_value = -1
                else:
                    repeat_matches = re.findall(pattern, repeat_result.stdout)
                    repeat_metrics = {metric: float(value) for metric, value in repeat_matches}
                    repeat_full = repeat_metrics.get("test_miou_full", 0)
                    repeat_per = repeat_metrics.get("test_miou_per", 0)
                    repeat_value = float((0.8 * repeat_full) + (0.2 * repeat_per))
                writer.writerow([row['number'], row['value'], repeat_value])
                consistent_values.append(repeat_value)
            
            # Check if all 3 runs are consistent with each other
            all_consistent = True
            for val in consistent_values:
                if val == -1:  # Skip error values
                    all_consistent = False
                    break
                for other_val in consistent_values:
                    if other_val != -1 and not is_within_5pct(val, other_val):
                        all_consistent = False
                        break
                if not all_consistent:
                    break
            
            if all_consistent:
                print(f"All 3 runs are consistent for config {row['number']}: {consistent_values}")
                found_consistent = True
                break
            else:
                print(f"Runs not consistent for config {row['number']}: {consistent_values}, continuing to next config")