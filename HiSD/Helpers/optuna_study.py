import optuna
import subprocess
import json
import re
import pandas as pd
import joblib
import os
import argparse

def save_study_progress(study, save_dir, csv_filename, study_filename):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, csv_filename), index=False)
    joblib.dump(study, os.path.join(save_dir, study_filename))

def objective(trial, args):
    """Objective function to optimize hyperparameters"""
    # Define hyperparameter search space
    params = {
        "alpha-train": trial.suggest_float("alpha-train", 0.01, 1, step = 0.01),
        "alpha-eval": trial.suggest_float("alpha-eval", 0.01, 1, step = 0.01),
        "lambda-frames-train": trial.suggest_float("lambda-frames-train", 0.01, 0.1, step = 0.01),
        "lambda-actions-train": trial.suggest_float("lambda-actions-train", 0.01, 0.1, step = 0.01),
        "lambda-frames-eval": trial.suggest_float("lambda-frames-eval", 0.01, 0.1, step = 0.01),
        "lambda-actions-eval": trial.suggest_float("lambda-actions-eval", 0.01, 0.1, step = 0.01),
        "eps-train": trial.suggest_float("eps-train", 0.001, 0.5, step = 0.001),
        "eps-eval": trial.suggest_float("eps-eval", 0.001, 0.5, step = 0.001),
        "radius-gw": trial.suggest_float("radius-gw", 0.001, 0.1, step = 0.001),
        "learning-rate": trial.suggest_categorical("learning-rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "weight-decay": trial.suggest_categorical("weight-decay", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batch-size": trial.suggest_categorical("batch-size", [2, 8]),
        "n-epochs": trial.suggest_int("n-epochs", 5, 50, step = 5),
        "ub-frames": trial.suggest_categorical("ub-frames", [True, False]),
        "ub-actions": trial.suggest_categorical("ub-actions", [True, False]),
        "std-feats": trial.suggest_categorical("std-feats", [True, False]),
        "rho": trial.suggest_float("rho", 0.001, 0.3, step = 0.001),
        "n-frames": trial.suggest_int("n-frames", 2, 30, step = 2),
    }

    # Fixed parameters (not part of tuning)
    fixed_params = {
        "dataset": args.dataset,
        "feature-name": args.feature_name,
        "save-directory": args.dataset,
        "n-clusters": args.clusters,
        "layers": args.layers,
        "val-freq": 100,
        "log": False,
        "visualize": False,
        "seed": 0,
    }

    # Build CLI string from parameters
    cli_args = " ".join(
        [f"--{k} {v}" for k, v in {**params, **fixed_params}.items() if not isinstance(v, bool)]
    )

    # Add boolean flags if True
    for flag in ["ub-frames", "ub-actions", "std-feats", "log", "visualize"]:
        if params.get(flag, False) or fixed_params.get(flag, False):
            cli_args += f" --{flag}"

    try:
        result = subprocess.run(
            f"python src/train.py {cli_args}",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        pattern = r"(\btest_\w+\b)\s+([\d\.]+)"
        matches = re.findall(pattern, result.stdout)
        metrics = {metric: float(value) for metric, value in matches}
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        return float('-inf')
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Output received:\n{result.stdout}")
        return float('-inf')

    # Extract metrics
    test_miou_full = metrics.get("test_miou_full", 0)
    test_miou_per = metrics.get("test_miou_per", 0)

    return test_miou_full

# Entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use')
    parser.add_argument('--feature-name', type=str, default='pca_features', help='Name of the features folder')
    parser.add_argument('--clusters', type=int, default=3, help='Nb of clusters')
    parser.add_argument('--layers', '-ls',  type=str, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--trials',  default=1500, type=int, help='nb of trials')
    parser.add_argument('--filename', type=str, required=True, help='Base name used to build output filenames')
    args = parser.parse_args()

    directory = args.dataset

    # Derive filenames
    study_filename = f"optuna_study_{args.filename}.pkl"
    results_filename = f"optuna_results_{args.filename}.csv"
    best_filename = f"{args.filename}_best.csv"

    # Create base save dir: <dataset>/optuna/<filename>/
    base_save_dir = os.path.join(directory, "optuna", args.filename)
    os.makedirs(base_save_dir, exist_ok=True)

    study_file = os.path.join(base_save_dir, study_filename)
    if os.path.exists(study_file):
        print("Study file found. Resuming the study.")
        study = joblib.load(study_file)
    else:
        print("No study file found. Starting a new study.")
        study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.trials,
        callbacks=[lambda study, trial: save_study_progress(study, base_save_dir, results_filename, study_filename)]
    )

    save_study_progress(study, base_save_dir, results_filename, study_filename)

    # Save best trial to <dataset>/optuna/<filename>/<filename>_best.csv
    best_trial = study.best_trial
    df_all = study.trials_dataframe()
    df_best = df_all[df_all["number"] == best_trial.number]
    df_best.to_csv(os.path.join(base_save_dir, best_filename), index=False)
