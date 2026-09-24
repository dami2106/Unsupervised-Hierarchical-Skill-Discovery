# Unsupervised Hierarchical Skill Discovery (HiSD)

This repository contains the code for the paper **"Unsupervised Hierarchical Skill Discovery"** by Damion Harvey, Geraud Nangue Tasse, Branden Ingram, Benjamin Rosman, and Steven James.

> [arXiv:2601.23156](https://arxiv.org/abs/2601.23156)

## Overview

HiSD is a fully unsupervised framework for extracting reusable, multi-level skill hierarchies from observational trajectories in reinforcement learning domains. It operates in two stages:

1. **Skill Segmentation** — Unlabelled observation trajectories are segmented into discrete skills using temporal action segmentation.
2. **Hierarchy Induction** — The discovered skill sequences are compressed into compositional hierarchies using a modified Sequitur grammar induction algorithm.

HiSD requires no action labels, reward signals, or manual annotations — only a loose prior on the maximum number of skills. The discovered hierarchies can be deployed as options in downstream RL to accelerate learning.

The method is evaluated on **Craftax** (a 2D Minecraft-inspired environment) and the **full, unmodified version of Minecraft**.

## Repository Structure

```
├── HiSD/                   # Core HiSD implementation
│   ├── src/                #   training, metrics, utilities
│   ├── sequitur/           #   Modified Sequitur algorithm & hierarchy metrics
│   ├── Helpers/            #   Optuna hyperparameter search, averaging runs
│   └── Scripts/            #   Shell scripts for training & hierarchy induction
│
├── Craftax/                # Craftax environment & experiments
│   ├── craftax/            #   Modified Craftax environment (fully observable)
│   ├── Skill_Learning/     #   Behavioural cloning & PU learning for options
│   ├── RL_Scripts/         #   PPO training scripts (hierarchy, skills, baselines)
│   ├── Scripts/            #   BC & PU model training scripts
│   ├── generate_data.py    #   Expert trajectory generation via A* planner
│   ├── get_pca_features.py #   PCA feature extraction from pixel observations
│   └── top_down_env_gymnasium*.py  # Gymnasium wrappers for RL deployment
│
├── Minecraft/              # Minecraft environment & experiments
│   ├── lib/                #   VPT model architecture (policy, action heads, etc.)
│   ├── Learn_Skill_Data/   #   BC (GRU) & PU learning for Minecraft skills
│   ├── RL/                 #   PPO training scripts for Minecraft
│   ├── agent.py            #   VPT agent for trajectory collection
│   ├── run_agent.py        #   Data collection entry point
│   └── process_raw_data.py #   Raw trajectory processing & ground-truth extraction
│
└── environment_nobuilds.yml  # Conda environment specification
```

## Setup

### Prerequisites

- Python 3.10
- CUDA 11.8+ (for GPU training)
- Conda

### Installation

**Craftax & HiSD:**

```bash
conda env create -f environment_nobuilds.yml
conda activate hisd
```

**Minecraft** (requires a separate environment due to older dependency versions):

```bash
conda env create -f Minecraft/environment.yml
conda activate minecraft
```

You will also need to download the [MineCLIP](https://github.com/MineDojo/MineCLIP) model weights and place them in the `Minecraft/` directory. These are used for visual feature extraction from Minecraft observations.

## Usage

All commands below are run from the repository root unless otherwise noted.

### 1. Data Generation

**Craftax** — Generate expert trajectories using the A* planner:

```bash
cd Craftax
python generate_data.py
```

**Minecraft** — Activate the Minecraft environment first, then collect trajectories using a pretrained VPT agent:

```bash
conda activate minecraft
cd Minecraft
python run_agent.py
python process_raw_data.py
```

### 2. Feature Extraction

**Craftax** — Extract PCA features from pixel observations:

```bash
cd Craftax
python get_pca_features.py
```

**Minecraft** — Features are extracted via MineCLIP embeddings during data processing.

### 3. Skill Segmentation 

Train the HiSD model to segment trajectories into skills. Example for the Craftax Stone Pickaxe task:

```bash
cd HiSD
bash Scripts/train_asot.sh
```

Edit `Scripts/train_asot.sh` to point to the desired dataset and adjust hyperparameters. Refer to the paper's Appendix E for optimal configurations.

### 4. Hierarchy Induction (Sequitur)

Run the modified Sequitur algorithm on the predicted skill sequences:

```bash
cd HiSD
bash Scripts/run_sequitur.sh
```

### 5. Downstream RL Deployment

Train option policies (BC) and initiation/termination models (PU learning):

```bash
cd Craftax
bash Scripts/train_bc_resnet.sh
bash Scripts/train_pu_start.sh
bash Scripts/train_pu_end.sh
```

Train a hierarchical PPO agent using the discovered skills and hierarchy:

```bash
cd Craftax
bash RL_Scripts/ppo_hierarchy_asot.sh
```

Equivalent scripts for Minecraft are located in `Minecraft/RL/` and `Minecraft/Learn_Skill_Data/`.

## Bayesian-nonparametric extension

This branch adds three ablation-ready components: K-free segmentation via a learned DP
marginal in ASOT (A), a Pitman–Yor noisy-channel grammar over skill strings (B), and
duration-aware option termination with soft initiation masks (C). See
[`HiSD/BNP_EXTENSION.md`](HiSD/BNP_EXTENSION.md) for the method, usage, and Craftax
verification results.

## Citation

```bibtex
@misc{unsupervisedhierarchicalskilldiscovery,
      title={Unsupervised Hierarchical Skill Discovery}, 
      author={Damion Harvey and Geraud Nangue Tasse and Branden Ingram and Benjamin Rosman and Steven James},
      year={2026},
      eprint={2601.23156},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.23156}, 
}
```
