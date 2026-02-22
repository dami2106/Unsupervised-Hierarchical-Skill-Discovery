import os
import json
import argparse
from glob import glob
from typing import Optional

import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from sb3_contrib import MaskablePPO

from .helpers import PROJECT_ROOT, get_unique_skills
from .hierarchy_helpers_gru import (
    load_unique_hierarchies,
    compile_composites_from_hierarchies,
    make_masked_hierarchy_env_gru,
)


def latest_checkpoint(path: str, prefix: str) -> Optional[str]:
    files = sorted(glob(os.path.join(path, f"{prefix}_*.zip")))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchy PPO over GRU-BC skills using MineCLIP embeddings."
    )
    parser.add_argument("--ppo_seed", type=int, default=888)
    parser.add_argument("--env_seed", type=int, default=13579)

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/rl_checkpoints_gt_hierarchy_fs_gru_fixed",
    )
    parser.add_argument(
        "--rl_ckpt_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/bc_gru_gt",
        help="Directory with GRU BC checkpoints (bc_gru_model_*.pt).",
    )
    parser.add_argument(
        "--start_models_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/pu_start_models_gt",
    )
    parser.add_argument(
        "--end_models_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/pu_end_models_gt",
    )

    # Ground-truth skill labels (leaf skills)
    parser.add_argument(
        "--skills_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/groundTruth",
    )

    # Hierarchies
    parser.add_argument(
        "--hierarchy_file",
        type=str,
        default="Data/minecraft_cobblestone_mapped/hierarchy_data/ground_truth_hierarchy",
        help="Path to a directory containing JSON hierarchy files (or single file for backward compatibility).",
    )
    parser.add_argument(
        "--symbol_map_path",
        type=str,
        default="Data/minecraft_cobblestone_mapped/mapping/ground_truth_mapping.json",
        help="JSON mapping from leaf 'symbol' (string) to leaf skill name.",
    )

    parser.add_argument("--run_name", type=str, default="gt_hierarchy_fs_gru")

    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--pretrained_model_path", type=str, default="ViT-B-16.pt")
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=500,
        help="Save PPO model every N steps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest PPO checkpoint if present.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=8,
        help="Frame skip (default 8). Must match BC/PU setup.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for both environment and PPO model (default: cpu).",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=800,
        help="Maximum episode steps in real Minecraft ticks. With skip=8, this becomes max_episode_steps/8 steps (default: 2500 -> 312 steps).",
    )
    parser.add_argument(
        "--max_skill_steps",
        type=int,
        default=64,
        help="Maximum steps a skill can run before being forced to end (default: 250).",
    )
    parser.add_argument(
        "--all_hierarchy",
        action="store_true",
        help="If set, include all unique internal production nodes in addition to root nodes and leaf skills.",
    )
    parser.add_argument(
        "--disable_pu_end",
        action="store_true",
        help="If set, disable PU end models and run every skill for max_skill_steps regardless of PU end model predictions.",
    )
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_prefix = f"hierarchy_ppo_mapped_{args.run_name}"
    vecnorm_path = os.path.join(args.ckpt_dir, f"{ckpt_prefix}_vecnormalize.pkl")

    # Leaf skills from groundTruth labels
    skill_files = os.listdir(args.skills_dir)
    leaf_skills = sorted(get_unique_skills(args.skills_dir, skill_files))
    print("[Leaf skills]", leaf_skills)

    # Load hierarchies & symbol map, compile composites
    with open(args.symbol_map_path, "r") as f:
        symbol_map = json.load(f)

    hierarchies, root_name_map = load_unique_hierarchies(args.hierarchy_file)
    composite_specs = compile_composites_from_hierarchies(
        hierarchies, symbol_map, all_hierarchy=args.all_hierarchy
    )
    print("[Composite specs]", {k: v for k, v in sorted(composite_specs.items())})
    if root_name_map:
        print("[Root name mapping]", root_name_map)

    # Calculate action space size
    num_composites = len(composite_specs)
    num_leaf_skills = len(leaf_skills)
    num_total_skills = num_composites + num_leaf_skills
    selector_size = num_total_skills + 1  # +1 for primitive action (selector=0)
    print(f"[Action Space] Composites: {num_composites}, Leaf skills: {num_leaf_skills}, Total skills: {num_total_skills}, Selector size: {selector_size}")

    # Build vectorized env using GRU hierarchy-aware wrapper
    def _make_one_env():
        return make_masked_hierarchy_env_gru(
            leaf_skills=leaf_skills,
            composite_specs=composite_specs,
            ckpt_dir=args.rl_ckpt_dir,
            start_models_dir=args.start_models_dir,
            end_models_dir=args.end_models_dir,
            project_root=args.project_root,
            pretrained_model_path=args.pretrained_model_path,
            device=args.device,
            target_item="log",
            target_count=1,
            max_episode_steps=args.max_episode_steps,
            seed=args.env_seed,
            skip=args.skip,
            max_skill_steps=args.max_skill_steps,
            disable_pu_end=args.disable_pu_end,
        )

    vec_env = make_vec_env(_make_one_env, n_envs=2, seed=args.env_seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    # Print full action space information
    action_space = vec_env.action_space
    if hasattr(action_space, 'nvec'):
        total_action_space_size = np.prod(action_space.nvec)
        print(f"[Action Space] Full action space size: {total_action_space_size} (nvec: {action_space.nvec})")
    else:
        print(f"[Action Space] Action space: {action_space}")

    # Maybe resume PPO
    model: Optional[MaskablePPO] = None
    if args.resume:
        last = latest_checkpoint(args.ckpt_dir, ckpt_prefix)
        if last and os.path.exists(vecnorm_path):
            print(f"[Resume] Loading VecNormalize from {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = True
            vec_env.norm_reward = False

            print(f"[Resume] Loading model from {last}")
            model = MaskablePPO.load(
                last, env=vec_env, device=args.device, seed=args.ppo_seed
            )
        elif last:
            print(
                f"[Resume] VecNormalize state not found, loading model from {last}"
            )
            model = MaskablePPO.load(
                last, env=vec_env, device=args.device, seed=args.ppo_seed
            )
        else:
            print("[Resume] No checkpoint found, starting fresh.")

    if model is None:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log="./tb_logs_ppo_minecraft",
            device=args.device,
            seed=args.ppo_seed,
            n_steps=512,
            batch_size=128,
            learning_rate=1e-4,
            ent_coef=0.01,  # CHANGED: old value was 0.01
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.ckpt_every,
        save_path=args.ckpt_dir,
        name_prefix=ckpt_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb])

    try:
        model.learn(
            total_timesteps=120_000,
            log_interval=1,
            tb_log_name=f"ppo_hierarchy_mapped_{args.run_name}",
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print(
            "\n[Signal] KeyboardInterrupt received, saving emergency checkpoint..."
        )
    except Exception as e:
        print(
            f"\n[Error] Exception during training: {e}\nSaving emergency checkpoint..."
        )
    finally:
        last_path = os.path.join(args.ckpt_dir, f"{ckpt_prefix}_last")
        model.save(last_path)
        vec_env.save(vecnorm_path)
        print(
            f"[Saved] Emergency/final checkpoint: {last_path}.zip and {vecnorm_path}"
        )

    out_name = f"hierarchy_ppo_mapped_{args.run_name}"
    model.save(out_name)
    vec_env.save(vecnorm_path)
    print(f"Saved model to {out_name} and vecnorm to {vecnorm_path}")


if __name__ == "__main__":
    main()


