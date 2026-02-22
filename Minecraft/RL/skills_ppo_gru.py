import os
import argparse
from glob import glob
from typing import Optional

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from .helpers import PROJECT_ROOT, get_unique_skills
from .helpers_gru import make_masked_env_gru


def latest_checkpoint(path: str, prefix: str) -> Optional[str]:
    files = sorted(glob(os.path.join(path, f"{prefix}_*.zip")))
    return files[-1] if files else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO over GRU-BC skills using MineCLIP embeddings."
    )
    parser.add_argument("--ppo_seed", type=int, default=888)
    parser.add_argument("--env_seed", type=int, default=13579)

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/rl_checkpoints_gt_skills_gru",
        help="Directory to save PPO checkpoints.",
    )
    parser.add_argument(
        "--bc_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/bc_gru_gt",
        help="Directory containing GRU BC checkpoints (bc_gru_model_*.pt).",
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
    parser.add_argument(
        "--skills_dir",
        type=str,
        default="Data/minecraft_cobblestone_mapped/groundTruth",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="gt_skills_frameskip_gru",
    )

    # MineCLIP env params
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument(
        "--pretrained_model_path", type=str, default="ViT-B-16.pt"
    )

    # Checkpoint config
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
        help="Frame skip (default 8). Must match BC training setup.",
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
        help="Maximum episode steps in real Minecraft ticks. With skip=8, this becomes max_episode_steps/8 steps (default: 2000 -> 250 steps).",
    )
    parser.add_argument(
        "--max_skill_steps",
        type=int,
        default=64,
        help="Maximum steps a skill can run before being forced to end (default: 250).",
    )
    parser.add_argument(
        "--disable_pu_end",
        action="store_true",
        help="If set, disable PU end models and run every skill for max_skill_steps regardless of PU end model predictions.",
    )
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_prefix = f"skills_ppo_mapped_{args.run_name}"
    vecnorm_path = os.path.join(args.ckpt_dir, f"{ckpt_prefix}_vecnormalize.pkl")

    skill_files = os.listdir(args.skills_dir)
    skills = sorted(get_unique_skills(args.skills_dir, skill_files))
    print("[Skills]", skills)

    def _make_one_env():
        return make_masked_env_gru(
            skills=skills,
            ckpt_dir=args.bc_dir,
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

    # Build env
    vec_env = make_vec_env(_make_one_env, n_envs=2, seed=args.env_seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Maybe resume
    model = None
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
            ent_coef=0.005,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
        )

    # Checkpoint callback (saves model + vecnormalize)
    checkpoint_cb = CheckpointCallback(
        save_freq=args.ckpt_every,
        save_path=args.ckpt_dir,
        name_prefix=ckpt_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb])

    # Train with crash-safe saving
    try:
        model.learn(
            total_timesteps=120_000,
            log_interval=1,
            tb_log_name=f"ppo_skills_mapped_{args.run_name}",
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

    out_name = f"skills_ppo_mapped_{args.run_name}"
    model.save(out_name)
    vec_env.save(vecnorm_path)
    print(f"Saved model to {out_name} and vecnorm to {vecnorm_path}")


