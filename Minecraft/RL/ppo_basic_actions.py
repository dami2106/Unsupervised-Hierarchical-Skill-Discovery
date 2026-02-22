# PPO_Basic.py
import argparse
import numpy as np
import os
from glob import glob
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from .helpers import make_env_mineclip, PROJECT_ROOT


def latest_checkpoint(path: str, prefix: str) -> Optional[str]:
    files = sorted(glob(os.path.join(path, f"{prefix}_*.zip")))
    return files[-1] if files else None


parser = argparse.ArgumentParser(description="PPO over basic actions using MineCLIP embeddings.")
parser.add_argument("--ppo_seed", type=int, default=888)
parser.add_argument("--env_seed", type=int, default=13579)

# Checkpoint / run config (mirrors GRU scripts)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="Data/minecraft_cobblestone_mapped/rl_checkpoints_basic_actions",
    help="Directory to save PPO checkpoints.",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="basic_actions_mapped",
    help="Run name suffix for checkpoints and logging.",
)
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

# MineCLIP / env params (keep consistent with other scripts)
parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
parser.add_argument("--pretrained_model_path", type=str, default="ViT-B-16.pt")
parser.add_argument(
    "--skip",
    type=int,
    default=8,
    help="Frame skip (default 8).",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for both environment and PPO model (default: cuda).",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=2000,
    help="Maximum episode steps in real Minecraft ticks.",
)

args, _ = parser.parse_known_args()


if __name__ == "__main__":
    print("Starting (MineCLIP obs, basic actions)")

    # Build env
    vec_env = make_vec_env(
        lambda: make_env_mineclip(
            project_root=args.project_root,
            pretrained_model_path=args.pretrained_model_path,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            target_item="log",
            target_count=1,
            seed=args.env_seed,
            skip=args.skip,
        ),
        n_envs=2,
        seed=args.env_seed,
        vec_env_cls=DummyVecEnv,
        monitor_dir=None,
    )

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,     # Helps MLP learn from MineCLIP features
        norm_reward=False, # Critical for sparse rewards (+10)
        clip_obs=10.0,
    )

    # Checkpoint paths
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_prefix = f"ppo_basic_actions_mapped_{args.run_name}"
    vecnorm_path = os.path.join(args.ckpt_dir, f"{ckpt_prefix}_vecnormalize.pkl")

    print("Env Setup complete")

    # Maybe resume
    model: Optional[PPO] = None
    if args.resume:
        last = latest_checkpoint(args.ckpt_dir, ckpt_prefix)
        if last and os.path.exists(vecnorm_path):
            print(f"[Resume] Loading VecNormalize from {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = True
            vec_env.norm_reward = False

            print(f"[Resume] Loading model from {last}")
            model = PPO.load(last, env=vec_env, device=args.device, seed=args.ppo_seed)
        elif last:
            print(f"[Resume] VecNormalize state not found, loading model from {last}")
            model = PPO.load(last, env=vec_env, device=args.device, seed=args.ppo_seed)
        else:
            print("[Resume] No checkpoint found, starting fresh.")

    # Fresh model if not resumed
    if model is None:
        model = PPO(
            policy="MlpPolicy",   # vector obs → use MLP
            env=vec_env,
            verbose=1,
            tensorboard_log="./tb_logs_ppo_minecraft",
            device=args.device,
            seed=args.ppo_seed,
            n_steps=512,              # match hierarchy/skills PPO defaults
            batch_size=128,
            learning_rate=1e-4,       # stable for fixed MineCLIP
            ent_coef=0.005,           # encourage exploration
            gamma=0.99,
            gae_lambda=0.95,          # good bias-variance tradeoff
            vf_coef=0.5,              # value function loss
            max_grad_norm=0.5,        # gradient clipping
            n_epochs=10,              # optimize each batch more
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
            tb_log_name=f"ppo_basic_actions_mapped_{args.run_name}",
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[Signal] KeyboardInterrupt received, saving emergency checkpoint...")
    except Exception as e:
        print(f"\n[Error] Exception during training: {e}\nSaving emergency checkpoint...")
    finally:
        last_path = os.path.join(args.ckpt_dir, f"{ckpt_prefix}_last")
        model.save(last_path)
        vec_env.save(vecnorm_path)
        print(
            f"[Saved] Emergency/final checkpoint: {last_path}.zip and {vecnorm_path}"
        )

    out_name = f"ppo_basic_actions_mapped_{args.run_name}"
    model.save(out_name)
    vec_env.save(vecnorm_path)
    print(f"Saved model to {out_name} and vecnorm to {vecnorm_path}")
