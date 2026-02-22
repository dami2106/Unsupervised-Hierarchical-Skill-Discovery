# PPO_Skills.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker  # <-- ensures masks used in rollout

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium import CraftaxTopDownEnv
from top_down_env_gymnasium_options import OptionsOnTopEnv

import imageio


import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from sb3_contrib.common.wrappers import ActionMasker

from option_helpers import FixedSeedAlways, to_gif_frame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_primitives", type=int, default=16)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_skill_len", type=int, default=25)

parser.add_argument("--skill_list", nargs="+", default=['wood_pick'])
parser.add_argument("--root", type=str, default='Traces/stone_pickaxe_easy')
parser.add_argument("--bc_checkpoint_dir", type=str, default='bc_checkpoints_resnet')
parser.add_argument("--pca_model_path", type=str, default='pca_models/pca_model_750.joblib')
parser.add_argument("--pu_start_models_dir", type=str, default='pu_start_models')
parser.add_argument("--pu_end_models_dir", type=str, default='pu_end_models')

parser.add_argument("--run_name", type=str, default='test_ppo_options')
parser.add_argument("--ppo_seed", type=int, default=888)

args, _ = parser.parse_known_args()

def make_options_env(*, seed: int, render_mode=None,  max_episode_steps=100):
    def _thunk():
        base = CraftaxTopDownEnv(
            render_mode=render_mode,
            reward_items=[],
            done_item="wood_pickaxe",
            include_base_reward=False,
            return_uint8=True,
        )

        

        core = OptionsOnTopEnv(
            base,
            num_primitives=args.num_primitives,
            gamma=args.gamma,
            max_skill_len=args.max_skill_len,
            skill_list=args.skill_list,
            root=args.root,
            bc_checkpoint_dir=args.bc_checkpoint_dir,
            pca_model_path=args.pca_model_path,
            pu_start_models_dir=args.pu_start_models_dir,
            pu_end_models_dir=args.pu_end_models_dir,
        )

        # 1) ActionMasker wraps the env that has `action_masks`
        def mask_fn(e):  # no unwrapping needed
            return e.action_masks()
        masked = ActionMasker(core, mask_fn)

        # 2) Add outer wrappers afterwards
        capped   = TimeLimit(masked, max_episode_steps=max_episode_steps)
        fixed    = FixedSeedAlways(capped, seed=seed)
        logged   = RecordEpisodeStatistics(fixed)
        return logged
    return _thunk


def mask_fn(env):
    return env.action_masks()


# Helper: unwrap wrappers to reach the env that exposes `action_masks()`
def get_action_masks(env_or_vec):
    """Return the current action mask from a (possibly vectorized and wrapped) env.

    Works with DummyVecEnv and common Gym wrappers by walking down `.env` until
    an object with `action_masks()` is found.
    """
    # If a VecEnv is provided, grab the first sub-env
    env = env_or_vec.envs[0] if hasattr(env_or_vec, "envs") else env_or_vec

    # Walk through wrapper chain
    cur = env
    while True:
        if hasattr(cur, "action_masks") and callable(getattr(cur, "action_masks")):
            return cur.action_masks()
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break

    # Final check (in case the base env itself had it but loop ended)
    if hasattr(cur, "action_masks") and callable(getattr(cur, "action_masks")):
        return cur.action_masks()

    raise AttributeError("No action_masks() found in env wrapper stack.")


if __name__ == "__main__":
    K = 5
    TRAIN_SEED = 888  # set your fixed training seed here

    train_env = DummyVecEnv([make_options_env(seed=TRAIN_SEED, render_mode=None)])
    train_env = VecTransposeImage(train_env) 
    train_env = VecMonitor(train_env)

    print("Training PPO on options to get wood_pickaxe SEED : ", args.ppo_seed)

    model = MaskablePPO(
        "CnnPolicy",                   # pixels -> CNN
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
        seed=args.ppo_seed,
    )



    model.learn(
        total_timesteps=100_000,
        tb_log_name=args.run_name,   
        log_interval=1,
        progress_bar=True,

    )
    model.save(args.run_name)


    # obs = eval_env_vec.reset()
    # # Convert the initial observation to a GIF-safe frame (HWC uint8) to match subsequent frames
    # frames = [to_gif_frame(obs)]

    # done = False
    # steps = 0
    # while not done and steps < 100:
    #     masks = get_action_masks(eval_env_vec)
    #     action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    #     obs, reward, terminated, info = eval_env_vec.step(action)
    #     frames.append(to_gif_frame(obs))
    #     done = bool(terminated[0])

    #     print("Step:", steps, "Action:", action, "Reward:", reward, "Done:", done)

    #     steps += 1

    # imageio.mimsave("craftax_ppo_wood_pick_options_eval.gif", frames, fps=1)