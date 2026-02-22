#PPO_Basic.py

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

from option_helpers import FixedSeedAlways, to_gif_frame



parser = argparse.ArgumentParser()
parser.add_argument("--ppo_seed", type=int, default=888)

args, _ = parser.parse_known_args()

def make_env(seed: int):
    def _thunk():
        env = CraftaxTopDownEnv(
            seed=seed,               # initialize env RNG deterministically
            render_mode=None,
            reward_items=[],
            done_item="wood_pickaxe",
            include_base_reward=False,
        )
        env = TimeLimit(env, max_episode_steps=100)
        env = FixedSeedAlways(env, seed=seed)  # keep the SAME seed every reset
        return env
    return _thunk

if __name__ == "__main__":
    SEED = 888  # single training seed; every episode uses this seed

    train_env = DummyVecEnv([make_env(seed=SEED)])
    train_env = VecTransposeImage(train_env)  # HWC -> CHW for CnnPolicy
    train_env = VecMonitor(train_env)

    # model = PPO.load("ppo_craftax_wood_ppo_actions")

    print ("Training PPO on basic actions to get wood_pickaxe SEED : ", args.ppo_seed)

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        seed=args.ppo_seed,  # set SB3/torch RNGs too for full determinism
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
        gamma=0.99,
    )

    model.learn(
        total_timesteps=100_000,
        log_interval=1,
        tb_log_name="ppo_wood_pick_actions",
        progress_bar=True,
    )
    
    model.save("ppo_craftax_wood_pick_actions")

    # -------- Eval vec env (choose a seed; use same wrapper for fixed-seed eval) --------
    

    # obs = eval_env.reset()
    # images = [to_gif_frame(obs)]

    # done = False
    # steps = 0
    # while not done and steps < 100:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, infos = eval_env.step(action)
    #     images.append(to_gif_frame(obs))
    #     done = bool(dones[0])

    #     print("Step:", steps, "Action:", action, "Reward:", rewards, "Done:", done)

    #     steps += 1

    # imageio.mimsave(f"craftax_ppo_wood_pick_actions_eval.gif", images, fps=1)
