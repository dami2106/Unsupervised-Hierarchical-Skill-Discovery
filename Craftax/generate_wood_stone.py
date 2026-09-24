"""
Generate Craftax skill datasets (Wood-Stone Collection "wsws", or the paper's Stone
Pickaxe plan with --task stone_pickaxe) directly in the RLDataset
layout consumed by HiSD (``groundTruth/``, ``mapping/mapping.txt``, ``top_down_obs/``,
``pixel_obs/`` (egocentric local view), ``actions/``).

Ground-truth skill labels are the plan step that produced each frame.  In the
wood_stone task (K = 2) the agent starts with a wooden pickaxe so both skills
(collect wood, collect stone) are always executable; stone_pickaxe (K = 5) runs
wood, wood, table, wood, wooden_pickaxe, stone, wood, stone_pickaxe.

  --order static : every episode runs wood, stone, wood, stone
  --order random : every episode runs a random sequence of 3-6 wood/stone steps

Example:
    python generate_wood_stone.py --samples 100 --order random --path Traces/wsws_random/
"""
import argparse
import json
import logging

import os
import sys

# Always use the repository's modified Craftax (Craftax/craftax), never a pip-installed one
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from craftax.craftax_env import make_craftax_env_from_name
import craftax
assert os.path.dirname(os.path.abspath(craftax.__file__)) == os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "craftax"), f"using a non-local craftax: {craftax.__file__}"

import block_types as bt
import action_types as at
from astar import plan_to_object, plan_to_object_with_mining, set_astar_log_level
from helpers import get_top_down_obs

jax.config.update("jax_platform_name", "cpu")
logging.getLogger().setLevel(logging.ERROR)
set_astar_log_level(logging.ERROR)

STEPS = {
    "wood": (bt.TREE, [at.DO]),
    "stone": (bt.STONE, [at.DO]),
    "table": (bt.GRASS, [at.PLACE_TABLE]),
    "wooden_pickaxe": (bt.CRAFTING_TABLE, [at.MAKE_WOOD_PICKAXE]),
    "stone_pickaxe": (bt.CRAFTING_TABLE, [at.MAKE_STONE_PICKAXE]),
}
INVENTORY_KEY = {"wooden_pickaxe": "wood_pickaxe"}
# the paper's Stone Pickaxe plan (Craftax/generate_data.py)
STONE_PICKAXE_PLAN = ["wood", "wood", "table", "wood", "wooden_pickaxe", "stone", "wood", "stone_pickaxe"]


def sample_plan(rng, order, task):
    if task == "stone_pickaxe":
        return list(STONE_PICKAXE_PLAN)
    if order == "static":
        return ["wood", "stone", "wood", "stone"]
    n = rng.integers(3, 7)
    return list(rng.choice(["wood", "stone"], size=n))


def run_step(step_fn, key, state, env_params, target, extra_actions):
    start = tuple(int(x) for x in state.player_position.tolist())
    path = plan_to_object_with_mining(state.map, start, target)
    if path is None:
        path = plan_to_object(state.map, start, target)
    if path is None:
        raise RuntimeError(f"no path to {target}")
    obs_list, state_list, actions = [], [], []
    for ac in list(path) + list(extra_actions):
        obs, state, _, _, _ = step_fn(key, state, ac, env_params)
        obs_list.append(np.asarray(obs))
        state_list.append(state)
        actions.append(int(ac))
    return state, obs_list, state_list, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--order", choices=["static", "random"], default="random")
    parser.add_argument("--task", choices=["wood_stone", "stone_pickaxe"], default="wood_stone",
                        help="wood_stone (K=2, starts with a wooden pickaxe) or the paper's stone_pickaxe plan (K=5)")
    parser.add_argument("--path", type=str, default="Traces/wsws_random/")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--downscale", type=int, default=2,
                        help="integer downscale of the 274x274 top-down image (memory)")
    args = parser.parse_args()

    for sub in ["groundTruth", "mapping", "top_down_obs", "pixel_obs", "actions"]:
        os.makedirs(os.path.join(args.path, sub), exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    env_params = env.default_params.replace(max_timesteps=100000, day_length=99999,
                                            mob_despawn_distance=0)
    step_fn = jax.jit(env.step, static_argnums=(3,))
    rng = np.random.default_rng(args.base_seed)

    lengths, plans = [], []
    seed, n_ok, n_fail = args.base_seed, 0, 0
    pbar = tqdm(total=args.samples)
    while n_ok < args.samples:
        seed += 1
        if n_fail > 10 * args.samples:
            raise RuntimeError("too many failed episodes")
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key, env_params)
        if args.task == "wood_stone":
            state = state.replace(inventory=state.inventory.replace(wood_pickaxe=jnp.asarray(1, dtype=jnp.int32)))
        plan = sample_plan(rng, args.order, args.task)
        frames = [get_top_down_obs(state, np.asarray(obs))]
        local = [np.asarray(obs)]  # egocentric local view (the env's native pixel obs)
        truths = [plan[0]]
        actions = []
        try:
            for skill in plan:
                target, extra = STEPS[skill]
                inv_key = INVENTORY_KEY.get(skill, skill)
                if inv_key == "table":
                    inv_before = int((state.map == bt.CRAFTING_TABLE).sum())
                else:
                    inv_before = int(getattr(state.inventory, inv_key))
                state, o, s, a = run_step(step_fn, key, state, env_params, target, extra)
                inv_after = int((state.map == bt.CRAFTING_TABLE).sum()) if inv_key == "table" \
                    else int(getattr(state.inventory, inv_key))
                if inv_after <= inv_before:
                    raise RuntimeError(f"step {skill} did not succeed")
                frames.extend(get_top_down_obs(si, oi) for si, oi in zip(s, o))
                local.extend(o)
                truths.extend([skill] * len(o))
                actions.extend(a)
        except RuntimeError:
            n_fail += 1
            continue
        actions.append(0)
        frames = np.asarray(frames, dtype=np.float32)
        if args.downscale > 1:
            d = args.downscale
            h = frames.shape[1] // d * d
            frames = frames[:, :h, :h].reshape(len(frames), h // d, d, h // d, d, 3).mean(axis=(2, 4))
        name = f"craftax_{n_ok}"
        np.save(os.path.join(args.path, "top_down_obs", name + ".npy"), (frames * 255).astype(np.uint8))
        np.save(os.path.join(args.path, "pixel_obs", name + ".npy"), (np.asarray(local) * 255).astype(np.uint8))
        np.save(os.path.join(args.path, "actions", name + ".npy"), np.asarray(actions))
        with open(os.path.join(args.path, "groundTruth", name), "w") as f:
            f.write("\n".join(truths))
        lengths.append(len(truths))
        plans.append(plan)
        n_ok += 1
        pbar.update(1)

    with open(os.path.join(args.path, "mapping", "mapping.txt"), "w") as f:
        for i, skill in enumerate(sorted({x for p in plans for x in p})):
            f.write(f"{i} {skill}\n")
    with open(os.path.join(args.path, "trace_config.json"), "w") as f:
        json.dump({"parameters": vars(args), "plans": plans,
                   "stats": {"min_len": int(np.min(lengths)), "avg_len": float(np.mean(lengths)),
                             "max_len": int(np.max(lengths))}}, f, indent=2)
    print(f"Generated {n_ok} episodes; mean length {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
