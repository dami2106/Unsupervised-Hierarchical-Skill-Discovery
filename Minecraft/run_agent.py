from argparse import ArgumentParser
import pickle
import gzip
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS
import gym
import os
import copy


class InventoryDoneWrapper(gym.Wrapper):
    """
    Gym wrapper that ends an episode when the inventory contains
    at least `target_count` of `target_item`.
    """

    def __init__(self, env, target_item: str, target_count: int = 1):
        super().__init__(env)
        self.target_item = target_item
        self.target_count = target_count
        self._item_collected = False

    def reset(self, **kwargs):
        self._item_collected = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        inventory = obs.get("inventory", {})
        count = inventory.get(self.target_item, 0)
        if count >= self.target_count:
            done = True
            self._item_collected = True
        # expose flag in info
        info["collected_target"] = self._item_collected
        return obs, reward, done, info


def setup_agent_env(
    model_path: str, weights_path: str, target_item: str, target_qty: int
):
    """
    Load a pretrained agent and wrap the environment.
    """
    env = HumanSurvival(**ENV_KWARGS).make()
    env = InventoryDoneWrapper(env, target_item=target_item, target_count=target_qty)

    try:
        with open(model_path, "rb") as f:
            params = pickle.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading model file '{model_path}': {e}")

    policy_kwargs = params["model"]["args"]["net"]["args"]
    pi_head_kwargs = params["model"]["args"]["pi_head_opts"].copy()
    pi_head_kwargs["temperature"] = float(pi_head_kwargs.get("temperature", 1.0))

    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)

    try:
        agent.load_weights(weights_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading weights from '{weights_path}': {e}")

    return agent, env


if __name__ == "__main__":
    parser = ArgumentParser(description="Run pretrained models on MineRL environment")
    parser.add_argument(
        "--model",
        type=str,
        default="foundation-model-3x.model",
        help="Path to the pickled model file.",
    )
    # parser.add_argument("--model", type=str, default="2x.model", help="Path to the pickled model file.")
    parser.add_argument(
        "--weights",
        type=str,
        default="foundation-model-3x.weights",
        help="Path to the model weights.",
    )
    # parser.add_argument("--weights", type=str, default="rl-from-early-game-2x.weights", help="Path to the model weights.")
    parser.add_argument(
        "--target-item", type=str, default="oak_log", help="Inventory item to collect."
    )
    parser.add_argument(
        "--target-qty",
        type=int,
        default=2,
        help="Quantity of the target item to collect.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Max steps per episode."
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of successful episodes to save."
    )
    # parser.add_argument("--save-dir", type=str, default="Data/iron_ingot", help="Base directory to save observations.")

    args = parser.parse_args()

    args.save_dir = f"Data/{args.target_item}"
    obs_dir = os.path.join(args.save_dir, "raw_obs")
    os.makedirs(obs_dir, exist_ok=True)

    # Detect existing compressed or uncompressed episodes
    existing_ids = set()
    for fn in os.listdir(obs_dir):
        if not fn.startswith("minecraft_"):
            continue
        tail = fn[len("minecraft_") :]
        for ext in (".pkl", ".pkl.gz"):
            if tail.endswith(ext):
                idx = tail[: -len(ext)]
                if idx.isdigit():
                    existing_ids.add(int(idx))

    # Compute which episode IDs are missing in [1 .. args.episodes]
    all_ids = set(range(1, args.episodes + 1))
    missing_ids = sorted(all_ids - existing_ids)
    if not missing_ids:
        print(f"All {args.episodes} episodes already exist! Nothing to do.")
        exit(0)

    print(f"Found {len(missing_ids)} missing episodes: {missing_ids}")

    # Prepare agent and environment once
    agent, env = setup_agent_env(
        model_path=args.model,
        weights_path=args.weights,
        target_item=args.target_item,
        target_qty=args.target_qty,
    )

    # Generate each missing episode
    for ep_id in missing_ids:
        print(f"\n=== Generating episode {ep_id} ===")
        obs = env.reset()
        obs["isGuiOpen"] = False
        trajectory = []
        done_flag = False

        for step in range(args.max_steps):
            action = agent.get_action(obs)

            print("Action taken:", action)

            record = copy.deepcopy(obs)
            record["action"] = action
            trajectory.append(record)

            obs, reward, done, info = env.step(action)
            obs["isGuiOpen"] = info.get("isGuiOpen", False)

            if done:
                if info.get("collected_target", False):
                    done_flag = True
                    print(f"  Collected target at step {step + 1}.")
                else:
                    print(
                        f"  Episode ended without collecting target at step {step + 1}."
                    )
                break

        if not done_flag:
            print(
                f"  Did NOT collect target within {args.max_steps} steps; skipping save."
            )
            continue  # move on to next missing ep

        # Save completed episode
        out_path = os.path.join(obs_dir, f"minecraft_{ep_id}.pkl.gz")
        with gzip.open(out_path, "wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved episode {ep_id} to {out_path}.")

    env.close()
    print(f"\nDone! Filled in {len(missing_ids)} missing episodes.")
