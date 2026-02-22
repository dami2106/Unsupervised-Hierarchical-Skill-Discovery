# options_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error
from top_down_env_gymnasium import CraftaxTopDownEnv

# IMPORTANT: import the updated helpers
from option_helpers import (
    available_skills,
    bc_policy_hierarchy,
    load_all_models_hierarchy,
    new_call_id,
    should_terminate,          # phase/instance-aware
)

class OptionsOnTopEnv(gym.Env):
    """
    Discrete action space:
      [0..P-1]      -> primitive actions (single step)
      [P..P+K-1]    -> options (macro policy via BC, but we execute EXACTLY ONE primitive per env.step)

    Commitment semantics:
      - When NO option is active: primitives are valid; options valid iff available_skills(...)
      - When an option IS active: ONLY that option is valid (primitives & other options masked out)
      - Option ends if: episode ends, should_terminate(...), or step budget hits 0
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        base_env,
        num_primitives: int = 17,
        gamma: float = 0.99,
        max_skill_len: int = 100,   # per-leaf cap; composites scale by number of leaves

        skill_list=['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
        hierarchies_dir='Traces/stone_pickaxe_easy/hierarchy_data/Simple',
        symbol_map={
            "0": "stone",
            "1": "stone_pickaxe",
            "2": "table",
            "3": "wood",
            "4": "wood_pickaxe",
        },
        root: str = 'Traces/stone_pickaxe_easy',
        backbone_hint:  str = 'resnet34',
        bc_checkpoint_dir: str = 'bc_checkpoints_resnet',
        pca_model_path: str = 'pca_models/pca_model_750.joblib',
        pu_start_models_dir: str = 'pu_start_models',
        pu_end_models_dir: str = 'pu_end_models',

    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)


        # Models / skills (supports composite skills + instance-scoped runtime)
        self.models = load_all_models_hierarchy(
            skill_list,
            hierarchies_dir,
            symbol_map,
            root,
            backbone_hint,
            bc_checkpoint_dir,
            pca_model_path,
            pu_start_models_dir,
            pu_end_models_dir,
        )

        self.skills = self.models["skills"]
        self.num_options = len(self.skills)

        print("Number of options:", self.num_options)
        print("Available skills:", self.skills)
        print("symbol_map:", symbol_map)

        # ---- Action space mapping
        self.num_primitives = int(num_primitives)
        assert self.num_primitives > 0, "Need at least 1 primitive action"
        assert self.num_primitives <= self.env.action_space.n, \
            f"num_primitives={self.num_primitives} exceeds base primitive actions ({self.env.action_space.n})"

        self.action_space = spaces.Discrete(self.num_primitives + self.num_options)
        self.observation_space = self.env.observation_space

        # Book-keeping
        self.elapsed_macro_steps = 0
        self.current_obs = None

        # Seeding consistent with base
        self._seed = getattr(self.env, "_seed", 0)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

        # Commitment state (active option)
        self.active_option_idx = None    # index into self.skills, or None
        self.active_call_id = None       # unique call id for this macro execution
        self.option_steps_left = 0       # budget counter (see _skill_budget)

    # ---------- utilities ----------
    def _skill_budget(self, skill_name: str) -> int:
        """
        Primitive skills get max_skill_len.
        Composite skills get (#leaf_skills) * max_skill_len.
        We infer composite by checking models["bc_models"][skill_name] == ("__COMPOSITE__", seq).
        """
        entry = self.models["bc_models"].get(skill_name)
        if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
            seq = entry[1]             # list of leaf skill names
            return len(seq) * self.max_skill_len
        return self.max_skill_len

    @staticmethod
    def _as_uint8_frame(obs):
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                return obs
            return (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        raise TypeError(f"Unsupported obs type for BC: {type(obs)}")

    # ---------- MaskablePPO hook ----------
    def action_masks(self):
        P, K = self.num_primitives, self.num_options
        if K == 0:
            return np.ones(P, dtype=bool)

        if self.current_obs is None:
            # Not reset yet: allow only primitives
            prim_mask = np.ones(P, dtype=bool)
            opt_mask  = np.zeros(K, dtype=bool)
            return np.concatenate([prim_mask, opt_mask], axis=0)

        # If committed to an option, expose ONLY that option
        if self.active_option_idx is not None:
            prim_mask = np.zeros(P, dtype=bool)
            opt_mask  = np.zeros(K, dtype=bool)
            opt_mask[self.active_option_idx] = True
            return np.concatenate([prim_mask, opt_mask], axis=0)

        # No active option: primitives ON, options according to availability
        prim_mask = np.ones(P, dtype=bool)
        frame = self._as_uint8_frame(self.current_obs)
        full_mask = available_skills(self.models, frame)  # aligned to models["skills"]

        # Map any internal order to self.skills order (if different)
        skills_order = self.models["skills"]
        idx_map = {s: i for i, s in enumerate(skills_order)}
        opt_mask = np.array([full_mask[idx_map[s]] for s in self.skills], dtype=bool)
        return np.concatenate([prim_mask, opt_mask], axis=0)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_macro_steps = 0
        self.current_obs = obs

        # Clear composite/instance runtime across episodes
        if "composite_runtime" in self.models:
            self.models["composite_runtime"].clear()
        if "call_id_ctr" in self.models:
            self.models["call_id_ctr"] = 0

        # Clear commitment
        self.active_option_idx = None
        self.active_call_id = None
        self.option_steps_left = 0

        return obs, info

    def _bc_one_step_for_active(self, frame_uint8: np.ndarray) -> int:
        """
        Compute exactly ONE primitive action from BC for the active option instance.
        """
        assert self.active_option_idx is not None
        assert self.active_call_id is not None
        skill_name = self.skills[self.active_option_idx]
        prim = int(
            bc_policy_hierarchy(
                self.models,
                frame_uint8,
                skill_name,
                self.active_call_id,
                max_leaf_len=self.max_skill_len,  # per-leaf cap
            )
        )
        if prim < 0 or prim >= self.num_primitives:
            raise error.InvalidAction(
                f"bc_policy returned invalid primitive {prim} (P={self.num_primitives})"
            )
        return prim

    def step(self, a):
        # Normalize scalar action
        if not np.isscalar(a):
            a = int(np.asarray(a).item())
        if a < 0 or a >= self.action_space.n:
            raise error.InvalidAction(f"Action {a} out of range for Discrete({self.action_space.n})")

        P, K = self.num_primitives, self.num_options

        # Decide which primitive to execute THIS step
        if self.active_option_idx is None:
            # No active option yet
            if a < P:
                # Chose a primitive → execute it directly
                prim_action = a
            else:
                # Start a new option; commit and run ONE BC step
                picked_idx = a - P
                if picked_idx < 0 or picked_idx >= K:
                    raise error.InvalidAction(f"Invalid option id {picked_idx}")
                self.active_option_idx = picked_idx
                self.active_call_id = new_call_id(self.models)  # unique instance id
                skill_name = self.skills[self.active_option_idx]
                self.option_steps_left = self._skill_budget(skill_name)  # total budget for this macro
                frame = self._as_uint8_frame(self.current_obs)
                prim_action = self._bc_one_step_for_active(frame)
        else:
            # Option is active → ignore 'a' (mask only exposes the same option anyway)
            frame = self._as_uint8_frame(self.current_obs)
            prim_action = self._bc_one_step_for_active(frame)

        # ---- exactly one primitive env step ----
        obs, r, terminated, truncated, info = self.env.step(prim_action)
        self.current_obs = obs
        self.elapsed_macro_steps += 1

        # ---- termination checks for active option (if any) ----
        if self.active_option_idx is not None:
            self.option_steps_left -= 1
            stop = False

            if terminated or truncated:
                stop = True
            else:
                skill_name = self.skills[self.active_option_idx]
                if should_terminate(self.models, self._as_uint8_frame(obs), skill_name, self.active_call_id):
                    stop = True
                elif self.option_steps_left <= 0:
                    stop = True

            if stop:
                # Clear commitment
                self.active_option_idx = None
                self.active_call_id = None
                self.option_steps_left = 0

        return obs, float(r), bool(terminated), bool(truncated), info

    # Optional passthroughs
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

if __name__ == "__main__":
    # Build envs
    base = CraftaxTopDownEnv(
        render_mode="rgb_array",
        reward_items=[],
        done_item="wood_pickaxe",
        include_base_reward=False,
        return_uint8=True,
    )
    env = OptionsOnTopEnv(base_env=base, num_primitives=16, gamma=0.99, max_skill_len=25)
    # env.max_skill_len = 30  
    # env.max_skill_len = 20     


    # Reproducibility
    seed = 888
    random.seed(seed)
    np.random.seed(seed)

    # Reset and capture first frame
    frames = []
    env.debug_record_frames = frames
    obs, info = env.reset(seed=seed)
    frames.append(obs.copy())

    num_primitives = env.num_primitives
    num_options = env.num_options
    print(f"Env with {num_primitives} primitives + {num_options} options (total {env.action_space.n})")
    print("Available skills:", env.skills)

    # Now this list can contain both primitives and options directly
    # Available skills: ['wood',    'stone',     'wood_pickaxe',     'stone_pickaxe',      'table']
    # Available skills: ['16',      '17',        '18',               '19',                 '20']
    # Available skills: ['wood' , 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table', 'Production_0', 'Production_1', 'Production_8 WTW 23', 'Production_10', 'Production_9', 'Production_14']


    skills_seq = [ 23] # w w w 



    mask = env.action_masks()
    print_options_mask("BEFORE first run", mask, num_primitives, num_options)

    rewards = []

    def run_action(a, tag):
        mask = env.action_masks()
        valid = bool(mask[a]) if a < len(mask) else False
        label = str(a)

        if a >= num_primitives and hasattr(env, "skills"):
            skill_idx = a - num_primitives
            if 0 <= skill_idx < len(env.skills):
                label += f" (option:{env.skills[skill_idx]})"

        print(f"{tag}: running action {label}, valid={valid}")
        obs, r, terminated, truncated, info = env.step(a)
        frames.append(obs.copy())
        print(f"{tag}: reward={r}, terminated={terminated}, truncated={truncated}")

        mask_after = env.action_masks()
        print_options_mask(f"AFTER {tag}", mask_after, num_primitives, num_options)

        # if terminated or truncated:
        #     print(f"{tag}: episode ended → resetting")
        #     obs, _ = env.reset()
        #     frames.append(obs.copy())

        print("===========")
        return r

    # Execute the sequence (mix primitives + options)
    for i, a in enumerate(skills_seq, 1):
        r = run_action(a, f"Run {i}")
        rewards.append(r)

    # Save GIF
    # Annotate each frame with the frame number in the top left
    annotated_frames = []
    for idx, frame in enumerate(frames):
        # Ensure frame is in uint8 format
        img = frame.copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # Add text
        cv2.putText(
            img,
            f"Frame: {idx}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated_frames.append(img)

    seq_str = "-".join(map(str, skills_seq))
    rewards_str = "_".join(str(float(r)) for r in rewards)
    out_path = f"craftax_seq_.gif"
    imageio.mimsave(out_path, annotated_frames, fps=10)
    print(f"Saved GIF to: {out_path}")