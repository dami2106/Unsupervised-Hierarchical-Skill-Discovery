import numpy as np
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
import gym
from gym.wrappers import TimeLimit
from gym import spaces
import os, sys, torch
from contextlib import contextmanager
from PIL import Image
import torchvision.transforms as T
import json
from typing import List, Tuple, Dict, Optional
import torch.nn as nn
from joblib import load as joblib_load
from sb3_contrib.common.wrappers import ActionMasker



@contextmanager
def pushd(new_dir: str):
    prev = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

CAM_VALUES = np.array(
    [
        -10.0,
        -5.80948313,
        -3.21536913,
        -1.60949864,
        -0.61539427,
        0.0,
        0.61539427,
        1.60949864,
        3.21536913,
        5.80948313,
        10.0,
    ],
    dtype=np.float32,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM = 512
NUM_BUTTONS = 20
NUM_CAM = 11
# Was 300. With skip=8, 300 steps is 2400 ticks (longer than episode).
# We want ~300 ticks / 8 = ~37 steps. Let's set it to 50 to be safe.
MAX_SKILL_STEPS = 50
# 3 steps * 8 ticks = 24 ticks of checking the "End Classifier".
# Increased from 2 to provide more stability with frame skip.
END_PATIENCE_STEPS = 3


# ======================= BC model (consumes MineCLIP features) ===============

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=512, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.btn_head = nn.Linear(hidden, NUM_BUTTONS)
        self.yaw_head = nn.Linear(hidden, NUM_CAM)
        self.pitch_head = nn.Linear(hidden, NUM_CAM)

    def forward(self, x):
        h = self.backbone(x)
        return self.btn_head(h), self.yaw_head(h), self.pitch_head(h)

def load_bc(skill_name: str, ckpt_dir: str):
    ckpt_path = os.path.join(ckpt_dir, f"bc_model_{skill_name}.pt")
    thr_path = os.path.join(ckpt_dir, f"thresholds_{skill_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    if not os.path.exists(thr_path):
        raise FileNotFoundError(thr_path)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model = PolicyNet().to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    mean = state["mean"].to(DEVICE)
    std = state["std"].to(DEVICE)
    thresholds = torch.load(thr_path, map_location=DEVICE).to(DEVICE)
    return model, mean, std, thresholds

def _to_feat_tensor(obs_feat: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """Ensure observation feature vector is a float32 torch tensor of shape [1, D]."""
    if isinstance(obs_feat, torch.Tensor):
        t = obs_feat
    else:
        t = torch.from_numpy(np.asarray(obs_feat))
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)

@torch.no_grad()
def bc_action_multidiscrete(model, mean, std, thresholds, feat_1xD):
    """Compute MultiDiscrete action from MineCLIP feature tensor [1, D]."""
    z = (feat_1xD - mean) / std
    btn_l, yaw_l, pitch_l = model(z)
    btn = (torch.sigmoid(btn_l[0]) >= thresholds).to(torch.int64).cpu().numpy()
    yaw_idx = int(torch.argmax(yaw_l[0]).item())
    pitch_idx = int(torch.argmax(pitch_l[0]).item())
    vec = np.zeros(22, dtype=np.int64)
    vec[:20] = btn; vec[20] = yaw_idx; vec[21] = pitch_idx
    return vec

# ======================= PU start/end utils ==================================

def _require_pu_model_and_threshold(models_dir: str, skill: str):
    clf_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path = os.path.join(models_dir, f"{skill}_meta.json")
    if not os.path.isfile(clf_path) or not os.path.isfile(meta_path):
        print(f"[ERROR] Missing PU model or meta for skill '{skill}' in {models_dir}")
        raise FileNotFoundError(f"Expected files: {clf_path} and {meta_path}")
    clf = joblib_load(clf_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if "threshold" not in meta:
        print(f"[ERROR] Missing 'threshold' in meta for skill '{skill}' at {meta_path}")
        raise KeyError(f"threshold not found in {meta_path}")
    thr = float(meta["threshold"])
    return clf, thr

def _proba_geq(model, thr: float, feat_1xD_torch: torch.Tensor) -> bool:
    x = feat_1xD_torch.detach().to("cpu").numpy().astype(np.float32)
    p = float(model.predict_proba(x)[:, 1][0])
    return p >= thr

# ======================= Skill multiplexer (strict lock) ======================

class SkillMuxWrapper(gym.Wrapper):
    """
    MultiDiscrete action:
      - action[0] in {0..K}  (0 = primitive, 1..K = BC skill i)
      - action[1:] = primitive MultiDiscrete (same as base env)

    Strict locking:
      - If NO active skill: selector valid = {0} ∪ { i : START_i >= thr_i }.
      - If a skill is ACTIVE: selector valid = { active } ONLY.
        (Primitives and other skills are masked.)
    Primitive sub-actions remain unmasked in the vector, but are ignored while locked.
    """

    def __init__(
        self,
        env,
        skills: List[str],
        ckpt_dir: str,
        device: str,
        start_models_dir: str,
        end_models_dir: str,
        end_patience_steps: int = END_PATIENCE_STEPS,
    ):
        super().__init__(env)
        self.skills = list(skills)
        self.num_skills = len(self.skills)
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.start_models_dir = start_models_dir
        self.end_models_dir = end_models_dir
        self.end_patience_steps = int(max(0, end_patience_steps))

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.primitives_nvec = env.action_space.nvec
        nvec = np.concatenate(([self.num_skills + 1], self.primitives_nvec))
        self.action_space = spaces.MultiDiscrete(nvec)
        self.observation_space = env.observation_space

        # Load BC policies
        self._bc_cache: Dict[str, Tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for s in self.skills:
            self._bc_cache[s] = load_bc(s, ckpt_dir)

        # REQUIRED: Load PU start/end models + thresholds
        self._start_models: Dict[str, Tuple[object, float]] = {}
        self._end_models: Dict[str, Tuple[object, float]] = {}
        for s in self.skills:
            self._start_models[s] = _require_pu_model_and_threshold(self.start_models_dir, s)
            self._end_models[s] = _require_pu_model_and_threshold(self.end_models_dir, s)

        self._active_skill_idx: Optional[int] = None  # 1..K
        self._steps_in_skill: int = 0
        self._last_obs = None
        self._startable_mask_cache: Optional[np.ndarray] = None  # (K,)
        self._end_fired_recently: int = 0

    # ---------- Mask helpers ----------

    def _compute_startable_skills(self, obs_feat_vec) -> np.ndarray:
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)  # [1,D]
        out = np.zeros(self.num_skills, dtype=bool)
        for i, s in enumerate(self.skills):
            model, thr = self._start_models[s]
            out[i] = _proba_geq(model, thr, feat)
        return out

    def _end_should_fire(self, obs_feat_vec) -> bool:
        if self._active_skill_idx is None:
            self._end_fired_recently = 0
            return False
        skill_name = self.skills[self._active_skill_idx - 1]
        model, thr = self._end_models[skill_name]
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)
        fired = _proba_geq(model, thr, feat)
        self._end_fired_recently = self._end_fired_recently + 1 if fired else 0
        return self._end_fired_recently >= self.end_patience_steps or \
               (self._steps_in_skill >= MAX_SKILL_STEPS)

    # Called by ActionMasker each step
    def compute_action_mask(self) -> np.ndarray:
        sel_size = self.num_skills + 1

        if self._startable_mask_cache is None and self._last_obs is not None:
            self._startable_mask_cache = self._compute_startable_skills(self._last_obs)

        selector_mask = np.zeros(sel_size, dtype=bool)

        if self._active_skill_idx is None:
            # allow primitives
            selector_mask[0] = True
            # allow only startable skills
            if self._startable_mask_cache is not None:
                selector_mask[1:] = self._startable_mask_cache
            else:
                # at very first mask call before obs, be permissive
                selector_mask[1:] = True
        else:
            # STRICT LOCK: only the active skill is valid
            selector_mask[self._active_skill_idx] = True
            # primitives (0) intentionally NOT allowed here

        primitive_mask = np.concatenate([np.ones(n, dtype=bool) for n in self.primitives_nvec])
        return np.concatenate([selector_mask, primitive_mask])

    def reset(self, **kwargs):
        self._active_skill_idx = None
        self._steps_in_skill = 0
        self._end_fired_recently = 0
        seed = kwargs.pop("seed", None)
        if seed is not None:
            # Older Gym/MineRL APIs do not accept seed in reset(); use env.seed() instead
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self._last_obs = self.env.reset(**kwargs)
        self._startable_mask_cache = self._compute_startable_skills(self._last_obs)
        return self._last_obs

    def _bc_step_action(self, obs_vec):
        assert self._active_skill_idx is not None
        skill_name = self.skills[self._active_skill_idx - 1]
        model, mean, std, thresholds = self._bc_cache[skill_name]
        feat = _to_feat_tensor(obs_vec, device=self.device)
        return bc_action_multidiscrete(model, mean, std, thresholds, feat)

    def step(self, action):
        action = np.asarray(action)
        selector = int(action[0])
        primitive_vec = action[1:].astype(np.int64)

        # Check end condition on current obs BEFORE consuming action
        if self._active_skill_idx is not None and self._end_should_fire(self._last_obs):
            self._active_skill_idx = None
            self._steps_in_skill = 0
            self._end_fired_recently = 0
            self._startable_mask_cache = self._compute_startable_skills(self._last_obs)

        # Decide env action
        if self._active_skill_idx is None:
            # free state: either primitives (0) or a startable skill (1..K)
            if selector == 0:
                env_action = primitive_vec
            else:
                # start the chosen skill
                self._active_skill_idx = selector
                self._steps_in_skill = 0
                self._end_fired_recently = 0
                env_action = self._bc_step_action(self._last_obs)
        else:
            # STRICT LOCK: ignore selector (mask should already enforce),
            # keep executing active skill
            env_action = self._bc_step_action(self._last_obs)

        # Step env
        obs, rew, done, info = self.env.step(env_action)

        # Update counters
        if self._active_skill_idx is not None:
            self._steps_in_skill += 1

        if done:
            self._active_skill_idx = None
            self._steps_in_skill = 0
            self._end_fired_recently = 0

        # Update caches for next step
        self._last_obs = obs
        self._startable_mask_cache = (
            self._compute_startable_skills(self._last_obs)
            if self._active_skill_idx is None
            else None
        )

        info = dict(info)
        info.update(
            {
                "active_skill": self.skills[self._active_skill_idx - 1] if self._active_skill_idx else None,
                "skill_steps_in_run": self._steps_in_skill if self._active_skill_idx else 0,
            }
        )
        return obs, rew, done, info

# ======================= Env factory =========================================

def _mask_fn(env):
    return env.compute_action_mask()


def make_masked_env(
    skills: List[str],
    ckpt_dir: str,
    start_models_dir: str = "Data/minecraft_cobblestone_mapped/pu_start_models",
    end_models_dir: str = "Data/minecraft_cobblestone_mapped/pu_end_models",
    project_root: str = PROJECT_ROOT,
    pretrained_model_path: str = "ViT-B-16.pt",
    device: str = DEVICE,
    target_item: str = "log",
    target_count: int = 1,
    max_episode_steps: int = 2000,
    seed: int = None,
    skip: int = 8,
):
    base = make_env_mineclip(
        project_root=project_root,
        pretrained_model_path=pretrained_model_path,
        device=device,
        target_item=target_item,
        target_count=target_count,
        max_episode_steps=max_episode_steps,
        seed=seed,
        skip=skip,
    )

    mux = SkillMuxWrapper(
        base,
        skills=skills,
        ckpt_dir=ckpt_dir,
        device=device,
        start_models_dir=start_models_dir,
        end_models_dir=end_models_dir,
        end_patience_steps=END_PATIENCE_STEPS,
    )
    masked = ActionMasker(mux, _mask_fn)
    return masked

def get_unique_skills(dir_, files):
    unique_skills = set()
    for file in files:
        with open(os.path.join(dir_, file), "r") as f:
            lines = f.read().splitlines()
        unique_skills.update(lines)
    return unique_skills

class InventoryDoneWrapper(gym.Wrapper):
    def __init__(self, env, target_item: str, target_count: int = 1):
        super().__init__(env)
        self.target_item = target_item
        self.target_count = target_count
        self._item_collected = False

    def reset(self, **kwargs):
        self._item_collected = False
        seed = kwargs.pop("seed", None)
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _base_reward, done, info = self.env.step(action)
        inventory = obs.get("inventory", {})
        total = sum(count for name, count in inventory.items() if self.target_item in name)

        # Replace underlying reward: +1 when threshold reached, else 0.
        if total >= self.target_count:
            reward = 10.0
            done = True  # terminate immediately when threshold reached
            self._item_collected = True
        else:
            reward = 0.0
            self._item_collected = False

        info = dict(info)
        info["collected_target"] = self._item_collected
        return obs, reward, done, info

class PovOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        pov_space = env.observation_space["pov"]
        assert isinstance(pov_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=pov_space.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return obs["pov"]

class MaxAndSkipEnv(gym.Wrapper):
    """
    Frame skip wrapper that repeats actions N times and returns max-pooled observations.
    Must be placed BEFORE MineCLIPObsWrapper to avoid computing embeddings for skipped frames.
    """
    def __init__(self, env, skip=8):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = False
        info = {}
        
        # Track last two frames seen (for max-pooling)
        last_two_frames = []
        
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            
            # Keep track of last two frames
            last_two_frames.append(obs["pov"])
            if len(last_two_frames) > 2:
                last_two_frames.pop(0)
            
            total_reward += reward
            if done:
                break
        
        # Max-pool over the last two frames to remove flickering (common in games)
        # If we only have one frame, just use it
        if len(last_two_frames) == 1:
            max_frame = last_two_frames[0]
        elif len(last_two_frames) == 2:
            max_frame = np.maximum(last_two_frames[0], last_two_frames[1])
        else:
            # Shouldn't happen, but fallback to last obs
            max_frame = obs["pov"]
        
        # Update the obs with the max frame so MineCLIP sees the "cleanest" view
        obs["pov"] = max_frame
        
        return obs, total_reward, done, info

class DictToMultiDiscreteActions(gym.ActionWrapper):
    """
    Expose a MultiDiscrete([2]*20 + [11, 11]) to SB3 and convert it
    to the env's expected action dict with:
      - 20 binary buttons in {0,1}
      - 'camera': 2 floats picked from CAM_VALUES (yaw, pitch)
    """

    # Put the 20 button keys in a **fixed order** that matches your data / env
    BUTTON_KEYS = [
        "attack",
        "back",
        "forward",
        "jump",
        "left",
        "right",
        "sneak",
        "sprint",
        "use",
        "drop",
        "inventory",
        "hotbar.1",
        "hotbar.2",
        "hotbar.3",
        "hotbar.4",
        "hotbar.5",
        "hotbar.6",
        "hotbar.7",
        "hotbar.8",
        "hotbar.9",
    ]

    def __init__(self, env, cam_values: np.ndarray):
        super().__init__(env)
        self.cam_values = cam_values
        # Sanity checks
        assert isinstance(env.action_space, spaces.Dict), "Expected a Dict action space"
        for k in self.BUTTON_KEYS + ["camera"]:
            assert k in env.action_space.spaces, f"Missing action key: {k}"
        assert env.action_space["camera"].shape == (2,), (
            "camera must be a 2D Box (yaw, pitch)"
        )

        # MultiDiscrete: 20 binaries + 2 camera indices
        nvec = np.array([2] * 20 + [len(cam_values), len(cam_values)], dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(nvec)

    def action(self, act_vec):
        """
        Convert MultiDiscrete vector -> dict the base env expects.
        act_vec: np.ndarray shape (22,)
          [0:20] in {0,1}, [20] in [0..10] (yaw idx), [21] in [0..10] (pitch idx)
        """
        act_vec = np.asarray(act_vec)
        btn_vals = act_vec[:20].astype(int)
        yaw_idx = int(act_vec[20])
        pitch_idx = int(act_vec[21])

        # map indices to float values
        yaw = float(self.cam_values[yaw_idx])
        pitch = float(self.cam_values[pitch_idx])

        a = {}
        for i, k in enumerate(self.BUTTON_KEYS):
            a[k] = int(btn_vals[i])  # or np.array([0/1]) if your env wants that
        a["camera"] = np.array([yaw, pitch], dtype=np.float32)  # shape (2,)

        return a

    def reverse_action(self, a_dict):
        """
        Optional: dict -> MultiDiscrete (useful for random plays / debugging)
        """
        out = np.zeros(22, dtype=np.int64)
        for i, k in enumerate(self.BUTTON_KEYS):
            out[i] = int(a_dict[k])
        # snap floats to nearest cam index
        yaw, pitch = np.asarray(a_dict["camera"], dtype=np.float32)
        yaw_idx = int(np.abs(self.cam_values - yaw).argmin())
        pitch_idx = int(np.abs(self.cam_values - pitch).argmin())
        out[20] = yaw_idx
        out[21] = pitch_idx
        return out

class PersistentSeedWrapper(gym.Wrapper):
    def __init__(self, env, default_seed: int):
        super().__init__(env)
        self._default_seed = default_seed
        self._original_seed = default_seed  # Store original seed, never change it

    def seed(self, seed=None):
        """
        Override seed() to ignore SB3's rank-based seeding.
        Always use the original seed to ensure all environments are identical.
        """
        # Ignore any seed changes from SB3's make_vec_env (which uses seed + rank)
        # Always use the original seed that was set during environment creation
        try:
            return self.env.seed(self._original_seed)
        except Exception:
            return None

    def reset(self, **kwargs):
        # Always use the original seed, ignoring any seed passed in kwargs
        # This ensures all environments remain identical even with n_envs > 1
        kwargs.pop("seed", None)  # Remove seed from kwargs if present
        try:
            self.env.seed(self._original_seed)
        except Exception:
            pass
        return self.env.reset(**kwargs)

class MineCLIPObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        project_root: str,
        clip4mc_subdir: str = "../Clip4MC",
        pretrained_model_path: str = "ViT-B-16.pt",
        device: str = "cpu",
        resize_hw=(160, 256),
        dtype=np.float32,
    ):
        super().__init__(env)
        self.dtype = dtype
        self.device = torch.device(device)

        # Resolve dirs/files
        clip4mc_dir = os.path.join(project_root, clip4mc_subdir)
        sys.path.insert(0, clip4mc_dir)
        from model.clip4mc import CLIP4MC  # after sys.path tweak

        # Resolve paths robustly
        if not os.path.isabs(pretrained_model_path):
            pretrained_model_path = os.path.join(clip4mc_dir, pretrained_model_path)
        config_yaml = os.path.join(clip4mc_dir, "config.yaml")

        # Debug prints (optional)
        print(f"Importing CLIP4MC from: {clip4mc_dir}")
        print(f"Using pretrained: {pretrained_model_path}")
        print(f"Expecting config.yaml at: {config_yaml}")

        # Preprocess transform
        self.transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor(),
        ])

        # Build model **inside** Clip4MC dir so relative 'config.yaml' works
        with pushd(clip4mc_dir):
            pretrained_clip = torch.jit.load(pretrained_model_path, map_location=self.device)
            self.clip4mc = CLIP4MC(
                frame_num=100,
                use_action=False,
                use_brief_text=False,
                pretrained_clip=pretrained_clip,
            ).to(self.device).eval()

        # Probe embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resize_hw[0], resize_hw[1], device=self.device)
            emb = self.clip4mc.get_image_embedding(dummy)
            feat_dim = int(emb.shape[-1])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32
        )

        assert isinstance(env.observation_space, gym.spaces.Dict) and "pov" in env.observation_space.spaces

    def _embed(self, frame_np: np.ndarray) -> np.ndarray:
        # frame_np: HWC uint8
        img = Image.fromarray(frame_np)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        with torch.no_grad():
            emb = self.clip4mc.get_image_embedding(img_tensor)  # (1, D)
        # Convert to numpy float32 (SB3 default)
        return emb.squeeze(0).detach().cpu().numpy().astype(self.dtype, copy=False)

    def observation(self, obs):
        # obs is a dict with 'pov'
        return self._embed(obs["pov"])

def make_env_mineclip(
    project_root: str = None,
    pretrained_model_path: str = "ViT-B-16.pt",
    device: str = "cuda",
    target_item: str = "log",
    target_count: int = 1,
    max_episode_steps: int = 2000,
    seed: int = None,
    skip: int = 8,
):
    """
    Factory that builds a MineRL env that emits MineCLIP embeddings instead of images.
    
    Args:
        skip: Number of frames to skip (default 8). Must be placed BEFORE MineCLIP wrapper
              to avoid computing embeddings for skipped frames.
    """
    if seed is None:
        seed = 888

    if project_root is None:
        project_root = os.path.dirname(__file__)

    ENV_KWARGS = dict(
        seed=seed,            # fixed Minecraft world seed
        # force_reset=True,
    )

    env = HumanSurvival(**ENV_KWARGS).make()
    env = InventoryDoneWrapper(env, target_item=target_item, target_count=target_count)

    # IMPORTANT: keep raw dict obs here (we'll embed 'pov' next)
    # env = PovOnlyWrapper(env)  # <- REMOVE this for MineCLIP; we need the dict to read 'pov'

    # Map dict actions to MultiDiscrete (unchanged)
    env = DictToMultiDiscreteActions(env, CAM_VALUES)

    # --- Frame skip: MUST be before MineCLIP to avoid computing embeddings for skipped frames ---
    if skip > 1:
        env = MaxAndSkipEnv(env, skip=skip)

    # Convert obs dict['pov'] -> MineCLIP vector
    env = MineCLIPObsWrapper(
        env,
        project_root=project_root,
        pretrained_model_path=pretrained_model_path,
        device=device,
        resize_hw=(160, 256),
    )

    # IMPORTANT: Adjust TimeLimit because max_episode_steps is in "skipped" steps
    # If you want 2000 'real' ticks, and skip is 8, max_steps should be 2000/8 = 250.
    env = TimeLimit(env, max_episode_steps=int(max_episode_steps / skip))
    env.seed(seed)
    # Enforce deterministic resets unless an explicit seed is provided at reset()
    env = PersistentSeedWrapper(env, default_seed=seed)

    return env
