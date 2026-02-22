import os
import torch
import numpy as np

from Learn_Skill_Data.minecraft_gru_fs import RecurrentPolicyNet
from .helpers import OBS_DIM, DEVICE, NUM_BUTTONS


def load_gru_bc(skill_name: str, ckpt_dir: str, device: str = DEVICE):
    """
    Load a trained GRU BC policy and its normalization stats + thresholds.

    Expects checkpoints saved by minecraft_gru_fs.train_eval_gru_bc_once:
      - bc_gru_model_{skill_name}.pt    (state dict + mean/std + hyperparams)
      - thresholds_{skill_name}.pt      (per-button decision thresholds)
    """
    ckpt_path = os.path.join(ckpt_dir, f"bc_gru_model_{skill_name}.pt")
    thr_path = os.path.join(ckpt_dir, f"thresholds_{skill_name}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    state = torch.load(ckpt_path, map_location=device)

    # Extract hyperparams from checkpoint if available, else fall back to defaults.
    hp = state.get("hyperparams", {})
    hidden = int(hp.get("hidden", 512))
    layers = int(hp.get("rnn_layers", 1))

    model = RecurrentPolicyNet(obs_dim=OBS_DIM, hidden=hidden, rnn_layers=layers).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    mean = state["mean"].to(device)
    std = state["std"].to(device)

    if os.path.exists(thr_path):
        thresholds = torch.load(thr_path, map_location=device).to(device)
    else:
        print(f"[WARN] No thresholds found for {skill_name}, using 0.5 default")
        thresholds = torch.full((NUM_BUTTONS,), 0.5, device=device, dtype=torch.float32)

    return model, mean, std, thresholds


def _to_feat_tensor(obs_feat: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """
    Ensure observation feature vector is a float32 torch tensor of shape (1, 1, D)
    suitable for feeding into the GRU policy.
    """
    if isinstance(obs_feat, torch.Tensor):
        t = obs_feat
    else:
        t = torch.from_numpy(np.asarray(obs_feat))

    # GRU expects (Batch, Seq, Dim). For single-step inference we use (1, 1, D).
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:
        # Assume (Batch, Dim) -> (Batch, 1, Dim)
        t = t.unsqueeze(1)

    return t.to(device=device, dtype=torch.float32)


@torch.no_grad()
def bc_gru_action_multidiscrete(model, mean, std, thresholds, feat_1_1_D, h_in=None):
    """
    Single-step inference for the GRU BC controller.

    Args:
        model: RecurrentPolicyNet
        mean, std: normalization tensors (1, OBS_DIM)
        thresholds: per-button thresholds tensor (NUM_BUTTONS,)
        feat_1_1_D: tensor (1, 1, OBS_DIM) or broadcastable equivalent
        h_in: optional hidden state from previous step (num_layers, 1, hidden)

    Returns:
        action_vec: np.ndarray of shape (22,) MultiDiscrete action
        h_out: updated hidden state (num_layers, 1, hidden)
    """
    # Normalize features
    z = (feat_1_1_D - mean) / std

    # Forward pass through GRU
    btn_logits, yaw_logits, pitch_logits, h_out = model(z, h_in)

    # Squeeze sequence dimension: (1, 1, ...) -> (1, ...)
    btn_l = btn_logits[:, 0, :]
    yaw_l = yaw_logits[:, 0, :]
    pitch_l = pitch_logits[:, 0, :]

    # Decode button presses with calibrated thresholds
    btn = (torch.sigmoid(btn_l[0]) >= thresholds).to(torch.int64).cpu().numpy()
    yaw_idx = int(torch.argmax(yaw_l[0]).item())
    pitch_idx = int(torch.argmax(pitch_l[0]).item())

    vec = np.zeros(22, dtype=np.int64)
    vec[:20] = btn
    vec[20] = yaw_idx
    vec[21] = pitch_idx

    return vec, h_out

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
from sb3_contrib.common.wrappers import ActionMasker

from .helpers import (
    CAM_VALUES,
    OBS_DIM,
    NUM_BUTTONS,
    NUM_CAM,
    DEVICE,
    MAX_SKILL_STEPS,
    END_PATIENCE_STEPS,
    make_env_mineclip,
    _require_pu_model_and_threshold,
    _proba_geq,
)
from Learn_Skill_Data.minecraft_gru_fs import RecurrentPolicyNet


def _to_feat_tensor(obs_feat: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """Ensure observation feature vector is a float32 torch tensor of shape [1, D]."""
    if isinstance(obs_feat, torch.Tensor):
        t = obs_feat
    else:
        t = torch.from_numpy(np.asarray(obs_feat))
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


def load_gru_bc(skill_name: str, ckpt_dir: str, device: str = DEVICE):
    """
    Load a GRU-based BC model checkpoint trained by `minecraft_gru_fs.py`.

    Expects files:
      - bc_gru_model_{skill}.pt

    The checkpoint must contain:
      - 'model': state_dict for RecurrentPolicyNet
      - 'mean', 'std': feature normalization tensors
      - 'hyperparams' (optional): used to recover hidden size / layers
    """
    ckpt_path = os.path.join(ckpt_dir, f"bc_gru_model_{skill_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    state = torch.load(ckpt_path, map_location=device)
    hparams = state.get("hyperparams", {})
    hidden = int(hparams.get("hidden", 512))
    rnn_layers = int(hparams.get("rnn_layers", 1))
    dropout = float(hparams.get("dropout", 0.1))

    model = RecurrentPolicyNet(
        obs_dim=OBS_DIM,
        hidden=hidden,
        rnn_layers=rnn_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    mean = state["mean"].to(device)
    std = state["std"].to(device)

    # Default per-button threshold; can be replaced by calibrated values if desired.
    thresholds = torch.full((NUM_BUTTONS,), 0.5, device=device)
    return model, mean, std, thresholds


@torch.no_grad()
def bc_gru_action_multidiscrete(
    model: RecurrentPolicyNet,
    mean: torch.Tensor,
    std: torch.Tensor,
    thresholds: torch.Tensor,
    feat_1xD: torch.Tensor,
    h_in: Optional[torch.Tensor],
):
    """
    Compute MultiDiscrete action from MineCLIP feature tensor [1, D] using GRU BC.

    Maintains recurrence via h_in / h_out.
    """
    # Normalize and wrap as a length-1 sequence
    z = (feat_1xD - mean) / std  # (1, D)
    z_seq = z.unsqueeze(1)  # (1, 1, D)

    btn_l, yaw_l, pitch_l, h_out = model(z_seq, h_in=h_in)

    # Take the single time step
    btn_logits = btn_l[0, 0]
    yaw_logits = yaw_l[0, 0]
    pitch_logits = pitch_l[0, 0]

    btn = (torch.sigmoid(btn_logits) >= thresholds).to(torch.int64).cpu().numpy()
    yaw_idx = int(torch.argmax(yaw_logits).item())
    pitch_idx = int(torch.argmax(pitch_logits).item())

    vec = np.zeros(22, dtype=np.int64)
    vec[:20] = btn
    vec[20] = yaw_idx
    vec[21] = pitch_idx
    return vec, h_out


class SkillMuxWrapperGRU(gym.Wrapper):
    """
    Same interface as `SkillMuxWrapper` in `helpers.py`, but using GRU BC policies.

    MultiDiscrete action:
      - action[0] in {0..K}  (0 = primitive, 1..K = BC skill i)
      - action[1:] = primitive MultiDiscrete (same as base env)
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
        max_skill_steps: int = MAX_SKILL_STEPS,
        disable_pu_end: bool = False,
    ):
        super().__init__(env)
        self.skills = list(skills)
        self.num_skills = len(self.skills)
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.start_models_dir = start_models_dir
        self.end_models_dir = end_models_dir
        self.end_patience_steps = int(max(0, end_patience_steps))
        self.max_skill_steps = int(max(1, max_skill_steps))
        self.disable_pu_end = bool(disable_pu_end)

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.primitives_nvec = env.action_space.nvec
        nvec = np.concatenate(([self.num_skills + 1], self.primitives_nvec))
        self.action_space = spaces.MultiDiscrete(nvec)
        self.observation_space = env.observation_space

        # Load GRU BC policies
        # cache: skill -> (model, mean, std, thresholds)
        self._bc_cache: Dict[str, Tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # hidden state per skill: skill -> h (num_layers, 1, hidden)
        self._gru_h: Dict[str, Optional[torch.Tensor]] = {}
        for s in self.skills:
            self._bc_cache[s] = load_gru_bc(s, ckpt_dir, device=self.device)
            self._gru_h[s] = None

        # PU start/end models + thresholds
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
        # Batch GPU->CPU transfer: do it once for all skills
        feat_cpu = feat.detach().to("cpu").numpy().astype(np.float32)
        out = np.zeros(self.num_skills, dtype=bool)
        for i, s in enumerate(self.skills):
            model, thr = self._start_models[s]
            p = float(model.predict_proba(feat_cpu)[:, 1][0])
            out[i] = p >= thr
        return out

    def _end_should_fire(self, obs_feat_vec) -> bool:
        if self._active_skill_idx is None:
            self._end_fired_recently = 0
            return False
        
        if self.disable_pu_end:
            # When disabled, only check max_skill_steps
            return self._steps_in_skill >= self.max_skill_steps
        
        skill_name = self.skills[self._active_skill_idx - 1]
        model, thr = self._end_models[skill_name]
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)
        # Batch GPU->CPU transfer
        feat_cpu = feat.detach().to("cpu").numpy().astype(np.float32)
        p = float(model.predict_proba(feat_cpu)[:, 1][0])
        fired = p >= thr
        self._end_fired_recently = self._end_fired_recently + 1 if fired else 0
        return self._end_fired_recently >= self.end_patience_steps or (
            self._steps_in_skill >= self.max_skill_steps
        )

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

        primitive_mask = np.concatenate([np.ones(n, dtype=bool) for n in self.primitives_nvec])
        return np.concatenate([selector_mask, primitive_mask])

    def reset(self, **kwargs):
        self._active_skill_idx = None
        self._steps_in_skill = 0
        self._end_fired_recently = 0
        # Reset all GRU hidden states
        for s in self.skills:
            self._gru_h[s] = None

        seed = kwargs.pop("seed", None)
        if seed is not None:
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
        h_in = self._gru_h.get(skill_name, None)
        action_vec, h_out = bc_gru_action_multidiscrete(
            model, mean, std, thresholds, feat, h_in=h_in
        )
        # Cache updated hidden state
        self._gru_h[skill_name] = h_out.detach()
        return action_vec

    def step(self, action):
        action = np.asarray(action)
        selector = int(action[0])
        primitive_vec = action[1:].astype(np.int64)

        # Check end condition on current obs BEFORE consuming action
        if self._active_skill_idx is not None and self._end_should_fire(self._last_obs):
            skill_name = self.skills[self._active_skill_idx - 1]
            self._active_skill_idx = None
            self._steps_in_skill = 0
            self._end_fired_recently = 0
            self._startable_mask_cache = self._compute_startable_skills(self._last_obs)
            # Reset hidden state when a skill ends
            self._gru_h[skill_name] = None

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
                # reset hidden state at skill start
                skill_name = self.skills[self._active_skill_idx - 1]
                self._gru_h[skill_name] = None
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
            if self._active_skill_idx is not None:
                skill_name = self.skills[self._active_skill_idx - 1]
                self._gru_h[skill_name] = None
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
                "active_skill": self.skills[self._active_skill_idx - 1]
                if self._active_skill_idx
                else None,
                "skill_steps_in_run": self._steps_in_skill if self._active_skill_idx else 0,
            }
        )
        return obs, rew, done, info


def _mask_fn(env):
    return env.compute_action_mask()


def make_masked_env_gru(
    skills: List[str],
    ckpt_dir: str,
    start_models_dir: str = "Data/minecraft_cobblestone_mapped/pu_start_models_gt",
    end_models_dir: str = "Data/minecraft_cobblestone_mapped/pu_end_models_gt",
    project_root: str = None,
    pretrained_model_path: str = "ViT-B-16.pt",
    device: str = DEVICE,
    target_item: str = "log",
    target_count: int = 1,
    max_episode_steps: int = 2000,
    seed: int = None,
    skip: int = 8,
    max_skill_steps: int = MAX_SKILL_STEPS,
    disable_pu_end: bool = False,
):
    """
    Env factory identical to `make_masked_env`, but wired to GRU-based BC controllers.
    """
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

    mux = SkillMuxWrapperGRU(
        base,
        skills=skills,
        ckpt_dir=ckpt_dir,
        device=device,
        start_models_dir=start_models_dir,
        end_models_dir=end_models_dir,
        end_patience_steps=END_PATIENCE_STEPS,
        max_skill_steps=max_skill_steps,
        disable_pu_end=disable_pu_end,
    )
    masked = ActionMasker(mux, _mask_fn)
    return masked


