import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gym
from gym import spaces
from sb3_contrib.common.wrappers import ActionMasker

from .helpers import (
    PROJECT_ROOT,
    MAX_SKILL_STEPS,
    END_PATIENCE_STEPS,
    make_env_mineclip,
    _require_pu_model_and_threshold,
)
from .helpers_gru import (
    load_gru_bc,
    bc_gru_action_multidiscrete,
    _to_feat_tensor,
)


# ==============================
# Hierarchy utilities (reuse structure from hierarchy_ppo)
# ==============================


def _canonicalize_tree(node: Dict[str, Any]) -> str:
    if "symbol" in node:
        return f"S:{node['symbol']}"
    pid = node.get("production", None)
    children = node.get("children", [])
    sig_children = ",".join(_canonicalize_tree(c) for c in children)
    return f"P:{pid}[{sig_children}]"


def _canonicalize_tree_ignore_root(node: Dict[str, Any], is_root: bool = True) -> str:
    """
    Canonicalize tree structure, but ignore production ID for root node
    (since all roots are Production 0 for plotting semantics).
    """
    if "symbol" in node:
        return f"S:{node['symbol']}"
    pid = node.get("production", None)
    children = node.get("children", [])
    sig_children = ",".join(_canonicalize_tree_ignore_root(c, is_root=False) for c in children)
    # For root node, use "ROOT" instead of the actual production ID
    if is_root:
        return f"P:ROOT[{sig_children}]"
    return f"P:{pid}[{sig_children}]"


def _iter_production_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if "production" in node:
        out.append(node)
    for c in node.get("children", []):
        out.extend(_iter_production_nodes(c))
    return out


def _leaf_symbols_inorder(node: Dict[str, Any]) -> List[str]:
    if "symbol" in node:
        return [str(node["symbol"])]
    seq: List[str] = []
    for c in node.get("children", []):
        seq.extend(_leaf_symbols_inorder(c))
    return seq


def load_unique_hierarchies(hierarchies_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load unique hierarchies from a directory of JSON files.
    Ignores root production ID (always 0) when checking uniqueness.
    Assigns unique root names to each unique tree structure.
    
    Args:
        hierarchies_path: Path to a directory containing JSON hierarchy files.
    
    Returns:
        Tuple of:
        - List of unique hierarchy dictionaries (with root names assigned)
        - Dictionary mapping root names to their original file names (for debugging)
    """
    uniq: Dict[str, Dict[str, Any]] = {}
    root_name_map: Dict[str, str] = {}  # root_name -> original filename
    
    # Check if it's a single file (backward compatibility)
    if os.path.isfile(hierarchies_path):
        if not hierarchies_path.lower().endswith(".json"):
            print(f"[WARN] File is not a JSON file: {hierarchies_path}")
            return [], {}
        try:
            with open(hierarchies_path, "r") as f:
                tree = json.load(f)
            # Use canonicalization that ignores root production ID
            sig = _canonicalize_tree_ignore_root(tree, is_root=True)
            h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
            if h not in uniq:
                root_name = f"Root_{len(uniq)}"
                # Assign unique root name (store in tree for later use)
                tree["__root_name__"] = root_name
                uniq[h] = tree
                root_name_map[root_name] = os.path.basename(hierarchies_path)
        except Exception as e:
            print(f"[WARN] Error loading hierarchy file {hierarchies_path}: {e}")
            return [], {}
        return list(uniq.values()), root_name_map
    
    # Otherwise, treat as directory
    if not os.path.isdir(hierarchies_path):
        print(f"[WARN] hierarchies_path not found (neither file nor directory): {hierarchies_path}")
        return [], {}
    
    for fname in sorted(os.listdir(hierarchies_path)):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(hierarchies_path, fname)
        try:
            with open(fpath, "r") as f:
                tree = json.load(f)
            # Use canonicalization that ignores root production ID
            sig = _canonicalize_tree_ignore_root(tree, is_root=True)
            h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
            if h not in uniq:
                root_name = f"Root_{len(uniq)}"
                # Assign unique root name (store in tree for later use)
                tree["__root_name__"] = root_name
                uniq[h] = tree
                root_name_map[root_name] = fname
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    
    print(f"[INFO] Loaded {len(uniq)} unique hierarchies from {hierarchies_path}")
    if root_name_map:
        print(f"[INFO] Root names: {', '.join(sorted(root_name_map.keys()))}")
    
    return list(uniq.values()), root_name_map


def _get_node_leaf_sequence(node: Dict[str, Any], symbol_map: Dict[str, str]) -> List[str]:
    """
    Get the inorder sequence of leaf skill names for a given node.
    Recursively flattens production nodes to their leaf sequences.
    """
    if "symbol" in node:
        # Leaf node: map symbol to skill name
        try:
            return [symbol_map[str(node["symbol"])]]
        except KeyError:
            return []
    
    # Production node: recursively get leaf sequences from children
    seq: List[str] = []
    for child in node.get("children", []):
        seq.extend(_get_node_leaf_sequence(child, symbol_map))
    return seq


def compile_composites_from_hierarchies(
    hierarchies: List[Dict[str, Any]],
    symbol_map: Dict[str, str],
    all_hierarchy: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract composites from hierarchies.
    
    When all_hierarchy=False (default):
      - Only root nodes are included in the action space.
      - Each root gets a composite name (e.g., 'Root_0')
      - Sequence is the inorder list of mapped leaf skill names from the entire tree.
    
    When all_hierarchy=True:
      - Root nodes AND all unique internal production nodes are included.
      - Each production node gets a name like 'Prod_<production_id>'
      - Sequence is the inorder list of mapped leaf skill names from that node's subtree.
      - If multiple nodes have the same production ID, they share the same composite.
    
    If any leaf symbol is missing from symbol_map, that composite is skipped.
    Returns {composite_skill_name: [leaf_skill_names...]}
    """
    composites: Dict[str, List[str]] = {}
    
    # Process root nodes (always included)
    for tree in hierarchies:
        root_name = tree.get("__root_name__", None)
        if root_name is None:
            root_name = f"Root_{len(composites)}"
            tree["__root_name__"] = root_name
        
        leaves = _leaf_symbols_inorder(tree)
        if not leaves:
            print(f"[WARN] Tree with root {root_name} has no leaves, skipping")
            continue
        
        try:
            seq = [symbol_map[str(s)] for s in leaves]
        except KeyError as e:
            print(
                f"[WARN] Missing symbol in symbol_map: {e}; skipping {root_name}"
            )
            continue
        
        if root_name in composites:
            print(f"[WARN] Duplicate root name {root_name}, this should not happen")
            root_name = f"{root_name}_{len(composites)}"
        
        composites[root_name] = seq
    
    if composites:
        print(f"[INFO] Added root composites: {', '.join(sorted(composites.keys()))}")
    
    # Process internal nodes if all_hierarchy=True
    if all_hierarchy:
        # Track unique production nodes by their production ID
        # We'll use the first occurrence's structure to define the composite
        seen_productions: Dict[int, Dict[str, Any]] = {}
        
        for tree in hierarchies:
            # Get all production nodes (including root, but we already processed roots)
            all_prod_nodes = _iter_production_nodes(tree)
            for node in all_prod_nodes:
                prod_id = node.get("production")
                if prod_id is None:
                    continue
                
                # Skip root nodes (production 0) as they're already processed
                if prod_id == 0:
                    continue
                
                # If we've seen this production ID before, skip (use first occurrence)
                if prod_id in seen_productions:
                    continue
                
                seen_productions[prod_id] = node
        
        # Create composites for each unique internal production node
        for prod_id, node in sorted(seen_productions.items()):
            prod_name = f"Prod_{prod_id}"
            
            # Get leaf sequence for this node's subtree
            try:
                seq = _get_node_leaf_sequence(node, symbol_map)
            except KeyError as e:
                print(
                    f"[WARN] Missing symbol in symbol_map for Prod_{prod_id}: {e}; skipping"
                )
                continue
            
            if not seq:
                print(f"[WARN] Prod_{prod_id} has no leaf sequence, skipping")
                continue
            
            if prod_name in composites:
                print(f"[WARN] Duplicate production name {prod_name}, this should not happen")
                prod_name = f"{prod_name}_{len(composites)}"
            
            composites[prod_name] = seq
        
        if seen_productions:
            internal_names = sorted([f"Prod_{pid}" for pid in seen_productions.keys()])
            print(f"[INFO] Added internal production composites: {', '.join(internal_names)}")
    
    return composites


# ==============================
# Composite-aware Skill Mux (GRU BC)
# ==============================


class CompositeSkillMuxWrapperGRU(gym.Wrapper):
    """
    GRU-based version of the hierarchy CompositeSkillMuxWrapper.

    MultiDiscrete action:
      - selector(0..K) + primitive vector, as in the non-GRU version.
      - Uses GRU BC per leaf skill and supports composites as ordered sequences of leaves.
    """

    def __init__(
        self,
        env,
        skills: List[str],
        leaf_skills: List[str],
        composite_specs: Dict[str, List[str]],
        ckpt_dir: str,
        device: str,
        start_models_dir: str,
        end_models_dir: str,
        end_patience_steps: int = END_PATIENCE_STEPS,
        max_skill_steps: int = MAX_SKILL_STEPS,
        disable_pu_end: bool = False,
    ):
        super().__init__(env)
        self.skills = list(skills)  # composites + leaves
        self.leaf_skills = set(leaf_skills)
        self.composite_specs = dict(composite_specs)
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

        # BC policies (GRU) for leaves, composites are sentinel with their sequence
        self._bc_cache: Dict[str, Any] = {}
        # Per-leaf GRU hidden state
        self._gru_h: Dict[str, Optional[torch.Tensor]] = {}
        for s in self.skills:
            if s in self.leaf_skills:
                self._bc_cache[s] = load_gru_bc(s, ckpt_dir, device=self.device)
                self._gru_h[s] = None
            else:
                seq = self.composite_specs.get(s, [])
                if not seq:
                    raise ValueError(f"Composite '{s}' has empty sequence")
                self._bc_cache[s] = ("__COMPOSITE__", list(seq))

        # PU start/end for leaves; composites alias first/last leaf
        self._start_models: Dict[str, Tuple[object, float]] = {}
        self._end_models: Dict[str, Tuple[object, float]] = {}

        for s in self.leaf_skills:
            self._start_models[s] = _require_pu_model_and_threshold(
                self.start_models_dir, s
            )
            self._end_models[s] = _require_pu_model_and_threshold(
                self.end_models_dir, s
            )

        for comp, seq in self.composite_specs.items():
            first_leaf, last_leaf = seq[0], seq[-1]
            if first_leaf not in self._start_models:
                raise FileNotFoundError(
                    f"Missing start PU for leaf '{first_leaf}' (needed by {comp})"
                )
            if last_leaf not in self._end_models:
                raise FileNotFoundError(
                    f"Missing end PU for leaf '{last_leaf}' (needed by {comp})"
                )
            self._start_models[comp] = self._start_models[first_leaf]
            self._end_models[comp] = self._end_models[last_leaf]

        # Runtime caps
        self._cap_steps: Dict[str, int] = {}
        for s in self.skills:
            if s in self.leaf_skills:
                self._cap_steps[s] = int(self.max_skill_steps)
            else:
                seq = self.composite_specs[s]
                self._cap_steps[s] = int(len(seq) * self.max_skill_steps)

        # State
        self._active_skill_idx: Optional[int] = None  # 1..K
        self._steps_in_skill: int = 0
        self._phase_idx: int = 0
        self._steps_in_phase: int = 0
        self._end_fired_recently: int = 0
        self._last_obs = None
        self._startable_mask_cache: Optional[np.ndarray] = None

    # ---------- Mask helpers ----------

    def _proba_geq(self, model, thr: float, feat_1xD_torch) -> bool:
        x = feat_1xD_torch.detach().to("cpu").numpy().astype(np.float32)
        p = float(model.predict_proba(x)[:, 1][0])
        return p >= thr

    def _compute_startable_skills(self, obs_feat_vec) -> np.ndarray:
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)
        # Batch GPU->CPU transfer: do it once for all skills
        feat_cpu = feat.detach().to("cpu").numpy().astype(np.float32)
        out = np.zeros(self.num_skills, dtype=bool)
        for i, s in enumerate(self.skills):
            model, thr = self._start_models[s]
            p = float(model.predict_proba(feat_cpu)[:, 1][0])
            out[i] = p >= thr
        return out

    def _child_end_should_fire(self, child_skill: str, obs_feat_vec) -> bool:
        if self.disable_pu_end:
            # When disabled, only check max_skill_steps
            return self._steps_in_phase >= self.max_skill_steps
        
        model, thr = self._end_models[child_skill]
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)
        # Batch GPU->CPU transfer
        feat_cpu = feat.detach().to("cpu").numpy().astype(np.float32)
        p = float(model.predict_proba(feat_cpu)[:, 1][0])
        fired = p >= thr
        self._end_fired_recently = self._end_fired_recently + 1 if fired else 0
        return (self._end_fired_recently >= self.end_patience_steps) or (
            self._steps_in_phase >= self.max_skill_steps
        )

    def _leaf_end_should_fire(self, leaf_skill: str, obs_feat_vec) -> bool:
        if self.disable_pu_end:
            # When disabled, only check max_skill_steps
            return self._steps_in_skill >= self.max_skill_steps
        
        model, thr = self._end_models[leaf_skill]
        feat = _to_feat_tensor(obs_feat_vec, device=self.device)
        # Batch GPU->CPU transfer
        feat_cpu = feat.detach().to("cpu").numpy().astype(np.float32)
        p = float(model.predict_proba(feat_cpu)[:, 1][0])
        fired = p >= thr
        self._end_fired_recently = self._end_fired_recently + 1 if fired else 0
        return (self._end_fired_recently >= self.end_patience_steps) or (
            self._steps_in_skill >= self.max_skill_steps
        )

    # Called by ActionMasker
    def compute_action_mask(self) -> np.ndarray:
        sel_size = self.num_skills + 1

        if self._startable_mask_cache is None and self._last_obs is not None:
            self._startable_mask_cache = self._compute_startable_skills(self._last_obs)

        selector_mask = np.zeros(sel_size, dtype=bool)

        if self._active_skill_idx is None:
            selector_mask[0] = True
            if self._startable_mask_cache is not None:
                selector_mask[1:] = self._startable_mask_cache
            else:
                selector_mask[1:] = True
        else:
            selector_mask[self._active_skill_idx] = True

        primitive_mask = np.concatenate(
            [np.ones(n, dtype=bool) for n in self.primitives_nvec]
        )
        return np.concatenate([selector_mask, primitive_mask])

    # ---------- Gym API ----------

    def reset(self, **kwargs):
        self._active_skill_idx = None
        self._steps_in_skill = 0
        self._phase_idx = 0
        self._steps_in_phase = 0
        self._end_fired_recently = 0
        # reset all GRU hidden states
        for s in self.leaf_skills:
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

    def _active_skill_name(self) -> Optional[str]:
        if self._active_skill_idx is None:
            return None
        return self.skills[self._active_skill_idx - 1]

    def _current_child_skill(self, comp_name: str) -> str:
        seq = self.composite_specs[comp_name]
        return seq[self._phase_idx]

    def _bc_step_action_leaf(self, leaf_name: str, obs_vec):
        model, mean, std, thresholds = self._bc_cache[leaf_name]
        feat = _to_feat_tensor(obs_vec, device=self.device)
        h_in = self._gru_h.get(leaf_name, None)
        action_vec, h_out = bc_gru_action_multidiscrete(
            model, mean, std, thresholds, feat, h_in=h_in
        )
        self._gru_h[leaf_name] = h_out.detach()
        return action_vec

    def _bc_step_action(self, skill_name: str, obs_vec):
        if skill_name in self.leaf_skills:
            return self._bc_step_action_leaf(skill_name, obs_vec)
        # composite: run BC of current child skill
        _, seq = self._bc_cache[skill_name]
        child = seq[self._phase_idx]
        return self._bc_step_action_leaf(child, obs_vec)

    def step(self, action):
        action = np.asarray(action)
        selector = int(action[0])
        primitive_vec = action[1:].astype(np.int64)

        # End checks BEFORE consuming the action
        if self._active_skill_idx is not None:
            active = self._active_skill_name()
            if active in self.leaf_skills:
                if self._leaf_end_should_fire(active, self._last_obs):
                    self._active_skill_idx = None
                    self._steps_in_skill = 0
                    self._end_fired_recently = 0
                    # reset leaf GRU hidden
                    self._gru_h[active] = None
                    self._startable_mask_cache = self._compute_startable_skills(
                        self._last_obs
                    )
            else:
                comp = active
                child = self._current_child_skill(comp)
                comp_cap = self._cap_steps[comp]

                if self._child_end_should_fire(child, self._last_obs):
                    # reset child GRU hidden when moving to the next phase
                    self._gru_h[child] = None
                    self._phase_idx += 1
                    self._steps_in_phase = 0
                    self._end_fired_recently = 0

                if self._phase_idx >= len(self.composite_specs[comp]) or (
                    self._steps_in_skill >= comp_cap
                ):
                    self._active_skill_idx = None
                    self._steps_in_skill = 0
                    self._phase_idx = 0
                    self._steps_in_phase = 0
                    self._end_fired_recently = 0
                    self._startable_mask_cache = self._compute_startable_skills(
                        self._last_obs
                    )

        # Decide env action
        if self._active_skill_idx is None:
            if selector == 0:
                env_action = primitive_vec
            else:
                self._active_skill_idx = selector
                self._steps_in_skill = 0
                self._phase_idx = 0
                self._steps_in_phase = 0
                self._end_fired_recently = 0
                name = self._active_skill_name()
                # reset GRU hidden(s) at start
                if name in self.leaf_skills:
                    self._gru_h[name] = None
                else:
                    for leaf in self.composite_specs[name]:
                        self._gru_h[leaf] = None
                env_action = self._bc_step_action(name, self._last_obs)
        else:
            name = self._active_skill_name()
            env_action = self._bc_step_action(name, self._last_obs)

        # Step env
        obs, rew, done, info = self.env.step(env_action)

        if self._active_skill_idx is not None:
            self._steps_in_skill += 1
            active = self._active_skill_name()
            if active not in self.leaf_skills:
                self._steps_in_phase += 1

        if done:
            # clear all GRU hiddens on episode end
            for s in self.leaf_skills:
                self._gru_h[s] = None
            self._active_skill_idx = None
            self._steps_in_skill = 0
            self._phase_idx = 0
            self._steps_in_phase = 0
            self._end_fired_recently = 0

        self._last_obs = obs
        self._startable_mask_cache = (
            self._compute_startable_skills(self._last_obs)
            if self._active_skill_idx is None
            else None
        )

        info = dict(info)
        active_name = self._active_skill_name()
        active_child = None
        if active_name is not None and active_name not in self.leaf_skills:
            seq = self.composite_specs[active_name]
            if 0 <= self._phase_idx < len(seq):
                active_child = seq[self._phase_idx]

        info.update(
            {
                "active_skill": active_name,
                "skill_steps_in_run": self._steps_in_skill if active_name else 0,
                "active_composite_phase_idx": self._phase_idx
                if active_child is not None
                else -1,
                "active_composite_child": active_child,
            }
        )
        return obs, rew, done, info


def _mask_fn(env):
    return env.compute_action_mask()


def make_masked_hierarchy_env_gru(
    leaf_skills: List[str],
    composite_specs: Dict[str, List[str]],
    ckpt_dir: str,
    start_models_dir: str,
    end_models_dir: str,
    project_root: str = PROJECT_ROOT,
    pretrained_model_path: str = "ViT-B-16.pt",
    device: str = "cuda",
    target_item: str = "log",
    target_count: int = 1,
    max_episode_steps: int = 2000,
    seed: int = None,
    skip: int = 8,
    max_skill_steps: int = MAX_SKILL_STEPS,
    disable_pu_end: bool = False,
):
    """
    Builds a MineCLIP env and wraps it with a CompositeSkillMuxWrapperGRU + ActionMasker.
    Exposed skills = composites ∪ leaves, with GRU BC controllers per leaf.
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

    comp_names = sorted(composite_specs.keys())
    leaf_names = sorted(leaf_skills)
    skills = comp_names + leaf_names

    mux = CompositeSkillMuxWrapperGRU(
        base,
        skills=skills,
        leaf_skills=leaf_names,
        composite_specs=composite_specs,
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


