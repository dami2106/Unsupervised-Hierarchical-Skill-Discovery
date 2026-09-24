# Bayesian-nonparametric HiSD (BNP-HiSD): components A, B, C

This branch adds three independent, ablation-ready components to HiSD. Each can be
switched on or off, so all eight cells of the 2³ design (Old HiSD, A, A+B, A+C,
A+B+C, B+C, B-only, C-only) can be run from the same code.

| Comp. | What changes | Code | Switch |
|---|---|---|---|
| **A** | K-free segmentation: ASOT with a learned DP (truncated stick-breaking) target marginal plus redundancy pruning. `--n-clusters` becomes K_max. | `src/bnp_marginal.py`, `src/asot.py` (`q=`), `src/train.py` | `--marginal dp` |
| **B** | Noise-robust hierarchy: a Pitman–Yor mixture over episode "sentence types" with an edit-distance noisy channel. Sequitur then runs on the denoised corpus. Also produces posterior hierarchy samples. | `grammar/np_grammar.py`, `grammar/run_np_grammar.py` | run `grammar/run_np_grammar.py` instead of `sequitur/sequitur.py` |
| **C** | Duration-aware options: Bayesian duration hazards, calibrated end-state models, noisy-OR or product-of-experts termination, composite convolution, and soft initiation masks. | `options/*.py`, `Craftax/option_helpers.py`, `Craftax/top_down_env_gymnasium_hierarchy.py`, `Craftax/ppo_hierarchy.py` | `--termination_mode`, `--durations_path`, `--soft_masks` |

C is defined only on segment outputs (per-frame labels) and states, so it runs on
any segmenter (B+C, C-only).

---

## A — K-free ASOT (`--marginal dp`)

ASOT solves `min_Γ ⟨C,Γ⟩ + α R_temp(Γ) + λ KL(Γᵀ1 ‖ q)` with q uniform over K. Here
q is the posterior-mean stick-breaking weight vector of a truncated DP(γ):

    v_k | m ~ Beta(1 + m_k, γ + Σ_{l>k} m_l),   q_k = E[v_k] Π_{l<k} (1 − E[v_l])

- Sticks are ordered by usage at every q-step. This is equivalent to re-sorting the
  prototypes, but leaves the optimiser state alone.
- Usage m is counted in *segments*: each episode's soft usage is rescaled to its
  number of contiguous segments. Frames inside a segment are not independent draws.
- An optional Gamma(a₀, b₀) hyperprior on γ is available (`--dp-gamma-prior 1 1`),
  updated with the Escobar–West conditional mean.

### Deviations from the plan (and why)

The closed-form q-step alone does not infer K. Two failure modes showed up
experimentally:

1. **Fixed point.** With a balanced (hard) action marginal, the realised usage *is*
   q, so the posterior never moves.
2. **Collapse.** Feeding the learned q back into self-labelled representation
   learning collapses everything onto one prototype (K̂ = 1 on every dataset). This
   is the degenerate solution that equipartition in SeLa/ASOT exists to prevent.

The working version therefore:

- **Measures usage** for the q-step on the *unbalanced relaxation* of the same OT
  problem (`--dp-usage-lambda`, default 0.01).
- **Trains** with equipartition over the *active* prototypes
  (`--dp-train-marginal uniform`, the default). The learned q is used at evaluation.
- **Prunes** prototypes in two ways:
  - by stick mass (`--dp-prune-eps`, default 1/(2 K_max));
  - by a **small-variance / DP-means redundancy test** (Kulis & Jordan 2012): the
    most redundant prototype is dropped when its frames cost less than δ more on
    their runner-up prototype (`--dp-merge-delta`, default 0.1 cosine distance).
- **Only allows pruning after a warmup** (`--dp-warmup-frac`, default 0.5 of
  training). Redundancy cannot be identified until the embedding has settled.

  Diagnostic, Stone-Pickaxe, K_max = 10, trained without pruning: the surviving
  prototypes had runner-up gains ≈ 0.66–0.71. The duplicates had ≈ 0.01, with
  prototype cosine similarity 0.99. Pruning at step 20, as in the original
  schedule, removed everything.

New test metrics: `test_k_hat` (active prototypes), `test_k_used` (distinct
predicted labels), `test_k_err` (|K_used − K_gt|) and `test_nmi`. The final q is
written to `<log_dir>/dp_marginal.json`.

A3 (HDP-HSMM / BP-AR-HMM) was not needed and is not implemented.

---

## B — Pitman–Yor noisy-channel grammar (`grammar/`)

- **Sentence-level adaptor.** Episodes sit at tables of a PY(d, θ) process. Each
  table carries a prototype skill string (a latent clean derivation). This is an
  adaptor on the Sentence non-terminal: reuse follows the PY power law, and
  near-duplicate derivations collapse.
- **Noisy channel.** An observed string is an edit-channel emission (substitution,
  insertion, deletion; pair-HMM forward algorithm) of its table's prototype. The
  channel rates are re-estimated from Viterbi alignments under Beta priors. On clean
  data the channel learns ≈0 noise and nothing is merged. For example, on
  wsws_random ground truth it keeps all 11 true types, with learned noise rates
  < 0.005.
- **Base distribution G0.** Geometric length × a smoothed bigram chain over skills.
- **Inference.**
  - Collapsed Gibbs sampling with prototype resampling (`--method gibbs`), giving a
    MAP state plus posterior samples.
  - Greedy **Bayesian model merging** of types under the same posterior
    (`--method bmm`), the plan's fallback.
- **Grammar.** HiSD's modified Sequitur (φ boundaries are never inside a rule) runs
  on the denoised MAP corpus. Output uses HiSD's exact format (`tree_i.json`,
  `grammar.txt`, `structure_metrics.json`), so the option code is unchanged.
- **Metrics** (`np_metrics.json`, each paired with an exact-match Sequitur baseline):
  - MAP unique trees and unique types
  - posterior distribution of type counts
  - held-out per-symbol perplexity (fit on 80% of episodes)
  - MDL (grammar bits over the type inventory, plus seating and channel bits)
- **Posterior hierarchies** are written to `posterior_samples/sample_s/`.
  `ppo_hierarchy.py --hierarchy_posterior_sample` gives each PPO seed its own
  posterior draw. This is run-level sampling, which keeps the option set fixed
  within a run.

Not implemented: full adaptor-grammar MCMC over every non-terminal, HDP-PCFG, and
fragment grammars. The sentence-level PY adaptor plus the channel is where the
denoising happens. With `--no-channel`, B reduces exactly to Sequitur.

---

## C — Duration-aware options (`options/`)

- **`duration.py`.** Per-skill Poisson–Gamma model (d − 1 ~ Poisson(λ),
  λ ~ Gamma(a₀, b₀)). Its posterior predictive is negative binomial. Provides:
  - hazard β_dur(t) = p(t) / P(d ≥ t)
  - percentile caps
  - composite durations as the convolution of the children's pmfs
- **`nnpu.py`.** Non-negative PU learning (Kiryo et al. 2017) with the logistic
  loss. The prior comes from the segmentation (#segments / #frames). Temperature is
  calibrated by maximising the SCAR PU likelihood.
- **`termination.py`.** Termination combiners:
  - `noisy_or`: β = 1 − (1 − β_dur)(1 − β_state)
  - `bayes` (product of experts): odds = hazard odds × state likelihood ratio
  - `duration`, `state`, `horizon`
  - all with a 95th-percentile safety cap
  - `soft_initiation_mask`
- **`soft_mask_ppo.py`.** `SoftMask{Mlp,Cnn,MultiInput}Policy` for sb3-contrib
  MaskablePPO. A float "mask" p_a adds a log p_a logit bias; p = 0 is still a hard
  mask, and boolean masks behave as before. MaskablePPO's rollout buffer already
  stores masks as float32.
- **`fit_durations.py`.** Writes duration JSON for the option env. **`eval_termination.py`**
  evaluates offline (see below).
- **Craftax wiring.** Defaults reproduce the original behaviour.
  - `ppo_hierarchy.py --termination_mode {pu_horizon,duration,noisy_or,bayes} --durations_path <json> [--soft_masks]`
  - The env passes the option's elapsed steps to `should_terminate`. Composites
    stop at their convolved cap or when the last leaf fires.
  - With `--soft_masks`, `action_masks()` returns calibrated initiation
    probabilities.

Deviation: noisy-OR, as specified in the plan, is the *worst* combiner in every
regime we measured. Even when the state model is certain, it lets the duration
hazard fire early. The product-of-experts `bayes` combiner is the recommended
default. It follows a confident state model, and falls back to the duration hazard
when the state model is uninformative. The density-ratio (uLSIF/KLIEP) state model
was not implemented.

---

## Running it

```bash
# data (Craftax): the paper's stone-pickaxe plan, or wood/stone collection
cd Craftax
python generate_wood_stone.py --task stone_pickaxe --samples 150 --base-seed 2000 --path Traces/stone_pick/
python build_pca_features.py --data_dir Traces/stone_pick --components 256

cd ../HiSD
# A: fair protocol — tune the fixed-K baseline at K = GT, freeze, then sweep
python Helpers/bnp_sweep.py tune  --dataset ../Craftax/Traces/stone_pick --k-gt 5 --trials 18 --out runs/bnp_eval/stone_pick
python Helpers/bnp_sweep.py sweep --dataset ../Craftax/Traces/stone_pick --params runs/bnp_eval/stone_pick/best_params.json \
       --ks 3 4 5 6 7 --kmaxes 6 10 20 --seeds 0 1 2 3 4 --out runs/bnp_eval/stone_pick/sweep
# single K-free run (saves predicted_skills with --log --visualize)
python src/train.py --dataset ../Craftax/Traces/stone_pick --feature-name pca_features --n-clusters 10 --marginal dp ... --log --visualize

# B on any segmentation (or ground truth)
python grammar/run_np_grammar.py --input-dir <run>/version_0 --skill-folder predicted_skills --out-dir <run>/version_0/hierarchy_data/np
# C offline
python options/eval_termination.py --dataset ../Craftax/Traces/stone_pick --skills-dir <run>/predicted_skills --gt-dir ../Craftax/Traces/stone_pick/groundTruth
python options/fit_durations.py --skills-dir <run>/predicted_skills --out ../Craftax/Traces/stone_pick/durations_pred.json
# all offline cells of the 2^3 matrix from one fixed-K run and one DP run
python Helpers/ablation_matrix.py --dataset ../Craftax/Traces/stone_pick --fixed-run <fixedK run> --dp-run <dp run> --out runs/bnp_eval/stone_pick/ablation
```

Self-tests: `python src/bnp_marginal.py`, `python grammar/np_grammar.py`,
`python options/duration.py`, `python options/soft_mask_ppo.py`.

---

## Verification (Craftax, CPU-only container)

### Setup

**Task.** Craftax Stone Pickaxe, generated with `generate_wood_stone.py --task stone_pickaxe`:

- plan: wood, wood, table, wood, wooden_pickaxe, stone, wood, stone_pickaxe
- K_gt = 5; 150 episodes of 19–78 frames (mean 32)
- random worlds
- features: 256-d PCA of the top-down frame

**Shared hyperparameters** (`results/bnp_stone_pick/shared_params.json`) are the same
for every configuration. An 18-trial Optuna search of the fixed-K baseline at K = 5
(`tuned_params.json`) reached mIoU 0.437. That is a negligible gain over the shared
config (0.432 ± 0.019), and the tuned config breaks A (see limitations).

Three seeds per cell. All CSV/JSON results are in `results/bnp_stone_pick/`.

**What was not run.** Downstream PPO was not run. It needs ResNet-34 BC policies and
PU models per skill, which is impractical on this CPU-only container. Minecraft was
not run either. C is therefore evaluated offline by replaying held-out segments as
option executions (`options/eval_termination.py`). The RL wiring
(`ppo_hierarchy.py` flags, soft-mask policy) is implemented, and the soft-mask policy
was smoke-tested with MaskablePPO on a toy env.

### A — K-sensitivity (full/Hungarian mIoU, mean over 3 seeds)

| K (fixed) / K_max (DP) | fixed-K ASOT | DP-ASOT (current) | K̂ (current) | DP-ASOT v1 (no prune cooldown) | K̂ v1 |
|---|---|---|---|---|---|
| 3 | 0.268 | – | – | – | – |
| 4 | 0.389 | – | – | – | – |
| **5 = GT** | **0.432 ± 0.019** | – | – | – | – |
| 6 | 0.517 ± 0.032 | 0.517 ± 0.034 | 6 | 0.517 | 6 |
| 7 | 0.417 ± 0.056 | – | – | – | – |
| 10 (2·GT) | – | 0.460 ± 0.080 | 6 ± 0 | 0.454 ± 0.007 | **5 ± 0** |
| 20 (4·GT) | – | **0.485 ± 0.011** | 7.7 ± 0.6 | 0.330 ± 0.111 | 3.7 ± 1.2 |

- **K-sensitivity.** Fixed-K mIoU spans 0.25 across K = 3…7, with a non-monotone
  jump between K = 6 and 7. The current DP version spans 0.057 across
  K_max = 6…20.
- **At K_max = 4·GT**, DP (0.485 ± 0.011) beats fixed-K at the true K
  (0.432 ± 0.019). This passes criterion (i) of the plan's gate.
- **K̂ criterion.** "K̂ within ±1 of GT" holds at 2·GT but not at 4·GT: the current
  version over-estimates by 2–3. The earlier variant (v1: stale redundancy
  statistics, no cooldown) recovered K̂ = 5 exactly at K_max = 10 in every seed, but
  collapsed at K_max = 20. Full-mIoU with Hungarian matching rewards mild
  over-segmentation (fixed K = 6 > K = 5); segmental F1 does not (0.743 at K = 5 vs
  0.576 for DP at K_max = 20).
- **Tuned config.** With the Optuna config tuned for the fixed-K baseline (lr 1e-5,
  15 epochs, rho 0.29), DP collapses to K̂ = 1 (mIoU 0.10). In that regime the
  embedding barely trains, all prototypes are cosine-0.99 apart, and ASOT segments
  mostly through its temporal-position prior. The features carry no evidence about
  K (`ksweep_tuned_config.csv`).

**Verdict on A.** A is a partial pass. It removes the K cliff in mIoU, but K̂ is
biased upwards at large K_max, and it depends on a representation that actually
learns cluster structure.

### A / B / C ablation matrix

DP runs use K_max = 10; fixed-K runs use K = 5. Values are mean ± sd over 3 seeds.
Termination is scored against **ground-truth** segment ends on 30% held-out
episodes.

| config | mIoU | NMI | unique trees (GT 1) | unique types | MDL bits | held-out ppl | term. exact | term. MAE (steps) |
|---|---|---|---|---|---|---|---|---|
| Old HiSD | 0.432 ± .019 | 0.490 | 18.7 ± 9.1 | 12.7 | 481 | 1.240 | 0.236 | 4.01 ± 2.58 |
| A | 0.460 ± .080 | 0.517 | 16.7 ± 4.2 | 13.0 | 484 | 1.212 | 0.178 | 2.99 |
| A+B | 0.460 | 0.517 | **9.3 ± 4.0** | 7.0 | 422 | 1.203 | 0.178 | 2.99 |
| A+C | 0.460 | 0.517 | 16.7 | 13.0 | 484 | 1.212 | 0.194 | 2.56 |
| A+B+C | 0.460 | 0.517 | 9.3 | 7.0 | 422 | 1.203 | 0.194 | 2.56 |
| B+C | 0.432 | 0.490 | 12.0 ± 7.6 | 7.0 | 409 | 1.243 | **0.319** | **1.91 ± 0.09** |
| B-only | 0.432 | 0.490 | 12.0 | 7.0 | 409 | 1.243 | 0.236 | 4.01 |
| C-only | 0.432 | 0.490 | 18.7 | 12.7 | 481 | 1.240 | 0.319 | 1.91 |

With the v1 pruning (exact K̂ = 5, `ablation_matrix_dp_v1_seed*.csv`), A alone gave
9.3 unique trees and A+B gave 5.0 ± 1.0. So the value of A for the hierarchy hinges
on getting K̂ exactly right.

- **B** (the grammar):
  - Fewer unique trees on every segmentation: −36% on fixed-K, −44% on DP. Also
    fewer types, lower MDL, and held-out perplexity within noise.
  - Sanity check: on clean ground-truth strings B learns ≈0 channel noise and
    merges nothing.
- **C** (termination):
  - Duration-aware product-of-experts termination (`bayes`) halves the error of the
    paper's PU + horizon rule on discovered skills: MAE 4.0 → 1.9, exact-stop rate
    0.24 → 0.32.
  - C helps less on A's segmentation, where one skill is split in two.
- **Interactions.**
  - The best termination comes from the fixed-K segmentation, because K = 5 matches
    the true skills.
  - "Is A necessary?" (B+C vs A+B+C): A+B+C has fewer trees (9.3 vs 12.0) but worse
    termination (2.56 vs 1.91).

### C — termination regimes (ground-truth skills, `termination_gt_state_noise*.json`)

| rule | clean features: exact / MAE | Minecraft-like (state noise σ = 2): exact / MAE |
|---|---|---|
| fixed horizon | 0.143 / 14.2 | 0.143 / 14.2 |
| PU (Elkan–Noto) + horizon (paper) | **0.959 / 0.17** | 0.267 / 5.53 |
| duration hazard only | 0.318 / 2.43 | 0.337 / 2.27 |
| noisy-OR (plan) | 0.902 / 0.35 | 0.114 / 2.93 |
| product of experts (Elkan–Noto state) | 0.914 / 0.30 | 0.302 / 2.17 |
| product of experts (nnPU state) | 0.629 / 1.06 | 0.321 / **1.84** |

End-state frame F1:

| state model | clean | noisy |
|---|---|---|
| Elkan–Noto | 0.94 | 0.47 |
| nnPU | 0.63 | 0.44 |

For comparison, the paper reports Minecraft PU-End F1 of 0.088–0.108.

- **Clean regime** (Craftax-like, where end states are visually obvious): the
  paper's rule is already near-perfect, and C should not replace it.
- **Degraded regime:** PU + horizon falls back towards the horizon, while the
  duration-aware rules degrade gracefully (MAE 1.8–2.3 vs 5.5). This is the
  Minecraft failure mode C targets.
- **Noisy-OR** is dominated by the product-of-experts rule in both regimes.
- **Same pattern on wsws_static:** with σ = 2, Elkan–Noto F1 is 0.08. There,
  PU + horizon has MAE 5.18 vs duration-only 1.09.

### Other findings

- **Wood-Stone (wsws, K = 2) is unsuitable for ASOT with these features.** Both
  top-down and local-view PCA features are linearly informative (probe accuracy
  0.80), but fixed-K ASOT reaches NMI ≈ 0. With balanced marginals it cuts every
  episode into an early half and a late half, following the monotone drift of the
  map. The generator still supports the task (`--task wood_stone`).
- **Pre-existing pipeline bugs fixed on this branch:**
  - `sequitur/helpers.py` read `predicted_skills/*_segments.txt` as episodes.
  - `utils.plot_segmentation_gt` crashed when more clusters are predicted than GT
    classes; this hit `--visualize` for any K > K_gt.
  - `metrics.ClusteringMetrics` broke with current torch/lightning.
  - The validation-figure spacing divided by zero on small datasets.
  - `utils.py` had an unused `wandb` import.

### Limitations / not done

- Downstream PPO (Craftax Wooden Pickaxe, Minecraft Collect Log), rliable IQM, and
  the action-label N-sweep were not run.
- Minecraft segmentation / hierarchy were not run.
- Only 3 seeds (the plan asks for ≥5). All results are on one generated Craftax
  dataset; they are not the paper's data.
- Not implemented: A3 (HDP-HSMM); full adaptor-grammar MCMC over all non-terminals;
  HDP-PCFG; fragment grammars; the density-ratio state model; per-invocation
  posterior sampling of composite expansions (sampling is per PPO run).

