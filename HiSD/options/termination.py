"""
Component C (part 3): combining duration and state evidence into option termination,
and soft initiation masks for Maskable PPO.

    beta(s, t) = 1 - (1 - beta_dur(t)) * (1 - beta_state(s))       (noisy-OR)
    terminate w.p. beta(s, t), or deterministically once t >= cap
    cap = duration percentile (e.g. 95th) of the (composite) duration model

An alternative combiner, 'bayes', treats the duration hazard as the prior and the
state classifier as evidence (product of experts):

    odds(end | s, t) = [beta_dur(t) / (1 - beta_dur(t))] * LR(s),
    LR(s) = [p_state(s) / (1 - p_state(s))] / [pi / (1 - pi)]

where pi is the end-state prior the classifier was trained under.  A confident state
model dominates, and an uninformative one (LR ~ 1, the Minecraft regime) falls back to
the duration hazard; noisy-OR instead adds the hazard even when the state model is sure.

For a composite option the duration model is the convolution of its children's; it
terminates when its *last* child fires (child hazard OR child state model) or when
the convolved cap is reached.  Modes allow the ablations: 'duration', 'state',
'noisy_or', 'bayes', 'horizon' (the paper's fixed-horizon fallback).
"""
import numpy as np


class OptionTerminator:
    def __init__(self, duration_model=None, state_model=None, mode='noisy_or', cap_quantile=0.95,
                 fixed_horizon=64, threshold=None, rng=None, state_prior=None):
        self.dur, self.state = duration_model, state_model
        self.state_prior = state_prior if state_prior is not None else getattr(state_model, 'prior', 0.5)
        self.mode = mode
        self.cap = duration_model.percentile(cap_quantile) if (duration_model is not None and cap_quantile) else None
        self.fixed_horizon = fixed_horizon
        self.threshold = threshold  # None = sample Bernoulli(beta); float = deterministic beta >= thr
        self.rng = rng or np.random.default_rng(0)

    def beta_state(self, feat):
        if self.state is None or feat is None:
            return 0.
        return float(self.state.predict_proba(np.atleast_2d(feat))[0, 1])

    def beta(self, feat, t):
        """Termination probability after the option has run t >= 1 steps."""
        if self.mode == 'horizon':
            return 1. if t >= self.fixed_horizon else 0.
        b_dur = self.dur.hazard(t) if (self.dur is not None and self.mode in ('duration', 'noisy_or', 'bayes')) else 0.
        b_state = self.beta_state(feat) if self.mode in ('state', 'noisy_or', 'bayes') else 0.
        if self.mode == 'bayes':
            return float(combine_bayes(b_dur, b_state, self.state_prior))
        return 1. - (1. - b_dur) * (1. - b_state)

    def should_terminate(self, feat, t):
        if self.mode != 'horizon' and self.cap is not None and t >= self.cap:
            return True
        b = self.beta(feat, t)
        if self.threshold is not None:
            return b >= self.threshold
        return bool(self.rng.random() < b)


def soft_initiation_mask(init_probs, floor=0.02, hard_zero_below=None):
    """Probabilities used as a *soft* action mask (log-prob logit bias) for Maskable PPO.

    Options keep a floor probability so marginally-valid options still get gradient;
    optionally options below `hard_zero_below` are hard-masked."""
    p = np.clip(np.asarray(init_probs, dtype=np.float32), floor, 1.)
    if hard_zero_below is not None:
        p[np.asarray(init_probs) < hard_zero_below] = 0.
    return p


def combine_bayes(b_dur, p_state, prior, eps=1e-6):
    """Product-of-experts termination probability (vectorised)."""
    b_dur = np.clip(b_dur, eps, 1 - eps)
    p_state = np.clip(p_state, eps, 1 - eps)
    prior = float(np.clip(prior, eps, 1 - eps))
    log_odds = (np.log(b_dur) - np.log1p(-b_dur) + np.log(p_state) - np.log1p(-p_state)
                - np.log(prior) + np.log1p(-prior))
    return 1. / (1. + np.exp(-log_odds))
