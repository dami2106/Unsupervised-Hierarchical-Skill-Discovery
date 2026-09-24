"""
Component C (part 2): calibrated state-based termination via non-negative PU learning
(Kiryo et al., NeurIPS 2017).

Positives: last frame of each segment of the skill.  Unlabeled: every other frame of
that skill's segments (the end state may recur, so they are not clean negatives).
The class prior pi = P(end | frame of this skill) is known from the segmentation
itself: #segments / #frames of the skill.

We train with the logistic loss (a proper scoring rule), so sigmoid(g(x)) estimates
P(end | x); a temperature is then fitted on held-out episodes by matching the PU
likelihood, and reliability can be checked against ground truth where available.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNPUClassifier:
    def __init__(self, hidden=64, lr=1e-3, weight_decay=1e-4, epochs=200, beta=0., gamma=1.,
                 batch_size=512, seed=0):
        self.hidden, self.lr, self.wd, self.epochs = hidden, lr, weight_decay, epochs
        self.beta, self.gamma, self.batch_size, self.seed = beta, gamma, batch_size, seed
        self.temperature = 1.

    def _net(self, d):
        if self.hidden:
            return nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
        return nn.Linear(d, 1)

    def fit(self, X_pos, X_unl, prior):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu = np.concatenate([X_pos, X_unl]).mean(0)
        self.sd = np.concatenate([X_pos, X_unl]).std(0) + 1e-6
        P = torch.tensor((X_pos - self.mu) / self.sd, dtype=torch.float32)
        U = torch.tensor((X_unl - self.mu) / self.sd, dtype=torch.float32)
        self.prior = float(np.clip(prior, 1e-4, 1 - 1e-4))
        self.model = self._net(P.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_pos = lambda g: F.softplus(-g)  # logistic loss for label +1
        loss_neg = lambda g: F.softplus(g)   # logistic loss for label -1
        n_batches = max(1, int(np.ceil(len(U) / self.batch_size)))
        for _ in range(self.epochs):
            u_perm = rng.permutation(len(U))
            for b in range(n_batches):
                ub = U[u_perm[b::n_batches]]
                pb = P[rng.integers(0, len(P), size=max(1, len(P) // n_batches + 1))]
                gp, gu = self.model(pb).squeeze(1), self.model(ub).squeeze(1)
                r_pos = self.prior * loss_pos(gp).mean()
                r_neg = loss_neg(gu).mean() - self.prior * loss_neg(gp).mean()
                opt.zero_grad()
                if r_neg.item() < -self.beta:  # nnPU: step on the negative part only
                    (-self.gamma * r_neg).backward()
                else:
                    (r_pos + r_neg).backward()
                opt.step()
        return self

    def decision_function(self, X):
        with torch.no_grad():
            x = torch.tensor((np.atleast_2d(X) - self.mu) / self.sd, dtype=torch.float32)
            return self.model(x).squeeze(1).numpy()

    def predict_proba(self, X):
        p = 1. / (1. + np.exp(-self.decision_function(X) / self.temperature))
        return np.stack([1 - p, p], axis=1)

    def calibrate(self, X_pos, X_unl, grid=np.exp(np.linspace(-2, 2, 41))):
        """Pick the temperature maximising the PU (positive vs unlabeled) log-likelihood:
        a labelled positive has P(s=1|x) = c * p(x) and an unlabeled frame 1 - c * p(x),
        with c = |P| / (pi * (|P| + |U|)) (the SCAR label frequency)."""
        g_p, g_u = self.decision_function(X_pos), self.decision_function(X_unl)
        c = len(X_pos) / max(self.prior * (len(X_pos) + len(X_unl)), 1e-9)
        c = float(np.clip(c, 1e-3, 1.))
        best, best_t = -np.inf, 1.
        for t in grid:
            pp, pu = 1 / (1 + np.exp(-g_p / t)), 1 / (1 + np.exp(-g_u / t))
            ll = np.log(c * pp + 1e-12).sum() + np.log(1 - c * pu + 1e-12).sum()
            if ll > best:
                best, best_t = ll, t
        self.temperature = float(best_t)
        return self
