"""
Component A: a learned Dirichlet-process (truncated stick-breaking / GEM) target
marginal for ASOT, replacing the uniform-over-K action marginal.

ASOT solves  min_T <C,T> + alpha*R_temp(T) + lambda*KL(T^T 1 || q)  with q uniform
over K.  Here q is instead the posterior-mean stick-breaking weight vector of a
truncated DP(gamma) with truncation K_max, updated in closed form from the realised
skill usage:

    v_k | m ~ Beta(1 + m_k, gamma + sum_{l>k} m_l)
    q_k     = E[v_k] * prod_{l<k} (1 - E[v_l])

Stick-breaking is not exchangeable, so the sticks are ordered by usage (descending)
every update; this is equivalent to re-sorting the prototypes without touching the
optimiser state.  Prototypes whose weight falls below ``prune_eps`` are pruned; the
number of surviving prototypes is the inferred number of skills K_hat.

Usage counts are measured in *segments* rather than frames: an episode's soft frame
usage is rescaled to sum to its number of contiguous segments, so a DP "customer" is
one skill execution.  This keeps gamma on a meaningful scale (frames inside a
segment are not independent draws).

Optionally gamma gets a Gamma(a0, b0) hyperprior and is updated with the posterior
mean of the Escobar & West (1995) auxiliary-variable conditional.
"""
import math

import torch


class DPStickBreakingMarginal(torch.nn.Module):
    def __init__(self, k_max, gamma=1.0, prune_eps=None, gamma_prior=None, decay=0.99,
                 warmup_updates=20, min_active=1):
        super().__init__()
        self.k_max = k_max
        self.prune_eps = prune_eps if prune_eps is not None else 1. / (2. * k_max)
        self.gamma_prior = gamma_prior  # (a0, b0) or None
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.min_active = min_active
        self.register_buffer('gamma', torch.tensor(float(gamma)))
        self.register_buffer('counts', torch.zeros(k_max))
        self.register_buffer('active', torch.ones(k_max, dtype=torch.bool))
        self.register_buffer('n_updates', torch.tensor(0))
        self.register_buffer('q', torch.full((k_max,), 1. / k_max))
        # running transport-cost gain of each prototype over its runner-up (DP-means pruning)
        self.register_buffer('red_sum', torch.zeros(k_max))
        self.register_buffer('red_cnt', torch.zeros(k_max))

    @property
    def k_hat(self):
        return int(self.active.sum().item())

    @staticmethod
    def segment_usage(plan, mask):
        """Soft usage per prototype in segment units. plan: (B, N, K), mask: (B, N)."""
        plan = plan * mask.unsqueeze(2)
        frame_mass = plan.sum(dim=1)  # (B, K)
        labels = plan.argmax(dim=2)
        changes = ((labels[:, 1:] != labels[:, :-1]) & mask[:, 1:]).sum(dim=1) + 1  # (B,)
        frame_mass = frame_mass / frame_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return (frame_mass * changes.unsqueeze(1).float()).sum(dim=0)

    def stick_weights(self, counts=None):
        """Posterior-mean stick-breaking weights over active prototypes (usage-sorted)."""
        counts = self.counts if counts is None else counts
        idx = torch.nonzero(self.active).squeeze(1)
        m = counts[idx]
        order = torch.argsort(m, descending=True)
        m_sorted = m[order]
        tail = torch.flip(torch.cumsum(torch.flip(m_sorted, [0]), 0), [0]) - m_sorted  # sum_{l>k} m_l
        a = 1. + m_sorted
        b = self.gamma + tail
        ev = a / (a + b)
        ev[-1] = 1.  # truncation: last stick takes the remaining mass
        log_rest = torch.cat([torch.zeros(1, device=ev.device),
                              torch.cumsum(torch.log1p(-ev[:-1].clamp(max=1 - 1e-12)), 0)])
        w_sorted = ev * torch.exp(log_rest)
        q = torch.zeros_like(counts)
        q[idx[order]] = w_sorted
        return q

    def _update_gamma(self):
        if self.gamma_prior is None:
            return
        a0, b0 = self.gamma_prior
        n = float(self.counts[self.active].sum().item())
        k = float(self.k_hat)
        if n <= 1:
            return
        g = float(self.gamma.item())
        # E[log eta] for eta ~ Beta(g + 1, n) (digamma difference), then the mean of the
        # two-component Gamma mixture conditional of Escobar & West.
        e_log_eta = torch.digamma(torch.tensor(g + 1.)) - torch.digamma(torch.tensor(g + 1. + n))
        rate = b0 - float(e_log_eta)
        odds = (a0 + k - 1.) / (n * rate)
        pi = odds / (1. + odds)
        mean = pi * (a0 + k) / rate + (1. - pi) * (a0 + k - 1.) / rate
        self.gamma.fill_(max(mean, 1e-3))

    @torch.no_grad()
    def update(self, usage):
        """q-step: fold a batch's usage (K_max,) into running counts and recompute q."""
        usage = usage.detach().to(self.counts.device) * self.active
        self.counts.mul_(self.decay).add_(usage)
        self.n_updates += 1
        self._update_gamma()
        q = self.stick_weights()
        if self.n_updates.item() > self.warmup_updates:
            share = q / q.sum()
            prune = self.active & (share < self.prune_eps)
            # never prune below min_active prototypes
            n_keep = int((self.active & ~prune).sum().item())
            if n_keep < self.min_active:
                ranked = torch.argsort(share, descending=True)[:self.min_active]
                prune[ranked] = False
            if prune.any():
                self.active &= ~prune
                self.counts[prune] = 0.
                q = self.stick_weights()
        self.q.copy_(q / q.sum())
        return self.q

    @torch.no_grad()
    def accumulate_redundancy(self, sums, cnts):
        """Running (decayed) per-prototype transport-cost gain over the runner-up prototype."""
        if sums is None:
            return
        self.red_sum.mul_(self.decay).add_(sums.to(self.counts.device))
        self.red_cnt.mul_(self.decay).add_(cnts.to(self.counts.device))

    @torch.no_grad()
    def prune_redundant(self, delta):
        """Prune the most redundant active prototype if its mean cost gain is below delta
        (one per q-step so the survivors can re-absorb its frames)."""
        if self.n_updates.item() <= self.warmup_updates or self.k_hat <= self.min_active:
            return False
        mean_gain = torch.where(self.active & (self.red_cnt > 0), self.red_sum / self.red_cnt.clamp_min(1e-9),
                                torch.full_like(self.red_sum, float('inf')))
        # a prototype that owns no frames at all is redundant too
        mean_gain = torch.where(self.active & (self.red_cnt <= 1e-6), torch.zeros_like(mean_gain), mean_gain)
        k = int(torch.argmin(mean_gain).item())
        if mean_gain[k] >= delta:
            return False
        self.active[k] = False
        self.counts[k] = 0.
        self.red_sum[k] = 0.
        self.red_cnt[k] = 0.
        q = self.stick_weights()
        self.q.copy_(q / q.sum())
        return True

    @torch.no_grad()
    def merge_similar(self, prototypes, cos_thresh):
        """Merge active prototypes closer than cos_thresh: the less-used one is pruned and its
        usage is transferred (a DP-means style penalty on redundant clusters)."""
        if self.n_updates.item() <= self.warmup_updates:
            return
        idx = torch.nonzero(self.active).squeeze(1)
        if len(idx) <= self.min_active:
            return
        p = torch.nn.functional.normalize(prototypes[idx], dim=-1)
        sim = p @ p.T
        sim.fill_diagonal_(-1.)
        i, j = divmod(int(torch.argmax(sim).item()), len(idx))
        if sim[i, j] < cos_thresh:
            return
        keep, drop = (idx[i], idx[j]) if self.counts[idx[i]] >= self.counts[idx[j]] else (idx[j], idx[i])
        self.counts[keep] += self.counts[drop]
        self.counts[drop] = 0.
        self.active[drop] = False
        q = self.stick_weights()
        self.q.copy_(q / q.sum())

    def state_summary(self):
        return {'k_hat': self.k_hat, 'gamma': float(self.gamma.item()),
                'q': [round(float(x), 4) for x in self.q.tolist()],
                'active': self.active.tolist()}


def expected_num_clusters(gamma, n):
    """E[#clusters] under a CRP(gamma) with n customers (reference for choosing gamma)."""
    return sum(gamma / (gamma + i) for i in range(int(n))) if n > 0 else 0.0


def _selftest():
    torch.manual_seed(0)
    dp = DPStickBreakingMarginal(k_max=8, gamma=1.0, warmup_updates=2, decay=0.9)
    true_q = torch.tensor([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    for _ in range(50):
        dp.update(true_q * 20 + torch.rand(8) * 0.05)
    assert dp.k_hat == 3, dp.state_summary()
    assert math.isclose(float(dp.q.sum()), 1.0, rel_tol=1e-5)
    print('bnp_marginal selftest ok', dp.state_summary())


if __name__ == '__main__':
    _selftest()
