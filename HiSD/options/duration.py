"""
Component C (part 1): segmenter-agnostic Bayesian skill-duration models.

Everything here is computed from *segment outputs only* (per-frame skill labels from
any segmenter: fixed-K ASOT, DP-ASOT, ground truth, ...), so C composes with A or
without it (the B+C / C-only cells of the ablation).

Duration model (per skill):  d - 1 ~ Poisson(lam),  lam ~ Gamma(a0, b0).
The posterior predictive of d - 1 is negative binomial
    NB(r = a0 + sum(d_i - 1),  p = 1 / (b0 + n + 1)),
which is over-dispersed relative to a plug-in Poisson and shrinks towards the prior
for rarely seen skills.  The hazard
    beta_dur(t) = p(d = t) / P(d >= t)
is the probability that a skill that has run for t steps ends at step t.

Composite (grammar non-terminal) durations are the convolution of their children's
duration pmfs.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.stats import nbinom


def extract_segments(labels):
    """[(label, length), ...] for contiguous runs of a per-frame label sequence."""
    segs = []
    if len(labels) == 0:
        return segs
    start = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            segs.append((labels[start], t - start))
            start = t
    return segs


class PoissonGammaDuration:
    def __init__(self, a0=1.0, b0=0.1, t_max=512):
        self.a0, self.b0, self.t_max = a0, b0, t_max
        self.n, self.sum_excess = 0, 0.

    def fit(self, lengths):
        lengths = np.asarray(lengths, dtype=float)
        self.n = len(lengths)
        self.sum_excess = float(np.sum(lengths - 1)) if self.n else 0.
        self._build()
        return self

    def _build(self):
        r = self.a0 + self.sum_excess
        p = 1. / (self.b0 + self.n + 1.)  # scipy nbinom: failures before r successes, success prob 1-p
        t = np.arange(1, self.t_max + 1)
        pmf = nbinom.pmf(t - 1, r, 1. - p)
        tail = max(0., 1. - pmf.sum())
        pmf[-1] += tail  # truncate: remaining mass at t_max
        self.pmf = pmf

    @classmethod
    def from_pmf(cls, pmf):
        obj = cls(t_max=len(pmf))
        obj.pmf = np.asarray(pmf, dtype=float) / np.sum(pmf)
        return obj

    @property
    def mean(self):
        return float(np.sum(np.arange(1, len(self.pmf) + 1) * self.pmf))

    def cdf(self, t):
        if t < 1:
            return 0.
        return float(np.sum(self.pmf[:min(t, len(self.pmf))]))

    def hazard(self, t):
        """P(end at t | still running at t), t >= 1."""
        if t < 1:
            return 0.
        if t > len(self.pmf):
            return 1.
        surv = 1. - self.cdf(t - 1)
        return float(min(1., self.pmf[t - 1] / surv)) if surv > 1e-12 else 1.

    def hazard_curve(self):
        return np.array([self.hazard(t) for t in range(1, len(self.pmf) + 1)])

    def percentile(self, q):
        c = np.cumsum(self.pmf)
        return int(np.searchsorted(c, q) + 1)


def fit_skill_durations(label_sequences, a0=1.0, b0=0.1, t_max=512, exclude_last=False):
    """Fit one duration model per skill from a list of per-frame label sequences.

    exclude_last: drop each episode's final segment (right-censored by the episode end)."""
    lengths = defaultdict(list)
    for labels in label_sequences:
        segs = extract_segments(list(labels))
        if exclude_last and len(segs) > 1:
            segs = segs[:-1]
        for lab, ln in segs:
            lengths[lab].append(ln)
    return {lab: PoissonGammaDuration(a0, b0, t_max).fit(ls) for lab, ls in lengths.items()}, dict(lengths)


def composite_duration(children, duration_models, t_max=None):
    """Duration of a composite option executing `children` in sequence (pmf convolution)."""
    pmf = np.array([1.])  # delta at 0
    for c in children:
        cp = np.concatenate([[0.], duration_models[c].pmf])  # index = duration
        pmf = np.convolve(pmf, cp)
    pmf = pmf[1:]  # durations start at 1
    if t_max is not None and len(pmf) > t_max:
        pmf = np.concatenate([pmf[:t_max - 1], [pmf[t_max - 1:].sum()]])
    return PoissonGammaDuration.from_pmf(pmf)


def _selftest():
    rng = np.random.default_rng(0)
    lens = rng.poisson(9, size=200) + 1
    m = PoissonGammaDuration().fit(lens)
    assert abs(m.mean - lens.mean()) < 0.5, (m.mean, lens.mean())
    h = m.hazard_curve()
    assert h[0] < 0.01 and h[20] > 0.5
    comp = composite_duration(['a', 'b'], {'a': m, 'b': m})
    assert abs(comp.mean - 2 * m.mean) < 0.5
    segs = extract_segments([0, 0, 1, 1, 1, 0])
    assert segs == [(0, 2), (1, 3), (0, 1)]
    print('duration selftest ok: mean', round(m.mean, 2), 'p95', m.percentile(0.95),
          'composite mean', round(comp.mean, 2))


if __name__ == '__main__':
    _selftest()
