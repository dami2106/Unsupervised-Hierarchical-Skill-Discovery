"""
Component B: nonparametric, noise-robust hierarchy induction.

HiSD runs a deterministic Sequitur over the collapsed skill strings, so every
segmentation error mints new rules / new trees (e.g. 500 unique trees vs 293 GT on
Minecraft "All").  This module puts a Bayesian nonparametric model *under* the
grammar:

  * A Pitman-Yor process PY(d, theta) over latent "sentence types" (clean skill
    strings).  Each episode sits at a table; each table carries a prototype string.
    This is an adaptor (Johnson et al. 2006) on the Sentence non-terminal: reuse of a
    derivation follows the PY power law, and near-duplicate derivations collapse.
  * A noisy-channel terminal model: an observed episode string is an edit-channel
    emission (substitution / insertion / deletion) of its table's prototype, so
    residual label noise is absorbed instead of producing a new rule per error.
    Channel rates are re-estimated from Viterbi alignments under Beta priors.
  * A base distribution G0 over prototype strings (geometric length x smoothed
    bigram Markov chain over skills).

Inference is collapsed Gibbs sampling over table assignments with prototype
resampling (``method='gibbs'``), or greedy Bayesian model merging of types under the
same posterior (``method='bmm'``, Stolcke & Omohundro 1994) as the robust fallback.
Sequitur is then run on the denoised (MAP) corpus to obtain the rule hierarchy and
per-episode trees in exactly HiSD's format; posterior samples give distributions over
trees / tree counts.

The boundary token phi is never inside a rule: episodes are parsed independently and
Sequitur is fed Mark() separators exactly as in HiSD.
"""
import math
from collections import Counter, defaultdict

import numpy as np

NEG_INF = -np.inf


# --------------------------------------------------------------------------------------
# Noisy edit channel  p(obs | proto)
# --------------------------------------------------------------------------------------
class EditChannel:
    """Generative edit channel from a prototype string to an observed string.

    Before each prototype symbol (and at the end) a geometric number of uniformly
    random symbols is inserted (prob p_ins each); each prototype symbol is then
    deleted (p_del), substituted by a uniform other symbol (p_sub) or copied.
    """

    def __init__(self, n_symbols, p_sub=0.05, p_ins=0.05, p_del=0.05, enabled=True):
        self.V = max(int(n_symbols), 2)
        self.p_sub, self.p_ins, self.p_del = p_sub, p_ins, p_del
        self.enabled = enabled

    def _logs(self):
        V = self.V
        ins = math.log(self.p_ins / V)
        go = math.log1p(-self.p_ins)
        dele = go + math.log(self.p_del)
        sub = go + math.log(self.p_sub / (V - 1))
        match = go + math.log1p(-(self.p_del + self.p_sub))
        return ins, go, dele, sub, match

    def logp(self, obs, proto):
        """Forward algorithm (sum over alignments), vectorised over obs positions."""
        if not self.enabled:
            return 0.0 if tuple(obs) == tuple(proto) else NEG_INF
        ins, go, dele, sub, match = self._logs()
        obs = np.asarray([hash(o) for o in obs])
        n = len(obs)
        steps = np.arange(n + 1) * ins
        # row 0: only insertions
        prev = steps.copy()
        for p in proto:
            ph = hash(p)
            a = np.full(n + 1, NEG_INF)
            a = np.logaddexp(a, prev + dele)  # delete proto symbol, emit nothing
            emit = np.where(obs == ph, match, sub)
            a[1:] = np.logaddexp(a[1:], prev[:-1] + emit)
            # insertions within the row: F[j] = logsumexp_k<=j a[k] + (j-k)*ins
            prev = np.logaddexp.accumulate(a - steps) + steps
        return float(prev[n] + go)

    def viterbi_counts(self, obs, proto):
        """Counts of (match, sub, del, ins, proceed) on the best alignment."""
        if not self.enabled:
            return Counter()
        ins, go, dele, sub, match = self._logs()
        n, m = len(obs), len(proto)
        F = np.full((m + 1, n + 1), NEG_INF)
        bp = np.zeros((m + 1, n + 1), dtype=np.int8)  # 1 ins, 2 del, 3 sub/match
        F[0, 0] = 0.
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue
                best, arg = NEG_INF, 0
                if j > 0 and F[i, j - 1] + ins > best:
                    best, arg = F[i, j - 1] + ins, 1
                if i > 0 and F[i - 1, j] + dele > best:
                    best, arg = F[i - 1, j] + dele, 2
                if i > 0 and j > 0:
                    e = match if obs[j - 1] == proto[i - 1] else sub
                    if F[i - 1, j - 1] + e > best:
                        best, arg = F[i - 1, j - 1] + e, 3
                F[i, j], bp[i, j] = best, arg
        c = Counter()
        i, j = m, n
        c['proceed'] += 1  # final "stop inserting"
        while i > 0 or j > 0:
            a = bp[i, j]
            if a == 1:
                c['ins'] += 1
                j -= 1
            elif a == 2:
                c['del'] += 1
                c['proceed'] += 1
                i -= 1
            else:
                c['match' if obs[j - 1] == proto[i - 1] else 'sub'] += 1
                c['proceed'] += 1
                i, j = i - 1, j - 1
        return c

    def reestimate(self, counts, prior_noise=1., prior_clean=20., lo=1e-4, hi=0.3):
        ops = counts['match'] + counts['sub'] + counts['del']
        self.p_ins = float(np.clip((counts['ins'] + prior_noise) /
                                   (counts['ins'] + counts['proceed'] + prior_noise + prior_clean), lo, hi))
        self.p_del = float(np.clip((counts['del'] + prior_noise) / (ops + prior_noise + prior_clean), lo, hi))
        self.p_sub = float(np.clip((counts['sub'] + prior_noise) / (ops + prior_noise + prior_clean), lo, hi))

    def params(self):
        return {'p_sub': self.p_sub, 'p_ins': self.p_ins, 'p_del': self.p_del}


# --------------------------------------------------------------------------------------
# Base distribution over prototype strings
# --------------------------------------------------------------------------------------
class BigramBase:
    """G0(s) = Geom(len) x bigram Markov chain with add-alpha smoothing."""

    def __init__(self, strings, alpha=0.5):
        self.symbols = sorted({s for x in strings for s in x}, key=str)
        self.V = len(self.symbols)
        self.alpha = alpha
        mean_len = max(np.mean([len(x) for x in strings]), 1.)
        self.p_stop = 1. / (mean_len + 1.)
        self.big = defaultdict(Counter)
        for x in strings:
            prev = '<s>'
            for s in x:
                self.big[prev][s] += 1
                prev = s

    def logp(self, s):
        lp = len(s) * math.log1p(-self.p_stop) + math.log(self.p_stop)
        prev = '<s>'
        for x in s:
            row = self.big[prev]
            lp += math.log((row[x] + self.alpha) / (sum(row.values()) + self.alpha * (self.V + 1)))
            prev = x
        return lp


# --------------------------------------------------------------------------------------
# Pitman-Yor helpers
# --------------------------------------------------------------------------------------
def py_log_seating(sizes, d, theta):
    """log P(partition with table sizes) under PY(d, theta)."""
    sizes = [n for n in sizes if n > 0]
    n, T = sum(sizes), len(sizes)
    lp = 0.
    for t in range(1, T):
        lp += math.log(theta + t * d)
    lp -= math.lgamma(theta + n) - math.lgamma(theta + 1)
    for n_t in sizes:
        lp += math.lgamma(n_t - d) - math.lgamma(1 - d)
    return lp


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class PYNoisyChannelGrammar:
    def __init__(self, d=0.5, theta=1.0, channel=True, p_noise=0.05, learn_channel=True,
                 max_len_diff=None, seed=0):
        self.d, self.theta = d, theta
        self.use_channel = channel
        self.p_noise = p_noise
        self.learn_channel = learn_channel and channel
        self.max_len_diff = max_len_diff
        self.rng = np.random.default_rng(seed)

    # ---- setup ----
    def _prepare(self, strings):
        self.strings = [tuple(s) for s in strings]
        self.uniq = sorted(set(self.strings))
        self.uidx = {u: i for i, u in enumerate(self.uniq)}
        self.obs_u = np.array([self.uidx[s] for s in self.strings])
        self.base = BigramBase(self.uniq)
        n_sym = len({x for s in self.strings for x in s})
        self.channel = EditChannel(n_sym, self.p_noise, self.p_noise, self.p_noise, enabled=self.use_channel)
        self.logG0 = np.array([self.base.logp(u) for u in self.uniq])
        self._compute_L()

    def _compute_L(self):
        U = len(self.uniq)
        L = np.full((U, U), NEG_INF)
        for i, u in enumerate(self.uniq):
            for j, v in enumerate(self.uniq):
                if self.max_len_diff is not None and abs(len(u) - len(v)) > self.max_len_diff:
                    continue
                L[i, j] = self.channel.logp(u, v)
        self.L = L  # L[obs, proto] = log p(obs | proto)
        # log p_new(u) = log sum_v G0(v) p(u | v)  (candidate prototypes = observed types)
        self.log_pnew = np.logaddexp.reduce(self.L + self.logG0[None, :], axis=1)

    # ---- state ----
    def _init_state(self):
        # start from the Sequitur-equivalent state: one table per distinct string
        self.assign = self.obs_u.copy()
        self.protos = {u: u for u in range(len(self.uniq))}
        self.sizes = Counter(self.assign.tolist())

    def log_joint(self):
        tables = [t for t in self.sizes if self.sizes[t] > 0]
        lp = py_log_seating([self.sizes[t] for t in tables], self.d, self.theta)
        lp += sum(self.logG0[self.protos[t]] for t in tables)
        lp += float(sum(self.L[self.obs_u[e], self.protos[self.assign[e]]] for e in range(len(self.strings))))
        return lp

    def _new_table_id(self):
        return max(self.protos.keys(), default=-1) + 1

    def _gibbs_sweep(self):
        n = len(self.strings)
        for e in self.rng.permutation(n):
            t_old = self.assign[e]
            self.sizes[t_old] -= 1
            if self.sizes[t_old] == 0:
                del self.sizes[t_old]
                del self.protos[t_old]
            u = self.obs_u[e]
            tables = list(self.sizes.keys())
            T = len(tables)
            logits = [math.log(self.sizes[t] - self.d) + self.L[u, self.protos[t]] for t in tables]
            logits.append(math.log(self.theta + self.d * T) + self.log_pnew[u])
            logits = np.array(logits)
            p = np.exp(logits - logits.max())
            k = self.rng.choice(len(p), p=p / p.sum())
            if k < T:
                t_new = tables[k]
            else:
                t_new = self._new_table_id()
                lw = self.L[u] + self.logG0
                pw = np.exp(lw - lw.max())
                self.protos[t_new] = int(self.rng.choice(len(pw), p=pw / pw.sum()))
            self.assign[e] = t_new
            self.sizes[t_new] += 1
        self._resample_protos()

    def _resample_protos(self, greedy=False):
        members = defaultdict(list)
        for e, t in enumerate(self.assign):
            members[t].append(self.obs_u[e])
        for t, mem in members.items():
            lw = self.logG0 + self.L[np.array(mem)].sum(axis=0)
            if greedy:
                self.protos[t] = int(np.argmax(lw))
            else:
                pw = np.exp(lw - lw.max())
                self.protos[t] = int(self.rng.choice(len(pw), p=pw / pw.sum()))

    def _reestimate_channel(self):
        if not self.learn_channel:
            return
        c = Counter()
        for e in range(len(self.strings)):
            c += self.channel.viterbi_counts(self.strings[e], self.uniq[self.protos[self.assign[e]]])
        self.channel.reestimate(c)
        self._compute_L()

    def _snapshot(self):
        return {'assign': self.assign.copy(), 'protos': dict(self.protos), 'sizes': Counter(self.sizes),
                'log_joint': self.log_joint(), 'channel': self.channel.params()}

    def _restore(self, snap):
        self.assign = snap['assign'].copy()
        self.protos = dict(snap['protos'])
        self.sizes = Counter(snap['sizes'])

    # ---- inference ----
    def fit(self, strings, method='gibbs', n_iter=50, burn_in=20, thin=2, channel_every=5):
        self._prepare(strings)
        self._init_state()
        self.samples = []
        if method == 'bmm':
            self._fit_bmm(channel_every)
            self.samples = [self._snapshot()]
            self.map_state = self.samples[0]
            return self
        best = None
        for it in range(n_iter):
            self._gibbs_sweep()
            if channel_every and (it + 1) % channel_every == 0:
                self._reestimate_channel()
            snap = self._snapshot()
            if it >= burn_in and (it - burn_in) % thin == 0:
                self.samples.append(snap)
            if best is None or snap['log_joint'] > best['log_joint']:
                best = snap
        # MAP polish: greedy prototypes for the best state
        self._restore(best)
        self._resample_protos(greedy=True)
        self.map_state = self._snapshot()
        return self

    def _fit_bmm(self, channel_every):
        """Greedy Bayesian model merging of sentence types under the same posterior."""
        rounds = 0
        while True:
            self._resample_protos(greedy=True)
            base = self.log_joint()
            tables = list(self.sizes.keys())
            members = defaultdict(list)
            for e, t in enumerate(self.assign):
                members[t].append(self.obs_u[e])
            best_gain, best_pair = 0., None
            for a_i, a in enumerate(tables):
                for b in tables[a_i + 1:]:
                    gain = self._merge_gain(a, b, members)
                    if gain > best_gain:
                        best_gain, best_pair = gain, (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            self.assign[self.assign == a] = b
            self.sizes[b] += self.sizes.pop(a)
            del self.protos[a]
            rounds += 1
            if channel_every and rounds % channel_every == 0:
                self._reestimate_channel()
        self._resample_protos(greedy=True)

    def _merge_gain(self, a, b, members):
        sizes = [self.sizes[t] for t in self.sizes]
        before_seat = py_log_seating(sizes, self.d, self.theta)
        merged = [self.sizes[t] for t in self.sizes if t not in (a, b)] + [self.sizes[a] + self.sizes[b]]
        after_seat = py_log_seating(merged, self.d, self.theta)
        mem_a, mem_b = np.array(members[a]), np.array(members[b])
        before = (self.logG0[self.protos[a]] + self.L[mem_a, self.protos[a]].sum() +
                  self.logG0[self.protos[b]] + self.L[mem_b, self.protos[b]].sum())
        mem = np.concatenate([mem_a, mem_b])
        after = np.max(self.logG0 + self.L[mem].sum(axis=0))
        return (after_seat - before_seat) + (after - before)

    # ---- outputs ----
    def denoised(self, state=None):
        state = state or self.map_state
        return [self.uniq[state['protos'][t]] for t in state['assign']]

    def num_types(self, state=None):
        state = state or self.map_state
        return len({state['protos'][t] for t in state['sizes']})

    def posterior_type_counts(self):
        return [len({s['protos'][t] for t in s['sizes']}) for s in self.samples]

    def predictive_logp(self, s, state):
        """log p(s | state) under the PY posterior predictive with the edit channel."""
        s = tuple(s)
        n = sum(state['sizes'].values())
        T = len(state['sizes'])
        cache = {}

        def ch(obs, v):
            if (obs, v) not in cache:
                cache[(obs, v)] = self.channel.logp(obs, self.uniq[v]) if obs not in self.uidx \
                    else self.L[self.uidx[obs], v]
            return cache[(obs, v)]

        terms = [math.log(state['sizes'][t] - self.d) + ch(s, state['protos'][t]) for t in state['sizes']]
        cand = [self.logG0[v] + ch(s, v) for v in range(len(self.uniq))]
        if s not in self.uidx:  # the string itself as a candidate prototype
            cand.append(self.base.logp(s) + self.channel.logp(s, s))
        terms.append(math.log(self.theta + self.d * T) + np.logaddexp.reduce(cand))
        return float(np.logaddexp.reduce(terms) - math.log(n + self.theta))


# --------------------------------------------------------------------------------------
# Grammar construction + metrics
# --------------------------------------------------------------------------------------
def sequitur_grammar(strings):
    """Run HiSD's modified Sequitur (Mark() boundaries) and return (grammar, trees)."""
    from sksequitur import Parser, Grammar, Production, Mark

    parser = Parser()
    for s in strings:
        parser.feed(list(s))
        parser.feed([Mark()])
    grammar = Grammar(parser.tree)

    def subtree(prod):
        node = {"production": int(prod), "children": []}
        for tok in grammar[prod]:
            node["children"].append(subtree(tok) if isinstance(tok, Production) else {"symbol": tok})
        return node

    trees, cur = [], []
    for tok in grammar[Production(0)]:
        if isinstance(tok, Mark):
            trees.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        trees.append(cur)
    out = []
    for toks in trees:
        node = {"production": 0, "children": []}
        for tok in toks:
            node["children"].append(subtree(tok) if isinstance(tok, Production) else {"symbol": tok})
        out.append(node)
    assert len(out) == len(strings)
    return grammar, out


def grammar_description_length(strings):
    """Bits to encode the rule inventory Sequitur induces on a set of type strings.

    Rule 0 lists each type once (the corpus-level index is charged separately)."""
    from sksequitur import Production, Mark
    if not strings:
        return 0.
    grammar, _ = sequitur_grammar(sorted(set(tuple(s) for s in strings)))
    terminals = {x for s in strings for x in s}
    vocab = len(terminals) + len(grammar) + 1
    bits = 0.
    for head, rhs in grammar.items():
        bits += (len([t for t in rhs if not isinstance(t, Mark)]) + 1) * math.log2(vocab)
    return bits


def hierarchy_metrics(model, heldout=None):
    """Metrics for the NP grammar and the Sequitur-equivalent (exact, noise-free) baseline."""
    from collections import Counter as C
    raw = model.strings
    den = model.denoised()
    n = len(raw)
    res = {}
    # description length: grammar over type inventory + type index code + channel noise bits
    map_state = model.map_state
    seat_bits = -py_log_seating(list(map_state['sizes'].values()), model.d, model.theta) / math.log(2)
    noise_bits = -sum(model.L[model.obs_u[e], map_state['protos'][map_state['assign'][e]]]
                      for e in range(n)) / math.log(2)
    res['np_map_unique_trees'] = len(set(den))
    res['np_mdl_bits'] = grammar_description_length(den) + seat_bits + noise_bits
    counts = model.posterior_type_counts()
    res['np_posterior_unique_trees_mean'] = float(np.mean(counts)) if counts else float(res['np_map_unique_trees'])
    res['np_posterior_unique_trees_std'] = float(np.std(counts)) if counts else 0.
    res['np_posterior_unique_trees_hist'] = dict(C(counts))
    res['channel'] = model.channel.params()

    exact_sizes = list(C(raw).values())
    res['seq_unique_trees'] = len(set(raw))
    res['seq_mdl_bits'] = grammar_description_length(raw) - py_log_seating(exact_sizes, model.d, model.theta) / math.log(2)

    if heldout:
        syms = sum(len(s) + 1 for s in heldout)
        # NP: average predictive over posterior samples (MAP if no samples)
        states = model.samples or [map_state]
        lp_np = 0.
        for s in heldout:
            lps = [model.predictive_logp(s, st) for st in states]
            lp_np += float(np.logaddexp.reduce(lps) - math.log(len(lps)))
        # exact baseline: identity channel, tables = distinct training strings
        cnt = C(raw)
        T = len(cnt)
        lp_ex = 0.
        for s in heldout:
            s = tuple(s)
            p_new = math.log(model.theta + model.d * T) + model.base.logp(s)
            p_old = math.log(cnt[s] - model.d) if s in cnt else NEG_INF
            lp_ex += float(np.logaddexp(p_old, p_new) - math.log(n + model.theta))
        res['np_heldout_perplexity'] = float(math.exp(-lp_np / syms))
        res['seq_heldout_perplexity'] = float(math.exp(-lp_ex / syms))
    return res


def _selftest():
    rng = np.random.default_rng(0)
    clean = [tuple('abcd'), tuple('abab'), tuple('cdcd')]
    data = []
    for _ in range(120):
        s = list(clean[rng.integers(3)])
        if rng.random() < 0.3:  # corrupt one symbol
            s[rng.integers(len(s))] = 'abcd'[rng.integers(4)]
        data.append(tuple(s))
    for method in ['gibbs', 'bmm']:
        m = PYNoisyChannelGrammar(seed=0).fit(data, method=method, n_iter=30, burn_in=10)
        print(method, 'raw types', len(set(data)), '-> NP types', m.num_types(), m.channel.params())
        assert m.num_types() < len(set(data))
    print('np_grammar selftest ok')


if __name__ == '__main__':
    _selftest()
