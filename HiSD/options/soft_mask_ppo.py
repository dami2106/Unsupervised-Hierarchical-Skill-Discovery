"""
Soft initiation masks for sb3-contrib MaskablePPO.

Standard invalid-action masking (Huang & Ontanon 2020) sets masked logits to -1e8.
Here the "mask" an env returns may be a float vector of calibrated initiation
probabilities p_a in [0, 1]: the policy logits receive a log(p_a) bias
(p_a = 0 is still a hard mask, p_a = 1 leaves the logit unchanged).  Boolean masks
behave exactly as before, so hard masks remain available for Old-HiSD parity.

MaskablePPO's rollout buffer already stores masks as float32, so the probabilities
flow through collection and the PPO update unchanged.

    from soft_mask_ppo import SoftMaskCnnPolicy
    model = MaskablePPO(SoftMaskCnnPolicy, env, ...)
"""
import torch as th
from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import (MaskableActorCriticCnnPolicy, MaskableActorCriticPolicy,
                                                  MaskableMultiInputActorCriticPolicy)

HUGE_NEG = -1e8


class SoftMaskableCategorical(MaskableCategorical):
    def apply_masking(self, masks) -> None:
        if masks is None:
            self.masks = None
            logits = self._original_logits
        else:
            device = self.logits.device
            m = th.as_tensor(masks, dtype=self._original_logits.dtype, device=device).reshape(self.logits.shape)
            self.masks = m > 0  # support used by the (masked) entropy
            bias = th.where(self.masks, th.log(m.clamp_min(1e-12)), th.tensor(HUGE_NEG, dtype=m.dtype, device=device))
            logits = self._original_logits + bias
        # bypass MaskableCategorical.__init__'s re-masking; reinitialise the Categorical and
        # drop torch's lazily cached probs from any previous masking
        self.__dict__.pop('probs', None)
        super(MaskableCategorical, self).__init__(logits=logits, validate_args=self._validate_args)


class SoftMaskableCategoricalDistribution(MaskableCategoricalDistribution):
    def proba_distribution(self, action_logits: th.Tensor):
        self.distribution = SoftMaskableCategorical(logits=action_logits.view(-1, self.action_dim))
        return self


def _soft(cls):
    class Soft(cls):
        def _build(self, lr_schedule):
            self.action_dist = SoftMaskableCategoricalDistribution(int(self.action_space.n))
            super()._build(lr_schedule)
    Soft.__name__ = 'Soft' + cls.__name__
    return Soft


SoftMaskMlpPolicy = _soft(MaskableActorCriticPolicy)
SoftMaskCnnPolicy = _soft(MaskableActorCriticCnnPolicy)
SoftMaskMultiInputPolicy = _soft(MaskableMultiInputActorCriticPolicy)


def _selftest():
    logits = th.zeros(1, 4)
    d = SoftMaskableCategorical(logits=logits)
    d.apply_masking(th.tensor([[1., 0.5, 0.25, 0.]]))
    p = d.probs[0]
    assert p[3] < 1e-6 and abs(p[0] / p[1] - 2) < 1e-4 and abs(p[1] / p[2] - 2) < 1e-4
    d.apply_masking(th.tensor([[True, True, False, False]]))
    assert abs(d.probs[0, 0] - 0.5) < 1e-6
    print('soft mask selftest ok', p.tolist())


if __name__ == '__main__':
    _selftest()
