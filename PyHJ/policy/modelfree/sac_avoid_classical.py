# If you find any problem, please contact me at: james.jingqi.li@gmail.com
# Thanks for your support!

# We apply SAC to learn the classical reach-avoid value function, using the Bellman backup in arXiv:2112.12288.
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal

from PyHJ.data import Batch, ReplayBuffer
from PyHJ.exploration import BaseNoise
from PyHJ.policy.modelfree.ddpg_avoid_classical import (
    avoid_DDPGPolicy_annealing as DDPGPolicy,
)


class avoid_SACPolicy_annealing(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905,
        for learning the classical reach-avoid value function, arXiv:2112.12288.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        actor1: Optional[torch.nn.Module] = None,  # control policy
        actor1_optim: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None,
            None,
            tau,
            gamma,
            exploration_noise,
            reward_normalization,
            estimation_step,
            **kwargs,
        )
        if actor1 is not None and actor1_optim is not None:
            self.actor1: torch.nn.Module = actor1
            self.actor1_old = deepcopy(actor1)
            self.actor1_old.eval()
            self.actor1_optim: torch.optim.Optimizer = actor1_optim
        # self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "avoid_SACPolicy_annealing":
        self.training = mode
        self.actor1.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        model1 = getattr(self, "actor1")
        obs = batch[input]
        logits, hidden = model1(obs, state=state, info=batch.info)

        hidden = None
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + self.__eps).sum(
            -1, keepdim=True
        )
        return Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob,
        )

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        target_q = (
            torch.min(
                self.critic1_old(batch.obs_next, act_),
                self.critic2_old(batch.obs_next, act_),
            )
            - self._alpha * obs_next_result.log_prob
        )
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result1 = self(batch)
        act = obs_result1.act
        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        pure_critic_value1 = torch.min(current_q1a, current_q2a)

        rot_cost = torch.norm(act[:, 3:6], dim=1)
        actor1_loss = (
            self._alpha * obs_result1.log_prob.flatten() - pure_critic_value1
        ).mean() + 0.1 * rot_cost.mean()
        self.actor1_optim.zero_grad()
        actor1_loss.backward(retain_graph=True)
        self.actor1_optim.step()

        if self._is_auto_alpha:
            log_prob1 = obs_result1.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob1).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor1": -pure_critic_value1.mean().item(),
            # "loss/rotation": rot_cost.mean().item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch, new_expl=True
    ) -> Union[np.ndarray, Batch]:
        if new_expl:
            rand_act = np.random.uniform(-1, 1, act.shape)

            values = self.critic1(batch.obs, rand_act).cpu().detach().numpy()
            act = np.where(values < 0.0, act, rand_act)

        return act
