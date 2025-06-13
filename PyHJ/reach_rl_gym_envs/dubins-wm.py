from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt

from PyHJ.data.batch import Batch


class Dubins_WM_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self, params):
        if len(params) == 1:
            config = params[0]
        else:
            wm = params[0]
            past_data = params[1]
            config = params[2]
            self.set_wm(wm, past_data, config)

        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array(
            [
                1.1,
                1.1,
                np.pi,
            ]
        )
        self.low = np.array([-1.1, -1.1, -np.pi])
        self.device = "cuda:0"
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                1,
                1,
                544,
            ),
            dtype=np.float32,
        )
        image_size = config.size[0]  # 128
        img_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        bool_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        self.observation_space_full = gym.spaces.Dict(
            {
                "image": img_obs_space,
                # "obs_state": obs_space, # Don't use obs_state (sin, cos)
                "is_first": bool_space,
                "is_last": bool_space,
                "is_terminal": bool_space,
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # joint action space
        self.image_size = config.size[0]
        self.turnRate = config.turnRate

    def set_wm(self, wm, past_data, config):
        self.device = config.device
        self.encoder = wm.encoder.to(self.device)
        self.wm = wm.to(self.device)
        self.data = past_data

        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter

    def step(self, action):
        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = (
            torch.tensor([[action]], dtype=torch.float32).to(self.device)
            * self.turnRate
        )
        self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)

        self.feat = self.wm.dynamics.get_feat(self.latent)
        rew, cont = self.safety_margin(self.feat)  # rew is negative if unsafe
        self.feat = self.feat.detach().cpu().numpy()
        if cont < 0.75:
            terminated = True
        else:
            terminated = False
        truncated = False
        info = {"is_first": False, "is_terminal": terminated}
        return np.copy(self.feat), rew, terminated, truncated, info

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        init_traj = next(self.data)
        data = self.wm.preprocess(init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items():
            self.latent[k] = v[:, [-1]]
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()
        return np.copy(self.feat), {"is_first": True, "is_terminal": False}

    def safety_margin(self, feat):
        g_xList = []

        cont = self.wm.heads["cont"](feat)
        with torch.no_grad():  # Disable gradient calculation
            outputs = torch.tanh(self.wm.heads["margin"](feat))
            g_xList.append(outputs.detach().cpu().numpy())

        safety_margin = np.array(g_xList).squeeze()

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()

    def get_latent(self, wm, thetas, imgs):
        thetas = np.expand_dims(np.expand_dims(thetas, 1), 1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(thetas)[0], 1))
        firsts = np.ones((np.shape(thetas)[0], 1))
        lasts = np.zeros((np.shape(thetas)[0], 1))
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        states = np.concatenate([cos, sin], axis=-1)
        chunks = 21
        if np.shape(imgs)[0] > chunks:
            bs = int(np.shape(imgs)[0] / chunks)
        else:
            bs = int(np.shape(imgs)[0] / chunks)
        for i in range(chunks):
            if i == chunks - 1:
                data = {
                    "obs_state": states[i * bs :],
                    "image": imgs[i * bs :],
                    "action": dummy_acs[i * bs :],
                    "is_first": firsts[i * bs :],
                    "is_terminal": lasts[i * bs :],
                }
            else:
                data = {
                    "obs_state": states[i * bs : (i + 1) * bs],
                    "image": imgs[i * bs : (i + 1) * bs],
                    "action": dummy_acs[i * bs : (i + 1) * bs],
                    "is_first": firsts[i * bs : (i + 1) * bs],
                    "is_terminal": lasts[i * bs : (i + 1) * bs],
                }
            data = wm.preprocess(data)
            embeds = wm.encoder(data)
            if i == 0:
                embed = embeds
            else:
                embed = torch.cat([embed, embeds], dim=0)

        data = {
            "obs_state": states,
            "image": imgs,
            "action": dummy_acs,
            "is_first": firsts,
            "is_terminal": lasts,
        }
        data = wm.preprocess(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])

        feat = wm.dynamics.get_feat(post).detach()
        lz = torch.tanh(wm.heads["margin"](feat))
        return feat.squeeze().cpu().numpy(), lz.squeeze().detach().cpu().numpy()

    def evaluate_V(self, state, policy):
        tmp_obs = np.array(state)  # .reshape(1,-1)
        tmp_batch = Batch(obs=tmp_obs, info=Batch())
        tmp = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
        return tmp.cpu().detach().numpy().flatten()

    def get_eval_plot(self, cache, thetas, policy, config):
        fig1, axes1 = plt.subplots(len(thetas), 1, figsize=(3, 10))
        fig2, axes2 = plt.subplots(len(thetas), 1, figsize=(3, 10))

        for i in range(len(thetas)):
            theta = thetas[i]
            idxs, imgs_prev, thetas_prev = cache[theta]
            feat, lz = self.get_latent(wm=self.wm, thetas=thetas_prev, imgs=imgs_prev)
            vals = self.evaluate_V(state=feat, policy=policy)
            vals = np.minimum(vals, lz)
            axes1[i].imshow(
                vals.reshape(config.nx, config.ny).T > 0,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1,
                vmax=1,
                origin="lower",
            )
            axes2[i].imshow(
                vals.reshape(config.nx, config.ny).T,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1,
                vmax=1,
                origin="lower",
            )
        fig1.tight_layout()
        fig2.tight_layout()
        return fig1, fig2
