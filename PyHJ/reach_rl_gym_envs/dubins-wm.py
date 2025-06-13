from collections import defaultdict
from typing import Optional

import einops
import gymnasium as gym
import numpy as np
import torch
from generate_data_traj_cont import get_frame
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage import measure

from PyHJ.reach_rl_gym_envs.utils.dubins_gt_solver import DubinsHJSolver
from PyHJ.reach_rl_gym_envs.utils.env_eval_utils import get_metrics
from PyHJ.utils.eval_utils import evaluate_V


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

        self.config = config

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
        self.num_constraints = 1
        self.constraints_shape = 544
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(544,),
                    dtype=np.float32,
                ),
                "constraints": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.num_constraints,
                        self.constraints_shape + 1,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        image_size = config.size[0]  # 128
        img_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        bool_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        obs_dict = {
            "is_first": bool_space,
            "is_last": bool_space,
            "is_terminal": bool_space,
        }
        if "image" in config.encoder["cnn_keys"]:
            obs_dict["image"] = img_obs_space

        if "obs_state" in config.encoder["mlp_keys"]:
            obs_dict["obs_state"] = obs_space

        self.observation_space_full = gym.spaces.Dict(obs_dict)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # joint action space
        self.image_size = config.size[0]
        self.turnRate = config.turnRate

        self.solver = DubinsHJSolver(nx=config.nx, ny=config.ny, nt=config.nt)
        self.nominal_policy = "turn_right"
        if hasattr(self, "wm"):
            self.select_constraints()

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
        self.obs = {
            "state": self.feat.flatten(),
            "constraints": self.constraints,
        }
        info = {"is_first": False, "is_terminal": terminated}
        return self.obs, rew, terminated, truncated, info

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
        self.obs = {
            "state": self.feat.flatten(),
            "constraints": self.constraints,
        }
        return self.obs, {"is_first": True, "is_terminal": False}

    def safety_margin(self, feat):
        g_xList = []

        cont = self.wm.heads["cont"](feat)
        with torch.no_grad():  # Disable gradient calculation
            outputs = torch.tanh(self.wm.heads["margin"](feat))
            g_xList.append(outputs.detach().cpu().numpy())

        safety_margin = np.array(g_xList).squeeze()

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()

    def select_constraints(self):
        constraint = torch.tensor([0.0, 0.0, 0.0])
        img = get_frame(states=constraint, config=self.config)
        feat_c, __ = self.get_latent(
            wm=self.wm, thetas=constraint[-1].reshape(-1), imgs=[img]
        )
        feat_c = np.append(
            feat_c, 1.0
        )  # Append 1.0 to indicate that this constraint is active
        self.constraints = np.array(feat_c).reshape(self.num_constraints, -1)

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
        lz = self.safety_margin(feat)  # lz is the safety margin
        return feat.squeeze().cpu().numpy(), lz.squeeze().detach().cpu().numpy()

    def get_eval_plot(self, cache, thetas, policy, config):
        nx, ny, nt = config.nx, config.ny, 51
        fig1, axes1 = plt.subplots(2, len(thetas))
        fig2, axes2 = plt.subplots(2, len(thetas))
        fig3, axes3 = plt.subplots(1, len(thetas))

        constraint = np.array([0.0, 0.0, 0.5, 1.0]).reshape(1, -1)
        self.select_constraints()
        gt_values = self.solver.solve(
            constraints=constraint,
            constraints_shape=3,
        )

        all_metrics = []

        for i in range(len(thetas)):
            theta = thetas[i]
            idxs, imgs_prev, thetas_prev = cache[theta]
            with torch.no_grad():
                feat, lz = self.get_latent(
                    wm=self.wm,
                    thetas=thetas_prev,
                    imgs=imgs_prev,
                )
                obs = {
                    "state": feat,
                    "constraints": einops.repeat(
                        self.constraints, "1 C -> N 1 C", N=feat.shape[0]
                    ),
                }
                V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
            V = np.minimum(V, lz)

            nt_index = int(
                np.round((thetas[i] / (2 * np.pi)) * (nt - 1))
            )  # Convert theta to index in the grid

            V = V.reshape((nx, ny)).T  # Reshape to match the grid
            metrics = get_metrics(rl_values=V, gt_values=gt_values[:, :, nt_index].T)
            all_metrics.append(metrics)

            # Find contours for gt and rl Value functions
            contours_rl = measure.find_contours(
                np.array(V > 0).astype(float), level=0.5
            )
            contours_gt = measure.find_contours(
                np.array(gt_values[:, :, nt_index].T > 0).astype(float), level=0.5
            )

            # Show sub-zero level set
            axes1[0, i].imshow(V > 0, extent=(-1.0, 1.0, -1.0, 1.0), origin="lower")
            axes1[1, i].imshow(
                gt_values[:, :, nt_index].T > 0,
                extent=(-1.0, 1.0, -1.0, 1.0),
                origin="lower",
            )
            # Show value functions
            axes2[0, i].imshow(
                V, extent=(-1.0, 1.0, -1.0, 1.0), vmin=-1.0, vmax=1.0, origin="lower"
            )
            axes2[1, i].imshow(
                gt_values[:, :, nt_index].T,
                extent=(-1.0, 1.0, -1.0, 1.0),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )

            # Plot safety margin
            axes3[i].imshow(
                lz.reshape((nx, ny)),
                extent=(-1.0, 1.0, -1.0, 1.0),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )

            # Plot contours for RL Value function
            for contour in contours_rl:
                for axes in [axes1, axes2]:
                    [
                        ax.plot(
                            contour[:, 1] * (2.0 / (nx - 1)) - 1.0,
                            contour[:, 0] * (2.0 / (ny - 1)) - 1.0,
                            color="blue",
                            linewidth=2,
                            label="RL Value Contour",
                        )
                        for ax in axes[:, i]
                    ]
            # # Plot contours for GT Value function
            for contour in contours_gt:
                for axes in [axes1, axes2]:
                    [
                        ax.plot(
                            contour[:, 1] * (2.0 / (nx - 1)) - 1.0,
                            contour[:, 0] * (2.0 / (ny - 1)) - 1.0,
                            color="Green",
                            linewidth=2,
                            label="GT Value Contour",
                        )
                        for ax in axes[:, i]
                    ]

            # Add constraint patch
            [
                ax.add_patch(
                    Circle((0.0, 0.0), 0.5, color="red", fill=False, label="Fail Set")
                )
                for ax in axes1[:, i]
            ]
            [
                ax.add_patch(
                    Circle((0.0, 0.0), 0.5, color="red", fill=False, label="Fail Set")
                )
                for ax in axes2[:, i]
            ]

            axes1[0, i].set_title(
                "theta = {}".format(np.round(thetas[i], 2)),
                fontsize=12,
            )
            axes2[0, i].set_title(
                "theta = {}".format(np.round(thetas[i], 2)),
                fontsize=12,
            )
            axes1[1, i].set_title(
                "GT,\ntheta = {}".format(np.round(thetas[i], 2)),
                fontsize=12,
            )
            axes2[1, i].set_title(
                "GT,\ntheta = {}".format(np.round(thetas[i], 2)),
                fontsize=12,
            )
            axes3[i].set_title(
                "theta = {}".format(np.round(thetas[i], 2)),
                fontsize=12,
            )

        for axes in [axes1, axes2, axes3]:
            for ax in axes.flat:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect("equal")

        for fig, axes in zip([fig1, fig2, fig3], [axes1, axes2, axes3]):
            handles, labels = [], []
            for ax in axes.flat:
                h, label = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(label)

            # Remove duplicates while preserving order
            unique = dict(zip(labels, handles))

            # Create a single, global legend
            fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the legend

        aggregated = defaultdict(list)
        for metrics in all_metrics:
            for key, value in metrics.items():
                aggregated[key].append(value)

        # Compute averages
        averaged_metrics = {key: np.mean(values) for key, values in aggregated.items()}

        return (
            fig1,
            fig2,
            fig3,
            averaged_metrics,
        )
