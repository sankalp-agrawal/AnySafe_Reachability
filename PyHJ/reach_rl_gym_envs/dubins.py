from typing import Optional

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from matplotlib.patches import Circle
from skimage import measure

from PyHJ.reach_rl_gym_envs.utils.dubins_gt_solver import DubinsHJSolver
from PyHJ.utils import evaluate_V


class Dubins_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.high = np.array([1.1, 1.1, 1.1, 1.1])
        self.low = np.array([-1.1, -1.1, -1.1, -1.1])
        self.num_constraints = 3  # Number of constraints
        self.constraints_shape = 3  # Shape of one constraint, e.g. [x, y, r]
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=self.low, high=self.high, dtype=np.float32),
                "constraints": spaces.Box(  # [x, y, r]
                    low=einops.repeat(
                        np.array([-1.1, -1.1, 0, 0]), "C -> N C", N=self.num_constraints
                    ),
                    high=einops.repeat(
                        np.array([1.1, 1.1, 1.1, 1.0]),
                        "C -> N C",
                        N=self.num_constraints,
                    ),
                    shape=(self.num_constraints, self.constraints_shape + 1),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )  # joint action space

        self.u_max = 1.25
        self.solver = DubinsHJSolver()

    def step(self, action):
        # action is in -1, 1. Scale by self.u_max
        self.state = self.obs["state"]
        self.state[0] = self.state[0] + self.dt * self.state[3]
        self.state[1] = self.state[1] + self.dt * self.state[2]

        theta = np.arctan2(self.state[2], self.state[3])  # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)

        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        # l(x) = (x-x0)^2 + (y-y0)^2 - r^2
        rews = []
        for constraint in self.constraints:
            x, y, r, u = (
                constraint  # x, y are the center of the circle, r is the radius, u is the active flag
            )
            if u == 0:
                rew = (
                    np.inf
                )  # if the constraint is inactive, we set the reward to infinity
            else:
                rew = (self.state[0] - x) ** 2 + (self.state[1] - y) ** 2 - r**2
            rews.append(rew)

        rew = np.min(rews)  # take the minimum reward across all constraints

        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True

        info = {}
        self.obs = {
            "state": self.state,
            "constraints": self.constraints,
        }
        return self.obs, rew, terminated, truncated, info

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2 * np.pi)

            self.state = np.random.uniform(
                low=[-1, -1, -1, -1], high=[1, 1, 1, 1], size=(4,)
            )
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)
        else:
            self.state = initial_state
        self.state = self.state.astype(np.float32)
        self.obs = {
            "state": self.state,
            "constraints": self.select_constraints(),
        }
        return self.obs, {}

    def select_constraints(self):
        N = np.random.randint(1, self.num_constraints + 1)
        self.constraints = []
        for _ in range(N):
            self.constraints.append(
                np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
            )
        for _ in range(self.num_constraints - N):
            self.constraints.append(
                np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        0.0,  # This is used to say that this constraint is inactive
                    ]
                )
            )

        assert len(self.constraints) == self.num_constraints, (
            "Number of constraints should be {}, but got {}".format(
                self.num_constraints, len(self.constraints)
            )
        )

        return self.constraints

    def get_eval_plot(self, policy, critic):
        nx, ny, nt = 51, 51, 51
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2]

        fig1, axes1 = plt.subplots(2, len(thetas))
        fig2, axes2 = plt.subplots(2, len(thetas))
        X, Y = np.meshgrid(
            np.linspace(-1.0, 1.0, nx, endpoint=True),
            np.linspace(-1.0, 1.0, ny, endpoint=True),
        )
        self.select_constraints()
        gt_values = self.solver.solve(  # (nx, ny, nt)
            constraints=self.constraints, constraints_shape=self.constraints_shape
        )

        for i in range(len(thetas)):
            V = np.zeros_like(X)
            for ii in range(nx):
                for jj in range(ny):
                    tmp_point = np.array(
                        [
                            X[ii, jj],
                            Y[ii, jj],
                            np.sin(thetas[i]),
                            np.cos(thetas[i]),
                        ]
                    )
                    temp_obs = {
                        "state": tmp_point,
                        "constraints": np.array(self.constraints),
                    }
                    V[ii, jj] = evaluate_V(obs=temp_obs, policy=policy, critic=critic)

            nt_index = int(
                np.round((thetas[i] / (2 * np.pi)) * (nt - 1))
            )  # Convert theta to index in the grid

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
            # Plot contours for GT Value function
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

            # Add constraint patches
            for const in self.constraints:
                x, y, r, u = const
                if u == 0.0:
                    break

                # Add the circle to the axes
                [
                    ax.add_patch(
                        Circle((x, y), r, color="red", fill=False, label="Fail Set")
                    )
                    for ax in axes1[:, i]
                ]
                [
                    ax.add_patch(
                        Circle((x, y), r, color="red", fill=False, label="Fail Set")
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

        for axes in [axes1, axes2]:
            for ax in axes.flat:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect("equal")

        for fig, axes in zip([fig1, fig2], [axes1, axes2]):
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

        return fig1, fig2

    def get_eval_plot_f1(self, policy, critic):
        grid = np.load("/home/kensuke/HJRL/new_BRT_v1_w1.25.npy")

        plot_idxs = [0, 7, 14, 20]

        fig1, axes1 = plt.subplots(1, len(plot_idxs))
        fig2, axes2 = plt.subplots(1, len(plot_idxs))
        X, Y = np.meshgrid(
            np.linspace(-1.1, 1.1, grid.shape[0], endpoint=True),
            np.linspace(-1.1, 1.1, grid.shape[1], endpoint=True),
        )
        thetas_lin = np.linspace(0, 2 * np.pi, grid.shape[2], endpoint=True)

        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(thetas_lin)):
            V = np.zeros_like(X)
            for ii in range(grid.shape[0]):
                for jj in range(grid.shape[1]):
                    tmp_point = torch.tensor(
                        [
                            X[ii, jj],
                            Y[ii, jj],
                            np.sin(thetas_lin[i]),
                            np.cos(thetas_lin[i]),
                        ]
                    )
                    V[ii, jj] = evaluate_V(tmp_point, policy, critic)

            V_grid = grid[:, :, i]
            tp_grid = np.sum((V > 0) & (V_grid.T > 0))
            fp_grid = np.sum((V > 0) & (V_grid.T < 0))
            fn_grid = np.sum((V < 0) & (V_grid.T > 0))
            tn_grid = np.sum((V < 0) & (V_grid.T < 0))
            tp += tp_grid
            fp += fp_grid
            fn += fn_grid
            tn += tn_grid

            prec_grid = tp_grid / (tp_grid + fp_grid) if (tp_grid + fp_grid) > 0 else 0
            rec_grid = tp_grid / (tp_grid + fn_grid) if (tp_grid + fn_grid) > 0 else 0
            f1_grid = (
                2 * (prec_grid * rec_grid) / (prec_grid + rec_grid)
                if (prec_grid + rec_grid) > 0
                else 0
            )

            if i in plot_idxs:
                plot_idx = plot_idxs.index(i)
                axes1[plot_idx].imshow(
                    V > 0, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower"
                )
                axes2[plot_idx].imshow(
                    V,
                    extent=(-1.1, 1.1, -1.1, 1.1),
                    vmin=-1.0,
                    vmax=1.0,
                    origin="lower",
                )
                axes1[plot_idx].set_title(
                    "theta = {}".format(np.round(thetas_lin[i], 2)),
                    fontsize=12,
                )
                axes2[plot_idx].set_title(
                    "f1 = {}".format(np.round(f1_grid, 2)),
                    fontsize=12,
                )

                axes1[plot_idx].contour(
                    grid[:, :, i].T,
                    levels=[0.0],
                    colors="purple",
                    linewidths=2,
                    origin="lower",
                    extent=[-1.1, 1.1, -1.1, 1.1],
                )

        print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        return fig1, fig2, f1
