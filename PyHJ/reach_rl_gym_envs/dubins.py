from typing import Optional

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from matplotlib.patches import Circle

from PyHJ.reach_rl_gym_envs.utils.dubins_gt_solver import DubinsHJSolver
from PyHJ.reach_rl_gym_envs.utils.env_eval_utils import get_eval_plot
from PyHJ.utils import evaluate_V, find_a


class Dubins_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self, dist_type=None, nominal_policy=None):
        self.render_mode = None
        self.dt = 0.05
        self.v = 1
        self.high = np.array([1.1, 1.1, 1.1, 1.1])
        self.low = np.array([-1.1, -1.1, -1.1, -1.1])
        self.num_constraints = 1  # Number of constraints
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
        self.distribution_type = (
            dist_type  # What constraints are considered in distribution and OOD
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )  # joint action space

        self.u_max = 1.25
        self.solver = DubinsHJSolver()
        self.nominal_policy = nominal_policy  # Nominal policy for the agent

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
            "constraints": np.array(self.constraints),
        }
        return self.obs, rew, terminated, truncated, info

    def render(self, mode="human", unsafe=False, t=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(f"Nominal Policy: {self.nominal_policy}, 'time': {t}")
        for constraint in self.constraints:
            x, y, r, u = constraint
            if u == 0.0:
                break
            circle = Circle((x, y), r, color="red", fill=False, label="Fail Set")
            ax.add_patch(circle)
        state = self.obs["state"]
        agent_color = "red" if unsafe else "blue"
        plt.scatter(state[0], state[1], s=150, c=agent_color, zorder=3)
        plt.quiver(
            state[0],
            state[1],
            self.dt * self.v * state[3],  # cos
            self.dt * self.v * state[2],  # sin
            angles="xy",
            scale_units="xy",
            minlength=0,
            scale=0.5,  # Slightly smaller arrow
            width=0.05,
            color=agent_color,
            zorder=3,
        )

        handles, labels = [], []
        h, label = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(label)

        # Remove duplicates while preserving order
        unique = dict(zip(labels, handles))

        # Create a single, global legend
        fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

        # Return rgb array of the figure
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., 1:]
        plt.close(fig)  # Close the figure to free memory
        return img

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        in_distribution: bool = True,
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
            "constraints": self.select_constraints(in_distribution=in_distribution),
        }
        return self.obs, {}

    def select_one_constraint(self, in_distribution=True):
        if self.distribution_type == "four_corners":
            in_distribution_set = [
                np.array([-0.5, -0.5, 0.5, 1.0]),
                np.array([0.5, -0.5, 0.5, 1.0]),
                np.array([-0.5, 0.5, 0.5, 1.0]),
                np.array([0.5, 0.5, 0.5, 1.0]),
            ]
            if in_distribution:
                i = np.random.randint(0, len(in_distribution_set))
                return in_distribution_set[i]
            else:
                return np.array([0.0, 0.0, 0.5, 1.0])
        elif self.distribution_type == "four_corners_four_edges":
            in_distribution_set = [
                np.array([-0.5, -0.5, 0.5, 1.0]),
                np.array([0.5, -0.5, 0.5, 1.0]),
                np.array([-0.5, 0.5, 0.5, 1.0]),
                np.array([0.5, 0.5, 0.5, 1.0]),
                np.array([-0.5, 0.0, 0.5, 1.0]),
                np.array([0.5, 0.0, 0.5, 1.0]),
                np.array([0.0, -0.5, 0.5, 1.0]),
                np.array([0.0, 0.5, 0.5, 1.0]),
            ]
            if in_distribution:
                i = np.random.randint(0, len(in_distribution_set))
                return in_distribution_set[i]
            else:
                return np.array([0.0, 0.0, 0.5, 1.0])

        elif self.distribution_type == "right_half":
            if in_distribution:
                return np.array(
                    [
                        np.random.uniform(low=0.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
            else:
                return np.array(
                    [
                        np.random.uniform(low=-1.0, high=0.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        1.0,  # This is used to say that this constraint is active
                    ]
                )

        elif self.distribution_type == "big_radii":
            if in_distribution:
                return np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(
                            low=0.3, high=0.5
                        ),  # Big radii are in distribution
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
            else:
                return np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(
                            low=0.1, high=0.3
                        ),  # Small radii are out of distribution
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
        elif self.distribution_type == "vanilla":
            # Eval and test set are the same here
            return np.array(
                [
                    0.0,
                    0.0,
                    0.5,
                    1.0,
                ]  # This is used to say that this constraint is active
            )
        else:
            raise ValueError(
                "Unknown distribution type: {}".format(self.distribution_type)
            )

    def select_constraints(self, in_distribution=True):
        N = np.random.randint(1, self.num_constraints + 1)
        self.constraints = []
        for _ in range(N):
            self.constraints.append(
                self.select_one_constraint(in_distribution=in_distribution)
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

        return np.array(self.constraints)

    def get_eval_plot(self, policy, critic, in_distribution=True):
        return get_eval_plot(
            env=self, policy=policy, critic=critic, in_distribution=in_distribution
        )

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

    def generate_trajectory(self, policy, in_distribution=True):
        self.reset()
        self.select_constraints(in_distribution=in_distribution)
        obs = self.obs
        imgs = []
        done = False
        eps = 0.5
        t = 0
        while not done:
            V = evaluate_V(obs=obs, policy=policy, critic=policy.critic1)
            if V > eps:
                unsafe = False
                if self.nominal_policy == "turn_right":
                    action = np.array([-1.0], dtype=np.float32)
                elif self.nominal_policy == "random":
                    action = np.random.uniform(low=-1.0, high=1.0, size=(1,))
                else:
                    raise ValueError(
                        "Unknown nominal policy: {}".format(self.nominal_policy)
                    )
            else:
                unsafe = True
                action = find_a(obs, policy)
            obs, rew, done, _, _ = self.step(action)
            imgs.append(self.render(unsafe=unsafe, t=t))
            t += 1

        imgs = np.array(imgs)  # (T, W, H, C)
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # (T, C, W, H)
        return imgs
