import argparse
import os
import pathlib
import pickle
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml as yaml
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dreamerv3_torch")
)
sys.path.append(dreamer_dir)
import tools


def get_frame(states, config, fig, ax, circle_patch):
    ax.clear()
    ax.set_xlim([config.x_min, config.x_max])
    ax.set_ylim([config.y_min, config.y_max])
    ax.axis("off")

    if config.show_constraint and circle_patch:
        ax.add_patch(circle_patch)

    agent_color = "black"
    ax.quiver(
        states[0],
        states[1],
        config.dt * config.speed * torch.cos(states[2]),
        config.dt * config.speed * torch.sin(states[2]),
        angles="xy",
        scale_units="xy",
        minlength=0,
        width=0.1,
        scale=config.arrow_size,
        color=agent_color,
        zorder=3,
    )
    ax.scatter(states[0], states[1], s=20, c=agent_color, zorder=3)

    fig.canvas.draw()
    img_array = np.array(fig.canvas.buffer_rgba())
    return img_array[:, :, :3]  # Remove alpha channel


def get_init_state(config):
    states = torch.zeros(3)
    while (
        torch.linalg.norm(states[:2] - torch.tensor([config.obs_x, config.obs_y]))
        < config.obs_r
    ):
        states = torch.rand(3)
        states[0] = states[0] * (
            config.x_max - config.buffer - (config.x_min + config.buffer)
        ) + (config.x_min + config.buffer)
        states[1] = states[1] * (
            config.y_max - config.buffer - (config.y_min + config.buffer)
        ) + (config.y_min + config.buffer)

    states[2] = torch.atan2(-states[1], -states[0]) + torch.randn(1)
    states[2] = states[2] % (2 * torch.pi)
    return states


def gen_one_traj_img(config, fig, ax, circle_patch):
    states = get_init_state(config)

    state_obs_list = []
    img_obs_list = []
    state_gt_list = []
    dones_list = []
    acs_list = []
    u_max = config.turnRate
    dt = config.dt
    v = config.speed

    for t in range(config.data_length):
        ac = torch.rand(1) * 2 * u_max - u_max

        states_next = torch.empty(3)
        states_next[0] = states[0] + v * dt * torch.cos(states[2])
        states_next[1] = states[1] + v * dt * torch.sin(states[2])
        states_next[2] = states[2] + dt * ac

        state_obs_list.append(states[2])
        state_gt_list.append(states)

        done = 0
        if t == config.data_length - 1:
            done = 1
        elif (
            torch.abs(states[0]) > config.x_max - config.buffer
            or torch.abs(states[1]) > config.y_max - config.buffer
        ):
            done = 1
        dones_list.append(done)

        acs_list.append(ac)
        img_array = get_frame(states, config, fig, ax, circle_patch)
        img_obs_list.append(img_array)
        states = states_next
        if done == 1:
            break

    return (
        torch.stack(state_obs_list).squeeze(-1).numpy(),
        torch.stack(acs_list).squeeze(-1).numpy(),
        torch.stack(state_gt_list).numpy(),
        np.stack(img_obs_list),
        np.array(dones_list),
    )


def generate_trajs(config):
    demos = []

    fig, ax = plt.subplots()
    fig.set_size_inches(1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(config.size[0])

    circle_patch = None
    if config.show_constraint:
        circle_patch = patches.Circle(
            [config.obs_x, config.obs_y],
            config.obs_r,
            edgecolor=config.constraint_color,
            facecolor="none",
        )

    for i in tqdm(range(config.num_trajs), desc="Generating trajectories"):
        state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(
            config, fig, ax, circle_patch
        )
        demo = {}
        demo["obs"] = {"image": img_obs, "state": state_obs, "priv_state": state_gt}
        demo["actions"] = acs
        demo["dones"] = dones
        demos.append(demo)

        if i == 0:
            import imageio

            video_path = os.path.join("debug_rollout.mp4")
            imageio.mimsave(video_path, img_obs, fps=10)

    plt.close(fig)

    with open(config.dataset_path, "wb") as f:
        pickle.dump(demos, f)


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    config_args, remaining = parser.parse_known_args()

    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config = tools.set_wm_name(final_config)

    generate_trajs(final_config)
