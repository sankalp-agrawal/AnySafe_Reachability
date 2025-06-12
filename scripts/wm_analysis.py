import argparse
import os
import sys

import einops
import gymnasium  # as gym
import numpy as np
import torch

import PyHJ
import wandb
from PyHJ.utils import WandbLogger

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dreamerv3-torch")
)
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/PyHJ"))
sys.path.append(saferl_dir)
print(sys.path)
import collections
import io
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import models
import ruamel.yaml as yaml
import tools

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from generate_data_traj_cont import get_frame
from PIL import Image
from termcolor import cprint

# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
"""
    Note that, we can pass arguments to the script by using
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True

"""


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(
        # (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir + '/PyHJ'}/{config.expt_name}"
    # final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir + '/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config


dummy_variable = PyHJ

args = get_args()
config = args

env = gymnasium.make(args.task, params=[config])
config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

ckpt_path = config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path)
state_dict = {
    k[14:]: v for k, v in checkpoint["agent_state_dict"].items() if "_wm" in k
}
wm.load_state_dict(state_dict)
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

env.set_wm(wm, offline_dataset, config)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert("RGB")


def make_cache(config, thetas):
    nx, ny = config.nx, config.ny
    cache = {}
    for theta in thetas:
        v = np.zeros((nx, ny))
        xs = np.linspace(-1.1, 1.1, nx, endpoint=True)
        ys = np.linspace(-1.1, 1.1, ny, endpoint=True)
        key = theta
        print("creating cache for key", key)
        idxs, imgs_prev, thetas, thetas_prev = [], [], [], []
        xs_prev = xs - config.dt * config.speed * np.cos(theta)
        ys_prev = ys - config.dt * config.speed * np.sin(theta)
        theta_prev = theta
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x_prev = xs_prev[idx[0]]
            y_prev = ys_prev[idx[1]]
            thetas.append(theta)
            thetas_prev.append(theta_prev)
            imgs_prev.append(
                get_frame(torch.tensor([x_prev, y_prev, theta_prev]), config)
            )
            idxs.append(idx)
            it.iternext()
        idxs = np.array(idxs)
        theta_prev_lin = np.array(thetas_prev)
        cache[theta] = [idxs, imgs_prev, theta_prev_lin]

    return cache


def get_latent(
    wm,
    thetas,  # (N)
    imgs,  # N-List of (H, W, C)
):
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
    stoch = post["stoch"]  # z_t
    deter = post["deter"]  # h_t
    return feat.squeeze().cpu().numpy(), stoch, deter


def topographic_map(config, cache, thetas, constraint_state, similarity_metric):
    fig, axes = plt.subplots(
        1, len(thetas), figsize=(3 * len(thetas), 5), constrained_layout=True
    )

    constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
    constraint_img = get_frame(states=constraint_state, config=config)  # (H, W, C)

    feat_c, stoch_c, deter_c = get_latent(
        wm, thetas=np.array([constraint_state[-1].item()]), imgs=[constraint_img]
    )

    idxs, __, __ = cache[thetas[0]]
    feat_c = einops.repeat(feat_c, "c -> b c", b=idxs.shape[0])

    for i in range(len(thetas)):
        theta = thetas[i]
        axes[i].set_title(f"theta = {theta:.2f}")
        idxs, imgs_prev, thetas_prev = cache[theta]
        feat, stoch, deter = get_latent(wm, thetas_prev, imgs_prev)
        if similarity_metric == "Cosine_Similarity":  # negative cosine similarity
            numerator = np.sum(feat * feat_c, axis=1)
            denominator = np.linalg.norm(feat, axis=1) * np.linalg.norm(feat_c, axis=1)
            metric = -numerator / (denominator + 1e-8)  # (N)
        elif similarity_metric == "Euclidean Distance":
            metric = np.linalg.norm(feat - feat_c, axis=1)
        else:
            raise ValueError(
                f"Unknown similarity metric: {similarity_metric}. Supported: ['Cosine_Similarity', 'Euclidean Distance']"
            )

        metric = metric.reshape(config.nx, config.ny).T
        # axes[i].imshow(
        #     metric,
        #     extent=(-1.1, 1.1, -1.1, 1.1),
        #     vmin=-1,
        #     vmax=1,
        #     origin="lower",
        # )

        x = np.linspace(-1, 1, metric.shape[1])
        y = np.linspace(-1, 1, metric.shape[0])
        X, Y = np.meshgrid(x, y)

        contour = axes[i].contour(X, Y, metric, levels=5, colors="black", linewidths=1)
        axes[i].clabel(contour, inline=True, fontsize=8, fmt="%.2f")

        axes[i].imshow(
            constraint_img,
            extent=(-1.1, 1.1, -1.1, 1.1),
        )

    # set axes limits
    for ax in axes:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    fig.suptitle(f"Topographic Map using {similarity_metric}")
    plt.tight_layout()
    return fig


thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]
cache = make_cache(config, thetas)
logger = None
warmup = 1

similarity_metrics = ["Cosine_Similarity", "Euclidean Distance"]

logger = WandbLogger(name="WM Analysis")

for metric in similarity_metrics:
    fig = topographic_map(
        config=config,
        cache=cache,
        thetas=thetas,
        constraint_state=[0.0, 0.0, 0.0],
        similarity_metric=metric,
    )

    wandb.log(
        {
            f"{metric}": wandb.Image(fig),
        }
    )

    plt.close(fig)
