import argparse
import os
import sys

import einops
import gymnasium  # as gym
import numpy as np
import torch
import torch.nn as nn

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

config = tools.set_wm_name(config)

ckpt_path = config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path, weights_only=True)
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


def topographic_map(
    config, cache, thetas, constraint_state, similarity_metric, model=None
):
    if constraint_state[-1] is None:
        constraint_states = [
            np.array([constraint_state[0], constraint_state[1], t])
            for t in np.linspace(0, 2 * np.pi, 9)
        ]
        constraint_states = torch.tensor(constraint_states, dtype=torch.float32)
    else:
        constraint_states = torch.tensor([constraint_state], dtype=torch.float32)

    constraint_imgs = []
    for constraint_state in constraint_states:
        constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
        constraint_img = get_frame(states=constraint_state, config=config)  # (H, W, C)
        constraint_imgs.append(constraint_img)

    # Safety state
    safe_states = []
    for constraint_state in constraint_states:
        safe_state = torch.tensor(
            [-constraint_state[0], -constraint_state[1], constraint_state[2] + np.pi],
        )
        safe_states.append(safe_state)

    safe_states = torch.stack(safe_states, dim=0)

    safe_imgs = []
    for safe_state in safe_states:
        safe_state = torch.tensor(safe_state, dtype=torch.float32)
        safe_img = get_frame(states=safe_state, config=config)  # (H, W, C)
        safe_imgs.append(safe_img)

    with torch.no_grad():
        feat_c, stoch_c, deter_c = get_latent(  # [N, Z]
            wm, thetas=np.array(constraint_states[:, -1]), imgs=constraint_imgs
        )
        if feat_c.ndim == 1:
            feat_c = feat_c.reshape(1, -1)  # [1, Z]

        feat_s, __, __ = get_latent(  # [N, Z]
            wm, thetas=np.array(safe_states[:, -1]), imgs=safe_imgs
        )
        if feat_s.ndim == 1:
            feat_s = feat_s.reshape(1, -1)

    idxs, __, __ = cache[thetas[0]]
    feat_c = einops.repeat(feat_c, "N C -> B N C", B=idxs.shape[0])  # [B, N, Z]
    feat_s = einops.repeat(feat_s, "N C -> B N C", B=idxs.shape[0])  # [B, N, Z]

    fig, axes = plt.subplots(
        1, len(thetas) + 2, figsize=(3 * len(thetas), 5), constrained_layout=True
    )

    for i in range(len(thetas)):
        theta = thetas[i]
        i += 2  # offset for constraint and safe images
        axes[i].set_title(f"theta = {theta:.2f}")
        idxs, imgs_prev, thetas_prev = cache[theta]
        with torch.no_grad():
            feat, stoch, deter = get_latent(wm, thetas_prev, imgs_prev)  # [B, Z]
            feat = einops.repeat(feat, "B C -> B N C", N=feat_c.shape[1])  # [B, N, Z]
        if similarity_metric == "Cosine_Similarity":  # negative cosine similarity
            numerator = np.sum(feat * feat_c, axis=-1)  # (B, N)
            denominator = np.linalg.norm(feat, axis=-1) * np.linalg.norm(  # (B, N)
                feat_c, axis=-1
            )
            metric_const = -numerator / (denominator + 1e-8)  # (B, N)
            metric_const = np.min(metric_const, axis=-1)  # (B,)

            numerator = np.sum(feat * feat_s, axis=-1)  # (B, N)
            denominator = np.linalg.norm(feat, axis=-1) * np.linalg.norm(  # (B, N)
                feat_s, axis=-1
            )
            metric_safe = -numerator / (denominator + 1e-8)  # (B, N)
            metric_safe = np.min(metric_safe, axis=-1)
            metric = metric_const - np.clip(metric_safe, a_min=-0.5, a_max=0.5)  # (B,)
        elif similarity_metric == "Euclidean Distance":
            metric = -np.linalg.norm(feat - feat_c, axis=-1)  # (B, N)
            metric = np.min(metric, axis=-1)  # (B,)

        elif similarity_metric == "Learned":
            assert model is not None, (
                "Model must be provided for learned similarity metric."
            )
            feat = torch.tensor(feat, dtype=torch.float32)
            feat_c = torch.tensor(feat_c, dtype=torch.float32)
            metric = torch.tanh(model(feat, feat_c))  # (B, N)
            metric = metric.detach().cpu().numpy()  # (B, N)
            metric = np.min(metric, axis=-1)  # (B,)
        else:
            raise ValueError(
                f"Unknown similarity metric: {similarity_metric}. Supported: ['Cosine_Similarity', 'Euclidean Distance', 'Learned']"
            )

        metric = metric.reshape(config.nx, config.ny).T
        # axes[i].imshow(
        #     metric,
        #     extent=(-1.1, 1.1, -1.1, 1.1),
        #     vmin=-1,
        #     vmax=1,
        #     origin="lower",
        # )

        x = np.linspace(-1.1, 1.1, metric.shape[1])
        y = np.linspace(-1.1, 1.1, metric.shape[0])
        X, Y = np.meshgrid(x, y)

        contour = axes[i].contour(X, Y, metric, levels=5, colors="black", linewidths=1)
        axes[i].clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    for constraint_img in constraint_imgs:
        # Show the constraint image on the topographic map
        axes[0].imshow(
            constraint_img,
            extent=(config.x_min, config.x_max, config.y_min, config.y_max),
        )
        axes[0].set_title("Constraint Image")

    for safe_img in safe_imgs:
        # Show the safe image on the topographic map
        axes[1].imshow(
            safe_img, extent=(config.x_min, config.x_max, config.y_min, config.y_max)
        )
        axes[1].set_title("Safe Image")

    # set axes limits
    for ax in axes:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")

    fig.suptitle(f"Topographic Map using {similarity_metric}")
    plt.tight_layout()
    return fig


thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi, 3 * np.pi / 2]
cache = make_cache(config, thetas)
logger = None
warmup = 1


class SafetyMargin(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SafetyMargin, self).__init__()
        layers = []

        # Create hidden layers
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim

        # Final output layer (no activation here)
        layers.append(nn.Linear(last_dim, output_dim))
        # layers.append(nn.Tanh())  # Use Tanh to keep output in [-1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, z, z_const):
        input = torch.cat((z, z_const), dim=-1)
        return self.model(input)


safety_margin = SafetyMargin(input_dim=2 * 544, hidden_dims=[512, 256], output_dim=1)
# Load the pre-trained model
model_path = "safety_margin_model.pth"
if os.path.exists(model_path):
    safety_margin.load_state_dict(torch.load(model_path, map_location=config.device))
    print("Safety Margin model loaded successfully.")
else:
    print(f"Model file {model_path} not found. Using untrained model.")

similarity_metrics = ["Cosine_Similarity", "Euclidean Distance", "Learned"]

logger = WandbLogger(
    name=f"wm_Analysis_{config.wm_name}", config=config, project="Dubins"
)

for metric in similarity_metrics:
    constraint_list = [
        [0.0, 0.0, 0.0],  # 0.0],  # x, y, theta
        [0.5, 0.5, np.pi / 2],
        [-0.5, -0.5, -np.pi / 2],
        [0.5, -0.5, np.pi / 2],
        [-0.5, 0.5, -np.pi / 2],
    ]
    for constraint_state in constraint_list:
        cprint(
            f"Running topographic map for constraint state: {constraint_state}",
            "green",
            attrs=["bold"],
        )
        fig = topographic_map(
            config=config,
            cache=cache,
            thetas=thetas,
            constraint_state=constraint_state,
            similarity_metric=metric,
            model=safety_margin if metric == "Learned" else None,
        )

        wandb.log(
            {
                f"{metric}_constraint/{constraint_state}": wandb.Image(fig),
            }
        )

        plt.close(fig)
