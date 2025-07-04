import argparse
import os
import sys

import gymnasium  # as gym
import torch
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dreamerv3-torch")
)
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/PyHJ"))
sys.path.append(saferl_dir)
import einops
import torch.nn as nn
import torch.optim as optim

print(sys.path)
import collections
import pathlib
from datetime import datetime

import models
import requests
import ruamel.yaml as yaml
import tools

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from PIL import Image
from termcolor import cprint
from transformers import AutoImageProcessor, AutoModel

from PyHJ.exploration import GaussianNoise

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

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base")
patch_size = dino.config.patch_size


def encode_dino(image):
    inputs = processor(images=image, return_tensors="pt")
    batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = (
        img_height // patch_size,
        img_width // patch_size,
    )
    num_patches_flat = num_patches_height * num_patches_width

    outputs = dino(**inputs)
    last_hidden_states = outputs[0]
    # print(last_hidden_states.shape)  # [1, 1 + 256, 768]
    assert last_hidden_states.shape == (
        batch_size,
        1 + num_patches_flat,
        dino.config.hidden_size,
    )

    cls_token = last_hidden_states[:, 0, :]
    patch_features = last_hidden_states[:, 1:, :].unflatten(
        1, (num_patches_height, num_patches_width)
    )
    return cls_token, patch_features


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
    parser.add_argument("--debug", action="store_true", default=False)
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


def create_dataset(dataset, wm, r=0.5):
    Z = []
    priv_state = []
    # use tqdm
    for i in tqdm(range(4000), desc="Creating dataset", unit="sample"):
        # Get z value
        data = next(dataset)
        image = einops.rearrange(data["image"][:, 0, :, :], "N H W C -> N C H W")
        feat, __ = encode_dino(image)
        # data = wm.preprocess(data)
        # embed = wm.encoder(data)
        # post, __ = wm.dynamics.observe(embed, data["action"], data["is_first"])

        # feat = wm.dynamics.get_feat(post)

        Z.append(feat[0].detach().cpu())
        # Z.append(feat[0, 0, :].detach().cpu())
        priv_state.append(data["privileged_state"][0, 0, :])

    return torch.stack(Z), torch.stack(priv_state)


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


args = get_args()
config = args

GaussianNoise

env = gymnasium.make(args.task, params=[config])
config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

config = tools.set_wm_name(config)

ckpt_path = config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path)
state_dict = {
    k[14:]: v for k, v in checkpoint["agent_state_dict"].items() if "_wm" in k
}
wm.load_state_dict(state_dict)
wm.to(config.device)
wm.encoder.to(config.device)
wm.eval()


offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

tools = tools

# Create dummy data
input_dim = 10
output_dim = 1
num_samples = 1000
batch_size = 32


# Initialize model, loss, optimizer
model = SafetyMargin(input_dim=2 * 768, hidden_dims=[512, 256], output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

Z, GT_States = create_dataset(offline_dataset, wm, r=0.5)


def get_epoch_pair_batches(n, batch_size):
    # Shuffle once for both sets
    indices1 = torch.randperm(n)
    indices2 = torch.randperm(n)

    for i in range(0, n, batch_size):
        batch_idx1 = indices1[i : i + batch_size]
        batch_idx2 = indices2[i : i + batch_size]

        yield batch_idx1, batch_idx2


print("Running representation learning script...")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for indices_query, indices_const in get_epoch_pair_batches(len(Z), batch_size):
        z, gt_state, z_const, gt_state_const = (
            Z[indices_query],
            GT_States[indices_query],
            Z[indices_const],
            GT_States[indices_const],
        )
        optimizer.zero_grad()

        mask = torch.norm(gt_state[:, :2] - gt_state_const[:, :2], p=2, dim=-1) < 0.5

        lx_loss = 0.0
        preds = model(z, z_const)
        pos = model(z[~mask], z_const[~mask])
        neg = model(z[mask], z_const[mask])

        gamma = 0.75
        if pos.numel() > 0:
            lx_loss += torch.relu(gamma - pos).mean()
        if neg.numel() > 0:
            lx_loss += torch.relu(gamma + neg).mean()

        lx_loss.backward()
        optimizer.step()
        total_loss += lx_loss.item()

    avg_loss = total_loss / (len(Z) // batch_size)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# save the model
model_path = "safety_margin_model.pth"
torch.save(model.state_dict(), model_path)
