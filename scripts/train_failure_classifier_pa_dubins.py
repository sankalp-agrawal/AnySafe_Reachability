import argparse
import collections
import os
import random
import sys
from datetime import datetime

import einops
import gymnasium  # as gym
import matplotlib.pyplot as plt
import numpy as np
import PyHJ
import ruamel.yaml as yaml
import torch
import torch.nn.functional as F
import umap.umap_ as umap
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# note: need to include the dreamerv3 repo for this
from termcolor import cprint

dreamer_dir = os.path.abspath("/home/sunny/AnySafe_Reachability/dreamerv3_torch")
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath("/home/sunny/AnySafe_Reachability/PyHJ")
sys.path.append(saferl_dir)
print(sys.path)
import models
import tools
from dreamer import make_dataset
from matplotlib import colormaps

# note: need to include the dreamerv3 repo for this
from tqdm import *

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
    ]
)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

device = "cuda:0"

# Backbone Model
LOG_DIR = "logs_pa"


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
    with open("/home/sunny/AnySafe_Reachability/configs.yaml", "r") as f:
        configs = yml.load(f)

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

config = get_args()
config = tools.set_wm_name(config)

if config.pa["gpu_id"] != -1:
    torch.cuda.set_device(config.pa["gpu_id"])

config.nb_classes = config.grid_size**2 + 1  # four quadrants in the 2D space


# Setup wandb
def wandb_setup():
    # Wandb Initialization
    wandb_name_kwargs = {
        # "mrg": config.mrg,
        # "alpha": int(config.alpha),
        # "num_ex": (
        #     config.num_examples_per_class
        #     if config.num_examples_per_class is not None
        #     else "all"
        # ),
        "gs": config.grid_size,
        "split": config.train_test_split,
    }
    if config.pa["use_unlabeled_data"]:
        wandb_name_kwargs["ul"] = "T"
        wandb_name_kwargs["ul_ratio"] = (
            config.unlabeled_ratio
            if config.unlabeled_ratio != -1.0
            else f"all_{config.ratio_schedule}"
        )
        wandb_name_kwargs["beta"] = config.beta
        wandb_name_kwargs["temp"] = config.temp

    config.wandb_name = "".join(
        f"{key}_{value}_"
        for key, value in wandb_name_kwargs.items()
        if value is not None
    ).rstrip("_")
    wandb.init(name=config.wandb_name, project="ProxyAnchor")
    wandb.config.update(config)

    wandb.define_metric("num_updates", step_metric="num_updates")
    wandb.define_metric("*", step_metric="num_updates")


wandb_setup()


env = gymnasium.make(config.task, params=[config])

config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
model = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
ckpt_path = "logs/dreamer_dubins/dubins_mlp_obs_state_cnn_image_lz_None_sc_F_arrow_0.15/best_rssm_ckpt_0_10.pt"  # config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path)

state_dict = {
    k[14:]: v for k, v in checkpoint["agent_state_dict"].items() if "_wm" in k
}

model_state_dict = model.state_dict()
loaded_state_dict = {}

for name, param in state_dict.items():
    if name not in model_state_dict:
        print(f"Skipping '{name}' as it is not in the model.")
        continue

    if model_state_dict[name].shape != param.shape:
        print(
            f"Shape mismatch for '{name}': "
            f"model={model_state_dict[name].shape}, "
            f"checkpoint={param.shape}. Skipping."
        )
        continue

    loaded_state_dict[name] = param

# Load only the matching parameters
model.load_state_dict(loaded_state_dict, strict=False)

# model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

config = tools.set_wm_name(config)

for name, param in model.named_parameters():
    param.requires_grad = name.startswith("semantic_encoder")

# Decoder
decoder = torch.nn.Sequential(
    torch.nn.Linear(config.pa["sz_embedding"], 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 544),
).to(device)

offline_eps = collections.OrderedDict()
config.pa["batch_size"] = 1
config.pa["batch_length"] = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)


# 1. Flatten Trajectories to Timesteps
def flatten_trajectories(trajectories):
    flat_data = {}
    import ipdb

    ipdb.set_trace()
    keys = next(iter(trajectories.values())).keys()

    for key in keys:
        flat_data[key] = []

    for traj in trajectories.values():
        for key in keys:
            # Assume traj[key] is tensor or array with shape (T, ...)
            # We want to concatenate timesteps (dim=0) across trajectories
            # So just append the whole array, not flatten inside dimensions
            flat_data[key].append(torch.tensor(np.array(traj[key])))

    # Concatenate along time dimension (dim=0)
    for key in keys:
        flat_data[key] = torch.cat(flat_data[key], dim=0)
        if flat_data[key].ndim == 1:
            flat_data[key] = flat_data[key].unsqueeze(-1)

    # class labels
    points = flat_data["privileged_state"][:, :2]  # [B 2]

    # mask out points outside the square [-1, 1] x [-1, 1]
    mask = (
        (points[:, 0] >= -1)
        & (points[:, 0] <= 1)
        & (points[:, 1] >= -1)
        & (points[:, 1] <= 1)
    )
    flat_data = {k: v[mask] for k, v in flat_data.items()}
    points = flat_data["privileged_state"][:, :2]  # [B, 2]

    scaled = (points + 1) / 2

    # Convert to grid indices
    x_idx = (scaled[:, 0] * config.grid_size).clamp(0, max=config.grid_size - 1).long()
    y_idx = (scaled[:, 1] * config.grid_size).clamp(0, max=config.grid_size - 1).long()

    # label
    step = 2.0 / config.grid_size
    radius = step / 2 * 0.8
    coords = torch.linspace(
        -1 + step / 2, 1 - step / 2, config.grid_size, device=points.device
    )
    y_coords, x_coords = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    dists_squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
    mask = (dists_squared < radius**2).any(dim=1)

    # Row-major grid index: row * num_cols + col
    labels = y_idx * config.grid_size + x_idx
    labels = labels.to(torch.int64)
    labels[~mask] = (
        torch.max(labels) + 1
    )  # Assign a new class for points outside the square

    flat_data["label"] = labels.unsqueeze(-1)

    return flat_data


def visualize_data(flat_data, train_data, test_data):
    points = flat_data["privileged_state"][:, :2]  # [B, 2]
    labels = flat_data["label"].squeeze()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        points[:, 0].cpu().numpy(),
        points[:, 1].cpu().numpy(),
        c=labels,
        cmap=colormaps["rainbow"],
    )
    # add outline of square in the center
    square = plt.Rectangle(
        (-1, -1),
        2,
        2,
        color="black",
        alpha=0.5,
        fill=False,
        linewidth=2,
    )
    ax.add_patch(square)
    ax.set_title("Class Labels")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.tight_layout()
    wandb.log(
        {"class_labels": wandb.Image(fig), "num_updates": 0},
        step=0,
    )

    # Train/test split visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        train_data["privileged_state"][:, 0].cpu().numpy(),
        train_data["privileged_state"][:, 1].cpu().numpy(),
        c="blue",
        label="Train",
        alpha=0.5,
    )
    ax.scatter(
        test_data["privileged_state"][:, 0].cpu().numpy(),
        test_data["privileged_state"][:, 1].cpu().numpy(),
        c="red",
        label="Test",
        alpha=0.5,
    )
    ax.set_title("Train/Test Split")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.tight_layout()
    wandb.log(
        {"train_test_split": wandb.Image(fig), "num_updates": 0},
        step=0,
    )


# 2. Split at Timestep Level
def split_flat_data(flat_data, test_size=0.2, seed=42):
    points = flat_data["privileged_state"][:, :2]  # [B, 2]

    step = 2.0 / config.grid_size
    radius = step / 2
    coords = torch.linspace(
        -1 + step / 2, 1 - step / 2, config.grid_size, device=points.device
    )

    y_coords, x_coords = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)

    if config.train_test_split == "uni":
        mask = torch.rand(len(points)) > test_size
    elif config.train_test_split == "circ":
        dists_squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
        mask = (dists_squared < radius**2).any(dim=1)
    elif config.train_test_split == "sq":
        mask = (
            flat_data["label"].squeeze() != config.grid_size**2 // 2
        )  # Random square being test
    else:
        raise ValueError(f"Invalid train_test_split method: {config.train_test_split}")

    def extract(mask):
        return {k: v[mask] for k, v in flat_data.items()}

    return extract(mask), extract(torch.logical_not(mask))  # train, test data


# 3. Batch Generator
def timestep_batch_generator(data_dict, batch_size, shuffle=True):
    N = len(next(iter(data_dict.values())))
    indices = np.arange(N)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]

        yield {k: v[batch_idx] for k, v in data_dict.items()}


# Flatten trajectories to timestep-level data
offline_eps = flatten_trajectories(offline_eps)

# Split into train/test
train_data_labeled, test_data = split_flat_data(offline_eps, test_size=0.2)
visualize_data(offline_eps, train_data_labeled, test_data)

# Create batch generator
train_loader_labeled = timestep_batch_generator(
    train_data_labeled, batch_size=config.pa["sz_batch"], shuffle=True
)

# Get a batch
batch = next(train_loader_labeled)

for key, value in batch.items():
    print(f"{key}: {value.shape}")

# DML Losses
# criterion = losses.Proxy_Anchor(
#     nb_classes=config.nb_classes,
#     sz_embed=config.pa["sz_embedding"],
#     mrg=config.pa["mrg"],
#     alpha=config.pa["alpha"],
# ).cuda()

# Train Parameters
param_groups = [
    {
        "params": model.semantic_encoder.parameters(),  # Semantic encoder parameters
        "lr": float(config.pa["lr"]) * 1,
    },
    # {
    #     "params": criterion.parameters(),
    #     "lr": float(config.pa["lr"]) * 100,
    # },  # Just proxies
]
# Optimizer Setting
opt = torch.optim.AdamW(
    param_groups, lr=float(config.pa["lr"]), weight_decay=config.pa["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=config.pa["lr_decay_step"], gamma=config.pa["lr_decay_gamma"]
)

# Dataset Loader and Sampler
BS = config.pa["sz_batch"]  # batch size
BL = 1

# print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(config.pa["nb_epochs"]))
losses_list = []
pa_losses_list = []
ae_losses_list = []
best_epoch = 0
best_eval = -float("inf")

num_updates = 0

for epoch in tqdm(range(0, config.pa["nb_epochs"]), desc="Training Epochs", position=0):
    model.train()

    losses_per_epoch = {
        "loss": [],
        "pos_term": [],
        "neg_term": [],
        # "ae_loss": [],
    }
    auc_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    # TODO: implement warmup training if needed

    # # pbar = tqdm(enumerate(expert_loader))
    max_timesteps = 10_000  # Maximum number of timesteps to sample
    min_timesteps = 1_000  # Minimum number of timesteps to sample
    total_timesteps = len(train_data_labeled["discount"])

    if total_timesteps > max_timesteps:
        subset_indices = random.sample(range(total_timesteps), max_timesteps)
        subset = {k: v[subset_indices] for k, v in train_data_labeled.items()}
        train_loader_labeled = timestep_batch_generator(
            subset, batch_size=BS, shuffle=True
        )
        total_timesteps = max_timesteps
    elif (  # If small dataset and using unlabeled data
        total_timesteps < min_timesteps
    ):
        # Step 1: Include all original samples once
        all_indices = list(range(total_timesteps))
        extra_needed = min_timesteps - total_timesteps
        extra_indices = random.choices(all_indices, k=extra_needed)
        combined_indices = all_indices + extra_indices
        bootstrapped_subset = {
            k: v[combined_indices] for k, v in train_data_labeled.items()
        }

        train_loader_labeled = timestep_batch_generator(
            bootstrapped_subset, batch_size=BS, shuffle=True
        )
        total_timesteps = min_timesteps

    pbar = tqdm(
        enumerate(train_loader_labeled),
        total=total_timesteps // BS,
        position=1,
        leave=False,
    )

    # config.beta = np.linspace(0.2, 1.0, config.nb_epochs)[epoch]  # Linear increase of beta

    for batch_idx, data in pbar:
        # labels_gt = data["label"][:].to(device, dtype=torch.float32)

        image = data["image"].to(device)  # [B H W 3]
        state = data["privileged_state"][:, :2].to(device)  # [B, 2]
        diff = state.unsqueeze(0) - state.unsqueeze(1)  # [B, B, 2]
        dists = torch.norm(diff, dim=2)
        labels_gt = torch.clip(1 - 1 / (np.sqrt(2)) * dists, -1, 1)
        # labels_gt = torch.clip(1 - dists, -1, 1)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            semantic_features, feat_gt = model.semantic_embed(data)
            feat_pred = decoder(semantic_features)

            # Normalize along the embedding dimension
            sem_norm = F.normalize(
                semantic_features.squeeze(1), p=2, dim=1
            )  # Each row becomes unit norm

            # Compute cosine similarity via dot product → shape [B, B]
            cos_sim = sem_norm @ sem_norm.T

            # loss is MSE between labels gt and cosine similarity
            loss = F.mse_loss(cos_sim, labels_gt).to(torch.float16)
        # loss, pos_term, neg_term = criterion(
        #     X=semantic_features.float(),
        #     T=labels_gt.squeeze().cuda(),
        #     U=semantic_features_unlabeled_tensor.float()
        #     if config.pa["use_unlabeled_data"]  # and epoch >= 20
        #     else None,  # Warmup
        #     args=config.pa,
        # )
        # ae_loss = F.mse_loss(feat_pred.float(), feat_gt.float())
        # loss += ae_loss

        # if config.pa["use_unlabeled_data"]:
        #     wandb.log(
        #         {
        #             "train/data_ratio": semantic_features_unlabeled_tensor.shape[0]
        #             / semantic_features.shape[0]
        #         },
        #         step=num_updates,
        #     )

        # P = criterion.proxies.detach()  # Ensure P is in the same dtype as X
        semantic_features = einops.rearrange(
            semantic_features.float(), "B T Z -> (B T) Z"
        )  # Ensure X is in the correct shape

        # cos_sim_logits = F.softmax(
        #     F.linear(  # [(B T) N]
        #         losses.l2_norm(semantic_features), losses.l2_norm(P)
        #     ),
        #     dim=-1,
        # )

        # auc = roc_auc_score(
        #     y_true=einops.rearrange(labels_gt, "B T -> (B T)").cpu().numpy(),
        #     y_score=cos_sim_logits.detach().cpu().numpy(),
        #     multi_class="ovo",
        #     labels=np.arange(config.nb_classes),
        # )

        opt.zero_grad()
        loss.backward()
        num_updates += 1

        torch.nn.utils.clip_grad_value_(model.semantic_encoder.parameters(), 10)
        # torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch["loss"].append(loss.data.cpu().numpy())
        # losses_per_epoch["pos_term"].append(pos_term.data.cpu().numpy())
        # losses_per_epoch["neg_term"].append(neg_term.data.cpu().numpy())
        # losses_per_epoch["ae_loss"].append(ae_loss.data.cpu().numpy())
        # auc_per_epoch.append(auc)
        opt.step()

        pbar.set_description(
            "Train Epoch: {} Loss: {:.6f}".format(
                epoch,
                loss.item(),
            )
        )

    losses_list.append(np.mean(losses_per_epoch["loss"]))
    for key in losses_per_epoch:
        wandb.log(
            {
                f"train/{key}": np.mean(losses_per_epoch[key]),
                "num_updates": num_updates,
            },
            step=num_updates,
        )
    # wandb.log(
    #     {"train/AE Loss": np.mean(ae_losses_per_epoch), "num_updates": num_updates},
    #     step=num_updates,
    # )
    wandb.log(
        {"train/AUC": np.mean(auc_per_epoch), "num_updates": num_updates},
        step=num_updates,
    )

    scheduler.step()

    if epoch >= 0:
        metrics = {}
        # X = []
        y = []
        y_pred = []
        with torch.no_grad():
            test_loader = timestep_batch_generator(
                test_data, batch_size=config.pa["sz_batch"], shuffle=False
            )
            pbar = tqdm(
                enumerate(test_loader),
                total=len(test_data["discount"]) // config.pa["sz_batch"],
                desc="Evaluation",
                position=2,
                leave=False,
            )
            for batch_idx, data in pbar:
                # labels_gt = data["label"].to(device, dtype=torch.float32)  # [B, 1]
                state = data["privileged_state"][:, :2].to(device)  # [B, 2]
                diff = state.unsqueeze(0) - state.unsqueeze(1)  # [B, B, 2]
                dists = torch.norm(diff, dim=2)
                labels_gt = torch.clip(1 - 1 / (np.sqrt(2)) * dists, -1, 1)
                # labels_gt = torch.clip(1 - dists, -1, 1)

                semantic_features, __ = model.semantic_embed(data)  # [B, 1, 512]

                # Normalize all vectors for cosine similarity
                semantic_features_norm = F.normalize(
                    semantic_features, dim=-1
                )  # (10, 1, 512)

                # Compute cosine similarity
                sem_norm = F.normalize(
                    semantic_features.squeeze(1), p=2, dim=1
                )  # Each row becomes unit norm

                # Compute cosine similarity via dot product → shape [B, B]
                cos_sim = sem_norm @ sem_norm.T

                # loss is MSE between labels gt and cosine similarity
                loss = F.mse_loss(cos_sim, labels_gt).to(torch.float16)

                # X.append(cos_sim.flatten().cpu().numpy())
                y.append(labels_gt.flatten().cpu().numpy())
                y_pred.append(cos_sim.flatten().cpu().numpy())

            # X = einops.rearrange(np.concatenate(X, axis=0), "B T Z -> (B T) Z")
            y = np.concatenate(y, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            # Calculate metrics
            metrics["MSE"] = mean_squared_error(y, y_pred)
            # MAE
            metrics["MAE"] = mean_absolute_error(y, y_pred)
            # R^2 Score
            metrics["R^2"] = r2_score(y, y_pred)
            metrics["MSE Loss"] = loss.detach().cpu().numpy()
            # metrics["Cross Entropy Loss"].append(
            #     F.cross_entropy(cos_sim, gt_labels, reduction="mean").item()
            # )

        for key, value in metrics.items():
            metrics[key] = np.mean(value)
        wandb_log = {f"eval/{k}": v for k, v in metrics.items()}
        wandb_log["num_updates"] = num_updates
        wandb.log(wandb_log, step=num_updates)

        def UMAP_plot(X, max_samples=10_000):
            # ---- Optional Downsampling ----
            if max_samples is not None and X.shape[0] > max_samples:
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X = X[indices]
                y = y[indices]
            # ---- UMAP Setup ----
            umap_input = np.concatenate([X, proxies.detach().cpu().numpy()], axis=0)

            reducer = umap.UMAP(n_components=2, metric="cosine")
            umap_output = reducer.fit_transform(umap_input)

            X_umap = umap_output[: -config.nb_classes]
            proxies_umap = umap_output[-config.nb_classes :]

            # ---- Rainbow Color Setup ----
            cmap = colormaps.get_cmap("hsv")
            class_colors = {
                k: cmap(i / num_classes_eval) for i, k in enumerate(classes_eval)
            }

            # ---- Plot ----
            plt.figure(figsize=(8, 6))

            # Plot data points
            for class_idx, class_label in enumerate(classes_eval):
                idxs = y == class_label
                plt.scatter(
                    X_umap[idxs, 0],
                    X_umap[idxs, 1],
                    s=15,
                    color=class_colors[class_label],
                    label=f"Class {class_label} (data)",
                    alpha=0.7,
                )

            # Plot proxies on top
            for class_idx, class_label in enumerate(classes_eval):
                proxy = proxies_umap[class_idx]

                plt.scatter(
                    proxy[0],
                    proxy[1],
                    color=class_colors[class_label],
                    marker="X",
                    s=100,
                    edgecolor="black",
                    linewidth=1.2,
                    label=f"Class {class_label} (proxy)",
                    alpha=1.0,
                )

            # ---- Final Formatting ----
            plt.title("UMAP visualization of embeddings (cosine distance)")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.legend(loc="best", fontsize=8, frameon=True)
            plt.tight_layout()

            wandb.log(
                {"umap_plot": wandb.Image(plt), "num_updates": num_updates},
                step=num_updates,
            )
            plt.close()

        # if epoch % 10 == 0:
        #     UMAP_plot(
        #         X,
        #         y,
        #         criterion.proxies,
        #         num_classes_eval,
        #         classes_eval,
        #         max_samples=None,
        #     )

        # with torch.no_grad():
        #     model.proxies.copy_(criterion.proxies)

        if config.pa["save_model"]:
            model_name = config.wandb_name

            save_path = "logs/checkpoints_pa"
            # check if the directory exists, if not create it
            os.makedirs(save_path, exist_ok=True)

            torch.save(
                model.state_dict(),
                f"{save_path}/encoder_{model_name}.pth",
            )
            tqdm.write(f"Model saved to {save_path}/encoder_{model_name}.pth")

            if True:  # balanced_accuracy < best_eval:
                # best_eval = balanced_accuracy
                # print(f"New best at iter {i}, saving model.")
                torch.save(
                    model.state_dict(),
                    f"logs/checkpoints_pa/best_encoder_{model_name}.pth",
                )
