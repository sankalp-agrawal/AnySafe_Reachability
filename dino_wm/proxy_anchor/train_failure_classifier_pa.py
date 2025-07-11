import argparse
import copy
import os
import random
import sys

import einops
import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap.umap_ as umap
import wandb
from dino_wm.dino_models import VideoTransformer, normalize_acs
from dino_wm.test_loader import SplitTrajectoryDataset
from proxy_anchor.code import losses
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import *
from viz_traj_cosine_sim import data_from_traj

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


def make_parser():
    parser = argparse.ArgumentParser(
        description="Official implementation of `Proxy Anchor Loss for Deep Metric Learning`"
        + "Our code is modified from `https://github.com/dichotomies/proxy-nca`"
    )
    # export directory, training and val datasets, test datasets
    parser.add_argument(
        "--embedding-size",
        default=512,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch-size",
        default=150,
        type=int,
        dest="sz_batch",
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        default=60,
        type=int,
        dest="nb_epochs",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gpu-id", default=0, type=int, help="ID of GPU that is used for training."
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument("--model", default="bn_inception", help="Model for training")
    parser.add_argument("--loss", default="Proxy_Anchor", help="Criterion for training")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--lr-decay-step", default=10, type=int, help="Learning decay step setting"
    )
    parser.add_argument(
        "--lr-decay-gamma", default=0.5, type=float, help="Learning decay gamma setting"
    )
    parser.add_argument(
        "--alpha", default=32, type=float, help="Scaling Parameter setting"
    )
    parser.add_argument(
        "--mrg", default=0.1, type=float, help="Margin parameter setting"
    )
    parser.add_argument("--IPC", type=int, help="Balanced sampling, images per class")
    parser.add_argument("--warm", default=1, type=int, help="Warmup training epochs")
    parser.add_argument(
        "--bn-freeze", default=1, type=int, help="Batch normalization parameter freeze"
    )
    parser.add_argument("--l2-norm", default=1, type=int, help="L2 normlization")
    parser.add_argument("--remark", default="", help="Any remark")
    parser.add_argument(
        "--dont-save-model",
        dest="save_model",
        action="store_false",
        help="Don't save model",
    )
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Wandb Initialization
wandb.init(name=f"mrg_{args.mrg}_alpha_{args.alpha}", project="ProxyAnchor")
wandb.config.update(args)

# Dataset Loader and Sampler
BS = args.sz_batch  # batch size
BL = 4

hdf5_file = "/home/sunny/data/skittles/consolidated.h5"
hdf5_file_test = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"

expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)
expert_data_eval = SplitTrajectoryDataset(hdf5_file_test, BL, split="train", num_test=0)

expert_loader = DataLoader(expert_data, batch_size=BS, shuffle=True)
expert_loader_eval = DataLoader(expert_data_eval, batch_size=10, shuffle=True)

device = "cuda:0"

nb_classes = 2  # Safe and Failure

# Backbone Model
LOG_DIR = "logs_pa"

model = VideoTransformer(
    image_size=(224, 224),
    dim=384,  # DINO feature dimension
    ac_dim=10,  # Action embedding dimension
    state_dim=8,  # State dimension
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=BL - 1,
    dropout=0.1,
).to(device)
model.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
# model.load_state_dict(torch.load("../checkpoints_pa/encoder_0.1.pth"))

for name, param in model.named_parameters():
    param.requires_grad = name.startswith("semantic_encoder")

# DML Losses
criterion = losses.Proxy_Anchor(
    nb_classes=nb_classes,
    sz_embed=args.sz_embedding,
    mrg=args.mrg,
    alpha=args.alpha,
).cuda()

# Train Parameters
param_groups = [
    {
        "params": model.semantic_encoder.parameters(),  # Semantic encoder parameters
        "lr": float(args.lr) * 1,
    },
    {"params": criterion.parameters(), "lr": float(args.lr) * 100},  # Just proxies
]
# Optimizer Setting
opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

database = {}
with h5py.File(hdf5_file_test, "r") as hf:
    trajectory_ids = list(hf.keys())
    database = {
        i: data_from_traj(hf[traj_id]) for i, traj_id in enumerate(trajectory_ids)
    }

constraint1 = {
    "wrist": database[7]["robot0_eye_in_hand_image"][82],
    "front": database[7]["agentview_image"][82],
}  # weak unsafe frame
constraint2 = {
    "wrist": database[1]["robot0_eye_in_hand_image"][108],
    "front": database[1]["agentview_image"][108],
}  # unsafe frame


print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_epoch = 0
best_eval = -float("inf")

for epoch in range(0, args.nb_epochs):
    model.train()

    losses_per_epoch = []
    auc_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    # TODO: implement warmup training if needed

    max_trajectories = 10_000  # Maximum number of trajectories to sample
    total_trajectories = len(expert_data)
    subset_indices = random.sample(range(total_trajectories), max_trajectories)
    subset = Subset(expert_data, subset_indices)
    loader_subset = DataLoader(subset, batch_size=BS, shuffle=True)

    # pbar = tqdm(enumerate(expert_loader))
    pbar = tqdm(enumerate(loader_subset), total=len(loader_subset))

    for batch_idx, data in pbar:
        labels_gt = data["failure"][:, 1:].to(device, dtype=torch.float32)

        data1 = data["cam_zed_embd"].to(device)  # [B BL, 256, 384]
        data2 = data["cam_rs_embd"].to(device)  # [B BL, 256, 384]
        inputs1 = data1[:, :-1]  # [B BL-1, 256, 384]
        inputs2 = data2[:, :-1]  # [B BL-1, 256, 384]

        data_state = data["state"].to(device)
        states = data_state[:, :-1]  # [B BL-1, 8]

        data_acs = data["action"].to(device)
        acs = data_acs[:, :-1]  # [B BL-1, 10]
        acs = normalize_acs(acs, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            __, __, __, __, semantic_features = model(inputs1, inputs2, states, acs)

        unsafe_weak_mask = labels_gt == 2.0
        labels_gt_masked = copy.deepcopy(labels_gt)
        labels_gt_masked[unsafe_weak_mask] = 1.0  # Set

        loss = criterion(semantic_features.float(), labels_gt_masked.squeeze().cuda())

        P = criterion.proxies.detach()  # Ensure P is in the same dtype as X
        semantic_features = einops.rearrange(
            semantic_features.float(), "B T Z -> (B T) Z"
        )  # Ensure X is in the correct shape

        cos_sim_fail = F.linear(losses.l2_norm(semantic_features), losses.l2_norm(P))[
            :, -1
        ]

        if nb_classes == 2:
            auc = roc_auc_score(
                y_true=einops.rearrange(labels_gt_masked, "B T -> (B T)").cpu().numpy(),
                y_score=cos_sim_fail.detach().cpu().numpy(),
            )

        else:
            raise notImplementedError(
                "AUC calculation for more than 2 classes is not implemented."
            )
            metrics["AUC"].append(
                roc_auc_score(
                    y_true=losses.binarize(
                        einops.rearrange(labels_gt_masked, "B T -> (B T)"),
                        nb_classes=nb_classes,
                    )
                    .cpu()
                    .numpy(),
                    y_score=einops.rearrange(logits, "B T L -> (B T) L").cpu().numpy(),
                    multi_class="ovr",
                    average="macro",
                )
            )

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.semantic_encoder.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        auc_per_epoch.append(auc)
        opt.step()

        pbar.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch,
                batch_idx + 1,
                len(loader_subset),
                100.0 * batch_idx / len(loader_subset),
                loss.item(),
            )
        )

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({"train/Proxy Anchor Loss": losses_list[-1]})
    wandb.log({"train/AUC": np.mean(auc_per_epoch)})

    scheduler.step()

    if epoch >= 0:
        for data, constraint, t in zip(
            [database[7], database[1]], [constraint1, constraint2], [82, 108]
        ):
            inputs2 = data["cam_rs_embd"][t : BL - 1 + t, :].to(device).unsqueeze(0)
            inputs1 = data["cam_zed_embd"][t : BL - 1 + t, :].to(device).unsqueeze(0)
            acs = data["action"][t : BL - 1 + t, :].to(device).unsqueeze(0)
            acs = normalize_acs(acs, device=device)
            states = data["state"][t : BL - 1 + t, :].to(device).unsqueeze(0)

            __, __, __, __, semantic_feat = model(
                inputs1,
                inputs2,
                states,
                acs,
            )
            constraint.update({"semantic_feat": semantic_feat.squeeze()[-1]})

        accuracies = []
        metrics = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1-score": [],
            "Balanced Accuracy": [],
            "Proxy Anchor Loss": [],
            # "Cross Entropy Loss": [],
        }
        X = []
        y = []
        with torch.no_grad():
            print("**Evaluating...**")
            for batch_idx, data in enumerate(expert_loader_eval):
                # print(batch_idx)
                labels_gt = data["failure"][:, 1:].to(device, dtype=torch.float32)

                data1 = data["cam_zed_embd"].to(device)
                data2 = data["cam_rs_embd"].to(device)
                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = data["state"].to(device)
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = data["action"].to(device)
                acs = data_acs[:, :-1]
                acs = normalize_acs(acs, device)

                __, __, __, __, semantic_features = model(inputs1, inputs2, states, acs)

                unsafe_weak_mask = labels_gt == 2.0
                labels_gt_masked = copy.deepcopy(labels_gt)
                labels_gt_masked[unsafe_weak_mask] = 1.0  # Set

                # Normalize all vectors for cosine similarity
                semantic_features_norm = F.normalize(
                    semantic_features, dim=-1
                )  # (10, 3, 512)
                proxies_norm = F.normalize(criterion.proxies, dim=-1)  # (2, 512)

                # Broadcastable shapes: (10, 3, 1, 512) and (1, 1, 2, 512)
                semantic_features_exp = einops.rearrange(
                    semantic_features_norm, "B T Z -> B T 1 Z"
                )  # (10, 3, 1, 512)
                proxies_exp = einops.rearrange(
                    proxies_norm, "L Z -> 1 1 L Z"
                )  # (1, 1, 2, 512)

                # Compute cosine similarity
                cos_sim = (semantic_features_exp * proxies_exp).sum(
                    dim=-1
                )  # (10, 3, 2)

                # Choose the index (0 or 1) of the most similar vector
                logits = F.softmax(cos_sim, dim=-1)  # (10, 3, 2)
                pred_labels = logits.argmax(dim=-1)  # (10, 3)

                pred_labels = (
                    einops.rearrange(pred_labels, "B T -> (B T)").cpu().numpy()
                )  # Flatten
                gt_labels = (
                    einops.rearrange(labels_gt_masked, "B T -> (B T)").cpu().numpy()
                )  # Flatten
                cos_sim = einops.rearrange(cos_sim, "B T L -> (B T) L").cpu().numpy()

                X.append(semantic_features.cpu().numpy())
                y.append(labels_gt.cpu().numpy())

                accuracies.append(accuracy_score(gt_labels, pred_labels))

                # Calculate metrics
                metrics["Accuracy"].append(accuracy_score(gt_labels, pred_labels))
                metrics["Precision"].append(
                    precision_score(
                        gt_labels, pred_labels, average="macro", zero_division=0
                    )
                )
                metrics["Recall"].append(
                    recall_score(
                        gt_labels, pred_labels, average="macro", zero_division=0
                    )
                )
                metrics["F1-score"].append(
                    f1_score(gt_labels, pred_labels, average="macro", zero_division=0)
                )
                metrics["Balanced Accuracy"].append(accuracies[-1])
                metrics["Proxy Anchor Loss"].append(
                    criterion(
                        X=semantic_features.float(), T=labels_gt_masked.squeeze().cuda()
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                # metrics["Cross Entropy Loss"].append(
                #     F.cross_entropy(cos_sim, gt_labels, reduction="mean").item()
                # )

        balanced_accuracy = np.mean(accuracies)

        X = einops.rearrange(np.concatenate(X, axis=0), "B T Z -> (B T) Z")
        y = einops.rearrange(np.concatenate(y, axis=0), "B T -> (B T)")
        num_classes_eval = len(np.unique(y))

        cosine_sims = {
            i: {j: [] for j in range(i, num_classes_eval)}
            for i in range(num_classes_eval)
        }

        def cosine_sim_plot_eval(X, y):
            y_masked = copy.deepcopy(y)
            y_masked[y == 2] = 1  # Set weak unsafe to unsafe

            X_class = {
                k: X[y_masked == k]
                / (np.linalg.norm(X[y_masked == k], axis=1, keepdims=True) + 1e-8)
                for k in np.unique(y_masked)
            }

            fig, ax = plt.subplots(figsize=(10, 8))
            class_to_label = {0: "Safe", 1: "Fail", 2: "Weak Fail"}

            plt.title("Cosine Similarity Distribution per Class")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Density")

            class_pairs = []
            for i in range(len(np.unique(y_masked))):
                for j in range(i, len(np.unique(y_masked))):
                    class_pairs.append((i, j))

            cmap = plt.cm.rainbow
            colors = [cmap(i / len(class_pairs)) for i in range(len(class_pairs))]

            for idx, (i, j) in enumerate(class_pairs):
                cos_sim = X_class[i] @ X_class[j].T

                if i == j:  # Avoid self-comparison and double counting
                    mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
                    cos_sim = np.where(mask, cos_sim, -2.0)

                cos_sim = cos_sim[cos_sim != -2.0].flatten()

                if len(cos_sim) > 1000:
                    cos_sim_sampled = np.random.choice(cos_sim, 1000, replace=False)
                else:
                    cos_sim_sampled = cos_sim

                kde_cs = gaussian_kde(cos_sim_sampled)
                x_cs = np.linspace(cos_sim.min() - 1e-3, 1, 1000)
                y_pdf = kde_cs(x_cs)

                color = colors[idx]
                label = f"{class_to_label[i]}-{class_to_label[j]}"
                ax.plot(x_cs, y_pdf, label=label, color=color)

                # Statistics
                mean_val = np.mean(cos_sim_sampled)
                lower, upper = np.percentile(cos_sim_sampled, [2.5, 97.5])

                # Get KDE values at stat locations
                mean_y = kde_cs(mean_val)
                lower_y = kde_cs(lower)
                upper_y = kde_cs(upper)

                # Plot short vertical lines
                ax.vlines(
                    mean_val, 0, mean_y, color=color, linestyle="dashed", alpha=0.8
                )
                ax.vlines(lower, 0, lower_y, color=color, linestyle="dotted", alpha=0.5)
                ax.vlines(upper, 0, upper_y, color=color, linestyle="dotted", alpha=0.5)

                # Text labels with white background
                label_kwargs = dict(
                    ha="center",
                    fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0),
                )
                ax.text(
                    mean_val,
                    mean_y + 0.01,
                    f"μ={mean_val:.2f}",
                    color=color,
                    **label_kwargs,
                )
                ax.text(
                    lower, lower_y + 0.01, f"↓{lower:.2f}", color=color, **label_kwargs
                )
                ax.text(
                    upper, upper_y + 0.01, f"↑{upper:.2f}", color=color, **label_kwargs
                )

            # Add dummy lines for legend explanation
            ax.plot([], [], linestyle="dashed", color="black", label="μ = Mean")
            ax.plot(
                [], [], linestyle="dotted", color="black", label="↓ ↑ = 95% Interval"
            )

            ax.legend()
            plt.tight_layout()
            wandb.log({"eval/cosine_sim_plot": wandb.Image(fig)})

        cosine_sim_plot_eval(X, y)

        def TP_TN_plot(X, y, const1, const2):
            y_masked = copy.deepcopy(y)
            y_masked[y == 2] = 1  # Set weak unsafe to unsafe

            P = criterion.proxies.detach()  # Ensure P is in the same dtype as X

            cos_sim_fail = -F.linear(
                losses.l2_norm(torch.tensor(X, device=P.device).float()),
                losses.l2_norm(P),
            )[:, -1]

            cos_sim_const1 = -F.linear(
                losses.l2_norm(torch.tensor(X, device=P.device).float()),
                losses.l2_norm(const1["semantic_feat"].unsqueeze(0).float()),
            )[:, -1].detach()

            cos_sim_const2 = -F.linear(
                losses.l2_norm(torch.tensor(X, device=P.device).float()),
                losses.l2_norm(const2["semantic_feat"].unsqueeze(0).float()),
            )[:, -1].detach()

            thresholds = np.linspace(-1, 1, 100)
            fail_proxy_data = {
                "cos_sim": cos_sim_fail.cpu().numpy(),
                "tp_rates": [],
                "tn_rates": [],
                "fp_rates": [],
                "fn_rates": [],
            }
            const1_data = {
                "cos_sim": cos_sim_const1.cpu().numpy(),
                "tp_rates": [],
                "tn_rates": [],
                "fp_rates": [],
                "fn_rates": [],
            }
            const2_data = {
                "cos_sim": cos_sim_const2.cpu().numpy(),
                "tp_rates": [],
                "tn_rates": [],
                "fp_rates": [],
                "fn_rates": [],
            }
            for t in thresholds:
                for data in [fail_proxy_data, const1_data, const2_data]:
                    cos_sim = data["cos_sim"]

                    tp = ((cos_sim > t) & (y_masked == 0)).sum()
                    fp = ((cos_sim > t) & (y_masked == 1)).sum()
                    tn = ((cos_sim <= t) & (y_masked == 1)).sum()
                    fn = ((cos_sim <= t) & (y_masked == 0)).sum()

                    data["tp_rates"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    data["tn_rates"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    data["fp_rates"].append(fp / (fp + tn) if (fp + tn) > 0 else 0)
                    data["fn_rates"].append(fn / (fn + tp) if (fn + tp) > 0 else 0)

            for data in [fail_proxy_data, const1_data, const2_data]:
                data["tp_rates"] = np.array(data["tp_rates"])
                data["tn_rates"] = np.array(data["tn_rates"])
                thresholds = np.array(thresholds)
                diff = np.abs(data["tp_rates"] - data["tn_rates"])
                intersect_idx = np.argmin(diff)
                intersect_threshold = thresholds[intersect_idx]
                intersect_value = data["tp_rates"][
                    intersect_idx
                ]  # or tn_rates[intersect_idx]
                data["intersect_threshold"] = intersect_threshold
                data["intersect_value"] = intersect_value

            # Plot all the metrics
            fig, axes = plt.subplots(1, 3, figsize=(30, 8))
            for ax, (data, title) in zip(
                axes,
                [
                    (fail_proxy_data, "Fail Proxy"),
                    (const1_data, "Constraint 1 Conditioned"),
                    (const2_data, "Constraint 2 Conditioned"),
                ],
            ):
                ax.set_aspect("equal")
                thresholds = np.array(thresholds)

                ax.set_title(title)

                ax.plot(
                    thresholds,
                    data["tp_rates"],
                    label="True Positive Rate",
                    color="blue",
                )
                ax.plot(
                    thresholds,
                    data["tn_rates"],
                    label="True Negative Rate",
                    color="orange",
                )
                # ax.plot(thresholds, fp_rates, label="False Positive Rate", color="red")
                # ax.plot(thresholds, fn_rates, label="False Negative Rate", color="green")

                # Add vertical line and label at intersection
                ax.axvline(
                    data["intersect_threshold"],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )
                ax.text(
                    data["intersect_threshold"],
                    0.05,  # slightly above bottom
                    f"Threshold = {data['intersect_threshold']:.2f}, TPR = {data['intersect_value']:.2f}",
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    backgroundcolor="white",
                    fontsize=9,
                )

                ax.set_xlabel("Cosine Similarity Threshold")
                ax.set_ylabel("Rate")
                ax.legend()
            plt.tight_layout()

            wandb.log({"eval/metric_plot": wandb.Image(fig)})

            # Plot AUC curve
            fig, axes = plt.subplots(1, 3, figsize=(30, 8))
            for ax, (data, title) in zip(
                axes,
                [
                    (fail_proxy_data, "Fail Proxy"),
                    (const1_data, "Constraint 1 Conditioned"),
                    (const2_data, "Constraint 2 Conditioned"),
                ],
            ):
                ax.set_aspect("equal")
                fp_rates = np.array(data["fp_rates"])
                tp_rates = np.array(data["tp_rates"])
                ax.plot(fp_rates, tp_rates, label="ROC Curve", color="blue")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(title)
                ax.legend()
            plt.tight_layout()
            wandb.log({"eval/roc_curve": wandb.Image(fig)})
            return cos_sim_fail

        cos_sim_fail = TP_TN_plot(X, y, constraint1, constraint2)

        if nb_classes == 2:
            y_masked = copy.deepcopy(y)
            y_masked[y == 2] = 1
            auc = roc_auc_score(
                y_true=y_masked,
                y_score=-cos_sim_fail.cpu().numpy(),
            )
            wandb.log({"eval/AUC": auc})

        else:
            auc = roc_auc_score(
                y_true=losses.binarize(
                    einops.rearrange(labels_gt_masked, "B T -> (B T)"),
                    nb_classes=nb_classes,
                )
                .cpu()
                .numpy(),
                y_score=einops.rearrange(logits, "B T L -> (B T) L").cpu().numpy(),
                multi_class="ovr",
                average="macro",
            )
            wandb.log({"eval/AUC": auc})

        for key, value in metrics.items():
            metrics[key] = np.mean(value)
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        # ---- Flatten and Prepare Data ----
        if len(X) == 0 or len(y) == 0:
            print("No data to visualize.")
            continue

        print("Visualizing embeddings with UMAP...")

        # ---- UMAP Setup ----
        umap_input = np.concatenate(
            [X, criterion.proxies.detach().cpu().numpy()], axis=0
        )

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        umap_output = reducer.fit_transform(umap_input)

        X_umap = umap_output[:-nb_classes]
        proxies_umap = umap_output[-nb_classes:]

        # ---- Rainbow Color Setup ----
        cmap = cm.get_cmap("hsv")
        class_colors = [cmap(i / num_classes_eval) for i in range(num_classes_eval)]

        # ---- Plot ----
        plt.figure(figsize=(8, 6))

        # Plot data points
        for class_idx in range(num_classes_eval):
            idxs = y == class_idx
            plt.scatter(
                X_umap[idxs, 0],
                X_umap[idxs, 1],
                s=15,
                color=class_colors[class_idx],
                label=f"Class {class_idx} (data)",
                alpha=0.7,
            )

        # Plot proxies
        for i, proxy in enumerate(proxies_umap):
            plt.scatter(
                proxy[0],
                proxy[1],
                color=class_colors[i],
                marker="X",
                s=100,
                edgecolor="black",
                linewidth=1.2,
                label=f"Class {i} (proxy)",
                alpha=1.0,
            )

        # ---- Final Formatting ----
        plt.title("UMAP visualization of embeddings (cosine distance)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(loc="best", fontsize=8, frameon=True)
        plt.tight_layout()

        wandb.log({"umap_plot": wandb.Image(plt)})

        with torch.no_grad():
            model.proxies.copy_(criterion.proxies)

        if args.save_model:
            torch.save(
                model.state_dict(),
                f"../checkpoints_pa/encoder_{args.mrg}.pth",
            )

            if balanced_accuracy < best_eval:
                best_eval = balanced_accuracy
                print(f"New best at iter {i}, saving model.")
                torch.save(
                    model.state_dict(),
                    f"../checkpoints/best_encoder_{args.mrg}.pth",
                )
