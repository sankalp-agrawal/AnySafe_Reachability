import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add directories to system path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
        os.path.join(base_dir, "model_based_irl_torch"),
        os.path.join(base_dir, "real_envs"),
    ]
)

# Load model
print(sys.path)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

# Import custom modules
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs
from test_loader import SplitTrajectoryDataset


def transition_from_data(data, transition, device, use_amp=True):
    data1 = data["cam_zed_embd"].to(device)
    data2 = data["cam_rs_embd"].to(device)

    inputs1 = data1[:, :-1]
    inputs2 = data2[:, :-1]

    states = data["state"].to(device)[:, :-1]
    acs = normalize_acs(data["action"].to(device)[:, :-1], device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
        with torch.no_grad():
            pred1, pred2, pred_state, pred_fail, semantic_feat = transition(
                inputs1, inputs2, states, acs
            )

    return pred1, pred2, pred_state, pred_fail, semantic_feat


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform1 = transforms.Compose(
    [
        transforms.Resize(520),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    BS, BL = 16, 4
    hdf5_file = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)
    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))

    device = "cuda:0"

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,
        ac_dim=10,
        state_dim=8,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    # transition.load_state_dict(torch.load("checkpoints_pa/best_encoder.pth"))
    transition.eval()

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("checkpoints/testing_decoder.pth"))
    decoder.eval()

    data = next(expert_loader)
    pred1, pred2, pred_state, pred_fail, semantic_feat = transition_from_data(
        data, transition, device
    )

    def init_nested_dict():
        return {
            "safe": {"safe": [], "unsafe": [], "weak_unsafe": []},
            "unsafe": {"unsafe": [], "weak_unsafe": []},
            "weak_unsafe": {"weak_unsafe": []},
        }

    cosine_similarities_cam_1 = init_nested_dict()
    cosine_similarities_cam_2 = copy.deepcopy(cosine_similarities_cam_1)
    euc_distances_cam_1 = copy.deepcopy(cosine_similarities_cam_1)
    euc_distances_cam_2 = copy.deepcopy(cosine_similarities_cam_1)

    cosine_similarities_sem = init_nested_dict()

    for _ in tqdm(range(len(expert_loader) - 1), desc="Computing cosine similarities"):
        data = next(expert_loader)
        pred1, pred2, pred_state, pred_fail, semantic_feat = transition_from_data(
            data, transition, device
        )
        embed_cam_1 = data["cam_zed_embd"].to(device)
        embed_cam_2 = data["cam_rs_embd"].to(device)

        semantic_data = semantic_feat.to(device)

        masks = {
            "safe": data["failure"][:, 0] == 0.0,
            "unsafe": data["failure"][:, 0] == 1.0,
            "weak_unsafe": data["failure"][:, 0] == 2.0,
        }

        embed_cam_1_class = {k: embed_cam_1[m] for k, m in masks.items()}
        embed_cam_2_class = {k: embed_cam_2[m] for k, m in masks.items()}
        semantic_feat_class = {k: semantic_data[m] for k, m in masks.items()}

        for k1 in cosine_similarities_cam_1:
            for k2 in cosine_similarities_cam_1[k1]:
                if len(embed_cam_1_class[k1]) > 0 and len(embed_cam_1_class[k2]) > 0:
                    for cam_data, cos_dict, euc_dict in [
                        (
                            embed_cam_1_class,
                            cosine_similarities_cam_1,
                            euc_distances_cam_1,
                        ),
                        (
                            embed_cam_2_class,
                            cosine_similarities_cam_2,
                            euc_distances_cam_2,
                        ),
                    ]:
                        anchors = torch.norm(cam_data[k1][:, -1], dim=-2)
                        queries = torch.norm(cam_data[k2][:, -1], dim=-2)

                        anchors_norm = F.normalize(anchors, p=2, dim=1)
                        queries_norm = F.normalize(queries, p=2, dim=1)

                        # Compute cosine similarity
                        cos_sim_matrix = anchors_norm @ queries_norm.T
                        if k1 == k2:  # Avoid self-comparison
                            cos_sim_matrix.fill_diagonal_(-2.0)
                        cos_sim = cos_sim_matrix[cos_sim_matrix != -2.0].flatten()
                        cos_dict[k1][k2].append(cos_sim.cpu().numpy())

                        # Compute Euclidean distances
                        euc_dist = torch.norm(
                            anchors.unsqueeze(1) - queries.unsqueeze(0), dim=2
                        )  # [N, M]

                        if k1 == k2:  # Avoid self-comparison
                            euc_dist.fill_diagonal_(-1.0)
                        euc_dist = euc_dist[euc_dist != -1.0]
                        euc_dict[k1][k2].append(euc_dist.cpu().numpy())

                    # Compute cosine similarities for semantic embeddings
                    anchors_norm = F.normalize(
                        semantic_feat_class[k1][:, -1], p=2, dim=1
                    )  # [N, Z]
                    queries_norm = F.normalize(
                        semantic_feat_class[k2][:, -1], p=2, dim=1
                    )  # [M, Z]

                    # Compute cosine similarity as dot product of normalized vectors
                    cos_sim_matrix = anchors_norm @ queries_norm.T

                    if k1 == k2:  # Avoid self-comparison
                        cos_sim_matrix.fill_diagonal_(-2.0)
                    cos_sim_sem = cos_sim_matrix[cos_sim_matrix != -2.0].flatten()
                    cosine_similarities_sem[k1][k2].append(cos_sim_sem.cpu().numpy())

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))
    fig.suptitle("DINO WM")
    titles = [
        "Cos Sim - Front Cam",
        "Cos Sim - Wrist Cam",
        "Euc Dist - Front Cam",
        "Euc Dist - Wrist Cam",
        "Cos Sim - Sem Feat",
        "Euc Dist - Sem Feat",
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_ylabel("Probability Density")

    for k1 in cosine_similarities_cam_1:
        for k2 in cosine_similarities_cam_1[k1]:
            if cosine_similarities_cam_1[k1][k2]:
                if k1 == "weak_unsafe" or k2 == "weak_unsafe":
                    continue
                cs1 = np.concatenate(cosine_similarities_cam_1[k1][k2])
                cs2 = np.concatenate(cosine_similarities_cam_2[k1][k2])
                ed1 = np.concatenate(euc_distances_cam_1[k1][k2])
                ed2 = np.concatenate(euc_distances_cam_2[k1][k2])
                sem_cs = np.concatenate(cosine_similarities_sem[k1][k2])

                kde_cs1, kde_cs2 = gaussian_kde(cs1), gaussian_kde(cs2)
                kde_ed1, kde_ed2 = gaussian_kde(ed1), gaussian_kde(ed2)
                kde_sem = gaussian_kde(sem_cs)

                x_cs = np.linspace(cs1.min() - 1e-3, 1, 1000)
                x_ed = np.linspace(ed1.min() - 1, ed1.max() + 1, 1000)
                x_sem = np.linspace(sem_cs.min() - 1e-3, 1, 1000)

                axes[0, 0].plot(x_cs, kde_cs1(x_cs), label=f"{k1}-{k2}")
                axes[0, 1].plot(x_cs, kde_cs2(x_cs), label=f"{k1}-{k2}")
                axes[1, 0].plot(x_ed, kde_ed1(x_ed), label=f"{k1}-{k2}")
                axes[1, 1].plot(x_ed, kde_ed2(x_ed), label=f"{k1}-{k2}")
                axes[2, 0].plot(x_sem, kde_sem(x_sem), label=f"{k1}-{k2}")

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique = dict(zip(labels, handles))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
    )
    plt.savefig("latent_analysis.png")
