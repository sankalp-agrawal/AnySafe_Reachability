import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# add to os sys path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../model_based_irl_torch")
)
sys.path.append(dreamer_dir)
env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../real_envs"))
sys.path.append(env_dir)
print(dreamer_dir)
print(sys.path)
from torch.utils.data import DataLoader

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

import random

import torch

# from dino_decoders_official import VQVAE
from dino_decoder import VQVAE
from torchvision import transforms

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
        transforms.CenterCrop(518),  # should be multiple of model patch_size
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)


DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        # transforms.CenterCrop(224), #should be multiple of model patch_size
        transforms.ToTensor(),
    ]
)
norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

import torch
from dino_models import VideoTransformer, normalize_acs
from test_loader import SplitTrajectoryDataset

if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16  # batch size
    BL = 4
    hdf5_file = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)

    expert_loader = iter(DataLoader(expert_data, batch_size=16, shuffle=True))

    device = "cuda:0"
    H = 3

    # decoder = Decoder().to(device)
    # decoder.load_state_dict(torch.load('checkpoints/best_decoder_10m.pth'))
    # decoder.eval()

    transition = VideoTransformer(
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
    # transition.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/claude_zero_wfail20500_rotvec.pth'))
    # transition.load_state_dict(torch.load("checkpoints/best_classifier.pth"))

    decoder = VQVAE().to(device)
    # decoder.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/best_decoder_10m.pth'))
    decoder.load_state_dict(torch.load("checkpoints/testing_decoder.pth"))
    decoder.eval()

    data = next(expert_loader)

    data1 = data[
        "cam_zed_embd"
    ].to(
        device
    )  # [transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
    data2 = data[
        "cam_rs_embd"
    ].to(
        device
    )  # transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]

    inputs2 = data2[:, :-1]
    output2 = data2[:, 1:]

    data_state = data["state"].to(device)
    states = data_state[:, :-1]
    output_state = data_state[:, 1:]

    data_acs = data["action"].to(device)
    acs = data_acs[:, :-1]
    acs = normalize_acs(acs, device=device)

    print(data.keys())

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            pred1, pred2, pred_state, pred_fail = transition(
                inputs1, inputs2, states, acs
            )
    transition.eval()

    cosine_similarities_cam_1 = {
        "safe": {
            "safe": [],
            "unsafe": [],
            "weak_unsafe": [],
        },
        "unsafe": {
            "unsafe": [],
            "weak_unsafe": [],
        },
        "weak_unsafe": {
            "weak_unsafe": [],
        },
    }
    cosine_similarities_cam_2 = copy.deepcopy(cosine_similarities_cam_1)

    euc_distances_cam_1 = copy.deepcopy(cosine_similarities_cam_1)
    euc_distances_cam_2 = copy.deepcopy(cosine_similarities_cam_1)

    for __ in tqdm(range(len(expert_loader) - 1), desc="Computing cosine similarities"):
        data = next(expert_loader)

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

        with torch.no_grad():
            pred1, pred2, pred_state, __ = transition(inputs1, inputs2, states, acs)

        safe_mask = data["failure"][:, 0] == 0.0
        unsafe_mask = data["failure"][:, 0] == 1.0
        weak_unsafe_mask = data["failure"][:, 0] == 2.0

        pred1_class = {
            "safe": pred1[safe_mask],
            "unsafe": pred1[unsafe_mask],
            "weak_unsafe": pred1[weak_unsafe_mask],
        }
        pred2_class = {
            "safe": pred2[safe_mask],
            "unsafe": pred2[unsafe_mask],
            "weak_unsafe": pred2[weak_unsafe_mask],
        }
        for key in cosine_similarities_cam_1.keys():
            for key2 in cosine_similarities_cam_1[key].keys():
                if len(pred1_class[key]) > 0 and len(pred1_class[key2]) > 0:
                    # if key != key2:
                    #     import ipdb

                    #     ipdb.set_trace()
                    pred_anchors = torch.norm(pred1_class[key][:, -1], dim=-1)
                    pred_queries = torch.norm(pred1_class[key2][:, -1], dim=-1)
                    for pred_anchor in pred_anchors:
                        # Cosine similarity
                        pred_anchor = pred_anchor.unsqueeze(0)
                        cos_sim = torch.nn.functional.cosine_similarity(
                            pred_anchor, pred_queries, dim=1
                        )
                        cos_sim = cos_sim[cos_sim != 1.0]  # Avoid self-similarity
                        cosine_similarities_cam_1[key][key2].append(
                            cos_sim.cpu().numpy()
                        )

                        # Euclidean distance
                        euc_dist = torch.norm(pred_anchor - pred_queries, dim=1)
                        euc_dist = euc_dist[euc_dist != 0.0]  # Avoid self-similarity
                        euc_distances_cam_1[key][key2].append(euc_dist.cpu().numpy())

                    pred_anchors = torch.norm(pred2_class[key][:, -1], dim=-1)
                    pred_queries = torch.norm(pred2_class[key2][:, -1], dim=-1)
                    for pred_anchor in pred_anchors:
                        pred_anchor = pred_anchor.unsqueeze(0)

                        # Cosine similarity
                        cos_sim = torch.nn.functional.cosine_similarity(
                            pred_anchor, pred_queries, dim=1
                        )
                        cos_sim = cos_sim[cos_sim != 1.0]  # Avoid self-similarity
                        cosine_similarities_cam_2[key][key2].append(
                            cos_sim.cpu().numpy()
                        )

                        # Euclidean distance
                        euc_dist = torch.norm(pred_anchor - pred_queries, dim=1)
                        euc_dist = euc_dist[euc_dist != 0.0]  # Avoid self-similarity
                        euc_distances_cam_2[key][key2].append(euc_dist.cpu().numpy())

    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("DINO WM")
    axes[0, 0].set_title("Cosine Similarities - Front Cam")
    axes[0, 1].set_title("Cosine Similarities - Wrist Cam")
    axes[1, 0].set_title("Euclidean Distances - Front Cam")
    axes[1, 1].set_title("Euclidean Distances - Wrist Cam")
    for ax in axes.flat:
        ax.set_ylabel("Probability Density")

    for key in cosine_similarities_cam_1.keys():
        for key2 in cosine_similarities_cam_1[key].keys():
            if len(cosine_similarities_cam_1[key][key2]) != 0:
                cosine_similarities_cam_1[key][key2] = np.concatenate(
                    cosine_similarities_cam_1[key][key2]
                )
                cosine_similarities_cam_2[key][key2] = np.concatenate(
                    cosine_similarities_cam_2[key][key2]
                )
                euc_distances_cam_1[key][key2] = np.concatenate(
                    euc_distances_cam_1[key][key2]
                )
                euc_distances_cam_2[key][key2] = np.concatenate(
                    euc_distances_cam_2[key][key2]
                )

                kde_cos_1 = gaussian_kde(cosine_similarities_cam_1[key][key2])
                kde_cos_2 = gaussian_kde(cosine_similarities_cam_2[key][key2])
                kde_euc_1 = gaussian_kde(euc_distances_cam_1[key][key2])
                kde_euc_2 = gaussian_kde(euc_distances_cam_2[key][key2])

                x_vals_cos = np.linspace(
                    cosine_similarities_cam_1[key][key2].min() - 1 / 1000,
                    1,
                    1000,
                )
                pdf_vals_cos_1 = kde_cos_1(x_vals_cos)
                pdf_vals_cos_2 = kde_cos_2(x_vals_cos)

                x_vals_euc = np.linspace(
                    euc_distances_cam_1[key][key2].min() - 1,
                    euc_distances_cam_1[key][key2].max() + 1,
                    1000,
                )
                pdf_vals_euc_1 = kde_euc_1(x_vals_euc)
                pdf_vals_euc_2 = kde_euc_2(x_vals_euc)

                # Plot the PDF
                axes[0, 0].plot(x_vals_cos, pdf_vals_cos_1, label=f"{key}-{key2}")
                axes[0, 1].plot(x_vals_cos, pdf_vals_cos_2, label=f"{key}-{key2}")
                axes[1, 0].plot(x_vals_euc, pdf_vals_euc_1, label=f"{key}-{key2}")
                axes[1, 1].plot(x_vals_euc, pdf_vals_euc_2, label=f"{key}-{key2}")

    # Collect all legend handles and labels
    handles_labels = [ax.get_legend_handles_labels() for ax in axes.flat]
    handles, labels = zip(*handles_labels)
    handles = sum(handles, [])  # flatten list of lists
    labels = sum(labels, [])  # flatten list of lists

    # Deduplicate labels
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    # Adjust spacing first to leave room at top
    plt.tight_layout(
        rect=[0, 0, 1, 0.92]
    )  # Reserve top 8% of figure for title + legend

    # Add global legend above the plots
    fig.legend(
        unique.values(),  # handles
        unique.keys(),  # labels
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),  # adjust vertical position
        frameon=False,
    )
    plt.savefig("latent_analysis.png")

    import ipdb

    ipdb.set_trace()
