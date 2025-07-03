import os
import random
import sys

import h5py
import imageio
import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Import custom modules
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms

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


def data_from_traj(traj):
    data = {}
    segment_length = traj["actions"].shape[0]
    data["robot0_eye_in_hand_image"] = torch.tensor(
        np.array(traj["camera_0"][:]) * 255.0, dtype=torch.uint8
    )
    data["agentview_image"] = torch.tensor(
        np.array(traj["camera_1"][:]) * 255.0, dtype=torch.uint8
    )
    data["cam_rs_embd"] = torch.tensor(
        np.array(traj["cam_rs_embd"][:]), dtype=torch.float32
    )
    data["cam_zed_embd"] = torch.tensor(
        np.array(traj["cam_zed_embd"][:]), dtype=torch.float32
    )
    data["state"] = torch.tensor(np.array(traj["states"][:]), dtype=torch.float32)
    data["action"] = torch.tensor(np.array(traj["actions"][:]), dtype=torch.float32)
    if "labels" in traj.keys():
        data["failure"] = torch.tensor(np.array(traj["labels"][:]), dtype=torch.float32)
    data["is_first"] = torch.zeros(segment_length)
    data["is_last"] = torch.zeros(segment_length)
    data["is_first"][0] = 1.0
    data["is_terminal"] = data["is_last"]
    data["discount"] = torch.ones(segment_length, dtype=torch.float32)
    return data


def make_comparison_video(output_dict, save_path="output_video.mp4", fps=5):
    """
    Creates a video comparing ground truth and imagination rollouts over time,
    now with constraint images added next to wrist and front views.

    Parameters:
        output_dict (dict): Must include keys "ground_truth" and "imagination",
                            each containing "imgs_wrist", "imgs_front",
                            and single static images "img_constraint1", "img_constraint2",
                            plus time-series "ken_fail", "cosine_sim_prox", etc.
        save_path (str): Path to save the output video or gif (e.g., 'out.mp4' or 'out.gif').
        fps (int): Frames per second.
    """

    output = output_dict
    T = len(output["ground_truth"]["ken_fail"])

    required_keys = [
        "imgs_wrist",
        "imgs_front",
        "ken_fail",
        "cosine_sim_prox",
        "cosine_sim_const1",
        "cosine_sim_const2",
        "gt_fail_label",
    ]
    assert all(
        len(output[k][s]) == T
        for k in ["ground_truth", "imagination"]
        for s in required_keys
    ), "Inconsistent sequence lengths"

    # Setup figure: 2 graph rows, 2 image rows with 4 columns (wrist, front, constraint1, constraint2)
    fig = plt.figure(figsize=(12, 8), dpi=100)
    plt.subplots_adjust(top=0.85)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(4, 8, figure=fig)  # more cols for image width

    # Axes
    gt_graph_ax = fig.add_subplot(gs[0, 0:4])
    im_graph_ax = fig.add_subplot(gs[2, 0:4])

    gt_wrist_ax = fig.add_subplot(gs[0, 4])
    gt_front_ax = fig.add_subplot(gs[0, 5])
    gt_const1_ax = fig.add_subplot(gs[0, 6])
    gt_const2_ax = fig.add_subplot(gs[0, 7])  # now moved to same row (visually grouped)

    im_wrist_ax = fig.add_subplot(gs[2, 4])
    im_front_ax = fig.add_subplot(gs[2, 5])
    im_const1_ax = fig.add_subplot(gs[2, 6])
    im_const2_ax = fig.add_subplot(gs[2, 7])  # also aligned

    frames = []

    def prepare_img(img):
        if img.dtype == np.float16:
            img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        return img.astype(np.uint8)

    for t in range(T):
        for ax in [
            gt_graph_ax,
            im_graph_ax,
            gt_wrist_ax,
            gt_front_ax,
            gt_const1_ax,
            gt_const2_ax,
            im_wrist_ax,
            im_front_ax,
            im_const1_ax,
            im_const2_ax,
        ]:
            ax.clear()

        # Plotting graphs
        time = np.arange(t + 1)
        (line_ken,) = gt_graph_ax.plot([], [], color="red", label="Ken Fail")
        (line_cos_prox,) = gt_graph_ax.plot(
            [], [], color="blue", label="Proxies Cos Sim"
        )
        (line_cos_const1,) = gt_graph_ax.plot(
            [], [], color="orange", label="Constraint 1 Cos Sim"
        )
        (line_cos_const2,) = gt_graph_ax.plot(
            [], [], color="purple", label="Constraint 2 Cos Sim"
        )
        (line_gt,) = gt_graph_ax.plot(
            [], [], color="green", linestyle="--", label="Ground Truth Fail Label"
        )

        for ax, key in zip([gt_graph_ax, im_graph_ax], ["ground_truth", "imagination"]):
            ax.plot(time, output[key]["ken_fail"][: t + 1], color="red")
            ax.plot(time, output[key]["cosine_sim_prox"][: t + 1], color="blue")
            ax.plot(time, output[key]["cosine_sim_const1"][: t + 1], color="orange")
            ax.plot(time, output[key]["cosine_sim_const2"][: t + 1], color="purple")
            ax.plot(
                time,
                output[key]["gt_fail_label"][: t + 1],
                color="green",
                linestyle="--",
            )
            ax.set_ylim(-2, 2)
            ax.set_xlim(0, T)
            ax.set_ylabel("l(z)")
            ax.set_xlabel("Time")
            ax.set_title(f"{key.replace('_', ' ').title()} Graph")

        fig.legend(
            handles=[
                line_ken,
                line_cos_prox,
                line_cos_const1,
                line_cos_const2,
                line_gt,
            ],
            labels=[
                "Ken Fail",
                "Proxies Cos Sim",
                "Const 1 Cos Sim",
                "Const 2 Cos Sim",
                "Ground Truth Fail Label",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=3,
            frameon=False,
        )

        # Show images
        gt_wrist_ax.imshow(prepare_img(output["ground_truth"]["imgs_wrist"][t]))
        gt_front_ax.imshow(prepare_img(output["ground_truth"]["imgs_front"][t]))
        gt_const1_ax.imshow(prepare_img(output["ground_truth"]["img_constraint1"][0]))
        gt_const2_ax.imshow(prepare_img(output["ground_truth"]["img_constraint2"][0]))

        im_wrist_ax.imshow(prepare_img(output["imagination"]["imgs_wrist"][t]))
        im_front_ax.imshow(prepare_img(output["imagination"]["imgs_front"][t]))
        im_const1_ax.imshow(prepare_img(output["imagination"]["img_constraint1"][0]))
        im_const2_ax.imshow(prepare_img(output["imagination"]["img_constraint2"][0]))

        for ax, title in zip(
            [gt_wrist_ax, gt_front_ax, gt_const1_ax, gt_const2_ax],
            ["Wrist View", "Front View", "Constraint 1", "Constraint 2"],
        ):
            ax.set_title(title)
            ax.axis("off")

        for ax, title in zip(
            [im_wrist_ax, im_front_ax, im_const1_ax, im_const2_ax],
            ["Wrist View", "Front View", "Constraint 1", "Constraint 2"],
        ):
            ax.set_title(title)
            ax.axis("off")

        # Render and save frame
        canvas.draw()
        renderer = canvas.get_renderer()
        buf = np.asarray(renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())

    # Save video/gif
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved to {save_path}")


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
    device = "cuda:0"

    hdf5_file = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"
    database = {}
    with h5py.File(hdf5_file, "r") as hf:
        trajectory_ids = list(hf.keys())
        database = {
            i: data_from_traj(hf[traj_id]) for i, traj_id in enumerate(trajectory_ids)
        }

    BL = 4
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
    transition.load_state_dict(torch.load("../checkpoints_pa/encoder_0.1.pth"))

    # transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
    transition.eval()

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("../checkpoints/testing_decoder.pth"))
    decoder.eval()

    constraint1 = {
        "wrist": database[7]["robot0_eye_in_hand_image"][82],
        "front": database[7]["agentview_image"][82],
    }  # weak unsafe frame
    constraint2 = {
        "wrist": database[1]["robot0_eye_in_hand_image"][108],
        "front": database[1]["agentview_image"][108],
    }  # unsafe frame

    for data, constraint, t in zip(
        [database[7], database[1]], [constraint1, constraint2], [82, 108]
    ):
        inputs2 = data["cam_rs_embd"][t : BL - 1 + t, :].to(device).unsqueeze(0)
        inputs1 = data["cam_zed_embd"][t : BL - 1 + t, :].to(device).unsqueeze(0)
        acs = data["action"][t : BL - 1 + t, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        states = data["state"][t : BL - 1 + t, :].to(device).unsqueeze(0)

        __, __, __, __, semantic_feat = transition(
            inputs1,
            inputs2,
            states,
            acs,
        )
        constraint.update({"semantic_feat": semantic_feat.squeeze()[-1]})

    for traj_id in range(len(database)):
        # Select weak_unsafe frame
        data = database[traj_id]
        traj_length = data["action"].shape[0]

        inputs2 = data["cam_rs_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        states = data["state"][0 : BL - 1, :].to(device).unsqueeze(0)

        inp1, inp2, state, pred_fail, semantic_feat = transition(
            inputs1,
            inputs2,
            states,
            acs,
        )
        front_hist = torch.cat([inputs1[:, 1:], inp1[:, [-1]]], dim=1)
        wrist_hist = torch.cat([inputs2[:, 1:], inp2[:, [-1]]], dim=1)
        state_hist = torch.cat([states[:, 1:], state[:, [-1]]], dim=1)
        ac_hist = acs

        output = {
            "imagination": {
                "imgs_wrist": [],
                "imgs_front": [],
                "img_constraint1": constraint1["front"].unsqueeze(0).cpu().numpy(),
                "img_constraint2": constraint2["front"].unsqueeze(0).cpu().numpy(),
                "ken_fail": [],
                "cosine_sim_prox": [],
                "cosine_sim_const1": [],
                "cosine_sim_const2": [],
                "gt_fail_label": [],
            },
            "ground_truth": {
                "imgs_wrist": [],
                "imgs_front": [],
                "img_constraint1": constraint1["front"].unsqueeze(0).cpu().numpy(),
                "img_constraint2": constraint2["front"].unsqueeze(0).cpu().numpy(),
                "ken_fail": [],
                "cosine_sim_prox": [],
                "cosine_sim_const1": [],
                "cosine_sim_const2": [],
                "gt_fail_label": [],
            },
        }

        for t in range(traj_length - BL + 1):
            ac_torch = action = data["action"][[t + BL - 1], :].to(device).unsqueeze(0)

            ac_hist = torch.cat([ac_hist[:, 1:], ac_torch], dim=1)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    latent = transition.forward_features(
                        front_hist, wrist_hist, state_hist, ac_hist
                    )

                    # Generate predictions
                    inp1 = transition.front_head(latent)
                    inp2 = transition.wrist_head(latent)
                    state = transition.state_pred(latent)
                    pred_fail = transition.failure_pred(latent)

                    # Calculate cos sim for failure margin
                    semantic_feat = transition.semantic_embed(latent)  # [1 N Z]
                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_feat.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T
                    cos_sim_fail = -cos_sim_matrix[-1, -1].item()

                    # Decode images
                    pred_img, __ = decoder(torch.cat([inp1, inp2], dim=0))
                    # pred_img = einops.rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)

            pred1, pred2 = torch.split(pred_img, [inp1.shape[1], inp2.shape[1]], dim=0)
            pred1 = pred1.squeeze().permute(0, 2, 3, 1)
            pred2 = pred2.squeeze().permute(0, 2, 3, 1)

            output["imagination"]["imgs_front"].append(pred1.cpu().numpy()[-1] * 255.0)
            output["imagination"]["imgs_wrist"].append(pred2.cpu().numpy()[-1] * 255.0)
            output["imagination"]["ken_fail"].append(
                pred_fail.detach().squeeze().cpu().numpy()[-1]
            )
            output["imagination"]["cosine_sim_prox"].append(cos_sim_fail)
            output["imagination"]["cosine_sim_const1"].append(
                -F.cosine_similarity(
                    semantic_feat.squeeze()[-1], constraint1["semantic_feat"], dim=0
                ).item()
            )
            output["imagination"]["cosine_sim_const2"].append(
                -F.cosine_similarity(
                    semantic_feat.squeeze()[-1], constraint2["semantic_feat"], dim=0
                ).item()
            )
            output["imagination"]["gt_fail_label"].append(
                -2 * data["failure"][t + BL - 1].cpu().numpy() + 1
            )

            input2_gt = data["cam_rs_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            input1_gt = data["cam_zed_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            state_gt = data["state"][[t + BL - 1], :].to(device).unsqueeze(0)

            front_hist = torch.cat([front_hist[:, 1:], input1_gt[:, [-1]]], dim=1)
            wrist_hist = torch.cat([wrist_hist[:, 1:], input2_gt[:, [-1]]], dim=1)
            state_hist = torch.cat([state_hist[:, 1:], state_gt[:, [-1]]], dim=1)

        # Do ground truth images
        inputs2 = data["cam_rs_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        states = data["state"][0 : BL - 1, :].to(device).unsqueeze(0)

        inp1, inp2, state, pred_fail, semantic_feat = transition(
            inputs1,
            inputs2,
            states,
            acs,
        )
        front_hist = torch.cat([inputs1[:, 1:], inp1[:, [-1]]], dim=1)
        wrist_hist = torch.cat([inputs2[:, 1:], inp2[:, [-1]]], dim=1)
        state_hist = torch.cat([states[:, 1:], state[:, [-1]]], dim=1)
        ac_hist = acs

        for t in range(traj_length - BL + 1):
            ac_torch = action = data["action"][[t + BL - 1], :].to(device).unsqueeze(0)

            ac_hist = torch.cat([ac_hist[:, 1:], ac_torch], dim=1)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    latent = transition.forward_features(
                        front_hist, wrist_hist, state_hist, ac_hist
                    )

                    pred_fail = transition.failure_pred(latent)

                    # Calculate cos sim for failure margin
                    semantic_feat = transition.semantic_embed(latent)  # [1 N Z]
                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_feat.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T
                    cos_sim_fail = -cos_sim_matrix[-1, -1].item()

            inputs2 = data["cam_rs_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            inputs1 = data["cam_zed_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            acs = data["action"][[t + BL - 1], :].to(device).unsqueeze(0)
            acs = normalize_acs(acs, device=device)
            states = data["state"][[t + BL - 1], :].to(device).unsqueeze(0)

            front_hist = torch.cat([front_hist[:, 1:], inputs1[:, [-1]]], dim=1)
            wrist_hist = torch.cat([wrist_hist[:, 1:], inputs2[:, [-1]]], dim=1)
            state_hist = torch.cat([state_hist[:, 1:], states[:, [-1]]], dim=1)

            ken_fail = pred_fail

            output["ground_truth"]["imgs_front"].append(
                data["agentview_image"][t + BL - 1].unsqueeze(0).cpu().numpy()[-1]
            )
            output["ground_truth"]["imgs_wrist"].append(
                data["robot0_eye_in_hand_image"][t + BL - 1]
                .unsqueeze(0)
                .cpu()
                .numpy()[-1]
            )
            output["ground_truth"]["ken_fail"].append(
                ken_fail.detach().squeeze().cpu().numpy()[-1]
            )
            output["ground_truth"]["cosine_sim_const1"].append(
                -F.cosine_similarity(
                    semantic_feat.squeeze()[-1], constraint1["semantic_feat"], dim=0
                ).item()
            )
            output["ground_truth"]["cosine_sim_const2"].append(
                -F.cosine_similarity(
                    semantic_feat.squeeze()[-1], constraint2["semantic_feat"], dim=0
                ).item()
            )
            output["ground_truth"]["cosine_sim_prox"].append(cos_sim_fail)
            output["ground_truth"]["gt_fail_label"].append(
                -2 * data["failure"][t + BL - 1].cpu().numpy() + 1
            )

        make_comparison_video(
            output_dict=output, save_path=f"results/output_video_{traj_id}.gif", fps=5
        )
