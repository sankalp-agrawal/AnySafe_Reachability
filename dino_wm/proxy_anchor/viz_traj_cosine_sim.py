import copy
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
from gymnasium import spaces

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
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
    # data["robot0_eye_in_hand_image"] = torch.tensor(
    #     np.array(traj["camera_0"][:]) * 255.0, dtype=torch.uint8
    # )
    data["agentview_image"] = torch.tensor(
        np.array(traj["camera_1"][:]) * 255.0, dtype=torch.uint8
    )
    # data["cam_rs_embd"] = torch.tensor(
    #     np.array(traj["cam_rs_embd"][:]), dtype=torch.float32
    # )
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


import imageio
from matplotlib import cm


def make_comparison_video(
    output_dict, keys_to_plot=None, save_path="output_video.mp4", fps=5
):
    """
    Creates a video comparing ground truth and imagination rollouts over time.
    Now supports custom keys to plot and evenly spaced rainbow colors.

    Parameters:
        output_dict (dict): Must include "ground_truth" and "imagination" with rollouts and constraint images.
        keys_to_plot (list or None): List of keys to plot. Default is all relevant keys.
        save_path (str): Output path.
        fps (int): Frames per second.
    """
    output = output_dict

    # Default keys if not provided
    all_keys = [
        "ken_fail",
        "cosine_sim_prox",
        "cosine_sim_const1",
        "cosine_sim_const2",
        "value_fn",
        "value_fn_ken",
        "gt_fail_label",
    ]
    if keys_to_plot is None:
        keys_to_plot = all_keys

    # Tanh activation on selected outputs
    for key in ["ground_truth", "imagination"]:
        for subkey in [
            "ken_fail",
            "cosine_sim_prox",
            "cosine_sim_const1",
            "cosine_sim_const2",
        ]:
            output[key][subkey] = np.tanh(2 * np.array(output[key][subkey]).squeeze())

    T = len(output["ground_truth"]["ken_fail"])

    # Consistency check
    assert all(
        len(output[k][s]) == T
        for k in ["ground_truth", "imagination"]
        for s in keys_to_plot
    ), "Inconsistent sequence lengths"

    # Setup figure
    fig = plt.figure(figsize=(12, 8), dpi=100)
    plt.subplots_adjust(top=0.85)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(4, 8, figure=fig)

    # Graph axes
    gt_graph_ax = fig.add_subplot(gs[0, 0:4])
    im_graph_ax = fig.add_subplot(gs[2, 0:4])

    # Image axes
    def init_img(ax, title):
        img_obj = ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        ax.set_title(title)
        ax.axis("off")
        return img_obj

    gt_wrist_img = init_img(fig.add_subplot(gs[0, 4]), "Wrist View")
    gt_front_img = init_img(fig.add_subplot(gs[0, 5]), "Front View")
    gt_const1_img = init_img(fig.add_subplot(gs[0, 6]), "Constraint 1")
    gt_const2_img = init_img(fig.add_subplot(gs[0, 7]), "Constraint 2")

    im_wrist_img = init_img(fig.add_subplot(gs[2, 4]), "Wrist View")
    im_front_img = init_img(fig.add_subplot(gs[2, 5]), "Front View")
    im_const1_img = init_img(fig.add_subplot(gs[2, 6]), "Constraint 1")
    im_const2_img = init_img(fig.add_subplot(gs[2, 7]), "Constraint 2")

    # Generate colors from rainbow colormap
    cmap = cm.get_cmap("rainbow")
    colors = (
        [cmap(i / (len(keys_to_plot) - 1)) for i in range(len(keys_to_plot))]
        if len(keys_to_plot) > 1
        else [cmap(0.5)]
    )

    # Create legend handles
    legend_handles = []
    for key, color in zip(keys_to_plot, colors):
        label = key.replace("_", " ").title()
        handle = gt_graph_ax.plot([], [], color=color, label=label)[0]
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )

    # Initialize line objects
    def init_lines(ax):
        lines = {}
        for key, col in zip(keys_to_plot, colors):
            if key == "gt_fail_label":
                # Special case for gt_fail_label
                lines[key] = ax.plot([], [], color=col, linestyle="--", label=key)[0]
            else:
                lines[key] = ax.plot([], [], color=col, label=key)[0]
        return lines

    gt_lines = init_lines(gt_graph_ax)
    im_lines = init_lines(im_graph_ax)

    # Graph axes formatting
    for ax, title in zip(
        [gt_graph_ax, im_graph_ax], ["Ground Truth Graph", "Imagination Graph"]
    ):
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, T)
        ax.set_ylabel("l(z)")
        ax.set_xlabel("Time")
        ax.set_title(title)

    # Prepare images
    def prepare_img(img):
        if img.dtype == np.float16:
            img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        return img.astype(np.uint8)

    time = np.arange(T)
    frames = []

    # Generate frames
    for t in tqdm(range(T), desc="Generating Frames"):
        t_slice = slice(t + 1)

        # Graph updates
        for lines, key in [(gt_lines, "ground_truth"), (im_lines, "imagination")]:
            for k in keys_to_plot:
                lines[k].set_data(time[t_slice], output[key][k][t_slice])

        # Image updates
        # gt_wrist_img.set_data(prepare_img(output["ground_truth"]["imgs_wrist"][t]))
        gt_front_img.set_data(prepare_img(output["ground_truth"]["imgs_front"][t]))
        gt_const1_img.set_data(
            prepare_img(output["ground_truth"]["img_constraint1"][0])
        )
        gt_const2_img.set_data(
            prepare_img(output["ground_truth"]["img_constraint2"][0])
        )

        # im_wrist_img.set_data(prepare_img(output["imagination"]["imgs_wrist"][t]))
        im_front_img.set_data(prepare_img(output["imagination"]["imgs_front"][t]))
        im_const1_img.set_data(prepare_img(output["imagination"]["img_constraint1"][0]))
        im_const2_img.set_data(prepare_img(output["imagination"]["img_constraint2"][0]))

        # Render
        canvas.draw()
        renderer = canvas.get_renderer()
        buf = np.asarray(renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())

    # Save video/gif
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

    hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
    database = {}
    with h5py.File(hdf5_file, "r") as hf:
        trajectory_ids = list(hf.keys())
        database = {
            i: data_from_traj(hf[traj_id]) for i, traj_id in enumerate(trajectory_ids)
        }

    BL = 4
    open_loop = True
    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,
        ac_dim=10,
        state_dim=3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    # load_state_dict_flexible(transition, "../checkpoints_pa/encoder_0.1.pth")
    # load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

    transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
    transition.eval()

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU

    critic_net = Net(
        state_shape=(1, 1, 786),
        action_shape=7,
        hidden_sizes=[512, 512, 512, 512],
        activation=critic_activation,
        concat=True,
        device=device,
    )

    critic = Critic(critic_net, device=critic_net.device).to(critic_net.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-3)

    from PyHJ.policy import avoid_DDPGPolicy_annealing_dinowm as DDPGPolicy

    print(
        "DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!"
    )

    actor_net = Net(
        state_shape=(1, 1, 786),
        hidden_sizes=[512, 512, 512, 512],
        activation=actor_activation,
        device=device,
    )
    actor = Actor(actor_net, action_shape=(7,), max_action=1.0, device=device).to(
        device
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

    policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=0.005,
        gamma=0.9999,
        exploration_noise=GaussianNoise(sigma=0.1),
        reward_normalization=False,
        estimation_step=1,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=1,
    )
    # policy.load_state_dict(
    #     torch.load(
    #         "/home/sunny/anysafe_project/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy.pth"
    #     )
    # )
    ken_policy = copy.deepcopy(policy)
    # ken_policy.load_state_dict(
    #     torch.load(
    #         "/home/sunny/anysafe_project/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy_ken.pth"
    #     )
    # )

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("../checkpoints/testing_decoder.pth"))
    decoder.eval()

    constraint1 = {
        # "wrist": database[7]["robot0_eye_in_hand_image"][82],
        "front": database[7]["agentview_image"][82],
    }  # weak unsafe frame
    constraint2 = {
        # "wrist": database[1]["robot0_eye_in_hand_image"][108],
        "front": database[1]["agentview_image"][108],
    }  # unsafe frame

    scale = 1.0

    for data, constraint, t in zip(
        [database[7], database[1]], [constraint1, constraint2], [82, 108]
    ):
        # inputs2 = (  # [1, 1, 256, 384]
        #     data["cam_rs_embd"][[t], :].to(device).unsqueeze(0)
        # )
        inputs1 = (  # [1, 1, 256, 384]
            data["cam_zed_embd"][[t], :].to(device).unsqueeze(0)
        )
        # acs = data["action"][t, :].to(device).unsqueeze(0)
        # acs = normalize_acs(acs, device=device)
        states = select_xyyaw_from_state(
            data["state"][[t], :].to(device).unsqueeze(0)
        )  # [1, 1, 3]

        semantic_feature = transition.semantic_embed(  # [embedding_dim]
            inp1=inputs1, state=states
        )
        constraint.update({"semantic_feat": semantic_feature.squeeze()})

    # for traj_id in tqdm(
    #     range(len(database)), desc="Processing Trajectories", position=0
    # ):
    for traj_id in tqdm(range(10), desc="Processing Trajectories", position=0):
        data = database[traj_id]

        none_list = [-1.0 for _ in range(BL - 1)]

        output_dict = {
            # "imgs_wrist": [],
            "imgs_front": [
                img.cpu().numpy() for img in data["agentview_image"][: BL - 1]
            ],
            "img_constraint1": constraint1["front"].unsqueeze(0).cpu().numpy(),
            "img_constraint2": constraint2["front"].unsqueeze(0).cpu().numpy(),
            "ken_fail": copy.deepcopy(none_list),
            "cosine_sim_prox": copy.deepcopy(none_list),
            "cosine_sim_const1": copy.deepcopy(none_list),
            "cosine_sim_const2": copy.deepcopy(none_list),
            # "value_fn_ken": copy.deepcopy(none_list),
            # "value_fn": copy.deepcopy(none_list),
            "gt_fail_label": -1 + 2 * (data["failure"][:] != -1.0) * 1.0,
        }
        output = {
            "imagination": copy.deepcopy(output_dict),
            "ground_truth": copy.deepcopy(output_dict),
        }

        traj_length = data["action"].shape[0]

        # Imagination Rollouts
        # inputs1: [1, BL-1, 256, 384], acs: [1, BL-1, 7], states: [1, BL-1, 8]
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        # all_acs: [1 64 A]
        all_acs = data["action"][:].unsqueeze(0).to(device)
        all_acs = normalize_acs(all_acs, device)
        inputs_states = select_xyyaw_from_state(
            data["state"][0 : BL - 1, :].to(device)
        ).unsqueeze(0)

        # Imagination Loop
        for t in tqdm(
            range(traj_length - BL + 1),
            desc=f"Imagining Trajectory {traj_id}",
            leave=False,
            position=1,
        ):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    # pred1: [1, H, N, P], pred_state: [1, H, S], pred_fail: [1, H, 1]
                    # semantic_features: [1, H, Z], latent: [1, H, N, (P + A + S)]
                    pred1, pred_state, pred_fail, semantic_features, latent = (
                        transition(
                            inputs1,
                            # wrist_hist,
                            inputs_states,
                            acs,
                            return_latent=True,
                        )
                    )
                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_features.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T
                    cos_sim_fail = -cos_sim_matrix[-1, -1].item()
                    # Decode images
                    # pred_ims: [1, C, H, W] H is height not horizon
                    pred_ims, _ = decoder(pred1[:, [-1]])
                    pred_img1 = pred_ims.permute(0, 2, 3, 1)

                    # pred_img = einops.rearrange(pred_img, "(b t) c h w -> b t c h w", t=1)
                    # getting next inputs
                # acs: [1 H A]
                if t + BL < len(all_acs[0]):  # if not on last step
                    acs = torch.cat(
                        [
                            acs[[0], 1:],
                            all_acs[0, BL - 1 + t].unsqueeze(0).unsqueeze(0),
                        ],
                        dim=1,
                    )
                # inputs1: [1 H N P]
                # inputs_states: [1 H S]
                if open_loop:
                    inputs1 = torch.cat(
                        [inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1
                    )
                    states = torch.cat(
                        [inputs_states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1
                    )
                else:
                    inputs1 = (
                        data["cam_zed_embd"][t : t + BL - 1, :].to(device).unsqueeze(0)
                    )
                    inputs_states = select_xyyaw_from_state(
                        data["state"][t : t + BL - 1, :].to(device).unsqueeze(0)
                    )

            # pred1_img: [1, H, W, C]
            output["imagination"]["imgs_front"].append(
                pred_img1[0].cpu().numpy() * 255.0,
            )
            output["imagination"]["ken_fail"].append(
                pred_fail.detach().squeeze().cpu().numpy()[-1]
            )
            output["imagination"]["cosine_sim_prox"].append(cos_sim_fail * scale)
            output["imagination"]["cosine_sim_const1"].append(
                -F.cosine_similarity(
                    semantic_features.squeeze()[-1], constraint1["semantic_feat"], dim=0
                ).item()
                * scale
            )
            output["imagination"]["cosine_sim_const2"].append(
                -F.cosine_similarity(
                    semantic_features.squeeze()[-1], constraint2["semantic_feat"], dim=0
                ).item()
                * scale
            )

            if t + BL >= len(data["action"]):  # Last step
                index = t + BL - 1
            else:
                index = t + BL
            # output["imagination"]["value_fn"].append(
            #     policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=normalize_acs(
            #             data["action"][[index], :]
            #             .to(device)
            #             .unsqueeze(0)  # Next action
            #         ),
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # output["imagination"]["value_fn_ken"].append(
            #     ken_policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=normalize_acs(
            #             data["action"][[index], :]
            #             .to(device)
            #             .unsqueeze(0)  # Next action
            #         ),
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )

            # input2_gt = data["cam_rs_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            input1_gt = data["cam_zed_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            state_gt = select_xyyaw_from_state(
                data["state"][[t + BL - 1], :].to(device)
            ).unsqueeze(0)

        lengths = [
            len(output["imagination"][key]) for key in output["imagination"].keys()
        ]
        # assert all(length == traj_length for length in lengths), (
        #     f"Inconsistent sequence lengths in imagination output: {lengths} should be {traj_length}"
        # )

        # Do ground truth images
        inputs1 = data["cam_zed_embd"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = data["action"][0 : BL - 1, :].to(device).unsqueeze(0)
        acs = normalize_acs(acs, device=device)
        states = select_xyyaw_from_state(
            data["state"][0 : BL - 1, :].to(device).unsqueeze(0)
        )

        output["ground_truth"]["imgs_front"] = [
            img for img in data["agentview_image"][:].cpu().numpy()
        ]

        for t in tqdm(
            range(traj_length - BL + 1), desc="GT Trajectory", position=1, leave=False
        ):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    # Forward pass through the transition model
                    # semantic_features: [1, (T-1), Z]
                    assert inputs1.shape == (1, BL - 1, 256, 384), (
                        f"Inputs1 shape mismatch, got {inputs1.shape}"
                    )
                    assert states.shape == (1, BL - 1, 3), (
                        f"States shape mismatch, got {states.shape}"
                    )
                    semantic_features = transition.semantic_embed(
                        inp1=inputs1, state=states
                    )

                    # latent: [1, (T-1), N, (P + A + S)]
                    latent = transition.forward_features(
                        video1=inputs1, states=states, actions=acs
                    )

                    # pred_fail: [1, (T-1), 1]
                    pred_fail = transition.failure_pred(latent)

                    # Calculate cos sim for failure margin
                    proxies = transition.proxies.to(device)  # [M Z]

                    queries_norm = F.normalize(
                        semantic_features.squeeze(), p=2, dim=1
                    )  # [N, Z]
                    proxies_norm = F.normalize(proxies, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = queries_norm @ proxies_norm.T
                    cos_sim_fail = -cos_sim_matrix[-1, -1].item()

            # inputs2 = data["cam_rs_embd"][[t + BL - 1], :].to(device).unsqueeze(0)
            inputs1 = data["cam_zed_embd"][t : t + BL - 1, :].to(device).unsqueeze(0)
            acs = data["action"][t : t + BL - 1, :].to(device).unsqueeze(0)
            acs = normalize_acs(acs, device=device)
            states = select_xyyaw_from_state(
                data["state"][t : t + BL - 1, :].to(device)
            ).unsqueeze(0)

            # pred_fail: [1 (T-1), 1] -> [1]
            ken_fail = pred_fail.squeeze().cpu().numpy()[-1]

            # output["ground_truth"]["imgs_wrist"].append(
            #     data["robot0_eye_in_hand_image"][t + BL - 1]
            #     .unsqueeze(0)
            #     .cpu()
            #     .numpy()[-1]
            # )
            output["ground_truth"]["ken_fail"].append(ken_fail)
            # output["ground_truth"]["cosine_sim_const1"].append(
            #     -F.cosine_similarity(
            #         semantic_features.squeeze()[-1], constraint1["semantic_feat"], dim=0
            #     ).item()
            #     * scale
            # )
            # output["ground_truth"]["cosine_sim_const2"].append(
            #     -F.cosine_similarity(
            #         semantic_features.squeeze()[-1], constraint2["semantic_feat"], dim=0
            #     ).item()
            #     * scale
            # )
            output["ground_truth"]["cosine_sim_prox"].append(cos_sim_fail * scale)
            # output["ground_truth"]["value_fn"].append(
            #     policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=acs,  # Next action
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # output["ground_truth"]["value_fn_ken"].append(
            #     ken_policy.critic(
            #         obs=latent[:, [-1]].mean(dim=2),
            #         act=acs,  # Next action
            #     )
            #     .detach()
            #     .squeeze()
            #     .cpu()
            #     .numpy()
            # )
            # output["ground_truth"]["gt_fail_label"].append(
            #     -2 * data["failure"][t + BL - 1].cpu().numpy() + 1
            # )

        line_keys = [
            "ken_fail",
            # "cosine_sim_prox",
            # "cosine_sim_const1",
            # "cosine_sim_const2",
            # "value_fn",
            # "value_fn_ken",
            "gt_fail_label",
        ]

        make_comparison_video(
            output_dict=output,
            save_path=f"results/output_video_{traj_id}.mp4",
            fps=10,
            keys_to_plot=line_keys,
        )
