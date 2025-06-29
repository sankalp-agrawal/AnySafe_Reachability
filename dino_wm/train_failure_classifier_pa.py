import argparse
import os
import random

import losses
import numpy as np
import torch
import utils
import wandb
from net.bn_inception import *
from net.googlenet import *
from net.resnet import *
from torch.utils.data import DataLoader
from tqdm import *

from dino_wm.dino_models import VideoTransformer
from dino_wm.test_loader import SplitTrajectoryDataset

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
    parser.add_argument("--remark", default="", help="Any reamrk")
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Wandb Initialization
wandb.init(project="ProxyAnchor")
wandb.config.update(args)

# Dataset Loader and Sampler
BS = args.sz_batch  # batch size
BL = 4
hdf5_file = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"

expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)
expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
device = "cuda:0"

nb_classes = 2  # Safe and Failure

# Backbone Model

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

decoder = VQVAE().to(device)
print("decoder with parameters", count_parameters(decoder))


if args.gpu_id == -1:
    model = nn.DataParallel(model)

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
        "params": list(
            set(model.parameters()).difference(set(model.model.embedding.parameters()))
        )
        if args.gpu_id != -1
        else list(
            set(model.module.parameters()).difference(
                set(model.module.model.embedding.parameters())
            )
        )
    },
    {
        "params": model.model.embedding.parameters()
        if args.gpu_id != -1
        else model.module.model.embedding.parameters(),
        "lr": float(args.lr) * 1,
    },
]
param_groups.append({"params": criterion.parameters(), "lr": float(args.lr) * 100})

# Optimizer Setting
opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall = [0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = (
            model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        )
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(
                criterion.parameters()
            )
        else:
            unfreeze_model_param = list(
                model.module.model.embedding.parameters()
            ) + list(criterion.parameters())

        if epoch == 0:
            for param in list(
                set(model.parameters()).difference(set(unfreeze_model_param))
            ):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(
                set(model.parameters()).difference(set(unfreeze_model_param))
            ):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        m = model(x.squeeze().cuda())
        loss = criterion(m, y.squeeze().cuda())

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == "Proxy_Anchor":
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch,
                batch_idx + 1,
                len(dl_tr),
                100.0 * batch_idx / len(dl_tr),
                loss.item(),
            )
        )

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({"loss": losses_list[-1]}, step=epoch)
    scheduler.step()

    if epoch >= 0:
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset == "Inshop":
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif args.dataset != "SOP":
                Recalls = utils.evaluate_cos(model, dl_ev)
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_ev)

        # Logging Evaluation Score
        if args.dataset == "Inshop":
            for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
        elif args.dataset != "SOP":
            for i in range(6):
                wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10**i): Recalls[i]}, step=epoch)

        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists("{}".format(LOG_DIR)):
                os.makedirs("{}".format(LOG_DIR))
            torch.save(
                {"model_state_dict": model.state_dict()},
                "{}/{}_{}_best.pth".format(LOG_DIR, args.dataset, args.model),
            )
            with open(
                "{}/{}_{}_best_results.txt".format(LOG_DIR, args.dataset, args.model),
                "w",
            ) as f:
                f.write("Best Epoch: {}\n".format(best_epoch))
                if args.dataset == "Inshop":
                    for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                        f.write(
                            "Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100)
                        )
                elif args.dataset != "SOP":
                    for i in range(6):
                        f.write(
                            "Best Recall@{}: {:.4f}\n".format(
                                2**i, best_recall[i] * 100
                            )
                        )
                else:
                    for i in range(4):
                        f.write(
                            "Best Recall@{}: {:.4f}\n".format(
                                10**i, best_recall[i] * 100
                            )
                        )
