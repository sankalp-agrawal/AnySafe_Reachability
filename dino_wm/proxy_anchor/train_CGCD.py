import argparse
import copy
from functools import partial

import dataset
import losses
import numpy as np
import torch
import torch.nn.functional as F
import utils
from proxy_anchor.CGCD.net.resnet import *
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from tqdm import *


def generate_dataset(dataset, index, index_target=None, target=None):
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.ys[v] = target[i]

    for i, v in enumerate(index):
        j = v - i
        dataset_.I.pop(j)
        dataset_.ys.pop(j)
        dataset_.im_paths.pop(j)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    # if len(dataset_n.classes) > len(dataset_.classes):
    #     dataset_.classes = dataset_n.classes
    dataset_.I.extend(dataset_n.I)
    dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.ys.extend(dataset_n.ys)

    return dataset_


def make_parser():
    parser = argparse.ArgumentParser(
        description="Official implementation of `Proxy Anchor Loss for Deep Metric Learning`"
        + "Our code is modified from `https://github.com/dichotomies/proxy-nca`"
    )
    # export directory, training and val datasets, test datasets
    parser.add_argument("--LOG_DIR", default="./logs", help="Path to log folder")
    parser.add_argument(
        "--dataset", default="cub", help="Training dataset, e.g. cub, cars, SOP, Inshop"
    )  # cub # mit # dog # air
    parser.add_argument(
        "--embedding-size",
        default=512,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch-size",
        default=120,
        type=int,
        dest="sz_batch",
        help="Number of samples per batch.",
    )  # 150
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
        default=0,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--model", default="resnet18", help="Model for training"
    )  # resnet50 #resnet18  VIT
    parser.add_argument(
        "--loss", default="Proxy_Anchor", help="Criterion for training"
    )  # Proxy_Anchor #Contrastive
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="Learning rate setting"
    )  # 1e-4
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--lr-decay-step", default=5, type=int, help="Learning decay step setting"
    )  #
    parser.add_argument(
        "--lr-decay-gamma", default=0.5, type=float, help="Learning decay gamma setting"
    )
    parser.add_argument(
        "--alpha", default=32, type=float, help="Scaling Parameter setting"
    )
    parser.add_argument(
        "--mrg", default=0.1, type=float, help="Margin parameter setting"
    )
    parser.add_argument(
        "--warm", default=5, type=int, help="Warmup training epochs"
    )  # 1
    parser.add_argument(
        "--bn-freeze",
        default=True,
        type=bool,
        help="Batch normalization parameter freeze",
    )
    parser.add_argument("--l2-norm", default=True, type=bool, help="L2 normlization")
    parser.add_argument("--remark", default="", help="Any reamrk")

    parser.add_argument("--use_split_module", type=bool, default=True)
    parser.add_argument("--use_GM_clustering", type=bool, default=True)  # False

    parser.add_argument("--exp", type=str, default="0")
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Wandb Initialization
wandb_name_kwargs = {
    "mrg": args.mrg,
    "alpha": args.alpha,
    "num_ex": (
        args.num_examples_per_class
        if args.num_examples_per_class is not None
        else "all"
    ),
}
wandb_name = "".join(
    f"{key}_{value}_" for key, value in wandb_name_kwargs.items() if value is not None
).rstrip("_")
wandb.init(name=wandb_name, project="ProxyAnchorCGCD")
wandb.config.update(args)

#### Dataset Loader and Sampler
# TODO: Import datasets and give nb_classes
nb_classes = dset_tr_0.nb_classes()

# TODO: Use DINO-WM for backbone
model = model.cuda()

#### DML Losses
criterion_pa = losses.Proxy_Anchor(
    nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha
).cuda()

#### Train Parameters
param_groups = [
    {
        "params": model.semantic_encoder.parameters(),  # Semantic encoder parameters
        "lr": float(args.lr) * 1,
    },
    {"params": criterion_pa.parameters(), "lr": float(args.lr) * 100},  # Just proxies
]

#### Optimizer
opt_pa = torch.optim.AdamW(
    param_groups, lr=float(args.lr), weight_decay=args.weight_decay
)
scheduler_pa = torch.optim.lr_scheduler.StepLR(
    opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs".format(args.nb_epochs))
losses_list = []
best_recall = [0]
best_epoch = 0

#### Load checkpoint.

dset_ev = dataset.load(  # Dataset for evaluation
    name=args.dataset,
    root=pth_dataset,
    mode="eval_0",
    transform=dataset.utils.make_transform(is_train=False),
)
dlod_ev = torch.utils.data.DataLoader(  # DataLoader for evaluation
    dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
)

for epoch in range(0, args.nb_epochs):
    model.train()

    losses_per_epoch = []

    # Warmup: Train only new params, helps stabilize learning.
    # TODO: implement warmup training if needed

    total, correct = 0, 0
    pbar = tqdm(
        enumerate(dlod_tr_0)
    )  # DataLoader for training at iteration 0 (labeled)
    for batch_idx, (x, y, z) in pbar:
        ####
        feats = model(x.squeeze().cuda())
        loss_pa = criterion_pa(feats, y.squeeze().cuda())

        opt_pa.zero_grad()
        loss_pa.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

        losses_per_epoch.append(loss_pa.data.cpu().numpy())
        opt_pa.step()

        pbar.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}".format(
                epoch,
                batch_idx + 1,
                len(dlod_tr_0),
                100.0 * batch_idx / len(dlod_tr_0),
                loss_pa.item(),
                0,
                0,
            )
        )

    losses_list.append(np.mean(losses_per_epoch))
    scheduler_pa.step()

    if epoch >= 0:
        with torch.no_grad():
            print("Evaluating..")
            Recalls = utils.evaluate_cos(model, dlod_ev, epoch)

        #### Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            torch.save(
                {
                    "model_pa_state_dict": model.state_dict(),
                    "proxies_param": criterion_pa.proxies,
                },
                "{}/{}_{}_best_step_0.pth".format(
                    pth_rst_exp, args.dataset, args.model
                ),
            )
            with open(
                "{}/{}_{}_best_results.txt".format(
                    pth_rst_exp, args.dataset, args.model
                ),
                "w",
            ) as f:
                f.write(
                    "Best Epoch: {}\tBest Recall@{}: {:.4f}\n".format(
                        best_epoch, 1, best_recall[0] * 100
                    )
                )

####
print("==> Resuming from checkpoint..")
pth_pth = (
    pth_rst_exp + "/" + "{}_{}_best_step_{}.pth".format(args.dataset, args.model, 0)
)

checkpoint = torch.load(pth_pth)
model.load_state_dict(checkpoint["model_pa_state_dict"])
criterion_pa.proxies = checkpoint["proxies_param"]

model = model.cuda()
model.eval()  # Saved model is in eval mode

####
dset_ev = dataset.load(  # Dataset for evaluation
    name=args.dataset,
    root=pth_dataset,
    mode="eval_0",
    transform=dataset.utils.make_transform(is_train=False),
)
dlod_ev = torch.utils.data.DataLoader(  # Evaluation DataLoader
    dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
)

####
print("==> Init. Evaluation..")
with torch.no_grad():
    feats, _ = utils.evaluate_cos_(model, dlod_ev)
    cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
    _, preds_lb = torch.max(cos_sim, dim=1)
    preds = preds_lb.detach().cpu().numpy()
    acc_0, _ = utils._hungarian_match_(np.array(dlod_ev.dataset.ys), preds)
print("Valid Epoch: {} Acc: {:.4f}".format(str(-1), acc_0))

####
args.nb_epochs = 60
args.warm = 10
args.steps = 1  # 2

dlod_tr_prv = dlod_tr_0  # Previous training dataset loader (labeled)
dset_tr_now_md = "train_1"  # 'train_2'
dset_ev_now_md = "eval_1"  # 'eval_2'
nb_classes_prv = nb_classes
nb_classes_evn = nb_classes  # nb_classes_evn + nb_classes_
# model = model_now
# model.eval()
# criterion_pa = criterion_pa_now

dset_tr_now = dataset.load(  # Current training dataset
    name=args.dataset,
    root=pth_dataset,
    mode=dset_tr_now_md,  # mode controls what part of the dataset to load
    transform=dataset.utils.make_transform(is_train=False),
)
dset_ev_now = dataset.load(  # Current evaluation dataset
    name=args.dataset,
    root=pth_dataset,
    mode=dset_ev_now_md,
    transform=dataset.utils.make_transform(is_train=False),
)
dlod_tr_now = torch.utils.data.DataLoader(
    dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
)
dlod_ev_now = torch.utils.data.DataLoader(
    dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
)

####
print("==> Calc. proxy mean and sigma for exemplar..")
with torch.no_grad():
    feats, _ = utils.evaluate_cos_(model, dlod_tr_prv)
    feats = losses.l2_norm(feats)
    expler_s = feats.std(dim=0).cuda()
####
print("==> Init. Split old and new..")
thres = 0.0
with torch.no_grad():
    feats, labels = utils.evaluate_cos_(
        model, dlod_tr_now
    )  # Predict the features and labels of the current training dataset (unlabeled)
    cos_sim = F.linear(  # How close are these features to the proxies?
        losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies)
    )
    preds_cs, _ = torch.max(  # Get the maximum cosine similarity for each feature
        cos_sim, dim=1
    )
    utils.show_OnN(  # Graph histogram of the cosine similarity
        m=feats,
        y=labels,
        v=preds_cs,
        nb_classes=nb_classes_prv,
        pth_result=pth_rst_exp,
        thres=thres,
        is_hist=True,
    )
####
print("==> Fine. Split old and new..")
if args.use_split_module:
    from splitNet import SplitModule

    ev_dataset = dataset.load(  # evaluation dataset for old and new split
        name=args.dataset,
        root=pth_dataset,
        mode=dset_tr_now_md,  # Dataset, train, now, mode= 'train_1'
        transform=dataset.utils.make_transform(is_train=False),
    )
    ev_dataset_train = (
        dataset.load(  # evaluation dataset for old and new split, training
            name=args.dataset,
            root=pth_dataset,
            mode=dset_tr_now_md,
            transform=dataset.utils.make_transform(is_train=True),
        )
    )

    split_module = SplitModule(save_path=pth_rst_exp)
    idx_n, idx_o = (
        split_module.split_old_and_new(  # Produce indices for old and new classes
            main_model=model,
            proxy=criterion_pa,
            old_new_dataset_eval=ev_dataset,
            old_new_dataset_train=ev_dataset_train,
            last_old_num=nb_classes,
            thres_cos=thres,
        )
    )  # , step=i)
    dset_tr_o = generate_dataset(dset_tr_now, idx_o)  # Generate dataset for old classes
    dset_tr_n = generate_dataset(dset_tr_now, idx_n)  # Generate dataset for new classes
    dlod_tr_o = torch.utils.data.DataLoader(  # DataLoader for old classes
        dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
    dlod_tr_n = torch.utils.data.DataLoader(  # DataLoader for new classes
        dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
else:
    idx = torch.where(preds_cs >= thres, 0, 1)
    idx_o = torch.nonzero(idx).squeeze()
    dset_tr_o = generate_dataset(dset_tr_now, idx_o)
    idx_n = torch.nonzero(1 - idx).squeeze()
    dset_tr_n = generate_dataset(dset_tr_now, idx_n)
    dlod_tr_o = torch.utils.data.DataLoader(
        dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
    dlod_tr_n = torch.utils.data.DataLoader(
        dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

####
print("==> Replace old labels..")
with torch.no_grad():
    feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
    cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
    _, preds_lb = torch.max(cos_sim, dim=1)
    preds_lb_o = preds_lb.detach().cpu().numpy()

####
print("==> Clustering splitted new and replace new labels..")
with torch.no_grad():
    feats, _ = utils.evaluate_cos_(model, dlod_tr_n)
clst_a = AffinityPropagation().fit(feats.cpu().numpy())  # 0.75
p, c = np.unique(clst_a.labels_, return_counts=True)
nb_classes_k = len(p)
# print(p, c)
preds_lb_n = clst_a.labels_

####
if args.use_GM_clustering:
    gm = GaussianMixture(
        n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params="kmeans"
    ).fit(feats.cpu().numpy())
    preds_lb_n = gm.predict(feats.cpu().numpy())

#### Re-generate datasets and loader
dset_tr_now = dataset.load(
    name=args.dataset,
    root=pth_dataset,
    mode=dset_tr_now_md,
    transform=dataset.utils.make_transform(is_train=True),
)
dset_tr_o = generate_dataset(dset_tr_now, idx_o)
dset_tr_n = generate_dataset(dset_tr_now, idx_n)
dset_tr_o.ys = preds_lb_o.tolist()
dset_tr_n.ys = (preds_lb_n + nb_classes_prv).tolist()
dset_tr_now_m = merge_dataset(dset_tr_o, dset_tr_n)
dlod_tr_now_m = torch.utils.data.DataLoader(
    dset_tr_now_m, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers
)

####
print("==> Training splitted new..")
nb_classes_now = nb_classes_prv + nb_classes_k
criterion_pa_now = losses.Proxy_Anchor(
    nb_classes=nb_classes_now,
    sz_embed=args.sz_embedding,
    mrg=args.mrg,
    alpha=args.alpha,
).cuda()
criterion_pa_now.proxies.data[:nb_classes_prv] = criterion_pa.proxies.data
criterion_pa_now.proxies.data[nb_classes_prv:] = torch.from_numpy(
    clst_a.cluster_centers_
).cuda()

bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0.0, 0.0, 0.0, 0.0, 0.0
bst_epoch_a, bst_epoch_o, bst_epoch_n = 0.0, 0.0, 0.0

model_now = copy.deepcopy(model)
model_now = model_now.cuda()

param_groups = [
    {
        "params": list(
            set(model_now.parameters()).difference(
                set(model_now.model.embedding.parameters())
            )
        )
        if args.gpu_id != -1
        else list(
            set(model_now.module.parameters()).difference(
                set(model_now.module.model.embedding.parameters())
            )
        )
    },
    {
        "params": model_now.model.embedding.parameters()
        if args.gpu_id != -1
        else model_now.module.model.embedding.parameters(),
        "lr": float(args.lr) * 1,
    },
]
param_groups.append(
    {"params": criterion_pa_now.parameters(), "lr": float(args.lr) * 100}
)
opt = torch.optim.AdamW(
    param_groups, lr=float(args.lr), weight_decay=args.weight_decay, betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

epoch = 0
for epoch in range(0, args.nb_epochs):
    model_now.train()

    ####

    pbar = tqdm(enumerate(dlod_tr_now_m))
    for batch_idx, (x, y, z) in pbar:
        feats = model_now(x.squeeze().cuda())

        #### Exemplar
        y_n = torch.where(y > nb_classes_prv, 1, 0)  # new classes
        y_o = y.size(0) - y_n.sum()  # old classes
        if y_o > 0:
            y_sp = torch.randint(nb_classes_prv, (y_o,))
            feats_sp = torch.normal(criterion_pa.proxies[y_sp], expler_s).cuda()
            y = torch.cat((y, y_sp), dim=0)
            feats = torch.cat((feats, feats_sp), dim=0)
        loss_pa = criterion_pa_now(feats, y.squeeze().cuda())

        #### KD
        y_o_msk = torch.nonzero(y_n)
        if y_o_msk.size(0) > 1:
            y_o_msk = torch.nonzero(y_n).squeeze()
            x_o = torch.unsqueeze(x[y_o_msk[0]], dim=0)
            feats_n = torch.unsqueeze(feats[y_o_msk[0]], dim=0)
            for kd_idx in range(1, y_o_msk.size(0)):
                x_o_ = torch.unsqueeze(x[y_o_msk[kd_idx]], dim=0)
                x_o = torch.cat((x_o, x_o_), dim=0)
                feats_n_ = torch.unsqueeze(feats[y_o_msk[kd_idx]], dim=0)
                feats_n = torch.cat((feats_n, feats_n_), dim=0)
            with torch.no_grad():
                feats_o = model(x_o.squeeze().cuda())
            feats_n = feats_n.cuda()
            # FRoST
            loss_kd = torch.dist(  # Torch distance
                F.normalize(
                    feats_o.view(feats_o.size(0) * feats_o.size(1), 1), dim=0
                ).detach(),
                F.normalize(feats_n.view(feats_o.size(0) * feats_o.size(1), 1), dim=0),
            )
        else:
            loss_kd = torch.tensor(0.0).cuda()

        loss = loss_pa * 1.0 + loss_kd * 10.0

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_description(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}/{:.6f}/{:.6f}".format(
                epoch,
                batch_idx + 1,
                len(dlod_tr_now_m),
                100.0 * batch_idx / len(dlod_tr_now_m),
                loss.item(),
                loss_pa.item(),
                loss_kd.item(),
            )
        )

    scheduler.step()

    ####
    print("==> Evaluation..")
    model_now.eval()
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model_now, dlod_ev_now)
        cos_sim = F.linear(
            losses.l2_norm(feats), losses.l2_norm(criterion_pa_now.proxies)
        )
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds_lb = preds_lb.detach().cpu().numpy()

        y = np.array(dlod_ev_now.dataset.ys)

        proj_all_new = utils.cluster_pred_2_gt(preds_lb.astype(int), y.astype(int))
        pacc_fun_all_new = partial(utils.pred_2_gt_proj_acc, proj_all_new)
        acc_a = pacc_fun_all_new(y.astype(int), preds_lb.astype(int))

        selected_mask = y < nb_classes
        acc_o = pacc_fun_all_new(
            y[selected_mask].astype(int), preds_lb[selected_mask].astype(int)
        )
        selected_mask = y >= nb_classes_evn
        acc_n = pacc_fun_all_new(
            y[selected_mask].astype(int), preds_lb[selected_mask].astype(int)
        )

    if acc_a > bst_acc_a:
        bst_acc_a = acc_a
        bst_epoch_a = epoch

    if acc_o > bst_acc_oo:
        bst_acc_on = acc_n
        bst_acc_oo = acc_o
        bst_epoch_o = epoch
    if acc_n > bst_acc_nn:
        bst_acc_nn = acc_n
        bst_acc_no = acc_o
        bst_epoch_n = epoch

    print(
        "Valid Epoch: {} Acc: {:.4f}/{:.4f}/{:.4f} Best result: {}/{}/{} {:.4f}/{:.4f}/{:.4f}".format(
            epoch,
            acc_a,
            acc_o,
            acc_n,
            bst_epoch_a,
            bst_epoch_o,
            bst_epoch_n,
            bst_acc_a,
            bst_acc_oo,
            bst_acc_nn,
        )
    )
    pth_rst_exp_log = pth_rst_exp + "/" + "result.txt"
    with open(pth_rst_exp_log, "a+") as fval:
        fval.write(
            "Valid Epoch: {} Acc: {:.4f}/{:.4f}/{:.4f}/{:.4f} Best result: {}/{}/{} {:.4f}/{:.4f}/{:.4f}\n".format(
                epoch,
                acc_0,
                acc_a,
                acc_o,
                acc_n,
                bst_epoch_a,
                bst_epoch_o,
                bst_epoch_n,
                bst_acc_a,
                bst_acc_oo,
                bst_acc_nn,
            )
        )

    step = 1
    torch.save(
        {
            "model_pa_state_dict": model_now.state_dict(),
            "proxies_param": criterion_pa_now.proxies,
        },
        "{}/{}_{}_model_last_step_{}.pth".format(
            pth_rst_exp, args.dataset, args.model, str(step)
        ),
    )
