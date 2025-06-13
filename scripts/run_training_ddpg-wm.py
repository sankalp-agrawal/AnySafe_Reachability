import argparse
import os
import sys

import gymnasium  # as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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

import models
import ruamel.yaml as yaml
import tools

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from generate_data_traj_cont import get_frame
from PIL import Image
from termcolor import cprint

import wandb
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

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


args = get_args()
config = args


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
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

env.set_wm(wm, offline_dataset, config)


# check if the environment has control and disturbance actions:
assert hasattr(
    env, "action_space"
)  # and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
if isinstance(env.observation_space, gymnasium.spaces.Dict):
    args.state_shape = (
        env.observation_space["state"].shape or env.observation_space["state"].n
    )
else:
    args.state_shape = env.observation_space.shape or env.observation_space.n
args.constraint_dim = env.constraints_shape
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]


train_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(args.task, params=[wm, offline_dataset, config])
        for _ in range(args.training_num)
    ]
)
test_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(args.task, params=[wm, offline_dataset, config])
        for _ in range(args.test_num)
    ]
)


# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model

if args.actor_activation == "ReLU":
    actor_activation = torch.nn.ReLU
elif args.actor_activation == "Tanh":
    actor_activation = torch.nn.Tanh
elif args.actor_activation == "Sigmoid":
    actor_activation = torch.nn.Sigmoid
elif args.actor_activation == "SiLU":
    actor_activation = torch.nn.SiLU

if args.critic_activation == "ReLU":
    critic_activation = torch.nn.ReLU
elif args.critic_activation == "Tanh":
    critic_activation = torch.nn.Tanh
elif args.critic_activation == "Sigmoid":
    critic_activation = torch.nn.Sigmoid
elif args.critic_activation == "SiLU":
    critic_activation = torch.nn.SiLU

if args.critic_net is not None:
    critic_net = Net(
        args.state_shape,
        obs_inputs=["state", "constraint"],
        action_shape=args.action_shape,
        hidden_sizes=args.critic_net,
        constraint_dim=args.constraint_dim,
        constraint_embedding_dim=args.constraint_embedding_dim,
        hidden_sizes_constraint=args.control_net_const,
        activation=critic_activation,
        concat=True,
        device=args.device,
    )
else:
    # report error:
    raise ValueError("Please provide critic_net!")

critic = Critic(critic_net, device=args.device).to(args.device)
critic_optim = torch.optim.AdamW(
    critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
)

log_path = None

from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

print(
    "DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!"
)

actor_net = Net(
    args.state_shape,
    obs_inputs=["state", "constraint"],
    hidden_sizes=args.control_net,
    activation=actor_activation,
    device=args.device,
    constraint_dim=args.constraint_dim,
    constraint_embedding_dim=args.constraint_embedding_dim,
    hidden_sizes_constraint=args.control_net_const,
)
actor = Actor(
    actor_net, args.action_shape, max_action=args.max_action, device=args.device
).to(args.device)
actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)


policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=args.actor_gradient_steps,
)

log_path = os.path.join(
    args.logdir + "/PyHJ",
    args.task,
    "wm_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}".format(
        args.actor_activation,
        args.critic_activation,
        args.actor_gradient_steps,
        args.tau,
        args.training_num,
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.gamma_pyhj,
    ),
)


# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs)

if args.warm_start_path is not None:
    policy.load_state_dict(torch.load(args.warm_start_path))
    args.kwargs = args.kwargs + "warmstarted"

epoch = 0
# writer = SummaryWriter(log_path, filename_suffix="_"+timestr+"epoch_id_{}".format(epoch))
# logger = TensorboardLogger(writer)
log_path = (
    log_path
    + "/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}".format(
        args.exploration_noise,
        args.actor_lr,
        args.critic_lr,
        args.batch_size_pyhj,
        args.step_per_epoch,
        args.kwargs,
        args.seed,
    )
)


if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(
        torch.load(os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"))
    )


if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir))
    # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
    epoch = args.continue_training_epoch


def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(),
        os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"),
    )


def stop_fn(mean_rewards):
    return False


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert("RGB")


if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path + "/epoch_id_{}".format(epoch))


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


if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path + "/epoch_id_{}".format(epoch))
thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2]
cache = make_cache(config, thetas)
logger = None
warmup = 1
# plot1, plot2, metrics = env.get_eval_plot(
#     cache=cache, thetas=thetas, config=config, policy=policy
# )

for iter in range(warmup + args.total_episodes):
    if iter < warmup:
        policy._gamma = 0  # for warmup the value fn
        policy.warmup = True
    else:
        policy._gamma = config.gamma_pyhj
        policy.warmup = False

    if args.continue_training_epoch is not None:
        print(
            "epoch: {}, remaining epochs: {}".format(
                epoch // args.epoch, args.total_episodes - iter
            )
        )
    else:
        print(
            "epoch: {}, remaining epochs: {}".format(iter, args.total_episodes - iter)
        )
    epoch = epoch + args.epoch
    print("log_path: ", log_path + "/epoch_id_{}".format(epoch))
    if args.total_episodes > 1:
        writer = SummaryWriter(
            log_path + "/epoch_id_{}".format(epoch)
        )  # filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    else:
        if not os.path.exists(log_path + "/total_epochs_{}".format(epoch)):
            print("Just created the log directory!")
            print("log_path: ", log_path + "/total_epochs_{}".format(epoch))
            os.makedirs(log_path + "/total_epochs_{}".format(epoch))
        writer = SummaryWriter(
            log_path + "/total_epochs_{}".format(epoch)
        )  # filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    if logger is None:
        task_name = args.task.split("-")[-1]  # Take everything before the last dash
        wandb_name = f"{task_name}_DDPG_{config.wm_name}"
        logger = WandbLogger(name=wandb_name)
        logger.load(writer)
    logger = TensorboardLogger(writer)

    # import pdb; pdb.set_trace()
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    save_best_fn(policy, epoch=epoch)
    plot1, plot2, metrics = env.get_eval_plot(
        cache=cache, thetas=thetas, config=config, policy=policy
    )
    wandb.log(
        {
            "binary_reach_avoid_plot": wandb.Image(plot1),
            "continuous_plot": wandb.Image(plot2),
            **{f"metric/{k}": v for k, v in metrics.items()},
        }
    )
