import argparse

import torch

from ppo import PPO

parser = argparse.ArgumentParser(
    description="PPO with relational inductive bias module."
)

parser.add_argument(
    "--env", "-e", type=str, default="Cartpole-v0", help="Gym environment."
)
parser.add_argument(
    "--mode",
    "-m",
    default="train",
    choices=["train", "test"],
    help="Training or test mode.",
)
parser.add_argument("--xpid", default=None, help="Experiment id (default: None).")

parser.add_argument(
    "--epochs", default=200, type=int, metavar="E", help="Total epochs to train for.",
)
parser.add_argument(
    "--episodes_per_epoch", default=64, type=int, help="Episodes to run per epoch."
)
parser.add_argument(
    "--max_traj_length",
    default=1000,
    type=int,
    help="Maximum number of timesteps in a trajectory.",
)
parser.add_argument(
    "--n_policy_updates",
    default=20,
    type=int,
    help="Number of times to take gradient steps on the policy each update.",
)
parser.add_argument(
    "--n_value_updates",
    default=20,
    type=int,
    help="Number of times to take gradient steps on the value function each update.",
)

parser.add_argument("--p_lr", default=1e-3, type=float, help="Policy learning rate.")
parser.add_argument(
    "--v_lr", default=1e-3, type=float, help="Value function learning rate."
)
parser.add_argument(
    "--discounting", default=0.99, type=float, help="Discounting factor."
)
parser.add_argument("--epsilon", default=0.1, type=float, help="Small threshold.")
parser.add_argument(
    "--grad_norm_clip", default=40.0, type=float, help="Global gradient norm clip."
)

parser.add_argument(
    "--verbose", "-v", action="store_true", help="Enable to log progres to console."
)
parser.add_argument(
    "--render", "-r", action="store_true", help="Enables rendering of environment."
)
parser.add_argument(
    "--checkpoint_every", type=int, default=10, help="Epochs to checkpoint model every."
)
parser.add_argument(
    "--logdir",
    default="~/logs/PPO/",
    help="Root dir where experiment data will be saved.",
)
parser.add_argument(
    "--log_tensorboard", action="store_true", help="Enable tensorboard logging"
)
parser.add_argument(
    "--enable_atari_wrapper",
    action="store_true",
    help="Enable atari specific wrapper for env.",
)
parser.add_argument("--enable_cuda", action="store_true", help="Enable CUDA.")


def make_env(flags):
    import gym

    env_name = flags.env
    if "Bullet" in env_name:
        import pybullet_envs  # noqa: F401

        try:
            env = gym.make(env_name, isDiscrete=False, renders=args.render)
        except TypeError:
            env = gym.make(env_name, renders=args.render)
        if args.enable_atari_wrapper:
            raise Exception("enable_atari_wrappers cannot be used with pybullet envs")
    else:
        env = gym.make(env_name)

    if args.enable_atari_wrapper:
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)

    return env


args = parser.parse_args()
env = make_env(args)


if args.enable_cuda:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("Cuda not available")
else:
    device = torch.device("cpu")

if args.mode.lower() == "train":
    model = PPO(env, device)
    model.train(args)
elif args.mode.lower() == "test":
    model = PPO(env, device)
    model.test(args)
