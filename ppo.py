# Proximal policy optimization in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://arxiv.org/pdf/1707.06347.pdf
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

from datetime import datetime
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import RelationalActorCritic

dtype = torch.float32


class Trajectory:
    def __init__(
        self, observations=[], actions=[], rewards=[], dones=[], logits=[],
    ):
        self.obs = observations
        self.a = actions
        self.r = rewards
        self.d = dones
        self.logits = logits
        self.len = 0

    def add(
        self,
        obs: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        logits: torch.Tensor,
    ):
        self.obs.append(obs)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)
        self.logits.append(logits)
        self.len += 1

    def disc_r(self, gamma, device, normalize=False):
        disc_rewards = []
        r = 0.0
        for reward in self.r[::-1]:
            r = reward + gamma * r
            disc_rewards.insert(0, r)
        disc_rewards = torch.tensor(disc_rewards, device=device, dtype=dtype)
        if normalize:
            disc_rewards = (disc_rewards - disc_rewards.mean()) / (
                disc_rewards.std() + np.finfo(np.float32).eps
            )
        return disc_rewards

    def __len__(self):
        return self.len


# Model
class PPO:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__
        self.ac = RelationalActorCritic(
            obs_shape=(
                1,
                env.observation_space.shape[0],
                env.observation_space.shape[1],
            ),
            a_dim=env.action_space.n,
            conv_dims=[8, 16, 16],
            feature_dim=8,
            lin_dims=[128],
        ).to(self.device)

    def _update(self, batch, args, policy_optim, value_optim):
        # process batch
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [
            traj.disc_r(args.discounting, device=self.device, normalize=True)
            for traj in batch
        ]
        a = [torch.stack(traj.a) for traj in batch]

        with torch.no_grad():
            v = [
                self.ac.forward(o.permute(0, 3, 1, 2))[1] for o in obs
            ]  # [self.value(o) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]
            old_logits = [torch.stack(traj.logits) for traj in batch]
            # print(old_logits[0].squeeze(1).shape)
            # print(a[0].shape)
            old_logprobs = [
                -F.cross_entropy(old_logits[i].squeeze(1), a[i].squeeze(-1))
                for i in range(len(batch))
            ]

        policy_loss = None
        value_loss = None

        # update policy
        for _ in range(args.n_policy_updates):
            policy_loss = torch.zeros(
                1, device=self.device, dtype=dtype, requires_grad=True
            )
            for i, traj in enumerate(batch):
                curr_logits = self.ac.forward(obs[i].permute(0, 3, 1, 2))[
                    0
                ]  # self.policy(obs[i])
                curr_logprobs = -F.cross_entropy(curr_logits, a[i].squeeze(-1))
                ratio = torch.exp(curr_logprobs - old_logprobs[i])
                clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
                policy_loss = (
                    policy_loss
                    + torch.min(ratio * adv[i], clipped_ratio * adv[i]).mean()
                )

            policy_loss = policy_loss / len(batch)
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

        # update value function
        for j in range(args.n_value_updates):
            value_loss = torch.zeros(
                1, device=self.device, dtype=dtype, requires_grad=True
            )
            for i in range(len(batch)):
                v = self.ac(obs[i].permute(0, 3, 1, 2))[1].squeeze(-1)
                value_loss = value_loss + F.mse_loss(v, disc_r[i])
            value_loss = value_loss / len(batch)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

        return policy_loss.item(), value_loss.item()

    def train(self, args):
        """ Trains both policy and value networks """
        start_time = datetime.now()
        if args.xpid is None:
            logdir = Path(args.logdir).joinpath(
                f"{self.__class__.__name__}-{self.env_name}-{start_time:%d%m%y-%H%M%S}"
            )
        else:
            logdir = Path(args.logdir).joinpath(f"{args.xpid}")
        logdir.mkdir(parents=True)
        with open(logdir.joinpath("hyperparameters.txt"), "w+") as f:
            f.write(f"Start time: {start_time:%d%m%y-%H%M%S}\n")
            f.write(f"{args}")
        if args.log_tensorboard:
            writer = SummaryWriter(logdir)
            writer.add_text("hyperparameters", f"{args}")
        else:
            writer = None
        print(
            f"\nStarting at {start_time:%d-%m-%y %H:%M:%S}"
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}"
            f"\nLogging to {logdir}"
            f"\nHyperparameters: \n{args}\n"
        )
        self.ac.train()
        value_optim = torch.optim.Adam(self.ac.parameters(), lr=args.v_lr)
        policy_optim = torch.optim.Adam(self.ac.parameters(), lr=args.p_lr)
        rewards = []
        e = 0

        try:
            for epoch in range(args.epochs):
                epoch_rewards = []
                batch = []

                # Sample trajectories
                for _ in range(args.episodes_per_epoch):
                    # initialise tracking variables
                    obs = self.env.reset()
                    obs = torch.tensor(obs, device=self.device, dtype=dtype)
                    if len(obs.shape) < 3:
                        obs = obs.unsqueeze(-1)
                    traj = Trajectory([obs], [], [], [], [])
                    d = False
                    e += 1

                    # run for single trajectory
                    for i in range(args.max_traj_length):
                        if args.render and (
                            e == 0
                            or (e % ((args.epochs * args.episodes_per_epoch) / 10)) == 0
                        ):
                            self.env.render()

                        a_logits, _ = self.ac.forward(obs.permute(2, 0, 1).unsqueeze(0))
                        a = torch.distributions.Categorical(logits=a_logits).sample()

                        obs, r, d, _ = self.env.step(a.item())

                        obs = torch.tensor(obs, device=self.device, dtype=dtype)
                        if len(obs.shape) < 3:
                            obs = obs.unsqueeze(-1)
                        r = torch.tensor(r, device=self.device, dtype=dtype)
                        traj.add(obs, a, r, d, a_logits)

                        if d:
                            break

                    epoch_rewards.append(sum(traj.r))
                    batch.append(traj)

                # Update value and policy
                p_loss, v_loss = self._update(batch, args, policy_optim, value_optim)

                # Log rewards and losses
                avg_episode_reward = np.mean(epoch_rewards[-args.episodes_per_epoch :])
                rewards.append(avg_episode_reward)
                if writer is not None:
                    writer.add_scalar("policy_loss", p_loss, epoch)
                    writer.add_scalar("value_loss", v_loss, epoch)
                    writer.add_scalar("rewards", avg_episode_reward, epoch)

                if args.verbose and (
                    epoch == 0 or ((epoch + 1) % (args.epochs / 10)) == 0
                ):
                    print(
                        f"Epoch {epoch+1}: Average Episodic Reward = {avg_episode_reward:.2f} |"
                        f" Value Loss = {p_loss:.2f} |"
                        f" Policy Loss = {v_loss:.2f}"
                    )

        except KeyboardInterrupt:
            print("Training interrupted by user\n")

        except Exception as e:
            print(f"Training interrupted by exception:\n{e}\n")
            traceback.print_exc()
            raise e

        finally:
            self.env.close()
            print(
                f"\nTraining Completed in {(datetime.now() - start_time).seconds} seconds"
            )
            plt.plot(rewards)
            plt.title(f"Training {self.__class__.__name__} on {self.env_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Rewards")
            plt.savefig(logdir.joinpath("rewards_plot.png"))
            self.save(logdir.joinpath("model.pt"))

    def save(self, path):
        """ Save model parameters """
        torch.save(
            {"ac_state_dict": self.ac.state_dict()}, path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        """ Load model parameters """
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint["ac_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def test(self, args, deterministic=True):
        """ Evaluates model performance """
        start_time = datetime.now()
        if args.xpid is None:
            logdir = Path(args.logdir).joinpath("latest")
        else:
            logdir = Path(args.logdir).joinpath(f"{args.xpid}")
        assert (
            args.test_episodes is not None
        ), "test_episodes needs to be specified for testing"
        print(
            f"\nStarting at {start_time:%d-%m-%y %H:%M:%S}"
            f"\nTesting model on {self.env_name} for {args.test_episodes} epsidoes"
        )
        self.load(logdir.joinpath("model.pt"))
        self.ac.eval()
        rewards = []

        for episode in range(args.test_episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=self.device, dtype=dtype)
            if len(observation.shape) == 2:
                observation = observation.unsqueeze(-1)
            done = False
            episode_rewards = []

            while not done:
                if args.mode.lower() == "test_render":
                    self.env.render()

                a_logits, _ = self.ac.forward(observation.permute(2, 0, 1).unsqueeze(0))
                action = torch.distributions.Categorical(logits=a_logits).sample()
                next_observation, reward, done, _ = self.env.step(action.detach())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(
                    next_observation, device=self.device, dtype=dtype
                )
                if len(next_observation.shape) == 2:
                    next_observation = next_observation.unsqueeze(-1)
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(
                f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}"
            )
            rewards.append(total_episode_reward)

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(
            f"Evaluation Completed in {(datetime.now() - start_time).seconds} seconds"
        )
