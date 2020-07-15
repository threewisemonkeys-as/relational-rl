# Proximal policy optimization in pytorch
# Atharv Sonwane <atharvs.twm@gmail.com>

# References -
# https://arxiv.org/pdf/1707.06347.pdf
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import RelationalActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Hyperparameters
# Low values because of compute constraints
# Used stride = 4
EPOCHS = 200
EPISODES_PER_EPOCH = 20  # 64
N_POLICY_UPDATES = 16
N_VALUE_UPDATES = 16
GAMMA = 0.99
EPSILON = 0.1
VALUE_FN_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 3e-4
MAX_TRAJ_LENGTH = 10  # 1000


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

    def disc_r(self, gamma, normalize=False):
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
    def __init__(
        self,
        env
    ):
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__
        self.ac = RelationalActorCritic((3, env.observation_space.shape[0], env.observation_space.shape[1]), env.action_space.n, [8], 3, [4])

    def _update(self, batch, hp, policy_optim, value_optim, writer):
        # process batch
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [traj.disc_r(hp["gamma"], normalize=True) for traj in batch]
        a = [torch.stack(traj.a) for traj in batch]

        with torch.no_grad():
            v = [self.ac.forward(o.permute(0, 3, 1, 2))[1] for o in obs]  # [self.value(o) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]
            old_logits = [torch.stack(traj.logits) for traj in batch]
            # print(old_logits[0].squeeze(1).shape)
            # print(a[0].shape)
            old_logprobs = [
                -F.cross_entropy(old_logits[i].squeeze(1), a[i].squeeze(-1)) for i in range(len(batch))
            ]

        # update policy
        for j in range(hp["n_policy_updates"]):
            policy_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i, traj in enumerate(batch):
                curr_logits = self.ac.forward(obs[i].permute(0, 3, 1, 2))[0]  # self.policy(obs[i])
                curr_logprobs = -F.cross_entropy(curr_logits, a[i].squeeze(-1))
                ratio = torch.exp(curr_logprobs - old_logprobs[i])
                clipped_ratio = torch.clamp(ratio, 1 - hp["epsilon"], 1 + hp["epsilon"])
                policy_loss = (
                    policy_loss
                    + torch.min(ratio * adv[i], clipped_ratio * adv[i]).mean()
                )

            policy_loss = policy_loss / len(batch)
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

        # update value function
        for j in range(hp["n_value_updates"]):
            value_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i in range(len(batch)):
                v = self.ac(obs[i].permute(0, 3, 1, 2))[1].squeeze(-1)
                value_loss = value_loss + F.mse_loss(v, disc_r[i])
            value_loss = value_loss / len(batch)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

        return policy_loss.item(), value_loss.item()

    def train(
        self,
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        n_value_updates=N_VALUE_UPDATES,
        n_policy_updates=N_POLICY_UPDATES,
        value_lr=VALUE_FN_LEARNING_RATE,
        policy_lr=POLICY_LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        max_traj_length=MAX_TRAJ_LENGTH,
        log_dir="./logs/",
        RENDER=False,
        PLOT_REWARDS=True,
        VERBOSE=False,
        TENSORBOARD_LOG=True,
    ):
        """ Trains both policy and value networks """
        hp = locals()
        start_time = datetime.datetime.now()
        print(
            f"Start time: {start_time:%d-%m-%Y %H:%M:%S}"
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}\n"
            f"Hyperparameters: \n{hp}\n"
        )
        log_path = Path(log_dir).joinpath(f"{start_time:%d%m%Y%H%M%S}")
        log_path.mkdir(parents=True, exist_ok=False)
        if TENSORBOARD_LOG:
            writer = SummaryWriter(log_path)
            writer.add_text("hyperparameters", f"{hp}")
        else:
            writer = None

        # self.policy.train()
        # self.value.train()
        value_optim = torch.optim.Adam(self.ac.parameters(), lr=value_lr)
        policy_optim = torch.optim.Adam(self.ac.parameters(), lr=policy_lr)
        rewards = []
        e = 0

        try:
            for epoch in range(epochs):

                epoch_rewards = []
                batch = []

                # Sample trajectories
                for _ in range(episodes_per_epoch):
                    # initialise tracking variables
                    obs = self.env.reset()
                    obs = torch.tensor(obs, device=device, dtype=dtype)
                    traj = Trajectory([obs], [], [], [], [])
                    d = False
                    e += 1

                    # run for single trajectory
                    for i in range(max_traj_length):
                        if RENDER and (
                            e == 0 or (e % ((epochs * episodes_per_epoch) / 10)) == 0
                        ):
                            self.env.render()

                        a_logits = self.ac.forward(obs.permute(2, 0, 1).unsqueeze(0).float())[0]
                        a = torch.distributions.Categorical(logits=a_logits).sample()

                        obs, r, d, _ = self.env.step(a.item())

                        obs = torch.tensor(obs, device=device, dtype=dtype)
                        r = torch.tensor(r, device=device, dtype=dtype)
                        traj.add(obs, a, r, d, a_logits)

                        if d:
                            break

                    epoch_rewards.append(sum(traj.r))
                    batch.append(traj)

                # Update value and policy
                p_loss, v_loss = self._update(
                    batch, hp, policy_optim, value_optim, writer
                )

                # Log rewards and losses
                avg_episode_reward = np.mean(epoch_rewards[-episodes_per_epoch:])
                rewards.append(avg_episode_reward)
                if writer is not None:
                    writer.add_scalar("policy_loss", p_loss, epoch)
                    writer.add_scalar("value_loss", v_loss, epoch)
                    writer.add_scalar("rewards", avg_episode_reward, epoch)

                if VERBOSE and (epoch == 0 or ((epoch + 1) % (epochs / 10)) == 0):
                    print(
                        f"Epoch {epoch+1}: Average Episodic Reward = {avg_episode_reward:.2f} |"
                        f" Value Loss = {p_loss:.2f} |"
                        f" Policy Loss = {v_loss:.2f}"
                    )

        except KeyboardInterrupt:
            print("\nTraining Interrupted!\n")

        finally:
            self.env.close()
            print(
                f"\nTraining Completed in {(datetime.datetime.now() - start_time).seconds} seconds"
            )
            model.save(
                log_path.joinpath(f"{self.__class__.__name__}_{self.env_name}.pt")
            )
            if PLOT_REWARDS:
                plt.plot(rewards)
                plt.savefig(
                    log_path.joinpath(
                        f"{self.__class__.__name__}_{self.env_name}_reward_plot.png"
                    )
                )

    def save(self, path):
        """ Save model parameters """
        torch.save(
            {
                "relational_ppo_state_dict": self.ac.state_dict()
                # "policy_state_dict": self.policy.state_dict(),
                # "value_state_dict": self.value.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        """ Load model parameters """
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint["relational_ppo_state_dict"])
        # self.policy.load_state_dict(checkpoint["policy_state_dict"])
        # self.value.load_state_dict(checkpoint["value_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):
        """ Evaluates model performance """

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = datetime.datetime.now()
        rewards = []

        for episode in range(episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if render:
                    self.env.render()

                logits = self.ac(observation.permute(2, 1, 0).unsqueeze(0))[0]
                action = torch.distributions.Categorical(logits=logits).sample()
                next_observation, reward, done, _ = self.env.step(action.item())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                )
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(
                f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}"
            )
            rewards.append(total_episode_reward)

        env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(
            f"Evaluation Completed in {(datetime.datetime.now() - start_time).seconds} seconds"
        )


if __name__ == "__main__":

    import gym

    env = gym.make("MsPacman-v0")
    # env = gym.make("LunarLander-v2")

    # from pybullet_envs import bullet
    # env = bullet.racecarGymEnv.RacecarGymEnv(renders=False, isDiscrete=True)

    model = PPO(env)
    model.train(VERBOSE=True, PLOT_REWARDS=True, TENSORBOARD_LOG=True)
    model.eval(10)
