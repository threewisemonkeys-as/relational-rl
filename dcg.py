import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, n_neighbors: int, n_actions: int) -> None:
        self.m = torch.zeros(n_neighbors, n_actions)
        self.new_m = torch.zeros(n_neighbors, n_actions)
        self.a = 0


class DCG:
    def __init__(
        self,
        nodes: list[Node],
        neighbours: list[list[int]],
        utility_hidden_dims: list[int],
        payoff_hidden_dims: list[int],
        obs_dim: int,
        n_actions: int,
        iterations: int,
        state_hidden_dim: int,
    ) -> None:
        self.nodes = nodes
        self.neighbours = neighbours
        self.iterations = iterations
        self.n_actions = n_actions

        self.n_nodes = len(self.nodes)
        self.n_edges = 0
        for n in self.neighbours:
            self.n_edges += len(n)
        self.n_edges //= 2
        self.actions = torch.zeros(self.n_nodes, n_actions)
        self.state = torch.zeros(1, self.n_nodes, state_hidden_dim)
        self.state_encoder = nn.GRU(
            input_size=obs_dim + 1, hidden_size=state_hidden_dim
        )

        # create utility network
        utility_hidden_dims = [state_hidden_dim, *utility_hidden_dims, n_actions]
        module_list = []
        for i in range(len(utility_hidden_dims) - 2):
            module_list.append(
                nn.Linear(utility_hidden_dims[i], utility_hidden_dims[i + 1])
            )
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(utility_hidden_dims[-2], utility_hidden_dims[-1]))
        self.utility = nn.Sequential(*module_list)

        # create payoff network
        payoff_hidden_dims = [
            state_hidden_dim * 2,
            *utility_hidden_dims,
            n_actions ** 2,
        ]
        module_list = []
        for i in range(len(payoff_hidden_dims) - 2):
            module_list.append(
                nn.Linear(payoff_hidden_dims[i], payoff_hidden_dims[i + 1])
            )
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(payoff_hidden_dims[-2], payoff_hidden_dims[-1]))
        self.payoff = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        # encode observation and previous actions into hidden state
        encoder_input = torch.cat([x.unsqueeze(0), self.actions.unsqueeze(0)], dim=-1)
        _, self.state = self.state_encoder(encoder_input, self.state)

        # compute utility values for all nodes
        u = (1 / self.n_nodes) * self.utility(self.state)

        # run message passing across the graph
        for _ in range(self.iterations):
            for i, n_i in enumerate(self.nodes):
                for j, n_j in enumerate(self.neighbours[i]):

                    # compute payoffs for each edge
                    p = (1 / self.n_edges) * self.payoff(
                        self.state[[i, j]].view(-1)
                    ).view(self.n_actions, self.n_actions)

                    # compute messages
                    m = u[i] + torch.sum(n_i.m, dim=0) - n_i.m[j]
                    m = (m + p.T).T
                    idx = self.neighbours[j].index(i)
                    n_j.new_m[idx] = torch.max(m, dim=0).values

            # update the messages for for all nodes
            for n in self.nodes:
                n.m = n.new_m

        # compute optimal action for all nodes and q value
        q = torch.zeros
        for i, n_i in enumerate(self.nodes):
            t = torch.max((u[i] + torch.sum(n_i.m, dim=0)).view(-1), dim=0)
            n_i.a = t.indices
            q += t.values
            self.actions[i] = n_i.a

        return self.actions, q


class DCGAgent:
    def __init__(self, env, graph, hpam):
        self.env = env
        self.dcg = DCG(
            graph["nodes"], graph["neighbours"], [128], [256], 18, 5, 10, 128
        )
        self.hpam = hpam
        self.utility_targ = copy.deepcopy(self.dcg.utility)
        self.payoff_targ = copy.deepcopy(self.dcg.payoff)

    def train(self, epochs):
        # raise NotImplementedError

        # Setting the optimizers
        optimizer1 = torch.optim.Adam(
            self.dcg.utility.parameters(), lr=self.hpam["utility_lr"]
        )
        optimizer2 = torch.optim.Adam(
            self.dcg.payoff.parameters(), lr=self.hpam["payoff_hpam"]
        )

        mean_epoch_rewards = []

        # Start training
        for i in range(epochs):
            obs = env.reset_world()
            obs = torch.tensor(obs, dtype=torch.float32)
            replay_buffer = deque(maxlen=self.hpam["max_buffer_length"])
            eps_rewards = []
            done = False

            # Collect trajectories
            for j in range(max_traj_length):
                actions, q = self.dcg.forward(obs)
                a = self.select_action(actions)
                next_obs, reward, done, _ = self.env.step(a)
                eps_rewards.append(reward)
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                reward = torch.tensor(reward, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)
                replay_buffer.append((obs, action, reward, done, next_obs))
                if done:
                    break

                loss = torch.tensor(0.0)
                if len(replay_buffer) >= BATCH_SIZE:
                    # Randomly sample from the replay buffer
                    list = random.sample(replay_buffer, BATCH_SIZE)
                    obss = torch.stack([row[0] for row in list])
                    actions = torch.stack([row[1].long() for row in list])
                    rewards = torch.stack([row[2] for row in list])
                    dones = torch.stack([row[3] for row in list])
                    next_obs = torch.stack([row[4] for row in list])

                    with torch.no_grad():
                        target_qs = (
                            rewards
                            + self.hpam["gamma"]
                            * torch.argmax(self.utility_targ.forward(next_obs)[-1])
                            - self.dcg.utility.forward(obss)[1]
                        )
                    qs = self.dcg.utility.foward(obss)
                    loss = loss + nn.MSELoss()(target_qs, qs.view(-1, 1))

                if j % self.hpam["update_every"] == 0:
                    loss = 1 / self.hpam["update_every"] * loss
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()

                    # Update using polyak averaging for some given interval
                    with torch.no_grad():
                        # Utility network
                        for p_target, p in zip(
                            self.utility_targ.parameters(),
                            self.dcg.utility.parameters(),
                        ):
                            p_target.data.mul_(self.hpam["polyak_const"])
                            p_target.data.add_((1 - self.hpam["polyak_const"]) * p.data)

                        # Payoff network
                        for p_target, p in zip(
                            self.payoff_targ.parameters(), self.dcg.payoff.parameters()
                        ):
                            p_target.data.mul_(self.hpam["polyak_const"])
                            p_target.data.add_((1 - self.hpam["polyak_const"]) * p.data)

                print(f"Episode {j} ended with mean reward {np.mean(eps_rewards)}")
                if j >= self.hpam["batch_size"] * self.hpam["update_every"]:
                    print(f"DQN Loss = ={loss.item()}")
                else:
                    print("---Collecting Experience---")

                mean_epoch_rewards.append(np.mean(eps_rewards))

            print(f"Epoch {i} ended with mean reward {mean_epoch_rewards[-1]}")

        plt.plot(mean_epoch_rewards)

    def save(self):
        # Save the utility and payoff and target networks
        pass

    def load(self):
        # Load the networks
        pass

    def select_action(self, act: torch.Tensor):
        # Should we add epsilon greedy exploration??
        return act.numpy()

    def eval(self, episodes):
        print(f"Evaluating for {episodes} episodes")
        start = time.time()
        total_rewards = []

        for i in range(episodes):
            obs = self.env.reset_world()
            episode_rewards = []
            done = False

            while not done:
                #    if render:
                #        self.env.render()

                obs = torch.tensor(obs, dtype=torch.float32)
                actions, q = self.dcg.forward(obs)
                a = self.select_action(actions)
                next_obs, reward, done, _ = self.env.step(a)
                episode_rewards.append(reward)
                obs = next_observation

            total_rewards.append(np.sum(episode_rewards))
            print(f"Episode - {i} Total Reward - {total_rewards[-1]:.2f}")

        self.env.close()
        print(f"Evaluation Completed in {time.time() - start} seconds")
        print(f"Average episodic reward = {np.mean(total_rewards)}")


if __name__ == "__main__":
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario_name = "simple_spread"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation
    )
