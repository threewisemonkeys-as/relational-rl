import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Node:
    """Node for Deep Coordination Graph

    Args:
        n_nodes (int): number of nodes in the graph
        n_actions (int): number of actions possible for an agent

    Attributes:
        m (torch.Tensor): Tensor with messages passed to node from every other node
        m (torch.Tensor): Updated tensor with messages passed to node from every other node
    """
    def __init__(self, n_nodes: int, n_actions: int) -> None:
        """[summary]

        Args:
            n_nodes (int): [description]
            n_actions (int): [description]
        """
        self.m = torch.zeros(n_nodes, n_actions)
        self.new_m = torch.zeros(n_nodes, n_actions)


class DCG:
    """[summary]

    Args:
        nodes (list[Node]): List of nodes in the graph
        edges (list[tuple): List of edges as tuples of node indices
        utility_hidden_dims (list[int]): List of hidden dims for utility mlp
        payoff_hidden_dims (list[int]): List of hidden dims for payoff mlp
        obs_dim (int): Size of observation
        n_actions (int): Number of possible actions
        iterations (int): Number of iterations to run message passing for each forward pass
        state_hidden_dim (int): Size of hidden state

    Attributes:
        nodes (list[Node]): List of nodes in the graph
        edges (list[tuple): List of edges as tuples of node indices
        n_actions (int): Number of possible actions
        iterations (int): Number of iterations to run message passing for each forward pass
        n_nodes (int): Number of nodes in the graph
        n_edges (int): Number of edges in the graph
        state (torch.Tensor): Hidden state tensor
        state_encoder (nn.Module): RRN to encode observations into hidden states over time
        utility (nn.Module): MLP to compute the utility of each node from its hidden state
        payoff (nn.Module): MLP to compute the payoff of each edge from the hidden states of nodes
    """
    def __init__(
        self,
        nodes: list[Node],
        edges: list[tuple(int, int)],
        utility_hidden_dims: list[int],
        payoff_hidden_dims: list[int],
        obs_dim: int,
        n_actions: int,
        iterations: int,
        state_hidden_dim: int,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.n_actions = n_actions
        self.iterations = iterations

        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self.actions = torch.zeros(self.n_nodes, self.n_actions)
        self.state = torch.zeros(1, self.n_nodes, state_hidden_dim)
        self.state_encoder = nn.GRU(
            input_size=obs_dim + 1, hidden_size=state_hidden_dim
        )

        # create utility network
        utility_hidden_dims = [state_hidden_dim, *utility_hidden_dims, n_actions]
        self.utility = self._create_mlp(utility_hidden_dims)

        # create payoff network
        payoff_hidden_dims = [
            state_hidden_dim * 2,
            *utility_hidden_dims,
            n_actions ** 2,
        ]
        self.payoff = self._create_mlp(payoff_hidden_dims)

    def q(self, u: torch.Tensor, p: list[list[torch.Tensor]], a: torch.Tensor) -> torch.Tensor:
        """Compute the Q value of given action

        Args:
            u (torch.Tensor): Utility values of each node
            p (list[list[torch.Tensor]]): Payoff values for edges of each node
            a (torch.Tensor): The action assigned to each node

        Returns:
            torch.Tensor: The computed Q value
        """
        # compute utility component
        q = u[[i for i in range(len(a))], a].sum()

        # compute payoff component
        for e_idx, (i, j) in enumerate(self.edges):
            q += p[e_idx][a[i], a[j]]

        return q

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        """Compute the best set of actions given an observation

        Args:
            x (torch.Tensor): The observation

        Returns:
            tuple(torch.Tensor, torch.Tensor): The optimal actions and associated Q value
        """
        # encode observation and previous actions into hidden state
        encoder_input = torch.cat([x.unsqueeze(0), self.actions.unsqueeze(0)], dim=-1)
        _, self.state = self.state_encoder(encoder_input, self.state)
        self.actions = torch.zeros(self.n_nodes, self.n_actions)

        # compute utility values for all nodes
        u = (1 / self.n_nodes) * self.utility(self.state)

        # compute payoff values for all nodes
        p = []
        for i, j in self.edges:
            # compute payoffs for each edge
            p_i_j = (1 / self.n_edges) * self.payoff(
                self.state[[i, j]].view(-1)
            ).view(self.n_actions, self.n_actions)
            p_j_i = (1 / self.n_edges) * self.payoff(
                self.state[[j, i]].view(-1)
            ).view(self.n_actions, self.n_actions)
            # compute avergae of both ways for symetry and permuation invariance
            p[i].append((p_i_j + p_j_i) / 2)
        p = torch.stack(p)

        # run message passing across the graph
        q_max = torch.zeros(1)
        for _ in range(self.iterations):

            # message passing is not backpropogated through
            with torch.no_grad:
                for e_idx, (i, j) in enumerate(self.edges):
                    # compute message for i --> j
                    m_i_j = u[i] + torch.sum(self.nodes[i].m, dim=0) - self.nodes[i].m[j]
                    m_i_j = (m_i_j + p[e_idx].T).T
                    m_i_j = torch.max(m_i_j, dim=0).values
                    m_i_j -= m_i_j.mean()
                    self.nodes[j].new_m[i] = m_i_j

                    # compute message for j --> i
                    m_j_i = u[j] + torch.sum(self.nodes[j].m, dim=0) - self.nodes[j].m[i]
                    m_j_i = m_j_i + p[e_idx]
                    m_i_j = torch.max(m_j_i, dim=0).values
                    m_i_j -= m_j_i.mean()
                    self.nodes[i].new_m[j] = m_j_i

                # update the messages for for all nodes
                for n in self.nodes:
                    n.m = n.new_m

                # compute optimal action for all nodes and q value according to messages
                a = torch.zeros_like(self.actions)
                for i, n_i in enumerate(self.nodes):
                    n_i.a = torch.argmax((u[i] + torch.sum(n_i.m, dim=0)).view(-1), dim=0)
                    a[i] = n_i.a

            # check if computed actions are the best so dar
            q_real = self.q(u, p, a)
            if q_real > q_max:
                q_max = q_real
                self.actions = a

        return self.actions, q_max

    def _create_mlp(self, hidden_dims: list[int]) -> nn.Module:
        """Create MLP with given dimensions

        Args:
            hidden_dims (list[int]): list of hidden dimesions

        Returns:
            nn.Module: The created MLP
        """
        module_list = []
        for i in range(len(hidden_dims) - 2):
            module_list.append(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        return nn.Sequential(*module_list)


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
