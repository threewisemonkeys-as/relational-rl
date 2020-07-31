from __future__ import annotations

import copy
from itertools import count
from collections import deque
from datetime import datetime
import pdb
import random

import torch
from torch import ne
import torch.nn as nn
import numpy as np


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
        self.m = torch.zeros(n_nodes, n_actions)
        self.new_m = torch.zeros(n_nodes, n_actions)


class DCG(nn.Module):
    """Deep coordination graph

    Reference:
        Bohmer W. et al. Deep Coordination Graphs.
        https://arxiv.org/pdf/1910.00091.pdf

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
        super(DCG, self).__init__()
        self.nodes = nodes
        self.edges = edges
        self.n_actions = n_actions
        self.iterations = iterations
        self.obs_dim = obs_dim
        self.state_hidden_dim = state_hidden_dim

        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self.state_encoder = nn.GRU(
            input_size=obs_dim + n_actions, hidden_size=state_hidden_dim
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

    def q(
        self, u: torch.Tensor, p: list[list[torch.Tensor]], a: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Q value of given action

        Args:
            u (torch.Tensor): Utility values of each node
            p (list[list[torch.Tensor]]): Payoff values for edges of each node
            a (torch.Tensor): The action assigned to each node

        Returns:
            torch.Tensor: The computed Q value
        """
        # Convert onehot to ints
        a = torch.max((a), dim=-1).indices

        # compute utility component
        q = torch.sum(u[[i for i in range(self.n_nodes)], a])

        # compute payoff component
        for e_idx, (i, j) in enumerate(self.edges):
            q += p[e_idx][a[i], a[j]]

        return q

    def compute_u(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the utility value for the current state.

        Returns:
            torch.Tensor: Computed utility value
        """
        return (1 / self.n_nodes) * self.utility(state).view(-1, self.n_nodes, self.n_actions)

    def compute_p(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the payoff value for the current state.

        Returns:
            torch.Tensor: Computed payoff value
        """
        p = []
        for i, j in self.edges:
            # compute payoffs for each edge
            p_i_j = (1 / self.n_edges) * self.payoff(state[0][0][[i, j]].view(-1)).view(
                -1, self.n_actions, self.n_actions
            )
            p_j_i = (1 / self.n_edges) * self.payoff(state[0][0][[j, i]].view(-1)).view(
                -1, self.n_actions, self.n_actions
            )
            # compute avergae of both ways for symetry and permuation invariance
            p.append((p_i_j + p_j_i) / 2)

        return torch.stack(p)

    def message_passing(self, u: torch.Tensor, p: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        # run message passing across the graph
        optimal_a = torch.zeros(self.n_nodes, self.n_actions)
        q_max = torch.zeros(1)
        for _ in range(self.iterations):

            # message passing is not backpropogated through
            with torch.no_grad():
                for e_idx, (i, j) in enumerate(self.edges):
                    # compute message for i --> j
                    m_i_j = (
                        u[i] + torch.sum(self.nodes[i].m, dim=0) - self.nodes[i].m[j]
                    )
                    m_i_j = (m_i_j + p[e_idx].T).T
                    m_i_j = torch.max(m_i_j, dim=0).values
                    m_i_j -= m_i_j.mean()
                    self.nodes[j].new_m[i] = m_i_j

                    # compute message for j --> i
                    m_j_i = (
                        u[j] + torch.sum(self.nodes[j].m, dim=0) - self.nodes[j].m[i]
                    )
                    m_j_i = m_j_i + p[e_idx]
                    m_j_i = torch.max(m_j_i, dim=0).values
                    m_j_i -= m_j_i.mean()
                    self.nodes[i].new_m[j] = m_j_i

                # update the messages for for all nodes
                for n in self.nodes:
                    n.m = n.new_m

                # compute optimal action for all nodes and q value according to messages
                a = torch.zeros(self.n_nodes, self.n_actions)
                for i, n_i in enumerate(self.nodes):
                    n_i.a = torch.argmax(
                        (u[i] + torch.sum(n_i.m, dim=0)).view(-1), dim=0
                    )
                    a[i][n_i.a] = 1

            # check if computed actions are the best so dar
            q_real = self.q(u, p, a)
            if q_real > q_max:
                q_max = q_real
                optimal_a = a

        return optimal_a, q_max

    def forward(self, x: torch.Tensor, state: torch.Tensor, prev_action: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """Compute the best set of actions given an observation

        Args:
            x (torch.Tensor): The observation
            state (torch.Tensor): The state of the system
            prev_action (torch.Tensor): Previous action

        Returns:
            tuple(torch.Tensor, torch.Tensor): The optimal actions and associated Q value
        """
        # encode observation and previous actions into hidden state
        print(state.shape)
        encoder_input = torch.cat([x.view(1, -1, self.obs_dim), prev_action.view(1, -1, self.n_actions)], dim=-1)
        _, state = self.state_encoder(encoder_input, state.view(1, -1, self.state_hidden_dim))
        state = state.view(-1, 1, self.n_nodes, self.state_hidden_dim)

        # compute utility values for all nodes
        u = self.compute_u(state)

        # compute payoff values for all nodes
        p = self.compute_p(state)

        # run message passing and compute optimal actions
        a, q_max = self.message_passing(u, p)

        return a, q_max, u, p, state

    def _create_mlp(self, hidden_dims: list[int]) -> nn.Module:
        """Create MLP with given dimensions

        Args:
            hidden_dims (list[int]): list of hidden dimesions

        Returns:
            nn.Module: The created MLP
        """
        module_list = []
        for i in range(len(hidden_dims) - 2):
            module_list.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        return nn.Sequential(*module_list)


class DCGAgent:
    """Q learning agent using DCGs to generate Q values for a MARL system.

    Args:
        env: MARL Environment
        nodes: List[Node]: List of nodes representing indivisual agents
        edges: List[Tuple[int, int]]: List of tuples denoting edges
    """
    def __init__(self, env, nodes, edges, hp):
        self.env = env
        self.hp = hp
        self.dcg = DCG(
            nodes=nodes,
            edges=edges,
            utility_hidden_dims=hp["utility_hidden_dims"],
            payoff_hidden_dims=hp["payoff_hidden_dims"],
            obs_dim=env.observation_space[0].shape[0],
            n_actions=env.action_space[0].n,
            iterations=hp["iterations"],
            state_hidden_dim=hp["state_hidden_dim"],
        )
        self.target_dcg = copy.deepcopy(self.dcg)
        for p in self.target_dcg.parameters():
            p.requires_grad = False

    def _update_params(self, replay_buffer, optim):
        # Randomly sample from the replay buffer
        sample = random.sample(replay_buffer, self.hp["batch_size"])
        prev_actions, states, obss, actions, rewards, dones, next_obs, next_states = [torch.stack(i) for i in zip(*sample)]

        # Compute target q values
        with torch.no_grad():
            a, _, u, p, _ = self.dcg(next_obs, next_states, actions)
            target_q = rewards + self.hp["gamma"] * self.target_dcg.q(u, p, a) * (~dones)

        # Update params by backprop
        _, _, u, p, _ = self.dcg(obss, states, prev_actions)
        q = self.dcg.q(u, p, actions)
        loss = nn.MSELoss()(target_q, q)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Update using polyak averaging for some given interval
        with torch.no_grad():
            for p_target, p in zip(
                self.target_dcg.parameters(), self.dcg.parameters(),
            ):
                p_target.data.mul_(self.hp["polyak_const"])
                p_target.data.add_((1 - self.hp["polyak_const"]) * p.data)

        return loss.item()

    def train(self, episodes):
        # Setting the optimizers
        optim = torch.optim.Adam(
            [
                {"params": self.dcg.utility.parameters(), "lr": self.hp["utility_lr"]},
                {"params": self.dcg.payoff.parameters(), "lr": self.hp["payoff_lr"]},
            ]
        )

        rewards = []
        replay_buffer = deque(maxlen=self.hp["max_buffer_length"])
        step_count = 0

        for _ in range(episodes):
            eps_rewards = []
            obs = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            d = False

            prev_actions = torch.zeros(self.env.n, env.action_space[0].n)
            state = torch.zeros(1, 1, self.env.n, hp["state_hidden_dim"])

            # Collect trajectory
            print("Collecting Experience ...")
            for j in count():
                step_count += 1
                a, _, _, _, next_state = self.dcg(obs, state, prev_actions)
                next_obs, r, d, _ = self.env.step(a.to(int))
                eps_rewards.append(sum(r))
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                r = torch.tensor(r, dtype=torch.float32).sum()
                d = torch.tensor(d, dtype=torch.float32).sum()
                replay_buffer.append((prev_actions, state, obs, a, r, d, next_obs, next_state))
                prev_actions = a
                state = next_state

                rewards.append(sum(eps_rewards))
                if (step_count >= self.hp["update_after"]) and (
                    (step_count % self.hp["update_every"]) == 0
                ):
                    loss = self._update_params(replay_buffer, optim)
                    print(
                        f"Step Count {step_count}: Mean Ep. Reward (last 10) = {np.mean(rewards[-10:])} | DQN Loss = {loss}"
                    )

                if d or j > self.hp["max_traj_length"]:
                    break

    def save(self):
        # Save the utility and payoff and target networks
        pass

    def load(self):
        # Load the networks
        pass

    def eval(self, episodes):
        print(f"Evaluating for {episodes} episodes")
        start = datetime.now()
        total_rewards = []

        for i in range(episodes):
            obs = self.env.reset_world()
            episode_rewards = []
            done = False

            while not done:
                #    if render:
                #        self.env.render()

                obs = torch.tensor(obs, dtype=torch.float32)
                a, _, _, _ = self.dcg.forward(obs)
                obs, reward, done, _ = self.env.step(a)
                episode_rewards.append(reward)
                obs = torch.tensor(obs)

            total_rewards.append(np.sum(episode_rewards))
            print(f"Episode - {i} Total Reward - {total_rewards[-1]:.2f}")

        self.env.close()
        print(f"Evaluation Completed in {(datetime.now() - start).seconds} seconds")
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

    hp = {
        "utility_hidden_dims": [64],
        "payoff_hidden_dims": [64],
        "state_hidden_dim": 64,
        "iterations": 10,
        "utility_lr": 1e-3,
        "payoff_lr": 1e-3,
        "batch_size": 64,
        "gamma": 0.99,
        "polyack_const": 0.9,
        "max_buffer_length": 2000,
        "max_traj_length": 200,
        "update_after": 100,
        "update_every": 5,
    }

    n_actions = env.action_space[0].n
    nodes = [Node(env.n, n_actions) for _ in range(env.n)]
    edges = [(0, 1), (0, 2), (1, 2)]
    agent = DCGAgent(env, nodes, edges, hp)
    agent.train(20)
