import torch
import torch.nn as nn


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
        self.utility = nn.Sequential(*module_list)

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

    def __init__(self, env, graph):
        self.env = env
        self.dcg = DCG(graph["nodes"], graph["neighbours"], [128], [256], 18, 5, 10, 128)

    def train(self, epochs):
        raise NotImplementedError


if __name__ == "__main__":
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario_name = "simple_spread"
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
