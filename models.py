# Atharv Sonwane <atharvs.twm@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


class Relational(nn.Module):
    def __init__(self, input_shape, nheads=1, hidden_dim=None, output_dim=None):
        super(Relational, self).__init__()
        self.input_shape = input_shape
        self.nheads = nheads
        self.features = input_shape[-1]
        if hidden_dim is None:
            self.hidden_dim = self.features
        else:
            self.hidden_dim = hidden_dim
        if output_dim is None:
            self.output_dim = self.features
        else:
            self.output_dim = output_dim

        self.q_projection = nn.Linear(self.features, self.hidden_dim)
        self.k_projection = nn.Linear(self.features, self.hidden_dim)
        self.v_projection = nn.Linear(self.features, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self._apply_self_attention(x)
        x = self.output_linear(x)
        return x

    def _apply_self_attention(self, x):
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        q = q.view(*q.shape[:-1], self.nheads, -1).transpose(-2, -3)
        k = k.view(*k.shape[:-1], self.nheads, -1).transpose(-2, -3)
        v = v.view(*v.shape[:-1], self.nheads, -1).transpose(-2, -3)

        d = torch.tensor([self.features], dtype=x.dtype)
        w = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(d), dim=-1)
        scores = torch.matmul(w, v)

        scores = scores.transpose(-2, -3)
        scores = scores.view(*scores.shape[:-2], -1)

        return scores


class RelationalActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        a_dim,
        conv_dims,
        feature_dim,
        lin_dims,
        relational_hidden_dim=None,
        relational_output_dim=None,
    ):
        super(RelationalActorCritic, self).__init__()
        self.obs_shape = obs_shape  # env.observation_space.shape
        self.a_dim = a_dim  # env.action_space.n

        conv_dims.insert(0, obs_shape[0])
        conv_dims.append(feature_dim)
        conv_module_list = []
        for i in range(len(conv_dims) - 1):
            conv_module_list.append(nn.Conv2d(conv_dims[i], conv_dims[i + 1], 2, 1))
            conv_module_list.append(nn.ReLU())
            conv_module_list.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*conv_module_list)

        var = torch.zeros(4, *obs_shape, requires_grad=False)
        var = self.conv(var)
        var = var.flatten(start_dim=-2).transpose(-1, -2)
        c = torch.zeros(*var.shape[:-1], 1)
        var = torch.cat([var, c, c], dim=-1)
        self.relational = Relational(
            tuple(var.shape[-2:]),
            hidden_dim=relational_hidden_dim,
            output_dim=relational_output_dim,
        )

        var = self.relational(var)
        var = torch.max(var, dim=-2).values
        lin_dims.insert(0, var.shape[-1])
        lin_dims.append(a_dim)
        lin_module_list = []
        for i in range(len(lin_dims) - 1):
            lin_module_list.append(nn.Linear(lin_dims[i], lin_dims[i + 1]))
            lin_module_list.append(nn.ReLU())
        self.linear = nn.Sequential(*lin_module_list)
        self.policy_head = nn.Linear(a_dim, a_dim)
        self.baseline_head = nn.Linear(a_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        ncols = x.shape[-1]
        x = x.flatten(start_dim=-2).transpose(-1, -2)
        c = torch.arange(x.shape[-2]).expand(*x.shape[:-1]).to(x.dtype)
        x_coord = (c % ncols).view(*x.shape[:-2], -1, 1)
        y_coord = (c // ncols).view(*x.shape[:-2], -1, 1)
        x = torch.cat([x, x_coord, y_coord], dim=-1)
        x = self.relational(x)
        x = torch.max(x, dim=-2).values
        x = self.linear(x)
        b = self.baseline_head(x)
        pi_logits = self.policy_head(x)
        return pi_logits, b

    def evaluate(self, obs, deterministic=False):
        pi_logits, b = self.forward(obs)
        if deterministic:
            a = torch.distributions.Categorical(logits=pi_logits).sample()
        else:
            a = torch.argmax(pi_logits)
        return a.item(), pi_logits, b

    def sample_action(self, obs, deterministic=False):
        a, _, _ = self.evaluate(obs, deterministic)
        return a


if __name__ == "__main__":
    ac = RelationalActorCritic((3, 210, 160), 2, [8], 8, [8])
    x = torch.randn(4, 3, 210, 160)
    pi_logits, b = ac(x)
    a = ac.sample_action(x)
    print(a, pi_logits.shape, b.shape)
