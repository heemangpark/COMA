import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    def __init__(self, input_size, args):  # input size -> obs_shape + n_actions + n_agents
        super(RNN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_size, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h


class Critic(nn.Module):
    def __init__(self, input_size, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_size, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, args.n_actions)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)

        return q
