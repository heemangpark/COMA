import numpy as np
import torch

from policy import COMA


class Agents:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = COMA(args)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon):
        inputs = obs.copy()
        avail_actions_idx = np.nonzero(avail_actions)[0]
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        # TODO policy.py -> COMA -> eval_hidden
        # hidden_state = self.policy.eval_hidden[:, agent_num, :]

        """(42,) -> (1,42)"""
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        inputs = inputs.cuda()
        # hidden_state = hidden_state.cuda()