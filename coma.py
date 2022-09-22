import os.path

import torch

from network import RNN, Critic
from utils import td_lambda_target


class COMA:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self.state_shape + self.obs_shape + self.n_agents + (self.n_actions * self.n_agents * 2)

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents

        """Declare actor, critic, target critic"""
        self.eval_rnn = RNN(actor_input_shape, args)
        self.eval_critic = Critic(critic_input_shape, self.args)
        self.target_critic = Critic(critic_input_shape, self.args)

        """GPU utilize"""
        self.eval_rnn.cuda()
        self.eval_critic.cuda()
        self.target_critic.cuda()

        """pre-trained model loading option"""
        self.model_dir = args.model_dir + '/' + args.map
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                path_coma = self.model_dir + '/critic_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location='cuda:0'))
                self.eval_critic.load_state_dict(torch.load(path_coma, map_location='cuda:0'))
            else:
                raise Exception("No Pre-trained Model Exists")

        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
        self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
        self.args = args

        """
        During learning, maintain an eval_hidden for each agent in each episode.
        During execution, maintain an eval_hidden for each agent.
        """
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        # TODO: mask ???
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)
        u, mask = u.cuda(), mask.cuda()
        q_values = self.train_critic(batch, max_episode_len, train_step)
        action_prob = self.action_prob(batch, max_episode_len, epsilon)
        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        """Advantage"""
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()

    def critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx], \
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        u_onehot = batch['u_onehot'][:, transition_idx]
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        if transition_idx == 0:
            u_onehot_last = torch.zeros_like(u_onehot)
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []
        """state"""
        inputs.append(s)
        inputs_next.append(s_next)
        """observation"""
        inputs.append(obs)
        inputs_next.append(obs_next)
        """last actions"""
        inputs.append(u_onehot_last)
        inputs_next.append(u_onehot)
        """current actions"""
        action_mask = (1 - torch.eye(self.n_agents))
        action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(u_onehot * action_mask.unsqueeze(0))
        inputs_next.append(u_onehot_next * action_mask.unsqueeze(0))
        """one-hot vector for agent number"""
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        """concat and reshape whole inputs"""
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = [obs]
        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        qs, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self.critic_inputs(batch, transition_idx, max_episode_len)
            inputs, inputs_next = inputs.cuda(), inputs_next.cuda()
            q, q_target = self.eval_critic(inputs), self.target_critic(inputs_next)
            q = q.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            qs.append(q)
            q_targets.append(q_target)
        qs = torch.stack(qs, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return qs, q_targets

    def action_prob(self, batch, max_episode_len, epsilon):  # pi^a
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self.actor_inputs(batch, transition_idx)
            inputs = inputs.cuda()
            self.eval_hidden = self.eval_hidden.cuda()
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):  # zero tensor [episode_num, n_agents, rnn_hidden_dim]
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def train_critic(self, batch, max_episode_len, train_step):
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)
        u, u_next, mask = u.cuda(), u_next.cuda(), mask.cuda()
        qs, q_next_target = self.get_q_values(batch, max_episode_len)
        q_values = qs.clone()

        qs = torch.gather(qs, dim=3, index=u).squeeze(3)
        q_next_target = torch.gather(q_next_target, dim=3, index=u_next).squeeze(3)
        targets = td_lambda_target(batch, max_episode_len, q_next_target.cpu(), self.args).cuda()
        td_error = targets.detach() - qs
        masked_td_error = mask * td_error

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return q_values

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_params.pkl')
