import torch


def td_lambda_target(batch, max_ep_len, q_targets, args):
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))

    """n_step_return"""
    n_step_return = torch.zeros((episode_num, max_ep_len, args.n_agents, max_ep_len))
    for tr_idx in range(max_ep_len - 1, -1, -1):
        n_step_return[:, tr_idx, :, 0] = \
            (r[:, tr_idx] + args.gamma * q_targets[:, tr_idx] * terminated[:, tr_idx]) * mask[:, tr_idx]
        for n in range(1, max_ep_len - tr_idx):
            n_step_return[:, tr_idx, :, n] = \
                (r[:, tr_idx] + args.gamma * n_step_return[:, tr_idx + 1, :, n - 1]) * mask[:, tr_idx]

    """lambda return"""
    """lambda_return.shape = (episode_num, max_episode_len，n_agents)"""
    lambda_return = torch.zeros((episode_num, max_ep_len, args.n_agents))
    for tr_idx in range(max_ep_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_ep_len - tr_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, tr_idx, :, n - 1]
        lambda_return[:, tr_idx] = \
            (1 - args.td_lambda) * returns + pow(args.td_lambda, max_ep_len - tr_idx - 1) * \
            n_step_return[:, tr_idx, :, max_ep_len - tr_idx - 1]

    return lambda_return
