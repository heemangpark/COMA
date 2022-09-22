import argparse


def setting_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='coma', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--buffer_size', type=int, default=5000, help='the size of buffer')
    args = parser.parse_args()
    return args


def coma_args(args):
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    args.td_lambda = 0.8
    args.save_cycle = 5000
    args.target_update_cycle = 200
    args.grad_norm_clip = 10

    return args
