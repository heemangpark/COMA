from smac.env import StarCraft2Env

from arguments import setting_args, coma_args
from runner import Runner

if __name__ == '__main__':
    for i in range(8):
        args = setting_args()
        args = coma_args(args)
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,  # steps to make an action
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('Win rate: {}'.format(win_rate))
            break
        env.close()
