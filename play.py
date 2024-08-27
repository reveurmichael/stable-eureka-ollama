from pathlib import Path
from stable_eureka import make_env, get_logger, RLEvaluator
import yaml


if __name__ == '__main__':

    exp_path = Path('/home/utseus22/stable-eureka-zhuxinning/experiments/bipedal_walker_llama3/2024-08-22-07-52/')

    model_path = exp_path / 'code' / f'iteration_1' / f'sample_4' / 'model.zip'
    # model_path = exp_path / 'code' / 'benchmark' / 'model.zip'

    config = yaml.safe_load(open(exp_path / 'config.yaml', 'r'))
    env_name = f'BipedalWalker'

    env = make_env(env_class=env_name,
                   env_kwargs=config['environment'].get('kwargs', None),
                   n_envs=1,
                   is_atari=config['rl']['training'].get('is_atari', False),
                   state_stack=config['rl']['training'].get('state_stack', 1),
                   multithreaded=config['rl']['training'].get('multithreaded', False))

    evaluator = RLEvaluator(model_path, algo=config['rl']['algo'])
    evaluator.run(env, seed=config['rl']['evaluation']['seed'],
                  n_episodes=config['rl']['evaluation']['num_episodes'],
                  logger=get_logger(), save_gif=config['rl']['evaluation']['save_gif'])
