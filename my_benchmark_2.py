from stable_eureka import RLTrainer, make_env, get_logger, RLEvaluator
from pathlib import Path
import yaml
import imageio
import gymnasium as gym
import numpy as np

if __name__ == '__main__':

    exp_path = Path(
        "/home/utseus22/stable-eureka-chenlunde/experiments/mountain_car_continuous_llama3/benchmark_2/"
    )

    config = yaml.safe_load(open(exp_path / 'config.yaml', 'r'))

    log_dir = exp_path
    log_dir.mkdir(parents=True, exist_ok=True)

    benchmark_env = make_env(env_class=config['environment']['benchmark'],
                             env_kwargs=config['environment'].get('kwargs', None),
                             n_envs=config['rl']['training'].get('num_envs', 1),
                             is_atari=config['rl']['training'].get('is_atari', False),
                             state_stack=config['rl']['training'].get('state_stack', 1),
                             multithreaded=config['rl']['training'].get('multithreaded', True))

    eval_env = make_env(env_class=config['environment']['benchmark'],
                        env_kwargs=config['environment'].get('kwargs', None),
                        n_envs=1,
                        is_atari=config['rl']['training'].get('is_atari', False),
                        state_stack=config['rl']['training'].get('state_stack', 1),
                        multithreaded=config['rl']['training'].get('multithreaded', True))

    rl_trainer = RLTrainer(benchmark_env, config=config['rl'], log_dir=log_dir)
    rl_trainer.run(eval_env=eval_env,
                   eval_seed=config['rl']['training']['eval']['seed'],
                   eval_episodes=config['rl']['training']['eval']['num_episodes'],
                   num_evals=config['rl']['training']['eval']['num_evals'],
                   logger=get_logger(),
                   is_benchmark=True)

    model_path = exp_path / 'model.zip'
    env_name = config["environment"]["benchmark"]

    env = make_env(env_class=env_name,
                   env_kwargs=config['environment'].get('kwargs', None),
                   n_envs=1,
                   is_atari=config['rl']['training'].get('is_atari', False),
                   state_stack=config['rl']['training'].get('state_stack', 1),
                   multithreaded=config['rl']['training'].get('multithreaded', False))

    evaluator = RLEvaluator(model_path, algo=config['rl']['algo'])
    evaluator.run(env, seed=config['rl']['evaluation']['seed'],
                  n_episodes=config['rl']['evaluation']['num_episodes'],
                  logger=get_logger(), save_gif=True)
