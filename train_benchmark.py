from stable_eureka import RLTrainer, make_env, get_logger
from pathlib import Path
import yaml


if __name__ == '__main__':

    exp_path = Path('/home/rsanchezmo/Projects/stable-eureka/experiments/mountain_car_continuous_llama3/2024-06-08/')

    config = yaml.safe_load(open(exp_path / 'config.yaml', 'r'))
    env_name = f'MountainCarContinuous-v0'

    log_dir = exp_path / 'code' / 'benchmark'
    log_dir.mkdir(parents=True, exist_ok=True)

    benchmark_env = make_env(env_class=config['environment']['benchmark'],
                             env_kwargs=config['environment'].get('kwargs', None),
                             n_envs=config['rl']['training'].get('num_envs', 1),
                             is_atari=config['rl']['training'].get('is_atari', False),
                             state_stack=config['rl']['training'].get('state_stack', 1),
                             multithreaded=config['rl']['training'].get('multithreaded', False))

    eval_env = make_env(env_class=config['environment']['benchmark'],
                        env_kwargs=config['environment'].get('kwargs', None),
                        n_envs=1,
                        is_atari=config['rl']['training'].get('is_atari', False),
                        state_stack=config['rl']['training'].get('state_stack', 1),
                        multithreaded=config['rl']['training'].get('multithreaded', False))

    rl_trainer = RLTrainer(benchmark_env, config=config['rl'], log_dir=log_dir)
    rl_trainer.run(eval_env=eval_env,
                   eval_seed=config['rl']['training']['eval']['seed'],
                   eval_episodes=config['rl']['training']['eval']['num_episodes'],
                   num_evals=config['rl']['training']['eval']['num_evals'],
                   logger=get_logger(),
                   is_benchmark=True)
