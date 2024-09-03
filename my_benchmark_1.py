from stable_eureka import RLTrainer, make_env, get_logger
from pathlib import Path
import yaml
import imageio
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

if __name__ == '__main__':

    exp_path = Path("/home/utseus22/stable-eureka-chenlunde/experiments/bipedal_walker_llama3/benchmark_1/")

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
    
    def save_gif(env, model, path, episodes=1):
        images = []
        for _ in range(episodes):
            obs, _ = env.reset()  
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)  
                done = terminated or truncated
                images.append(env.render())
        env.close()

        imageio.mimsave(path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)

    eval_env_for_render = gym.make(config['environment']['benchmark'], render_mode="rgb_array")
    eval_env_for_render = RecordVideo(eval_env_for_render, video_folder=str(exp_path / 'code' / 'videos'))

    save_gif(eval_env_for_render, rl_trainer.model, exp_path / 'code' / 'evaluation.gif', episodes=1)
