from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_eureka.utils import save_to_json
import imageio


class RLEvaluator:
    AVAILABLE_ALGOS = {'ppo': PPO}

    def __init__(self, model_path: Path, algo: str):

        if algo not in RLEvaluator.AVAILABLE_ALGOS.keys():
            raise ValueError(f"Algorithm {algo} not available. "
                             f"Choose from {RLEvaluator.AVAILABLE_ALGOS.keys()}")

        self._model = RLEvaluator.AVAILABLE_ALGOS[algo].load(model_path,
                                                             device='cuda' if torch.cuda.is_available() else 'cpu')

        self._log_dir = model_path.parent

    def run(self, env, seed, n_episodes, save_gif=False, logger=None):
        rewards = []
        video_images = []
        fitness_scores = []
        ep_lengths = []
        for i in range(n_episodes):
            env.seed(seed + i)
            state = env.reset()
            if save_gif:
                video_images.append(env.render(mode='rgb_array'))
            episode_reward = 0
            episode_fitness = 0
            done = [False]
            ep_len = 0
            while not done[0]:
                action, _ = self._model.predict(state)
                state, reward, done, infos = env.step(action)
                episode_reward += reward[0]
                ep_len += 1
                episode_fitness += infos[0].get('fitness_score', reward[0])
                if save_gif:
                    video_images.append(env.render(mode='rgb_array'))
            rewards.append(float(episode_reward))
            fitness_scores.append(float(episode_fitness))
            ep_lengths.append(ep_len)

        mean_, min_, max_, std_ = np.mean(rewards), np.min(rewards), np.max(rewards), np.std(rewards)
        results = {
            'seeds': [i for i in range(seed, seed + n_episodes)],
            'reward': {
                'values': rewards,
                'mean': float(mean_),
                'min': float(min_),
                'max': float(max_),
                'std': float(std_)
            }
        }

        mean_, min_, max_, std_ = np.mean(fitness_scores), np.min(fitness_scores), np.max(fitness_scores), np.std(
            fitness_scores)
        results['fitness_score'] = {
            'values': fitness_scores,
            'mean': float(mean_),
            'min': float(min_),
            'max': float(max_),
            'std': float(std_)
        }

        mean_, min_, max_, std_ = np.mean(ep_lengths), np.min(ep_lengths), np.max(ep_lengths), np.std(ep_lengths)
        results['episode_length'] = {
            'values': ep_lengths,
            'mean': float(mean_),
            'min': float(min_),
            'max': float(max_),
            'std': float(std_)
        }

        if logger is not None:
            logger.info(
                f"[REWARD] Mean: {results['reward']['mean']} - Min: {results['reward']['min']} "
                f"- Max: {results['reward']['max']}")
            logger.info(
                f"[FITNESS_SCORE] Mean: {results['fitness_score']['mean']} - Min: {results['fitness_score']['min']} "
                f"- Max: {results['fitness_score']['max']}")
            logger.info(
                f"[EPISODE_LENGTH] Mean: {results['episode_length']['mean']} - Min: {results['episode_length']['min']} "
                f"- Max: {results['episode_length']['max']}")

        save_to_json(self._log_dir / 'eval.json', results)

        # remove one frame out of two to reduce the size of the gif
        video_images = video_images[::2]

        imageio.mimsave(self._log_dir / 'eval.gif', video_images, fps=30, optimize=True)

        return results
