from typing import Any, Dict, Optional, Union
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import VecEnv
from pathlib import Path
import json


def evaluate_policy(
        model,
        env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        seed: Optional[int] = None,
        is_benchmark: bool = False):
    seed = seed if seed is not None else np.random.randint(0, 2 ** 32 - 1)

    rewards = []
    ep_lengths = []
    infos = {}
    for idx in range(n_eval_episodes):
        env.seed(seed + idx)
        obs = env.reset()
        done = [False]
        episode_reward = 0.0
        episode_length = 0
        info_master = {}

        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            info = info[0]
            info.pop('TimeLimit.truncated', None)
            info.pop('episode', None)
            info.pop('terminal_observation', None)

            if is_benchmark:
                if 'fitness_score' not in info_master:
                    info_master['fitness_score'] = reward[0]
                else:
                    info_master['fitness_score'] += reward[0]

            for key, value in info.items():
                if key not in info_master:
                    info_master[key] = value
                else:
                    info_master[key] += value

        rewards.append(episode_reward)
        ep_lengths.append(episode_length)

        for key, value in info_master.items():
            if key not in infos:
                infos[key] = [value]
            else:
                infos[key].append(value)

    # compute the mean of the infos
    for key, value in infos.items():
        infos[key] = float(np.mean(value))

    infos['reward'] = float(np.mean(rewards))
    infos['episode_length'] = float(np.mean(ep_lengths))

    return infos


class RLEvalCallback(EventCallback):

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            seed: Optional[int] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[Path] = None,
            is_benchmark: bool = False,
            logger: Optional[Any] = None,
            name: str = '1'
    ):
        super().__init__(None, verbose=0)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

        self.eval_env = eval_env
        self.log_path = log_path
        self.seed = seed
        self.is_benchmark = is_benchmark

        self.results_dict = {}
        self._logger = logger
        self._name = name

    def _init_callback(self):
        ...

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            results = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                seed=self.seed,
                is_benchmark=self.is_benchmark
            )

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(results['reward']))
            self.logger.record("eval/mean_ep_length", float(results['episode_length']))
            self.logger.record("eval/fitness_score", float(results['fitness_score']))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # save the json file
            for key, value in results.items():
                if key not in self.results_dict:
                    self.results_dict[key] = [value]
                else:
                    self.results_dict[key].append(value)

            with open(self.log_path / 'evals.json', 'w') as f:
                json.dump(self.results_dict, f, indent=4)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            if self._logger is not None:
                self._logger.info(f"[{self._name}] Eval at timestep "
                                  f"{self.num_timesteps} with fitness score: {results['fitness_Score']}")

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        if self.callback:
            self.callback.update_locals(locals_)
