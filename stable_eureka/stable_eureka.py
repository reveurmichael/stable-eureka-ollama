import time
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime
import ollama
import os
import shutil
from typing import Dict

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.utils import (read_from_file, generate_text,
                                 get_code_from_response, append_and_save_to_txt,
                                 indent_code, save_to_txt, save_to_json)
from stable_eureka.utils import make_env
from stable_eureka.rl_trainer import RLTrainer

from gymnasium.envs.registration import register

import multiprocessing
import importlib
import torch


class StableEureka:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self._config = yaml.safe_load(open(config_path, 'r'))

        self._root_path = Path(os.getcwd())
        self._experiment_path = self._root_path / self._config['experiment']['parent'] / self._config['experiment'][
            'name']

        if self._config['experiment']['use_datetime']:
            self._experiment_path /= datetime.utcnow().strftime('%Y-%m-%d')

        self._experiment_path.mkdir(parents=True, exist_ok=True)

        self._regex = [
            r'```python(.*?)```'
        ]

        self._prompts = {'initial_system': read_from_file(self._root_path
                                                          / 'stable_eureka'
                                                          / 'prompts'
                                                          / 'initial_system_prompt.txt'),

                         'coding_instructions': read_from_file(self._root_path
                                                               / 'stable_eureka'
                                                               / 'prompts'
                                                               / 'coding_instructions_prompt.txt'),

                         'task_description': read_from_file(self._root_path
                                                            / 'envs'
                                                            / self._config['environment']['name']
                                                            / 'task_description.txt'),

                         'env_code': read_from_file(self._root_path
                                                    / 'envs'
                                                    / self._config['environment']['name']
                                                    / 'step.py'),

                         'reward_reflection_init': read_from_file(self._root_path
                                                                  / 'stable_eureka'
                                                                  / 'prompts'
                                                                  / 'reward_reflection_init_prompt.txt'),

                         'reward_reflection_end': read_from_file(self._root_path
                                                                 / 'stable_eureka'
                                                                 / 'prompts'
                                                                 / 'reward_reflection_end_prompt.txt'),

                         'reward_reflection': ''
                         }

        self._best_reward = ('', -float('inf'))  # (reward code, fitness value)

        self._record_results: Dict = {}

        (self._experiment_path / 'code').mkdir(parents=True, exist_ok=True)  # Code folder
        for iteration in range(self._config['eureka']['iterations']):
            for sample in range(self._config['eureka']['samples']):
                (self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}').mkdir(parents=True,
                                                                                                       exist_ok=True)

                shutil.copy(self._root_path / 'envs' / self._config['environment']['name'] / 'env.py',
                            self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}' / 'env.py')

        torch.multiprocessing.set_start_method('spawn')  # required for multiprocessing

    def run(self, verbose: bool = True):

        if verbose:
            logger = get_logger()
        else:
            logger = EmptyLogger()

        logger.info(f"Starting stable-eureka optimization. Iterations: {self._config['eureka']['iterations']}, "
                    f"samples: {self._config['eureka']['samples']}")
        logger.info(f"Using LLM: {self._config['eureka']['model']} with T={self._config['eureka']['temperature']}")

        ollama_options = ollama.Options(
            temperature=self._config['eureka']['temperature'],
        )

        for iteration in range(self._config['eureka']['iterations']):
            prompt = (self._prompts['initial_system'] +
                      '\nCoding instructions: ' + self._prompts['coding_instructions'] +
                      '\nTask description: ' + self._prompts['task_description'] +
                      '\nEnvironment code:\n' + self._prompts['env_code'] +
                      '\nReward reflection:\n' + self._prompts['reward_reflection'])

            save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}' / 'prompt.txt', prompt)

            init_t = time.time()
            rewards = generate_text(model=self._config['eureka']['model'],
                                    options=ollama_options,
                                    prompt=prompt,
                                    k=self._config['eureka']['samples'])
            end_t = time.time()
            elapsed = end_t - init_t
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info(f"Iteration {iteration}/{self._config['eureka']['iterations'] - 1} - "
                        f"LLM generation time: {elapsed:.2f}s")

            reward_codes = []
            processes = []
            for idx, reward_response in enumerate(rewards):
                code = get_code_from_response(reward_response, self._regex)
                logger.info(f"Sample {idx}"
                            f"/{self._config['eureka']['samples'] - 1}")
                logger.info(f"Reward: \n{code}")
                logger.info("--------------------------------------------------")

                code = indent_code(code, signature='# Generated code by stable-eureka')

                reward_codes.append(code)

                append_and_save_to_txt(self._experiment_path
                                       / 'code'
                                       / f'iteration_{iteration}'
                                       / f'sample_{idx}'
                                       / 'env.py', code)

                log_dir = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{idx}'

                process = None
                try:
                    module_name = f"{self._config['experiment']['parent']}.{self._config['experiment']['name']}"
                    if self._config['experiment']['use_datetime']:
                        module_name += f".{datetime.utcnow().strftime('%Y-%m-%d')}"

                    module_name += f".code.iteration_{iteration}.sample_{idx}.env"

                    register(id=f'iteration_{iteration}_sample_{idx}_env-v0',
                             entry_point=f"{module_name}:{self._config['environment']['class_name']}",
                             max_episode_steps=self._config['environment']['max_episode_steps'])

                    env = make_env(env_class=f'iteration_{iteration}_sample_{idx}_env-v0',
                                   env_kwargs=self._config['environment'].get('kwargs', None),
                                   n_envs=self._config['rl']['training'].get('num_envs', 1),
                                   is_atari=self._config['rl']['training'].get('is_atari', False),
                                   state_stack=self._config['rl']['training'].get('state_stack', 1),
                                   multithreaded=self._config['rl']['training'].get('multithreaded', False))

                    rl_trainer = RLTrainer(env, self._config['rl'], log_dir)
                    process = multiprocessing.Process(target=rl_trainer.run)
                    process.start()

                except Exception as e:
                    logger.error(f"Error in training: {e}, for sample {idx}")

                processes.append(process)

            while active_processes := np.sum([process.is_alive() for process in processes if process is not None]):
                time.sleep(20)
                logger.info(f"Active processes: {active_processes}")

            for process in processes:
                if isinstance(process, multiprocessing.Process) and process.is_alive():
                    process.join()

            logger.info("Training loop finished...")

            save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}' / 'reward_codes.txt',
                        '\n\n'.join(reward_codes))

            # evaluate in the end the fitness and the intermediate rewards
            fitness_values = -np.inf * np.ones(self._config['eureka']['samples'])
            for idx in range(self._config['eureka']['samples']):
                # run training of each reward
                # try to load an agent, if fails, leave the -inf value
                score = 0  # evaluate_agent()
                fitness_values[idx] = score

            # select the best reward among them from fitness value
            best_sample = np.argmax(fitness_values)
            best_value = fitness_values[best_sample]
            best_reward_code = reward_codes[best_sample]

            self._record_results[iteration] = (best_reward_code, best_value)

            # create the reward reflection prompt
            reward_reflection = ''
            reward_reflection_prompt = 'Stable-Eureka best output: \n' + best_reward_code + '\n\n' + \
                                       self._prompts['reward_reflection_init'] + reward_reflection + '\n\n' + \
                                       self._prompts['reward_reflection_end']

            self._prompts['reward_reflection'] = reward_reflection_prompt

            # update the best reward tuple
            if best_value > self._best_reward[1]:
                logger.info(f"New best reward found with fitness score of: {best_value}, "
                            f"previous best: {self._best_reward[1]}")
                logger.info(f"Reward code:\n{best_reward_code}")
                self._best_reward = (best_reward_code, best_value)

                save_to_txt(self._experiment_path / 'code' / 'best_rewards.txt',
                            f'Reward code (score: {best_value}):\n' + best_reward_code + '\n\n')

            save_to_json(self._experiment_path / 'code' / 'best_iteration_rewards.json',
                         self._record_results)
