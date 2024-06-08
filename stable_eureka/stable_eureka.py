import copy
import time
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime
import os
import shutil
from typing import Dict

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.ollama_generator import OllamaGenerator
from stable_eureka.openai_generator import OpenAIGenerator
from stable_eureka.utils import (read_from_file,
                                 get_code_from_response, append_and_save_to_txt,
                                 indent_code, save_to_txt, save_to_json,
                                 make_env, reflection_component_to_str, read_from_json)
from stable_eureka.rl_trainer import RLTrainer
from stable_eureka.rl_evaluator import RLEvaluator
from gymnasium.envs.registration import register

import multiprocessing
import torch


class StableEureka:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self._config = yaml.safe_load(open(config_path, 'r'))

        self._root_path = Path(os.getcwd())
        self._experiment_path = self._root_path / self._config['experiment']['parent'] / self._config['experiment'][
            'name']

        self._experiment_datetime = None
        if self._config['experiment']['use_datetime']:
            self._experiment_datetime = datetime.utcnow().strftime('%Y-%m-%d')
            self._experiment_path /= self._experiment_datetime

        self._experiment_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, self._experiment_path / 'config.yaml')

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

        initial_reward_prompt_path = self._root_path / 'envs' / self._config['environment'][
            'name'] / 'initial_reward_prompt.txt'
        if self._config['eureka']['use_initial_reward_prompt'] and initial_reward_prompt_path.exists():
            self._prompts['initial_reward'] = read_from_file(initial_reward_prompt_path)

        self._best_reward = ('', -float('inf'), None, None)  # (reward code, fitness value, iteration, sample)

        self._record_results: Dict = {}

        (self._experiment_path / 'code').mkdir(parents=True, exist_ok=True)  # Code folder
        for iteration in range(self._config['eureka']['iterations']):
            for sample in range(self._config['eureka']['samples']):
                (self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}').mkdir(parents=True,
                                                                                                       exist_ok=True)

                shutil.copy(self._root_path / 'envs' / self._config['environment']['name'] / 'env.py',
                            self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}' / 'env.py')

        torch.multiprocessing.set_start_method('spawn')  # required for multiprocessing

        if self._config['eureka']['backend'] == 'ollama':
            self._llm_generator = OllamaGenerator(model=self._config['eureka']['model'])
        elif self._config['eureka']['backend'] == 'openai':
            self._llm_generator = OpenAIGenerator(model=self._config['eureka']['model'])
        else:
            raise ValueError(f"Backend {self._config['eureka']['backend']} not available. "
                             f"Choose from ['ollama', 'openai']")

    def run(self, verbose: bool = True):
        init_run_time = time.time()
        if verbose:
            logger = get_logger()
        else:
            logger = EmptyLogger()

        logger.info(f"Starting stable-eureka optimization. Iterations: {self._config['eureka']['iterations']}, "
                    f"samples: {self._config['eureka']['samples']}")
        logger.info(f"Using LLM: {self._config['eureka']['model']} with T={self._config['eureka']['temperature']}")

        if self._config['environment']['benchmark'] is not None:
            log_dir = self._experiment_path / 'code' / 'benchmark'
            log_dir.mkdir(parents=True, exist_ok=True)
            # train the benchmark id environment (in a parallel process)
            benchmark_env = make_env(env_class=self._config['environment']['benchmark'],
                                     env_kwargs=self._config['environment'].get('kwargs', None),
                                     n_envs=self._config['rl']['training'].get('num_envs', 1),
                                     is_atari=self._config['rl']['training'].get('is_atari', False),
                                     state_stack=self._config['rl']['training'].get('state_stack', 1),
                                     multithreaded=self._config['rl']['training'].get('multithreaded', False))

            eval_env = make_env(env_class=self._config['environment']['benchmark'],
                                env_kwargs=self._config['environment'].get('kwargs', None),
                                n_envs=1,
                                is_atari=self._config['rl']['training'].get('is_atari', False),
                                state_stack=self._config['rl']['training'].get('state_stack', 1),
                                multithreaded=self._config['rl']['training'].get('multithreaded', False))

            rl_trainer = RLTrainer(benchmark_env, config=self._config['rl'], log_dir=log_dir)

            is_benchmark = True
            process = multiprocessing.Process(target=rl_trainer.run,
                                              args=(eval_env,
                                                    self._config['rl']['training']['eval']['seed'],
                                                    self._config['rl']['training']['eval']['num_episodes'],
                                                    self._config['rl']['training']['eval']['num_evals'],
                                                    logger, is_benchmark))
            process.start()

        for iteration in range(self._config['eureka']['iterations']):
            prompt = self._prompts['initial_system'] + \
                     '\nCoding instructions: ' + self._prompts['coding_instructions'] + \
                     '\nTask description: ' + self._prompts['task_description'] + \
                     '\nEnvironment code:\n' + self._prompts['env_code']

            if iteration == 0 and 'initial_reward' in self._prompts:
                prompt += '\nInitial reward proposal:\n' + self._prompts['initial_reward']
                prompt += ('\nYou must provide a variation from the initial reward proposal! '
                           'This is just a suggestion! It is crucial that you provide the code for '
                           'the reward function using the previous coding tips!')
            else:
                prompt += '\nReward reflection:\n' + self._prompts['reward_reflection']
                if self._config['eureka']['pretraining_with_best_model']:
                    prompt += ('\nThe next training will take the best model weights '
                               'so it reuses some of the relevant information '
                               'from the previous training!')

            save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}' / 'prompt.txt', prompt)

            init_t = time.time()
            rewards = self._llm_generator.generate(
                temperature=self._config['eureka']['temperature'],
                prompt=prompt,
                k=self._config['eureka']['samples'],
                logger=logger)
            end_t = time.time()
            elapsed = end_t - init_t
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info(f"Iteration {iteration}/{self._config['eureka']['iterations'] - 1} - "
                        f"LLM generation time: {elapsed:.2f}s")

            if isinstance(self._llm_generator, OllamaGenerator):
                logger.info(f"Sleeping to avoid CUDA memory issues...")
                time.sleep(5 * 60)  # wait for 5 minutes (to avoid cuda memory issues: remove the model from gpu memory)

            reward_codes = []
            processes = []
            for idx, reward_response in enumerate(rewards):
                save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}'
                            / f'sample_{idx}' / 'llm_response.txt',
                            reward_response)
                code = get_code_from_response(reward_response, self._regex)

                save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}'
                            / f'sample_{idx}' / 'reward_code.txt', code)

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
                        module_name += f".{self._experiment_datetime}"

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

                    eval_env = make_env(env_class=f'iteration_{iteration}_sample_{idx}_env-v0',
                                        env_kwargs=self._config['environment'].get('kwargs', None),
                                        n_envs=1,
                                        is_atari=self._config['rl']['training'].get('is_atari', False),
                                        state_stack=self._config['rl']['training'].get('state_stack', 1),
                                        multithreaded=self._config['rl']['training'].get('multithreaded', False))

                    pretrained_model = None
                    if self._config['eureka'].get('pretraining_with_best_model', False):
                        best_iteration = self._best_reward[2]
                        best_sample = self._best_reward[3]

                        if best_iteration is not None and best_sample is not None:
                            best_model_path = self._experiment_path / 'code' / f'iteration_{best_iteration}' / f'sample_{best_sample}' / 'model.zip'
                            if best_model_path.exists():
                                pretrained_model = best_model_path

                    rl_trainer = RLTrainer(env, config=self._config['rl'], log_dir=log_dir,
                                           pretrained_model=pretrained_model)
                    process = multiprocessing.Process(target=rl_trainer.run,
                                                      args=(eval_env,
                                                            self._config['rl']['training']['eval']['seed'],
                                                            self._config['rl']['training']['eval']['num_episodes'],
                                                            self._config['rl']['training']['eval']['num_evals'],
                                                            logger,))
                    process.start()

                except Exception as e:
                    logger.error(f"Error in training: {e}, for sample {idx}")

                processes.append(process)

            while active_processes := np.sum([process.is_alive() for process in processes if process is not None]):
                logger.info(f"Active processes: {active_processes}")
                time.sleep(20)

            for process in processes:
                if isinstance(process, multiprocessing.Process) and process.is_alive():
                    process.join()

            logger.info("Training loop finished...")

            best_eval = None
            best_fitness = -float('inf')
            best_idx = -1
            for idx in range(self._config['eureka']['samples']):
                log_dir = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{idx}' / 'evals.json'
                if not log_dir.exists():
                    continue

                evals = read_from_json(log_dir)
                fitness_scores = evals['fitness_score']

                if np.max(fitness_scores) > best_fitness:
                    best_fitness = np.max(fitness_scores)
                    best_idx = idx
                    best_eval = copy.deepcopy(evals)

            if best_eval is None:
                logger.info("No successful train found for this iteration... "
                            "Moving to the next one with the same reward reflection as before!")
                continue

            best_reward_code = reward_codes[best_idx]
            self._record_results[iteration] = (best_reward_code, best_fitness)

            # create the reward reflection prompt
            reward_reflection = reflection_component_to_str(best_eval)
            reward_reflection_prompt = (self._prompts['reward_reflection_init'] + reward_reflection + '\n' +
                                        self._prompts['reward_reflection_end'] +
                                        'Stable-Eureka best iteration  (you should modify it!): \n' +
                                        best_reward_code + '\n')

            self._prompts['reward_reflection'] = reward_reflection_prompt

            # update the best reward tuple
            if best_fitness > self._best_reward[1]:
                logger.info(f"New best reward found with fitness score of: {best_fitness}, "
                            f"previous best: {self._best_reward[1]}")
                logger.info(f"Reward code:\n{best_reward_code}")
                self._best_reward = (best_reward_code, best_fitness, iteration, best_idx)

                save_to_json(self._experiment_path / 'code' / 'best_reward.json',
                             {'reward': best_reward_code, 'fitness': best_fitness, 'iteration': iteration,
                              'sample': best_idx})

            save_to_json(self._experiment_path / 'code' / 'best_iteration_rewards.json',
                         self._record_results)

        model_path = self._experiment_path / 'code' / f'iteration_{self._best_reward[2]}' / f'sample_{self._best_reward[3]}' / 'model.zip'
        env_name = f'iteration_{self._best_reward[2]}_sample_{self._best_reward[3]}_env-v0'

        env = make_env(env_class=env_name,
                       env_kwargs=self._config['environment'].get('kwargs', None),
                       n_envs=1,
                       is_atari=self._config['rl']['training'].get('is_atari', False),
                       state_stack=self._config['rl']['training'].get('state_stack', 1),
                       multithreaded=self._config['rl']['training'].get('multithreaded', False))

        evaluator = RLEvaluator(model_path, algo=self._config['rl']['algo'])
        evaluator.run(env, seed=self._config['rl']['evaluation']['seed'],
                      n_episodes=self._config['rl']['evaluation']['num_episodes'],
                      logger=logger, save_gif=self._config['rl']['evaluation']['save_gif'])

        end_run_time = time.time()
        delta_time = end_run_time - init_run_time
        logger.info(f"Stable-Eureka optimization finished in {delta_time:.2f}s ("
                    f"{delta_time / 60:.2f}m) ({delta_time / 3600:.2f}h)")
