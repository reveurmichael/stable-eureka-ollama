import time
from pathlib import Path
import yaml
from datetime import datetime
import ollama
import os
import shutil

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.utils import read_from_file, generate_text, get_code_from_response, append_and_save_code, indent_code


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

                         'best_reward': ''
                         }

        (self._experiment_path / 'code').mkdir(parents=True, exist_ok=True)  # Code folder
        for iteration in range(self._config['eureka']['iterations']):
            for sample in range(self._config['eureka']['samples']):
                (self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}').mkdir(parents=True,
                                                                                                       exist_ok=True)

                shutil.copy(self._root_path / 'envs' / self._config['environment']['name'] / 'env.py',
                            self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}' / 'env.py')

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
            prompt = self._prompts['initial_system'] + self._prompts['coding_instructions'] + \
                     '\nTask description: ' + self._prompts['task_description'] + \
                     '\nEnvironment code:\n' + self._prompts['env_code'] + \
                     '\nBest reward:\n' + self._prompts['best_reward']

            init_t = time.time()
            rewards = generate_text(model=self._config['eureka']['model'],
                                    options=ollama_options,
                                    prompt=prompt,
                                    k=self._config['eureka']['samples'])
            end_t = time.time()
            elapsed = end_t - init_t
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info(f"Iteration {iteration}/{self._config['eureka']['iterations']-1} - "
                        f"LLM generation time: {elapsed:.2f}s")

            for idx, reward_response in enumerate(rewards):
                code = get_code_from_response(reward_response, self._regex)
                logger.info(f"Sample {idx}"
                            f"/{self._config['eureka']['samples']-1}")
                logger.info(f"Reward: {code}")
                logger.info("--------------------------------------------------")

                code = indent_code(code, signature='# Generated code by stable-eureka')

                append_and_save_code(self._experiment_path
                                     / 'code'
                                     / f'iteration_{iteration}'
                                     / f'sample_{idx}'
                                     / 'env.py', code)
