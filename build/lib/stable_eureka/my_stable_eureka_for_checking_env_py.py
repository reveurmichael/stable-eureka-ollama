import copy
import time
from pathlib import Path

import numpy as np
import yaml
import os
import shutil
from typing import Dict

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.utils import (read_from_file,
                                 get_code_from_response, append_and_save_to_txt,
                                 indent_code, save_to_txt, save_to_json,
                                 make_env, reflection_component_to_str, read_from_json)
from stable_eureka.rl_trainer import RLTrainer
from gymnasium.envs.registration import register

import torch
from typing import Dict
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3
import torch
from pathlib import Path

class MyStableEurekaForCheckingEnvPy:
    def __init__(self, config_path: str, experiment_datetime: str, iteration: int, sample: int):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self._config = yaml.safe_load(open(config_path, 'r'))

        self._root_path = Path(os.getcwd())
        self._experiment_path = self._root_path / self._config['experiment']['parent'] / self._config['experiment'][
            'name']
        
        self.experiment_datetime = experiment_datetime
        self.iteration = iteration
        self.sample = sample

    def run(self):
        iteration = self.iteration
        sample = self.sample
        log_dir = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}'
        module_name = f"{self._config['experiment']['parent']}.{self._config['experiment']['name']}"
        if self._config['experiment']['use_datetime']:
            module_name += f".{self.experiment_datetime}"

        module_name += f".code.iteration_{iteration}.sample_{sample}.env_code.env"

        register(id=f'iteration_{iteration}_sample_{sample}_env-v0',
                    entry_point=f"{module_name}:{self._config['environment']['class_name']}",
                    max_episode_steps=self._config['environment']['max_episode_steps'])

        env = make_env(env_class=f'iteration_{iteration}_sample_{sample}_env-v0',
                        env_kwargs=self._config['environment'].get('kwargs', None),
                        n_envs=self._config['rl']['training'].get('num_envs', 1),
                        is_atari=self._config['rl']['training'].get('is_atari', False),
                        state_stack=self._config['rl']['training'].get('state_stack', 1),
                        multithreaded=False)
        
        
        _params = RLTrainer.AVAILABLE_ALGOS[self._config['rl']['algo']][1](env, self._config['rl'], log_dir)
        model = RLTrainer.AVAILABLE_ALGOS[self._config['rl']['algo']][0](**_params)
        # model.learn(total_timesteps=self._config['rl']['training']['total_timesteps'])
        model.learn(total_timesteps=10)
        

