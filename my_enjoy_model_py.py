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
from stable_eureka.rl_evaluator import RLEvaluator
from gymnasium.envs.registration import register

import torch
from typing import Dict
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3
import torch
from pathlib import Path


if __name__ == '__main__':
    exp_path = Path('/home/utseus22/stable-eureka-chenlunde/experiments/bipedal_walker_llama3/benchmark_3/')
    # model_path = exp_path / 'code' / 'iteration_0' / 'sample_1' / 'model'
    model_path = exp_path / 'model'
    model = PPO.load(model_path)
    env_name = f'BipedalWalker-v3'

    vec_env = make_env(env_class=env_name,
                    env_kwargs=None,
                    n_envs=1,
                    is_atari=False,
                    state_stack=1)
    
    obs = vec_env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
    vec_env.close()

    evaluator = RLEvaluator(model_path, algo='ppo')
    evaluator.run(vec_env, seed=5, n_episodes=3, logger=get_logger(), save_gif=True)






