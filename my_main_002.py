from pathlib import Path
from stable_eureka import make_env, get_logger, RLEvaluator
import yaml
import zipfile
import shutil
import os
import torch
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3

exp_path = Path('/home/utseus22/stable-eureka-chenlunde/experiments/bipedal_walker_llama3/2024-08-29-11-18/')
model_path = exp_path / 'code' / 'iteration_0' / 'sample_1' / 'model'

model = PPO.load(model_path)

print("Load OK")
