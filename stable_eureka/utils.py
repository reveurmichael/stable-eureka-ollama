import re
from pathlib import Path
import time
import numpy as np
from typing import List, Optional, Dict
import json
import ollama

from stable_baselines3.common.vec_env import VecFrameStack, VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env, make_atari_env


def read_from_file(path: Path) -> str:
    with open(path, "r") as file:
        return file.read()


def get_code_from_response(response: str, regex: List[str]) -> str:
    for reg in regex:
        code = re.search(reg, response, re.DOTALL)
        if code:
            return code.group(1).strip()

    return ''


def append_and_save_to_txt(path: Path, txt: str):
    with open(path, 'a') as file:
        file.write(txt)


def save_to_txt(path: Path, txt: str):
    with open(path, 'w') as file:
        file.write(txt)


def save_to_json(path: Path, data: dict):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def read_from_json(path: Path) -> dict:
    with open(path, 'r') as file:
        return json.load(file)


def indent_code(code: str, signature: Optional[str] = None) -> str:
    indented_code = ''
    if signature:
        indented_code += f'    {signature}\n'
    indented_code += '\n'.join(['    ' + line for line in code.split('\n')])
    return indented_code


def make_env(env_class, env_kwargs, n_envs, is_atari, state_stack, multithreaded: bool = False) -> VecEnv:
    vec_env_cls = SubprocVecEnv if multithreaded and n_envs > 1 else DummyVecEnv

    if is_atari:
        env = make_atari_env(env_id=env_class, env_kwargs=env_kwargs, n_envs=n_envs, vec_env_cls=vec_env_cls)
    else:
        env = make_vec_env(env_id=env_class, env_kwargs=env_kwargs, n_envs=n_envs, vec_env_cls=vec_env_cls)

    if state_stack > 1:
        env = VecFrameStack(env, n_stack=state_stack)

    return env


def reflection_component_to_str(component: Dict):
    txt = ''
    for key, value in component.items():
        txt += f'   {key}: {value}. Max: {np.max(value)} - Mean: {np.mean(value)} - Min: {np.min(value)} \n'
    return txt
