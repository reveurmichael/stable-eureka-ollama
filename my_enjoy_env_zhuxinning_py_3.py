import copy
import time
from pathlib import Path

import numpy as np
import yaml
import os
import shutil
from typing import Dict

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.utils import (
    read_from_file,
    get_code_from_response,
    append_and_save_to_txt,
    indent_code,
    save_to_txt,
    save_to_json,
    make_env,
    reflection_component_to_str,
    read_from_json,
)
from stable_eureka.rl_trainer import RLTrainer
from stable_eureka.rl_evaluator import RLEvaluator
from gymnasium.envs.registration import register
from stable_eureka.rl_eval_callback import RLEvalCallback

import torch
from typing import Dict
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3
import torch
from pathlib import Path


class MyEnjoyEnvPy:
    def __init__(
        self,
        experiment_name: str,
        experiment_datetime: str,
    ):

        self._root_path = Path(os.getcwd())
        self.experiment_name = experiment_name
        self.experiment_datetime = experiment_datetime
        self._experiment_path = (
            self._root_path / "experiments" / experiment_name / self.experiment_datetime
        )
        self.log_dir = (
            self._experiment_path
        )
        config_path = self._experiment_path / "config.yaml"

        if not Path(config_path).exists():
            raise ValueError(f"Config file {config_path} not found")

        self._config = yaml.safe_load(open(config_path, "r"))

    def run(self):
        module_name = f"{self._config['experiment']['parent']}.{self._config['experiment']['name']}"
        if self._config["experiment"]["use_datetime"]:
            module_name += f".{self.experiment_datetime}"

        module_name += f".env_code.env"

        register(
            id=f"env-v0",
            entry_point=f"{module_name}:{self._config['environment']['class_name']}",
            max_episode_steps=self._config["environment"]["max_episode_steps"],
        )

        env = make_env(
            env_class=f"env-v0",
            env_kwargs=self._config["environment"].get("kwargs", None),
            n_envs=self._config["rl"]["training"].get("num_envs", 1),
            is_atari=self._config["rl"]["training"].get("is_atari", False),
            state_stack=self._config["rl"]["training"].get("state_stack", 1),
            multithreaded=False,
        )

        _params = RLTrainer.AVAILABLE_ALGOS[self._config["rl"]["algo"]][1](
            env, self._config["rl"], self.log_dir
        )

        env_name = self._config["environment"]["benchmark"]
        env_eval = make_env(
            env_class=env_name,
            env_kwargs=self._config["environment"].get("kwargs", None),
            n_envs=1,
            is_atari=self._config["rl"]["training"].get("is_atari", False),
            state_stack=self._config["rl"]["training"].get("state_stack", 1),
            multithreaded=self._config["rl"]["training"].get("multithreaded", False),
        )

        eval_freq = max(
            1,
            int(
                self._config["rl"]["training"]["total_timesteps"]
                // self._config["rl"]["training"]["num_envs"]
                / self._config["rl"]["training"]["eval"]["num_evals"]
            ),
        )
        info_saver_callback = RLEvalCallback(
            env_eval,
            seed=self._config["rl"]["training"]["eval"]["seed"],
            n_eval_episodes=self._config["rl"]["training"]["eval"]["num_episodes"],
            eval_freq=eval_freq,
            log_path=self.log_dir,
            is_benchmark=True,
            logger=get_logger(),
            name=f"{self.experiment_datetime} | ",
        )

        model = RLTrainer.AVAILABLE_ALGOS[self._config["rl"]["algo"]][0](**_params)
        model.learn(
            total_timesteps=self._config["rl"]["training"]["total_timesteps"],
            tb_log_name="tensorboard",
            callback=info_saver_callback,
        )
        # model.learn(total_timesteps=100)

        model_path = self.log_dir / "model_enjoy_env"
        model.save(model_path)

        evaluator = RLEvaluator(model_path, algo=self._config["rl"]["algo"])
        evaluator.run(
            env_eval,
            seed=self._config["rl"]["evaluation"]["seed"],
            n_episodes=self._config["rl"]["evaluation"]["num_episodes"],
            logger=get_logger(),
            save_gif=True,
        )


if __name__ == "__main__":
    m_checker = MyEnjoyEnvPy(
        experiment_name="bipedal_walker_llama3",
        experiment_datetime="zhuxinning-1021",
    )
    m_checker.run()
