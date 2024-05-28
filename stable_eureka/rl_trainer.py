import time
from typing import Dict
from stable_baselines3 import PPO
import torch
from pathlib import Path


def get_ppo_params(env, config: Dict, log_dir: Path):
    policy_kwargs = {
        'activation_fn': getattr(torch.nn, config['architecture']['activation_fn']),
        'net_arch': config['architecture']['net_arch'],
        'share_features_extractor': config['architecture']['share_features_extractor'],
    }

    # TODO: set default params (get)
    ppo_params = {
        'policy': config['algo_params']['policy'],
        'env': env,
        'policy_kwargs': policy_kwargs,
        'learning_rate': config['algo_params']['learning_rate'],
        'n_steps': config['algo_params']['n_steps'],
        'batch_size': config['algo_params']['batch_size'],
        'n_epochs': config['algo_params']['n_epochs'],
        'gamma': config['algo_params']['gamma'],
        'gae_lambda': config['algo_params']['gae_lambda'],
        'clip_range': config['algo_params']['clip_range'],
        'ent_coef': config['algo_params']['ent_coef'],
        'vf_coef': config['algo_params']['vf_coef'],
        'max_grad_norm': config['algo_params']['max_grad_norm'],
        'seed': config['training']['seed'],
        'device': config['training']['device'],
        'tensorboard_log': log_dir
    }

    return ppo_params


class RLTrainer:
    AVAILABLE_ALGOS = {'ppo': (PPO, get_ppo_params)}

    def __init__(self, env, config: Dict, log_dir: Path):
        self._config = config
        self._log_dir = log_dir

        if self._config['algo'] not in RLTrainer.AVAILABLE_ALGOS.keys():
            raise ValueError(f"Algorithm {self._config['algo']} not available. "
                             f"Choose from {RLTrainer.AVAILABLE_ALGOS.keys()}")

        self._params = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][1](env, self._config, self._log_dir)

    def run(self):
        model = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][0](**self._params)
        model.learn(total_timesteps=self._config['training']['total_timesteps'], tb_log_name="tensorboard")
        model.save(self._log_dir / "model")
