import time
from typing import Dict
from stable_baselines3 import PPO
import torch
from pathlib import Path
from stable_eureka.rl_eval_callback import RLEvalCallback


def get_ppo_params(env, config: Dict, log_dir: Path):
    policy_kwargs = {
        'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
        'net_arch': config['architecture']['net_arch'],
        'share_features_extractor': config['architecture'].get('share_features_extractor', False)
    }

    # TODO: set default params (get)
    ppo_params = {
        'policy': config['algo_params'].get('policy', 'MlpPolicy'),
        'env': env,
        'policy_kwargs': policy_kwargs,
        'learning_rate': config['algo_params'].get('learning_rate', 3e-4),
        'n_steps': config['algo_params'].get('n_steps', 2048),
        'batch_size': config['algo_params'].get('batch_size', 64),
        'n_epochs': config['algo_params'].get('n_epochs', 10),
        'gamma': config['algo_params'].get('gamma', 0.99),
        'gae_lambda': config['algo_params'].get('gae_lambda', 0.95),
        'clip_range': config['algo_params'].get('clip_range', 0.2),
        'ent_coef': config['algo_params'].get('ent_coef', 0.0),
        'vf_coef': config['algo_params'].get('vf_coef', 0.5),
        'max_grad_norm': config['algo_params'].get('max_grad_norm', 0.5),
        'seed': config['training'].get('seed', None),
        'device': config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
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

    def run(self, eval_env, eval_seed, logger=None):

        eval_seed.seed(eval_seed)
        info_saver_callback = RLEvalCallback(eval_env)

        model = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][0](**self._params)
        model.learn(total_timesteps=self._config['training']['total_timesteps'], tb_log_name="tensorboard",
                    callback=info_saver_callback)
        model.save(self._log_dir / "model")
        if logger:
            logger.info(f"Training done for {self._log_dir} experiment!")
