from typing import Dict
from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3
import torch
from pathlib import Path
from stable_eureka.rl_eval_callback import RLEvalCallback


def get_ppo_params(env, config: Dict, log_dir: Path):
    policy_kwargs = None
    if config.get('architecture', None) is not None:
        policy_kwargs = {
            'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
            'net_arch': config['architecture'].get('net_arch', None),
            'share_features_extractor': config['architecture'].get('share_features_extractor', False)
        }

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


def get_sac_params(env, config: Dict, log_dir: Path):
    policy_kwargs = None
    if config.get('architecture', None) is not None:
        policy_kwargs = {
            'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
            'net_arch': config['architecture'].get('net_arch', None),
            'share_features_extractor': config['architecture'].get('share_features_extractor', False)
        }

    sac_params = {
        'policy': config['algo_params'].get('policy', 'MlpPolicy'),
        'policy_kwargs': policy_kwargs,
        'env': env,
        'learning_rate': config['algo_params'].get('learning_rate', 3e-4),
        'buffer_size': config['algo_params'].get('buffer_size', 1_000_000),
        'learning_starts': config['algo_params'].get('learning_starts', 100),
        'batch_size': config['algo_params'].get('batch_size', 256),
        'ent_coef': config['algo_params'].get('ent_coef', 'auto'),
        'gamma': config['algo_params'].get('gamma', 0.99),
        'tau': config['algo_params'].get('tau', 0.005),
        'train_freq': config['algo_params'].get('train_freq', 1),
        'gradient_steps': config['algo_params'].get('gradient_steps', 1),
        'target_update_interval': config['algo_params'].get('target_update_interval', 1),
        'target_entropy': config['algo_params'].get('target_entropy', 'auto'),
        'seed': config['training'].get('seed', None),
        'device': config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'tensorboard_log': log_dir
    }

    return sac_params


def get_dqn_params(env, config: Dict, log_dir: Path):
    policy_kwargs = None
    if config.get('architecture', None) is not None:
        policy_kwargs = {
            'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
            'net_arch': config['architecture'].get('net_arch', None),
            'share_features_extractor': config['architecture'].get('share_features_extractor', False)
        }

    dqn_params = {
        'policy': config['algo_params'].get('policy', 'MlpPolicy'),
        'env': env,
        'policy_kwargs': policy_kwargs,
        'learning_rate': config['algo_params'].get('learning_rate', 1e-4),
        'buffer_size': config['algo_params'].get('buffer_size', 1_000_000),
        'learning_starts': config['algo_params'].get('learning_starts', 100),
        'batch_size': config['algo_params'].get('batch_size', 32),
        'gamma': config['algo_params'].get('gamma', 0.99),
        'tau': config['algo_params'].get('tau', 1.0),
        'train_freq': config['algo_params'].get('train_freq', 4),
        'gradient_steps': config['algo_params'].get('gradient_steps', 1),
        'exploration_fraction': config['algo_params'].get('exploration_fraction', 0.1),
        'exploration_initial_eps': config['algo_params'].get('exploration_initial_eps', 1.0),
        'exploration_final_eps': config['algo_params'].get('exploration_final_eps', 0.05),
        'max_grad_norm': config['algo_params'].get('max_grad_norm', 10),
        'target_update_interval': config['algo_params'].get('target_update_interval', 10000),
        'seed': config['training'].get('seed', None),
        'device': config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'tensorboard_log': log_dir
    }

    return dqn_params


def get_ddpg_params(env, config: Dict, log_dir: Path):
    policy_kwargs = None
    if config.get('architecture', None) is not None:
        policy_kwargs = {
            'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
            'net_arch': config['architecture'].get('net_arch', None),
            'share_features_extractor': config['architecture'].get('share_features_extractor', False)
        }

    ddpg_params = {
        'policy': config['algo_params'].get('policy', 'MlpPolicy'),
        'env': env,
        'policy_kwargs': policy_kwargs,
        'learning_rate': config['algo_params'].get('learning_rate', 1e-4),
        'buffer_size': config['algo_params'].get('buffer_size', 1_000_000),
        'learning_starts': config['algo_params'].get('learning_starts', 100),
        'batch_size': config['algo_params'].get('batch_size', 256),
        'gamma': config['algo_params'].get('gamma', 0.99),
        'tau': config['algo_params'].get('tau', 0.005),
        'train_freq': config['algo_params'].get('train_freq', 1),
        'gradient_steps': config['algo_params'].get('gradient_steps', 1),
        'seed': config['training'].get('seed', None),
        'device': config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'tensorboard_log': log_dir
    }

    return ddpg_params


def get_td3_params(env, config: Dict, log_dir: Path):
    policy_kwargs = None
    if config.get('architecture', None) is not None:
        policy_kwargs = {
            'activation_fn': getattr(torch.nn, config['architecture'].get('activation_fn', 'ReLU')),
            'net_arch': config['architecture'].get('net_arch', None),
            'share_features_extractor': config['architecture'].get('share_features_extractor', False)
        }

    td3_params = {
        'policy': config['algo_params'].get('policy', 'MlpPolicy'),
        'env': env,
        'policy_kwargs': policy_kwargs,
        'learning_rate': config['algo_params'].get('learning_rate', 1e-3),
        'buffer_size': config['algo_params'].get('buffer_size', 1_000_000),
        'learning_starts': config['algo_params'].get('learning_starts', 100),
        'batch_size': config['algo_params'].get('batch_size', 256),
        'gamma': config['algo_params'].get('gamma', 0.99),
        'tau': config['algo_params'].get('tau', 0.005),
        'train_freq': config['algo_params'].get('train_freq', 1),
        'gradient_steps': config['algo_params'].get('gradient_steps', 1),
        'policy_delay': config['algo_params'].get('policy_delay', 2),
        'target_policy_noise': config['algo_params'].get('target_policy_noise', 0.2),
        'target_noise_clip': config['algo_params'].get('target_noise_clip', 0.5),
        'seed': config['training'].get('seed', None),
        'device': config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'tensorboard_log': log_dir
    }

    return td3_params


class RLTrainer:
    AVAILABLE_ALGOS = {'ppo': (PPO, get_ppo_params),
                       'sac': (SAC, get_sac_params),
                       'dqn': (DQN, get_dqn_params),
                       'ddpg': (DDPG, get_ddpg_params),
                       'td3': (TD3, get_td3_params),
                       }

    def __init__(self, env, config: Dict, log_dir: Path, pretrained_model=None, name='1'):
        self._config = config
        self._name = name
        self._log_dir = log_dir

        if self._config['algo'] not in RLTrainer.AVAILABLE_ALGOS.keys():
            raise ValueError(f"Algorithm {self._config['algo']} not available. "
                             f"Choose from {RLTrainer.AVAILABLE_ALGOS.keys()}")

        self._params = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][1](env, self._config, self._log_dir)

        self._pretrained_model = pretrained_model

    def run(self, eval_env, eval_seed, eval_episodes, num_evals, logger=None, is_benchmark=False):

        eval_freq = max(1, int(
            self._config['training']['total_timesteps'] // self._config['training']['num_envs'] / num_evals))
        info_saver_callback = RLEvalCallback(eval_env, seed=eval_seed,
                                             n_eval_episodes=eval_episodes,
                                             eval_freq=eval_freq,
                                             log_path=self._log_dir,
                                             is_benchmark=is_benchmark,
                                             logger=logger,
                                             name=self._name)

        if self._pretrained_model is None:
            model = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][0](**self._params)
        else:
            model = RLTrainer.AVAILABLE_ALGOS[self._config['algo']][0].load(path=self._pretrained_model,
                                                                            **self._params)

        if self._config['training'].get('torch_compile', False):
            torch.set_float32_matmul_precision('high')
            model.policy = torch.compile(model.policy)

        model.learn(total_timesteps=self._config['training']['total_timesteps'], tb_log_name="tensorboard",
                    callback=info_saver_callback)
        model.save(self._log_dir / "model")
        if logger:
            if is_benchmark:
                logger.info(f"Training done for benchmark!")
            else:
                logger.info(f"Training done for {self._log_dir} experiment!")
