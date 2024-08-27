from pathlib import Path
from stable_eureka import make_env, get_logger, RLEvaluator
import yaml
import zipfile
import shutil
import os
import torch

def extract_model_zip(zip_file_path, extract_to='.'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_checkpoint(checkpoint_path, extracted_dir='.'):
    # 解压模型文件
    extract_model_zip(checkpoint_path, extracted_dir)

    # 寻找解压后的真实模型文件路径
    for root, dirs, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith('policy.pth'):  # 模型文件通常以 .pth 或 .pt 结尾
                real_model_path = os.path.join(root, file)
                break

    # 加载并处理状态字典
    checkpoint = torch.load(real_model_path, map_location='cpu')
    keys_list = list(checkpoint['state_dict'].keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            new_key = key.replace('orig_mod.', '')
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    # 清理临时解压的文件
    shutil.rmtree(extracted_dir)

    return checkpoint

if __name__ == '__main__':
    exp_path = Path('/home/utseus22/stable-eureka-zhuxinning/experiments/bipedal_walker_llama3/2024-08-22-07-52/')
    model_path = exp_path / 'code' / 'iteration_1' / 'sample_4' / 'model.zip'
    config = yaml.safe_load((exp_path / 'config.yaml').open('r'))
    env_name = 'BipedalWalker-v3'

    env = make_env(env_class=env_name,
                   env_kwargs=config['environment'].get('kwargs', None),
                   n_envs=1,
                   is_atari=config['rl']['training'].get('is_atari', False),
                   state_stack=config['rl']['training'].get('state_stack', 1),
                   multithreaded=config['rl']['training'].get('multithreaded', False))

    # 加载并处理检查点
    processed_checkpoint = process_checkpoint(str(model_path), extracted_dir='./tmp')

    # 创建评估器，并使用处理后的检查点
    evaluator = RLEvaluator(processed_checkpoint, algo=config['rl']['algo'])
    evaluator.run(env, seed=config['rl']['evaluation']['seed'],
                  n_episodes=config['rl']['evaluation']['num_episodes'],
                  logger=get_logger(), save_gif=config['rl']['evaluation']['save_gif'])