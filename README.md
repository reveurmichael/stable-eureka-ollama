# Stable Eureka
Stable Eureka is an iterative llm-based reward designer for reinforcement learning. It integrates
stable-baselines3, open-source LLMs and gym-based environments. This repo is based on [NVIDIA Eureka](https://github.com/eureka-research/Eureka/tree/main).


## Installation

```bash
git clone https://github.com/rsanchezmo/stable-eureka.git
cd stable-eureka
pip install .
```

You must install ollama before running the code:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Available LLMs
You have to pull the LLMs you want to use from the ollama repository. For example, to pull the llama3 LLM:
```bash
ollama pull llama3
```

- llama3
- codellama
- mistral
- phi3
- gemma

## Configuration
You must fill a configuration file with the following structure:
```yaml
eureka:
    model: 'llama3'
    temperature: 0.5
    iterations: 5
    samples: 10

environment:
    name: 'bipedal_walker'
    max_episode_steps: 1600
    class_name: 'BipedalWalker'
    kwargs: null

experiment:
    parent: 'experiments'
    name: 'bipedal_walker'
    use_datetime: true

rl:
    algo: 'ppo'
    algo_params:
        policy: 'MlpPolicy'
        learning_rate: 0.0003
        n_steps: 2048
        batch_size: 64
        n_epochs: 10
        gamma: 0.99
        gae_lambda: 0.95
        clip_range: 0.2
        ent_coef: 0.0
        vf_coef: 0.5
        max_grad_norm: 0.5

    architecture:
        net_arch: {'pi': [128, 128], 'vf': [128, 128]}
        activation_fn: 'Tanh'
        share_features_extractor: false

    training:
        seed: 0
        eval_seed: 5
        total_timesteps: 500_000
        device: 'cuda'
        num_envs: 4
        state_stack: 1
        is_atari: false
```

## Environment
You must provide the env code in a `env.py` file for now. You should include take the step func into a `step.py` file, and must
create a `task_description.txt` file with the task description:

```
envs/
    bipedal_walker/
        env.py
        step.py
        task_description.txt
```

The code will copy the code into the experiments folder and append the reward function to it. The reward function should 
satisfy the signature:
```python
reward, intermediate_reward = self.compute_reward(param1, param2, param3)
```
By doing so, the code will be automatically executed by the experiment runner once the reward function is appended.

You must also implement the `self.compute_fitness_score`, the ground truth reward function that allows to compare between 
environments with different reward functions. You can see several implementations on the environments folder:
```python
fitness_score = self.compute_fitness_score(param1, param2, param3)
```

> [!NOTE] 
> The `compute_fitness_score` returns a part of the total fitness score, which is actually the sum over all the episode. 
> Same as reward is the sum of the intermediate rewards during the episode. If the total fitness score is a binary value such as 1 for success, 
> then you will provide always 0 until the episode ends where it will return a 1.